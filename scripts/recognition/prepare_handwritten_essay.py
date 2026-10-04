import csv
import json
import math
import os
import shutil
from pathlib import Path

from PIL import Image

from configs.recognition.benchmark_config import DATASETS
from utils.dataset_downloads import download_dataset_files


DATASET_NAME = "handwritten_essay"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def source_image_path(folder, annotation_name):
    path = Path(annotation_name)
    parts = path.stem.split("_")
    if len(parts) < 3 or parts[0] != "test":
        raise ValueError(f"Unexpected handwritten essay page name: {annotation_name}")
    return folder.joinpath(*parts[1:-1], f"{parts[-1]}{path.suffix}")


def iter_words(page):
    for block in page.get("blocks", []):
        for line in block.get("lines", []):
            yield from line.get("words", [])


def prepared_dataset_is_complete(config):
    labels_path = config["labels"]
    images_dir = config["images_dir"]
    if not labels_path.is_file() or not images_dir.is_dir():
        return False
    expected_version = config.get("preparation_version")
    if expected_version is not None:
        version_path = config["prepared_dir"] / ".preparation_version"
        if not version_path.is_file() or version_path.read_text(encoding="utf-8").strip() != str(expected_version):
            return False
    with labels_path.open("r", encoding="utf-8", newline="") as file:
        expected = sum(1 for _ in csv.DictReader(file))
    actual = sum(
        1 for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    return expected > 0 and expected == actual


def prepare_imagefolder_dataset(name, config):
    prepared_dir = config["prepared_dir"]
    temp_dir = prepared_dir.with_name(f"{prepared_dir.name}.tmp")
    root = config["download_dir"].resolve()
    if root not in temp_dir.resolve().parents or root not in prepared_dir.resolve().parents:
        raise RuntimeError("Prepared dataset paths must stay inside the dataset root")
    shutil.rmtree(temp_dir, ignore_errors=True)
    images_dir = temp_dir / "images"
    images_dir.mkdir(parents=True)

    with config["source_labels"].open("r", encoding="utf-8-sig", newline="") as file:
        source_rows = list(
            csv.DictReader(
                file,
                delimiter=config.get("source_delimiter", ","),
                fieldnames=config.get("source_fieldnames"),
            )
        )
    image_column = config.get("source_image_column", "filename")
    if not source_rows or not {image_column, "text"}.issubset(source_rows[0]):
        raise ValueError(f"Unexpected labels format: {config['source_labels']}")

    rows = []
    for index, row in enumerate(source_rows):
        relative_path = Path(row[image_column])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe image path in labels: {row[image_column]}")
        source = config["folder"] / relative_path
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = images_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        rows.append(
            {
                "image": relative_path.as_posix(),
                "text": row["text"],
                "page": "",
                "order": index,
            }
        )

    with (temp_dir / "labels.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["image", "text", "page", "order"])
        writer.writeheader()
        writer.writerows(rows)

    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)
    os.replace(temp_dir, prepared_dir)
    print(f"Prepared {len(rows)} recognition samples: {prepared_dir}")


def indexed_source_images(source_dir):
    images = {}
    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.name in images:
            raise ValueError(f"Duplicate image name in source directory: {path.name}")
        images[path.name] = path
    return images


def annotation_bbox(segmentation):
    parts = (
        segmentation
        if segmentation and isinstance(segmentation[0], list)
        else [segmentation]
    )
    coordinates = [value for part in parts if part for value in part]
    if len(coordinates) < 6:
        return None
    xs = coordinates[0::2]
    ys = coordinates[1::2]
    return (
        math.floor(min(xs)),
        math.floor(min(ys)),
        math.ceil(max(xs)),
        math.ceil(max(ys)),
    )


def prepare_coco_words_dataset(name, config):
    source = json.loads(config["source_annotations"].read_text(encoding="utf-8"))
    category_names = {item["id"]: item["name"] for item in source["categories"]}
    text_categories = set(config["text_categories"])
    available_categories = set(category_names.values())
    if not text_categories.issubset(available_categories):
        raise ValueError(
            f"Missing text categories in {config['source_annotations']}: "
            f"{sorted(text_categories - available_categories)}"
        )

    image_records = {item["id"]: item for item in source["images"]}
    source_images = indexed_source_images(config["folder"])
    prepared_dir = config["prepared_dir"]
    temp_dir = prepared_dir.with_name(f"{prepared_dir.name}.tmp")
    root = config["download_dir"].resolve()
    if root not in temp_dir.resolve().parents or root not in prepared_dir.resolve().parents:
        raise RuntimeError("Prepared dataset paths must stay inside the dataset root")
    shutil.rmtree(temp_dir, ignore_errors=True)
    images_dir = temp_dir / "images"
    images_dir.mkdir(parents=True)

    rows = []
    opened_image_id = None
    opened_image = None
    try:
        for annotation in source["annotations"]:
            if category_names.get(annotation["category_id"]) not in text_categories:
                continue
            text = str(
                (annotation.get("attributes") or {}).get(config["text_attribute"], "")
            ).strip()
            bbox = annotation_bbox(annotation.get("segmentation"))
            if not text or bbox is None:
                continue

            image_id = annotation["image_id"]
            if image_id != opened_image_id:
                if opened_image is not None:
                    opened_image.close()
                record = image_records[image_id]
                source_path = source_images.get(Path(record["file_name"]).name)
                if source_path is None:
                    raise FileNotFoundError(record["file_name"])
                opened_image = Image.open(source_path).convert("RGB")
                opened_image_id = image_id

            left = max(0, bbox[0])
            top = max(0, bbox[1])
            right = min(opened_image.width, bbox[2])
            bottom = min(opened_image.height, bbox[3])
            if right <= left or bottom <= top:
                continue

            filename = f"{len(rows):06d}.jpg"
            opened_image.crop((left, top, right, bottom)).save(
                images_dir / filename, quality=95
            )
            rows.append(
                {
                    "image": filename,
                    "text": text,
                    "page": image_records[image_id]["file_name"],
                    "order": annotation.get("id", len(rows)),
                }
            )
    finally:
        if opened_image is not None:
            opened_image.close()

    with (temp_dir / "labels.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["image", "text", "page", "order"])
        writer.writeheader()
        writer.writerows(rows)

    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)
    os.replace(temp_dir, prepared_dir)
    print(f"Prepared {len(rows)} recognition samples: {prepared_dir}")


def prepare_dataset(name=DATASET_NAME, config=None):
    config = config or DATASETS[name]
    download_dataset_files(name, config)
    if config.get("format") == "old_orthography":
        from scripts.prepare_old_orthography import prepare

        prepare(config)
        return
    if prepared_dataset_is_complete(config):
        print(f"Skip preparation: already completed ({config['prepared_dir']})")
        return
    if config.get("format") == "line_crops":
        from scripts.line_recognition.prepare_datasets import prepare_line_dataset

        prepare_line_dataset(name, config)
        return
    if config.get("format") == "imagefolder":
        prepare_imagefolder_dataset(name, config)
        return
    if config.get("format") == "coco_words":
        prepare_coco_words_dataset(name, config)
        return

    prepared_dir = config["prepared_dir"]
    temp_dir = prepared_dir.with_name(f"{prepared_dir.name}.tmp")
    root = config["download_dir"].resolve()
    if root not in temp_dir.resolve().parents or root not in prepared_dir.resolve().parents:
        raise RuntimeError("Prepared dataset paths must stay inside the dataset root")
    shutil.rmtree(temp_dir, ignore_errors=True)
    images_dir = temp_dir / "images"
    images_dir.mkdir(parents=True)

    pages = json.loads(config["page_annotations"].read_text(encoding="utf-8"))
    rows = []
    crop_id = 0
    for page_name in sorted(pages):
        image_path = source_image_path(config["folder"], page_name)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            for word in iter_words(pages[page_name]):
                polygon = word.get("polygon") or []
                if len(polygon) < 3:
                    continue
                xs = [float(point[0]) for point in polygon]
                ys = [float(point[1]) for point in polygon]
                left = max(0, math.floor(min(xs)))
                top = max(0, math.floor(min(ys)))
                right = min(source.width, math.ceil(max(xs)))
                bottom = min(source.height, math.ceil(max(ys)))
                if right <= left or bottom <= top:
                    continue

                filename = f"{crop_id:06d}.png"
                source.crop((left, top, right, bottom)).save(images_dir / filename)
                rows.append(
                    {
                        "image": filename,
                        "text": word.get("text", ""),
                        "page": page_name,
                        "order": word.get("order", crop_id),
                    }
                )
                crop_id += 1

    labels_path = temp_dir / "labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "text", "page", "order"])
        writer.writeheader()
        writer.writerows(rows)

    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)
    os.replace(temp_dir, prepared_dir)
    print(f"Prepared {len(rows)} word crops: {prepared_dir}")


def main():
    prepare_dataset()


if __name__ == "__main__":
    main()
