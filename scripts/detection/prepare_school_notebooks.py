"""Prepare the school_notebooks_RU detection validation split."""

import argparse
import json
import os
import shutil
from pathlib import Path


DEFAULT_SOURCE = Path(r"C:\benchmark\school_notebooks_RU")
DEFAULT_OUTPUT = DEFAULT_SOURCE / "benchmark_validation"
TEXT_CATEGORIES = {"pupil_text", "pupil_comment", "teacher_comment"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the school_notebooks_RU validation split for benchmarking."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite already prepared images.",
    )
    return parser.parse_args()


def segmentation_bbox(segmentation):
    parts = segmentation if isinstance(segmentation[0], list) else [segmentation]
    coordinates = [value for part in parts for value in part]
    xs = coordinates[0::2]
    ys = coordinates[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def convert_annotations(source_file: Path):
    with source_file.open("r", encoding="utf-8") as file:
        source = json.load(file)

    category_names = {item["id"]: item["name"] for item in source["categories"]}
    text_category_ids = {
        category_id
        for category_id, name in category_names.items()
        if name in TEXT_CATEGORIES
    }
    if {category_names[item] for item in text_category_ids} != TEXT_CATEGORIES:
        raise ValueError(
            f"Expected categories {sorted(TEXT_CATEGORIES)}, got "
            f"{sorted(category_names.values())}"
        )

    annotations = []
    for source_annotation in source["annotations"]:
        if source_annotation["category_id"] not in text_category_ids:
            continue
        segmentation = source_annotation.get("segmentation")
        if not segmentation:
            continue
        bbox = segmentation_bbox(segmentation)
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        annotation = {
            "id": len(annotations),
            "image_id": source_annotation["image_id"],
            "category_id": 0,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
            "segmentation": segmentation,
        }
        if "attributes" in source_annotation:
            annotation["attributes"] = source_annotation["attributes"]
        annotations.append(annotation)

    return {
        "categories": [{"id": 0, "name": "text"}],
        "images": source["images"],
        "annotations": annotations,
    }


def source_images(source_dir: Path):
    images = {}
    for path in source_dir.rglob("*"):
        if not path.is_file() or path.name.startswith("._"):
            continue
        filename = path.name
        if filename in images:
            raise ValueError(f"Duplicate image name in source directory: {filename}")
        images[filename] = path
    return images


def prepare_validation_images(
    source_dir: Path,
    images,
    output_dir: Path,
    force: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    available = source_images(source_dir)
    missing = sorted(
        image["file_name"] for image in images if image["file_name"] not in available
    )
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} validation images in {source_dir}: {missing[:5]}"
        )

    for image in images:
        filename = image["file_name"]
        destination = output_dir / filename
        if destination.is_file() and not force:
            continue
        if destination.exists():
            destination.unlink()
        try:
            os.link(available[filename], destination)
        except OSError:
            shutil.copy2(available[filename], destination)


def prepare_dataset(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT, force=False):
    source = Path(source)
    output = Path(output)
    annotation_source = source / "annotations_val.json"
    image_source = source / "images"
    image_dir = output / "images"
    annotation_output = output / "annotations.json"
    if not annotation_source.is_file() or not image_source.is_dir():
        raise FileNotFoundError(
            f"Extract images.zip manually into {source}. Expected "
            f"{annotation_source} and {image_source}"
        )
    if not force and annotation_output.is_file() and image_dir.is_dir():
        with annotation_output.open("r", encoding="utf-8") as file:
            prepared = json.load(file)
        expected = len(prepared.get("images", []))
        actual = sum(1 for path in image_dir.iterdir() if path.is_file())
        if expected > 0 and actual == expected:
            print(f"Skip preparation: already completed ({output})")
            return

    converted = convert_annotations(annotation_source)
    prepare_validation_images(image_source, converted["images"], image_dir, force)

    output.mkdir(parents=True, exist_ok=True)
    with annotation_output.open("w", encoding="utf-8") as file:
        json.dump(converted, file, ensure_ascii=False, indent=2)

    print(f"Images: {len(converted['images'])}")
    print(f"Text annotations: {len(converted['annotations'])}")
    print(f"Image directory: {image_dir}")
    print(f"Annotations: {annotation_output}")


def main():
    args = parse_args()
    prepare_dataset(args.source, args.output, args.force)


if __name__ == "__main__":
    main()
