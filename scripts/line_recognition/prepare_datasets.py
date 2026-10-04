"""Create line crops and references from line-detection ground truth."""

import csv
import json
import math
import os
import shutil
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from scripts.recognition.prepare_handwritten_essay import source_image_path


TEXT_CATEGORIES = {"pupil_text", "pupil_comment", "teacher_comment"}


def bbox_from_annotation(annotation, width, height):
    x, y, box_width, box_height = annotation["bbox"]
    left = max(0, math.floor(x))
    top = max(0, math.floor(y))
    right = min(width, math.ceil(x + box_width))
    bottom = min(height, math.ceil(y + box_height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def masked_line_crop(image, annotation, bbox):
    """Crop a line while whitening neighbouring lines outside its GT hull."""
    left, top, right, bottom = bbox
    crop = image.crop(bbox)
    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)
    segmentation = annotation.get("segmentation") or []
    parts = segmentation if segmentation and isinstance(segmentation[0], list) else [segmentation]
    for part in parts:
        points = [
            (float(part[index]) - left, float(part[index + 1]) - top)
            for index in range(0, len(part) - 1, 2)
        ]
        if len(points) >= 3:
            draw.polygon(points, fill=255)
    mask = mask.filter(ImageFilter.MaxFilter(9))
    return Image.composite(crop, Image.new("RGB", crop.size, "white"), mask)


def indexed_images(folder):
    result = {}
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            if path.name in result:
                raise ValueError(f"Duplicate source image name: {path.name}")
            result[path.name] = path
    return result


def essay_references(config):
    pages = json.loads(config["page_annotations"].read_text(encoding="utf-8"))
    references = {}
    for page_name, page in pages.items():
        lines = [
            line
            for block in page.get("blocks", [])
            for line in block.get("lines", [])
        ] + list(page.get("free_lines", []))
        lines.sort(key=lambda line: line.get("order", 0))
        for line_order, line in enumerate(lines):
            words = sorted(line.get("words", []), key=lambda word: word.get("order", 0))
            text = " ".join(str(word.get("text", "")).strip() for word in words).strip()
            references[(page_name, line_order)] = (text, words)
    return references


def school_references(config):
    source = json.loads(config["source_annotations"].read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in source["categories"]}
    references = {}
    grouped = {}
    for annotation in source["annotations"]:
        if categories.get(annotation["category_id"]) not in TEXT_CATEGORIES:
            continue
        group_id = annotation.get("group_id")
        text = str((annotation.get("attributes") or {}).get("translation", "")).strip()
        if group_id is None or not text:
            continue
        segmentation = annotation.get("segmentation") or []
        parts = segmentation if segmentation and isinstance(segmentation[0], list) else [segmentation]
        xs = [value for part in parts for value in part[0::2]]
        x_center = sum(xs) / len(xs) if xs else 0
        grouped.setdefault((annotation["image_id"], group_id), []).append(
            (x_center, text, annotation)
        )
    for key, words in grouped.items():
        ordered = sorted(words, key=lambda item: item[0])
        references[key] = (
            " ".join(text for _x, text, _annotation in ordered),
            [annotation for _x, _text, annotation in ordered],
        )
    return references


def build_records(name, config):
    lines = json.loads(config["line_annotations"].read_text(encoding="utf-8"))
    images = {item["id"]: item for item in lines["images"]}
    if config["prepare"] == "handwritten_essay_lines":
        references = essay_references(config)
        paths = {
            item["id"]: source_image_path(config["folder"], item["file_name"])
            for item in lines["images"]
        }
        def reference(annotation, image):
            return references.get(
                (image["file_name"], annotation.get("attributes", {}).get("line_order")),
                ("", []),
            )
    elif config["prepare"] == "school_notebooks_lines":
        references = school_references(config)
        source_paths = indexed_images(config["folder"])
        paths = {
            item["id"]: source_paths.get(Path(item["file_name"]).name)
            for item in lines["images"]
        }
        def reference(annotation, image):
            return references.get(
                (image["id"], annotation.get("attributes", {}).get("group_id")),
                ("", []),
            )
    else:
        raise ValueError(f"Unsupported line dataset preparation: {name}")

    records = []
    for annotation in lines["annotations"]:
        image = images[annotation["image_id"]]
        text, words = reference(annotation, image)
        source_path = paths[image["id"]]
        if not text or source_path is None or not source_path.is_file():
            continue
        records.append((source_path, image["file_name"], annotation, text, words))
    return records


def word_polygons(word):
    if word.get("polygon"):
        return [[coordinate for point in word["polygon"] for coordinate in point]]
    segmentation = word.get("segmentation") or []
    return segmentation if segmentation and isinstance(segmentation[0], list) else [segmentation]


def compose_word_line(image, words):
    crops = []
    for word in words:
        polygons = [part for part in word_polygons(word) if len(part) >= 6]
        if not polygons:
            continue
        coordinates = [value for part in polygons for value in part]
        annotation = {
            "bbox": [
                min(coordinates[0::2]),
                min(coordinates[1::2]),
                max(coordinates[0::2]) - min(coordinates[0::2]),
                max(coordinates[1::2]) - min(coordinates[1::2]),
            ],
            "segmentation": polygons,
        }
        bbox = bbox_from_annotation(annotation, image.width, image.height)
        if bbox is not None:
            crops.append(masked_line_crop(image, annotation, bbox))
    if not crops:
        return None
    height = max(crop.height for crop in crops)
    gap = max(4, round(statistics.median(crop.height for crop in crops) * 0.2))
    width = sum(crop.width for crop in crops) + gap * (len(crops) - 1)
    line = Image.new("RGB", (width, height), "white")
    left = 0
    for crop in crops:
        line.paste(crop, (left, height - crop.height))
        left += crop.width + gap
    return line


def prepare_line_dataset(name, config):
    records = build_records(name, config)
    prepared_dir = config["prepared_dir"]
    temporary_dir = prepared_dir.with_name(f"{prepared_dir.name}.tmp")
    root = config["download_dir"].resolve()
    if root not in temporary_dir.resolve().parents or root not in prepared_dir.resolve().parents:
        raise RuntimeError("Prepared line-recognition paths must stay inside the dataset root")
    shutil.rmtree(temporary_dir, ignore_errors=True)
    images_dir = temporary_dir / "images"
    images_dir.mkdir(parents=True)
    rows = []
    opened_path = None
    opened_image = None
    try:
        for source_path, page_name, annotation, text, words in records:
            if source_path != opened_path:
                if opened_image is not None:
                    opened_image.close()
                opened_image = Image.open(source_path).convert("RGB")
                opened_path = source_path
            line_image = compose_word_line(opened_image, words)
            if line_image is None:
                continue
            filename = f"{len(rows):06d}.jpg"
            line_image.save(images_dir / filename, quality=95)
            rows.append({
                "image": filename,
                "text": text,
                "page": page_name,
                "order": annotation.get("attributes", {}).get("line_order", annotation["id"]),
            })
    finally:
        if opened_image is not None:
            opened_image.close()

    with (temporary_dir / "labels.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "text", "page", "order"])
        writer.writeheader()
        writer.writerows(rows)
    (temporary_dir / ".preparation_version").write_text(
        str(config["preparation_version"]), encoding="utf-8"
    )
    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)
    os.replace(temporary_dir, prepared_dir)
    print(f"Prepared {len(rows)} line-recognition samples: {prepared_dir}")
