import argparse
import json
import shutil
import zipfile
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
        help="Overwrite already extracted images.",
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


def archive_images(archive: zipfile.ZipFile):
    images = {}
    for member in archive.infolist():
        if member.is_dir() or member.filename.startswith("__MACOSX/"):
            continue
        filename = Path(member.filename).name
        if filename in images:
            raise ValueError(f"Duplicate image name in archive: {filename}")
        images[filename] = member
    return images


def extract_validation_images(
    archive_file: Path,
    images,
    output_dir: Path,
    force: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_file) as archive:
        members = archive_images(archive)
        missing = sorted(
            image["file_name"] for image in images if image["file_name"] not in members
        )
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} validation images in {archive_file}: {missing[:5]}"
            )

        for image in images:
            filename = image["file_name"]
            destination = output_dir / filename
            if destination.is_file() and not force:
                continue
            with archive.open(members[filename]) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def main():
    args = parse_args()
    annotation_source = args.source / "annotations_val.json"
    archive_source = args.source / "images.zip"
    if not annotation_source.is_file() or not archive_source.is_file():
        raise FileNotFoundError(
            f"Expected annotations_val.json and images.zip in {args.source}"
        )

    converted = convert_annotations(annotation_source)
    image_dir = args.output / "images"
    extract_validation_images(
        archive_source,
        converted["images"],
        image_dir,
        args.force,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    annotation_output = args.output / "annotations.json"
    with annotation_output.open("w", encoding="utf-8") as file:
        json.dump(converted, file, ensure_ascii=False, indent=2)

    print(f"Images: {len(converted['images'])}")
    print(f"Text annotations: {len(converted['annotations'])}")
    print(f"Image directory: {image_dir}")
    print(f"Annotations: {annotation_output}")


if __name__ == "__main__":
    main()
