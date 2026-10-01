import argparse
import json
from pathlib import Path

from PIL import Image


DEFAULT_ROOT = Path(r"C:\benchmark\handwritten_essay_v1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the handwritten_essay_v1 test split to COCO."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def polygon_bbox(polygon):
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def main():
    args = parse_args()
    # Mendeley test/ contains train_*.png; the corrected split annotation is test_page.json.
    source_file = args.root / "test_page.json"
    image_dir = args.root / "images_train"
    output_dir = args.root / "benchmark_test"
    output_file = output_dir / "annotations.json"

    with source_file.open("r", encoding="utf-8") as file:
        pages = json.load(file)

    images = []
    annotations = []
    for image_id, filename in enumerate(sorted(pages)):
        image_path = image_dir / filename
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
            }
        )

        page = pages[filename]
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    polygon = word.get("polygon")
                    if not polygon or len(polygon) < 3:
                        continue
                    bbox = polygon_bbox(polygon)
                    if bbox[2] <= 0 or bbox[3] <= 0:
                        continue
                    annotations.append(
                        {
                            "id": len(annotations),
                            "image_id": image_id,
                            "category_id": 0,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                            "segmentation": [
                                [coordinate for point in polygon for coordinate in point]
                            ],
                            "attributes": {"transcription": word.get("text", "")},
                        }
                    )

    converted = {
        "categories": [{"id": 0, "name": "text"}],
        "images": images,
        "annotations": annotations,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(converted, file, ensure_ascii=False, indent=2)

    print(f"Images: {len(images)}")
    print(f"Text annotations: {len(annotations)}")
    print(f"Annotations: {output_file}")


if __name__ == "__main__":
    main()
