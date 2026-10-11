"""Build a compact ground-truth overview of the Leibniz layout classes."""

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "models" / "rfdetr-segmentation-leibniz-dataset" / "test"
OUTPUT = ROOT / "benchmark_results" / "previews" / "rfdetr_leibniz_all_classes.jpg"
COLORS = {
    1: "#e6194b", 2: "#3cb44b", 3: "#ffe119", 4: "#4363d8",
    5: "#f58231", 6: "#911eb4", 7: "#00bcd4", 8: "#f032e6", 9: "#9a6324",
}


def main():
    data = json.loads((TEST_DIR / "_annotations.coco.json").read_text(encoding="utf-8"))
    images = {item["id"]: item for item in data["images"]}
    names = {item["id"]: item["name"] for item in data["categories"] if item["id"]}
    by_image = defaultdict(list)
    for annotation in data["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    uncovered = set(names)
    selected = []
    while uncovered:
        image_id, annotations = max(
            by_image.items(),
            key=lambda item: len({a["category_id"] for a in item[1]} & uncovered),
        )
        newly_covered = {a["category_id"] for a in annotations} & uncovered
        if not newly_covered:
            break
        selected.append(image_id)
        uncovered -= newly_covered
        by_image.pop(image_id)

    selected_items = [(TEST_DIR, data, image_id) for image_id in selected]
    if uncovered:
        train_dir = TEST_DIR.parent / "train"
        train_data = json.loads((train_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
        train_images = {item["id"]: item for item in train_data["images"]}
        candidate_ids = {a["image_id"] for a in train_data["annotations"] if a["category_id"] in uncovered}
        candidate_ids = [image_id for image_id in candidate_ids if (train_dir / train_images[image_id]["file_name"]).is_file()]
        for image_id in candidate_ids:
            selected_items.append((train_dir, train_data, image_id))
            uncovered -= {a["category_id"] for a in train_data["annotations"] if a["image_id"] == image_id}
            if not uncovered:
                break

    font = ImageFont.load_default(size=18)
    panels = []
    for image_dir, source, image_id in selected_items:
        source_images = {item["id"]: item for item in source["images"]}
        info = source_images[image_id]
        image = Image.open(image_dir / info["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        for annotation in [a for a in source["annotations"] if a["image_id"] == image_id]:
            category_id = annotation["category_id"]
            color = COLORS[category_id]
            for polygon in annotation.get("segmentation", []):
                points = list(zip(polygon[0::2], polygon[1::2]))
                if len(points) >= 3:
                    draw.polygon(points, fill=color + "35", outline=color, width=4)
            x, y, width, height = annotation["bbox"]
            label = f"{category_id}: {names[category_id]}"
            box = draw.textbbox((x, y), label, font=font, stroke_width=2)
            draw.rectangle(box, fill="#000000cc")
            draw.text((x, y), label, fill=color, font=font, stroke_width=1, stroke_fill="black")
        image.thumbnail((700, 700))
        panels.append(image)

    legend_height = 52 + 30 * len(names)
    width = max(700, max(panel.width for panel in panels))
    canvas = Image.new("RGB", (width, legend_height + sum(panel.height + 20 for panel in panels)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), "RF-DETR Leibniz: ground-truth zone classes", fill="black", font=font)
    y = 48
    for category_id, name in names.items():
        draw.rectangle((20, y, 42, y + 18), fill=COLORS[category_id])
        draw.text((52, y), f"{category_id}: {name}", fill="black", font=font)
        y += 30
    top = legend_height
    for panel in panels:
        canvas.paste(panel, ((width - panel.width) // 2, top))
        top += panel.height + 20
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, quality=92)
    print(f"Saved: {OUTPUT}")
    print(f"Pages: {len(selected_items)}; covered classes: {sorted(set(names) - uncovered)}")


if __name__ == "__main__":
    main()
