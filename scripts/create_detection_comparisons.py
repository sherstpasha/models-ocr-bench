from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from configs.detection.benchmark_config import DATASETS as WORD_DATASET_CONFIGS
from configs.line_detection.benchmark_config import DATASETS as LINE_DATASET_CONFIGS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "benchmark_results"
OUTPUT_ROOT = RESULTS_ROOT / "prediction_collages"

LEVELS = {
    "word_detection": {
        "results": RESULTS_ROOT / "detection",
        "datasets": WORD_DATASET_CONFIGS,
        "selected": ["russian_old_orthography", "handwritten_essay", "TotalText"],
    },
    "line_detection": {
        "results": RESULTS_ROOT / "line_detection",
        "datasets": LINE_DATASET_CONFIGS,
        "selected": ["russian_old_orthography", "school_notebooks_ru", "gota_hovratt_seg"],
    },
}

DATASET_INFO = {
    "russian_old_orthography": (
        "Russian Old Orthography OCR",
        "https://huggingface.co/datasets/nevmenandr/russian-old-orthography-ocr",
    ),
    "handwritten_essay": (
        "Handwritten Essay",
        "https://huggingface.co/datasets/sherstpasha/handwritten_essay",
    ),
    "TotalText": ("Total-Text", "https://github.com/cs-chan/Total-Text-Dataset"),
    "school_notebooks_ru": (
        "School Notebooks RU",
        "https://huggingface.co/datasets/ai-forever/school_notebooks_RU",
    ),
    "gota_hovratt_seg": (
        "Göta Hovrätt Segmentation",
        "https://huggingface.co/datasets/Riksarkivet/gota_hovratt_seg",
    ),
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def model_order(result_root: Path) -> list[str]:
    with (result_root / "summary.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        return [row["model"] for row in csv.DictReader(stream)]


def prediction_path(result_root: Path, model: str, dataset: str) -> Path:
    return result_root / model / f"{dataset}_{model}_predictions.json"


def prediction_map(path: Path) -> dict[str, list[dict]]:
    payload = read_json(path)
    predictions = payload.get("predictions", payload)
    if not isinstance(predictions, dict):
        raise ValueError(f"Unsupported prediction format: {path}")
    return predictions


def choose_common_image(result_root: Path, models: list[str], dataset: str, seed: int) -> str | None:
    available: list[set[str]] = []
    for model in models:
        path = prediction_path(result_root, model, dataset)
        if path.exists():
            available.append(set(prediction_map(path)))
    common = set.intersection(*available) if available else set()
    candidates = sorted(common)
    if not candidates:
        print(f"Skip {dataset}: no complete local example; using dash")
        return None
    return random.Random(seed).choice(candidates)


def points_from_shape(shape) -> list[tuple[float, float]]:
    while isinstance(shape, list) and len(shape) == 1 and isinstance(shape[0], list):
        shape = shape[0]
    if not isinstance(shape, list) or not shape:
        return []
    if isinstance(shape[0], (int, float)):
        return [(float(shape[i]), float(shape[i + 1])) for i in range(0, len(shape) - 1, 2)]
    if isinstance(shape[0], list) and len(shape[0]) >= 2:
        return [(float(point[0]), float(point[1])) for point in shape]
    return []


def item_polygons(item: dict) -> list[list[tuple[float, float]]]:
    value = item.get("polygon") or item.get("segmentation")
    if value:
        direct = points_from_shape(value)
        if direct:
            return [direct]
        polygons = [points_from_shape(part) for part in value if isinstance(part, list)]
        polygons = [polygon for polygon in polygons if polygon]
        if polygons:
            return polygons
    bbox = item.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1, third, fourth = map(float, bbox)
        # Saved predictions use xyxy. COCO GT uses xywh and is converted before this function.
        return [[(x1, y1), (third, y1), (third, fourth), (x1, fourth)]]
    return []


def ground_truth(annotation_path: Path, filename: str) -> list[dict]:
    coco = read_json(annotation_path)
    image = next((item for item in coco.get("images", []) if item.get("file_name") == filename), None)
    if image is None:
        return []
    output = []
    for item in coco.get("annotations", []):
        if item.get("image_id") != image.get("id"):
            continue
        copied = dict(item)
        if not copied.get("segmentation") and copied.get("bbox"):
            x, y, width, height = copied["bbox"]
            copied["bbox"] = [x, y, x + width, y + height]
        output.append(copied)
    return output


def source_image_path(dataset_config: dict, filename: str) -> Path | None:
    folder = Path(dataset_config["folder"])
    direct = folder / filename
    if direct.is_file():
        return direct
    if dataset_config.get("filename_prefix") == "test_" and filename.startswith("test_"):
        path = Path(filename)
        parts = path.stem.split("_")
        if len(parts) >= 3:
            nested = folder.joinpath(*parts[1:-1], f"{parts[-1]}{path.suffix}")
            if nested.is_file():
                return nested
    return None


def render_overlay(image_path: Path, gt: list[dict], predictions: list[dict], output: Path) -> None:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
    line_width = max(3, round(min(image.size) / 280))
    for items, fill, outline in (
        (gt, (0, 176, 80, 38), (0, 150, 65, 235)),
        (predictions, (255, 32, 32, 45), (230, 20, 20, 235)),
    ):
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for item in items:
            for polygon in item_polygons(item):
                if len(polygon) >= 3:
                    draw.polygon(polygon, fill=fill)
                    draw.line(polygon + [polygon[0]], fill=outline, width=line_width, joint="curve")
                elif len(polygon) == 2:
                    draw.line(polygon, fill=outline, width=line_width, joint="curve")
        image = Image.alpha_composite(image, overlay)
    image.thumbnail((640, 640), Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, quality=78, optimize=True, progressive=True)


def render(level: str, seed: int) -> Path:
    config = LEVELS[level]
    result_root: Path = config["results"]
    models = model_order(result_root)
    selected = config["selected"]
    chosen = {
        dataset: choose_common_image(result_root, models, dataset, seed + index)
        for index, dataset in enumerate(selected)
    }
    for dataset, filename in list(chosen.items()):
        dataset_config = config["datasets"][dataset]
        if (
            filename is None
            or not Path(dataset_config["annotations"]).is_file()
            or source_image_path(dataset_config, filename) is None
        ):
            chosen[dataset] = None
    cache = {
        (model, dataset): prediction_map(prediction_path(result_root, model, dataset))
        for model in models
        for dataset in selected
        if prediction_path(result_root, model, dataset).exists()
    }
    lines = [
        "### Примеры предсказаний",
        "",
        "Зелёным показана эталонная разметка, красным — предсказания модели.",
        "",
    ]
    headers = ["Модель"]
    for dataset in selected:
        name, source = DATASET_INFO[dataset]
        filename = chosen[dataset]
        headers.append(f"[**{name}**]({source})<br>{f'`{filename}`' if filename else '—'}")
    lines.extend([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ])
    assets_dir = OUTPUT_ROOT / "assets" / level
    for model in models:
        cells = [f"**{model}**"]
        for dataset in selected:
            filename = chosen[dataset]
            dataset_config = config["datasets"][dataset]
            predictions = cache.get((model, dataset))
            if filename is None or predictions is None:
                cells.append("—")
                continue
            image_path = source_image_path(dataset_config, filename)
            annotation_path = Path(dataset_config["annotations"])
            if image_path is None or not annotation_path.is_file():
                cells.append("—")
                continue
            asset_name = f"{model}_{dataset}_{Path(filename).stem}.jpg"
            render_overlay(
                image_path,
                ground_truth(annotation_path, filename),
                predictions.get(filename, []),
                assets_dir / asset_name,
            )
            cells.append(f'<img src="assets/{level}/{asset_name}" width="300">')
        lines.append("| " + " | ".join(cells) + " |")
    output = OUTPUT_ROOT / f"{level}_predictions.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {output} ({len(models)} models, {len(selected)} datasets)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render compact detection prediction comparison tables")
    parser.add_argument("--level", choices=[*LEVELS, "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    levels = LEVELS if args.level == "all" else [args.level]
    for level in levels:
        render(level, args.seed)


if __name__ == "__main__":
    main()
