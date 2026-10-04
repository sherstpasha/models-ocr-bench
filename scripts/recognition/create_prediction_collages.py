from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "benchmark_results"
DATA_ROOT = Path(os.environ.get("OCR_BENCH_DATA_ROOT", r"C:\benchmark"))
OUTPUT_ROOT = RESULTS_ROOT / "prediction_collages"

DATASETS = {
    "handwritten_essay": {
        "name": "Handwritten Essay",
        "source": "https://huggingface.co/datasets/sherstpasha/handwritten_essay",
    },
    "school_notebooks_ru": {
        "name": "School Notebooks RU",
        "source": "https://huggingface.co/datasets/ai-forever/school_notebooks_RU",
    },
    "russian_old_orthography": {
        "name": "Russian Old Orthography OCR",
        "source": "https://huggingface.co/datasets/nevmenandr/russian-old-orthography-ocr",
    },
}

LEVELS = {
    "word": {
        "results": RESULTS_ROOT / "recognition",
        "images": {
            "handwritten_essay": DATA_ROOT / "handwritten_essay" / "recognition_validation" / "images",
            "school_notebooks_ru": DATA_ROOT / "school_notebooks_RU" / "recognition_validation" / "images",
            "russian_old_orthography": DATA_ROOT / "russian_old_orthography_ocr" / "word_recognition" / "images",
        },
        "title": "Распознавание слов: сравнение предсказаний",
    },
    "line": {
        "results": RESULTS_ROOT / "line_recognition",
        "images": {
            "handwritten_essay": DATA_ROOT / "handwritten_essay" / "line_recognition_validation" / "images",
            "school_notebooks_ru": DATA_ROOT / "school_notebooks_RU" / "line_recognition_validation" / "images",
            "russian_old_orthography": DATA_ROOT / "russian_old_orthography_ocr" / "line_recognition" / "images",
        },
        "title": "Распознавание строк: сравнение предсказаний",
    },
}


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["image"]: row for row in csv.DictReader(stream)}


def collect(level: str) -> tuple[list[str], dict[str, dict[str, dict[str, dict[str, str]]]]]:
    result_root: Path = LEVELS[level]["results"]
    summary_path = result_root / "summary.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as stream:
        models = [row["model"] for row in csv.DictReader(stream)]
    data: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    for model in models:
        model_data: dict[str, dict[str, dict[str, str]]] = {}
        for dataset in DATASETS:
            path = result_root / model / f"{dataset}_{model}.csv"
            model_data[dataset] = read_csv(path) if path.exists() else {}
        data[model] = model_data
    return models, data


def select_examples(
    level: str,
    models: list[str],
    data: dict[str, dict[str, dict[str, dict[str, str]]]],
    count: int,
    seed: int,
) -> list[tuple[str, str | None]]:
    selected: list[tuple[str, str | None]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        available = [set(data[model][dataset]) for model in models if data[model][dataset]]
        common = set.intersection(*available) if available else set()
        image_dir: Path = LEVELS[level]["images"][dataset]
        candidates = sorted(name for name in common if image_dir.is_dir() and (image_dir / name).is_file())
        if len(candidates) < count:
            print(f"Skip {level}/{dataset}: no complete local example; using dash")
            chosen = [None] * count
        else:
            chosen = random.Random(seed + dataset_index).sample(candidates, count)
        selected.extend((dataset, name) for name in chosen)
    return selected


def save_preview(path: Path, output: Path, width: int, height: int) -> None:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=88, optimize=True)


def markdown_text(text: str) -> str:
    return " ".join((text or "∅").split()).replace("|", "\\|")


def render(level: str, examples_per_dataset: int, seed: int) -> Path:
    models, data = collect(level)
    examples = select_examples(level, models, data, examples_per_dataset, seed)
    assets_dir = OUTPUT_ROOT / "assets" / level
    lines = ["### Примеры предсказаний", ""]
    headers = ["Модель"]
    for column, (dataset, filename) in enumerate(examples):
        if filename is None:
            headers.append(
                f"[**{DATASETS[dataset]['name']}**]({DATASETS[dataset]['source']})<br>—"
            )
            continue
        image_path: Path = LEVELS[level]["images"][dataset] / filename
        asset_name = f"{dataset}_{Path(filename).stem}.jpg"
        save_preview(
            image_path,
            assets_dir / asset_name,
            520 if level == "line" else 280,
            150 if level == "line" else 110,
        )
        reference = next(
            (data[model][dataset][filename].get("reference", "") for model in models if filename in data[model][dataset]),
            "",
        )
        headers.append(
            f"[**{DATASETS[dataset]['name']}**]({DATASETS[dataset]['source']})<br>"
            f'<img src="assets/{level}/{asset_name}" width="{520 if level == "line" else 280}"><br>'
            f"**GT:** {markdown_text(reference)}"
        )
    lines.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
    for model in models:
        cells = [f"**{model}**"]
        for dataset, filename in examples:
            row = data[model][dataset].get(filename, {}) if filename is not None else {}
            cells.append(markdown_text(row.get("prediction", "—")) if row else "—")
        lines.append("| " + " | ".join(cells) + " |")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_ROOT / f"{level}_recognition_predictions.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {output} ({len(models)} models, {len(examples)} examples)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic OCR prediction comparison collages")
    parser.add_argument("--level", choices=["word", "line", "all"], default="all")
    parser.add_argument("--examples-per-dataset", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    levels = LEVELS if args.level == "all" else [args.level]
    for level in levels:
        render(level, args.examples_per_dataset, args.seed)


if __name__ == "__main__":
    main()
