from __future__ import annotations

import argparse
import csv
import difflib
import os
import random
from pathlib import Path

from PIL import Image, ImageOps


csv.field_size_limit(2**31 - 1)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "benchmark_results"
DATA_ROOT = Path(os.environ.get("OCR_BENCH_DATA_ROOT", r"C:\benchmark"))
OUTPUT_ROOT = RESULTS_ROOT / "prediction_collages"

PREFERRED_EXAMPLES = {
    ("line", "rus_xviii_printed_geography"): "images/00023.png",
    ("word", "russian_old_orthography"): "0002613.jpg",  # «посѣтительницъ»
}

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
    "yenisei_gov_reports_hwr": {
        "name": "YeniseiGovReports-HWR",
        "source": "https://huggingface.co/datasets/sherstpasha/YeniseiGovReports-HWR",
    },
    "yenisei_gov_reports_prt": {
        "name": "YeniseiGovReports-PRT",
        "source": "https://huggingface.co/datasets/sherstpasha/YeniseiGovReports-PRT",
    },
    "cyrillic_handwriting": {
        "name": "Cyrillic Handwriting",
        "source": "https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset",
    },
    "donkeysmall_printed_15random": {
        "name": "DonkeySmallOCR Printed",
        "source": "https://huggingface.co/datasets/sherstpasha/DonkeySmallOCR-Numbers-Printed-15random",
    },
    "ru_ocr_benchmark_hard": {
        "name": "RU OCR Benchmark Hard",
        "source": "https://huggingface.co/datasets/vladlinv/ru-ocr-benchmark-hard",
    },
    "rus_xviii_printed_geography": {
        "name": "Russian XVIII Printed Geography",
        "source": "https://huggingface.co/datasets/dsmchr/rus_xviii_printed_geography",
    },
}

LEVELS = {
    "word": {
        "results": RESULTS_ROOT / "recognition",
        "images": {
            "handwritten_essay": DATA_ROOT / "handwritten_essay" / "recognition_validation" / "images",
            "school_notebooks_ru": DATA_ROOT / "school_notebooks_RU" / "recognition_validation" / "images",
            "russian_old_orthography": DATA_ROOT / "russian_old_orthography_ocr" / "word_recognition" / "images",
            "yenisei_gov_reports_hwr": DATA_ROOT / "YeniseiGovReports-HWR" / "recognition_test" / "images",
            "yenisei_gov_reports_prt": DATA_ROOT / "YeniseiGovReports-PRT" / "recognition_test" / "images",
            "cyrillic_handwriting": DATA_ROOT / "cyrillic-handwriting-dataset" / "recognition_test" / "images",
            "donkeysmall_printed_15random": DATA_ROOT / "DonkeySmallOCR-Numbers-Printed-15random" / "recognition_test" / "images",
        },
        "datasets": ["handwritten_essay", "school_notebooks_ru", "russian_old_orthography", "yenisei_gov_reports_hwr", "yenisei_gov_reports_prt", "cyrillic_handwriting", "donkeysmall_printed_15random"],
        "title": "Распознавание слов: сравнение предсказаний",
    },
    "line": {
        "results": RESULTS_ROOT / "line_recognition",
        "images": {
            "handwritten_essay": DATA_ROOT / "handwritten_essay" / "line_recognition_validation" / "images",
            "school_notebooks_ru": DATA_ROOT / "school_notebooks_RU" / "line_recognition_validation" / "images",
            "russian_old_orthography": DATA_ROOT / "russian_old_orthography_ocr" / "line_recognition" / "images",
            "ru_ocr_benchmark_hard": DATA_ROOT / "ru-ocr-benchmark-hard" / "line_recognition_test" / "images",
            "rus_xviii_printed_geography": DATA_ROOT / "rus_xviii_printed_geography" / "line_recognition_test" / "images",
        },
        "datasets": ["handwritten_essay", "school_notebooks_ru", "russian_old_orthography", "ru_ocr_benchmark_hard", "rus_xviii_printed_geography"],
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
        for dataset in LEVELS[level]["datasets"]:
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
    for dataset_index, dataset in enumerate(LEVELS[level]["datasets"]):
        available = [set(data[model][dataset]) for model in models if data[model][dataset]]
        common = set.intersection(*available) if available else set()
        image_dir: Path = LEVELS[level]["images"][dataset]
        candidates = sorted(name for name in common if image_dir.is_dir() and (image_dir / name).is_file())
        if len(candidates) < count:
            print(f"Skip {level}/{dataset}: no complete local example; using dash")
            chosen = [None] * count
        else:
            preferred = PREFERRED_EXAMPLES.get((level, dataset))
            if count == 1 and preferred in candidates:
                chosen = [preferred]
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


def math_text(text: str) -> str:
    """Escape arbitrary OCR text for MathJax text mode and Markdown tables."""
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "&": r"\&",
        "|": r"\|",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    return r"\text{" + "".join(replacements.get(char, char) for char in text) + "}"


def colored_math(text: str, rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    return f"${{\\color[RGB]{{{red},{green},{blue}}}{math_text(text)}}}$"


def colored_prediction(prediction: str, reference: str) -> str:
    """Highlight OCR edits with GitHub-compatible MathJax colors."""
    prediction = " ".join((prediction or "").split())
    reference = " ".join((reference or "").split())
    if not prediction:
        return colored_math("∅", (179, 92, 92))

    parts = []
    matcher = difflib.SequenceMatcher(None, reference, prediction, autojunk=False)
    for operation, ref_start, ref_end, pred_start, pred_end in matcher.get_opcodes():
        predicted = prediction[pred_start:pred_end]
        expected = reference[ref_start:ref_end]
        if operation == "equal":
            parts.append(colored_math(predicted, (79, 138, 91)))
        elif operation in {"replace", "insert"}:
            parts.append(colored_math(predicted, (179, 92, 92)))
        elif operation == "delete":
            parts.append(f"<s>{colored_math(expected, (179, 92, 92))}</s>")
    # Keep adjacent inline-math fragments from becoming an accidental `$$`
    # display-math delimiter without adding visible spaces to OCR output.
    return "<!-- -->".join(parts)


def render(level: str, examples_per_dataset: int, seed: int) -> Path:
    models, data = collect(level)
    examples = select_examples(level, models, data, examples_per_dataset, seed)
    assets_dir = OUTPUT_ROOT / "assets" / level
    lines = [
        "### Примеры предсказаний",
        "",
        f'{colored_math("Зелёным", (79, 138, 91))} отмечены правильные символы; '
        f'{colored_math("красным", (179, 92, 92))} — ошибочные и лишние; '
        f'<s>{colored_math("зачёркнутым", (179, 92, 92))}</s> — пропущенные.',
        "",
    ]
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
            cells.append(
                colored_prediction(
                    row.get("prediction", ""), row.get("reference", "")
                )
                if row
                else "—"
            )
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
