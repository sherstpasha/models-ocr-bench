"""Build the project README from inventories and current benchmark summaries."""

from pathlib import Path

from configs.detection.benchmark_config import BENCHMARKS as DETECTION_MODELS
from configs.line_detection.benchmark_config import BENCHMARKS as LINE_DETECTION_MODELS
from configs.recognition.benchmark_config import BENCHMARKS as RECOGNITION_MODELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION = "v1"
VERSION_DATE = "4 октября 2026 года"

MODEL_LICENSES = {
    "east_50_g1": "MIT",
    "yolo26s_obb_text_g1": "AGPL-3.0",
    "yolo26x_obb_text_g1": "AGPL-3.0",
    "yolo11n_text": "Apache-2.0",
    "yolo11x_dialectic": "AGPL-3.0",
    "craft_easyocr": "Apache-2.0",
    "paddleocr_v6_medium_word": "Apache-2.0",
    "doctr_db_resnet50": "Apache-2.0",
    "openocr_repvit_db": "Apache-2.0",
    "paddleocr_v6_medium_line": "Apache-2.0",
    "rfdetr_textline_textregion_2xl": "Apache-2.0",
    "mask2former_line_v0_prev": "Apache-2.0",
    "doc_ufcn_generic_historical_line": "MIT",
    "surya_text_line_detection": "Apache-2.0",
    "kraken_blla_default": "Apache-2.0",
    "pero_layout_general": "BSD-3-Clause",
    "riksarkivet_rtmdet_lines": "MIT",
    "trba_lite_g1": "MIT",
    "trba_base_g1": "MIT",
    "cyrillic_g1": "Apache-2.0",
    "cyrillic_g2": "Apache-2.0",
    "paddleocr_cyrillic_v5_mobile": "Apache-2.0",
    "paddleocr_eslav_v5_mobile": "Apache-2.0",
    "trocr_ru_1700s": "MIT",
    "trocr_dialectic_stackmix": "Apache-2.0",
    "trocr_dialectic": "Apache-2.0",
    "trocr_base_ru": "Apache-2.0",
    "trocr_base_handwritten_ru": "Не указана",
    "cyrillic_large_handwritten": "Apache-2.0",
    "turkicocr_svtrv2_b": "Apache-2.0",
    "kraken_ppocrv6_medium": "Apache-2.0",
    "tesseract_rus_best": "Apache-2.0",
    "tesseract_cyrillic_best": "Apache-2.0",
}

DATASETS = [
    ("russian_old_orthography", "https://huggingface.co/datasets/nevmenandr/russian-old-orthography-ocr", "MIT"),
    ("YeniseiGovReports-TD", "https://huggingface.co/datasets/anna4uonline/YeniseiGovReports-TD", "MIT"),
    ("school_notebooks_RU", "https://huggingface.co/datasets/ai-forever/school_notebooks_RU", "MIT"),
    ("handwritten_essay", "https://huggingface.co/datasets/sherstpasha/handwritten_essay", "CC-BY-NC-3.0"),
    ("ICDAR2015", "https://rrc.cvc.uab.es/?ch=4", "Условия ICDAR/RRC"),
    ("TotalText", "https://github.com/cs-chan/Total-Text-Dataset", "BSD-3-Clause"),
    ("gota_hovratt_seg", "https://huggingface.co/datasets/Riksarkivet/gota_hovratt_seg", "Не указана"),
    ("svea_hovratt_seg", "https://huggingface.co/datasets/Riksarkivet/svea_hovratt_seg", "Не указана"),
    ("bergskollegium_relationer_och_skrivelser_seg", "https://huggingface.co/datasets/Riksarkivet/bergskollegium_relationer_och_skrivelser_seg", "Не указана"),
    ("YeniseiGovReports-HWR", "https://huggingface.co/datasets/sherstpasha/YeniseiGovReports-HWR", "MIT"),
    ("YeniseiGovReports-PRT", "https://huggingface.co/datasets/sherstpasha/YeniseiGovReports-PRT", "MIT"),
    ("cyrillic_handwriting", "https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset", "CC0-1.0"),
    ("DonkeySmallOCR-Numbers-Printed-15random", "https://huggingface.co/datasets/sherstpasha/DonkeySmallOCR-Numbers-Printed-15random", "Не указана"),
]

RESULTS = [
    (
        "Результаты детекции слов",
        "benchmark_results/detection/summary.md",
        "Метрики в ячейке: `F1@0.5 / F1@0.5:0.95 / Dice F1 / Polygon H-mean@0.5`; для всех метрик больше — лучше.",
    ),
    (
        "Результаты детекции строк",
        "benchmark_results/line_detection/summary.md",
        "Метрики в ячейке: `F1@0.5 / F1@0.5:0.95 / Dice F1 / Polygon H-mean@0.5`; для всех метрик больше — лучше.",
    ),
    (
        "Результаты распознавания слов",
        "benchmark_results/recognition/summary.md",
        "Метрики в ячейке: `Character Similarity / Exact Match / CER / WER`; для первых двух больше — лучше, для CER и WER меньше — лучше.",
    ),
    (
        "Результаты распознавания строк",
        "benchmark_results/line_recognition/summary.md",
        "Метрики в ячейке: `Character Similarity / Exact Match / CER / WER`; для первых двух больше — лучше, для CER и WER меньше — лучше.",
    ),
]


def markdown_table(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    table = [line for line in lines if line.startswith("|")]
    if not table:
        raise RuntimeError(f"No Markdown table found in {path}")
    return table


def model_inventory():
    combined = {}
    for configs in (DETECTION_MODELS, LINE_DETECTION_MODELS, RECOGNITION_MODELS):
        combined.update(configs)
    if set(combined) != set(MODEL_LICENSES):
        missing = sorted(set(combined) - set(MODEL_LICENSES))
        stale = sorted(set(MODEL_LICENSES) - set(combined))
        raise RuntimeError(f"Update MODEL_LICENSES; missing={missing}, stale={stale}")
    lines = ["| Модель | Источник | Лицензия |", "| --- | --- | --- |"]
    for name, config in combined.items():
        lines.append(f"| `{name}` | [source]({config['origin']}) | {MODEL_LICENSES[name]} |")
    return lines


def dataset_inventory():
    lines = ["| Датасет | Источник | Лицензия |", "| --- | --- | --- |"]
    for name, source, license_name in DATASETS:
        lines.append(f"| `{name}` | [source]({source}) | {license_name} |")
    return lines


def build():
    lines = [
        "# Russian OCR Benchmark",
        "",
        f"Бенчмарк OCR для современного и исторического русского текста. Версия **{VERSION}**, дата фиксации — **{VERSION_DATE}**.",
        "",
        "Инструкция по полному воспроизведению окружений, загрузке моделей и датасетов, запуску задач и пересборке таблиц: [REPRODUCING.md](REPRODUCING.md).",
        "",
        f"Протестировано **{len(MODEL_LICENSES)} модели/конфигурации** на **{len(DATASETS)} наборах данных** в четырёх задачах: детекция слов, детекция строк, распознавание слов и распознавание строк.",
        "",
        "## Модели",
        "",
        *model_inventory(),
        "",
        "Лицензии взяты из карточек моделей или исходных репозиториев.",
        "",
        "## Датасеты",
        "",
        *dataset_inventory(),
        "",
        "Лицензии взяты из карточек датасетов или исходных репозиториев.",
    ]
    for title, relative, description in RESULTS:
        lines.extend([
            "", f"## {title}", "", description,
            "", "`Mean Metric Rank`: 1 — лучший средний ранг; меньше — лучше.",
            "", *markdown_table(PROJECT_ROOT / relative),
        ])
    lines.append("")
    return "\n".join(lines)


def main():
    target = PROJECT_ROOT / "README.md"
    target.write_text(build(), encoding="utf-8")
    print(f"README: {target}")


if __name__ == "__main__":
    main()
