import csv
import json

from configs.recognition.benchmark_config import (
    BENCHMARKS,
    DATASETS,
    RECOGNITION_LEVEL,
    RESULTS_ROOT,
)
from utils.metrics import (
    cer_score,
    character_similarity_score,
    normalize_text,
    wer_score,
)
from utils.ranking import mean_metric_ranks
from utils.dataset_sources import markdown_dataset_header
from utils.training_overlaps import mark_training_overlap


csv.field_size_limit(2**31 - 1)


SUMMARY_CSV = RESULTS_ROOT / "summary.csv"
SUMMARY_MARKDOWN = RESULTS_ROOT / "summary.md"


def result_paths(model_name, dataset_name):
    stem = f"{dataset_name}_{model_name}"
    output_dir = BENCHMARKS[model_name]["output_dir"]
    return output_dir / f"{stem}.json", output_dir / f"{stem}.csv"


def read_predictions(path):
    if not path.is_file():
        return None
    references = []
    predictions = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            reference = normalize_text(row.get("reference", ""))
            prediction = normalize_text(row.get("prediction", ""))
            references.append(reference)
            predictions.append(prediction)
    if not references:
        return None
    return (
        character_similarity_score(predictions, references),
        sum(prediction == reference for prediction, reference in zip(predictions, references))
        / len(references),
        cer_score(predictions, references),
        wer_score(predictions, references),
    )


def read_metrics(model_name, dataset_name):
    json_path, csv_path = result_paths(model_name, dataset_name)
    if not json_path.is_file():
        return None
    result = json.loads(json_path.read_text(encoding="utf-8"))
    if (
        BENCHMARKS[model_name].get("backend") == "tesseract"
        and result.get("benchmark_version") != 2
    ):
        return None
    stats = result.get("gpu") or result.get("cpu") or {}
    metrics = stats.get("accuracy_metrics", {})
    if not all(key in metrics for key in ("cer", "wer")):
        return None
    throughput = stats.get("throughput_words_s")
    predictions = read_predictions(csv_path)
    if throughput is None or predictions is None:
        return None
    return {
        "mean": (
            predictions[0],
            predictions[1],
            predictions[2],
            predictions[3],
        ),
        "throughput": float(throughput),
    }


def heat_color(value, low, high, higher_is_better=True):
    if high == low:
        score = 1.0
    else:
        score = (value - low) / (high - low)
    if not higher_is_better:
        score = 1.0 - score
    # Text-only heatmap: keep the red identical to the character-error
    # highlighting used in prediction examples and use similarly muted tones.
    bad = (179, 92, 92)
    middle = (138, 122, 82)
    good = (79, 138, 91)
    start, end, mix = (
        (bad, middle, score * 2)
        if score <= 0.5
        else (middle, good, (score - 0.5) * 2)
    )
    red, green, blue = (
        round(start[index] + (end[index] - start[index]) * mix)
        for index in range(3)
    )
    return f"#{red:02x}{green:02x}{blue:02x}"


def heat_span(text, value, value_range, higher_is_better, is_best=False):
    low, high = value_range
    content = f"<b>{text}</b>" if is_best else text
    color = heat_color(value, low, high, higher_is_better)
    return f'<span style="color:{color}">{content}</span>'


def format_cell(values, best=None, precision=6, ranges=None):
    if values is None:
        return "-"
    formatted = [f"{value:.{precision}f}" for value in values]
    if ranges is not None:
        directions = (True, True, False, False)
        formatted = [
            heat_span(
                text,
                value,
                value_range,
                direction,
                best is not None and value == target,
            )
            for text, value, value_range, direction, target in zip(
                formatted, values, ranges, directions, best or values
            )
        ]
    elif best is not None:
        formatted = [
            f"**{text}**" if value == target else text
            for text, value, target in zip(formatted, values, best)
        ]
    return " / ".join(formatted)


def best_by_dataset(raw, table_key):
    best = {}
    models = {model for model, _dataset in raw}
    for dataset in DATASETS:
        available = [
            raw[model, dataset][table_key]
            for model in models
            if raw[model, dataset] is not None
        ]
        best[dataset] = (
            (
                max(values[0] for values in available),
                max(values[1] for values in available),
                min(values[2] for values in available),
                min(values[3] for values in available),
            )
            if available
            else None
        )
    return best


def ranges_by_dataset(raw, table_key):
    ranges = {}
    models = {model for model, _dataset in raw}
    for dataset in DATASETS:
        available = [
            raw[model, dataset][table_key]
            for model in models
            if raw[model, dataset] is not None
        ]
        ranges[dataset] = (
            tuple((min(values), max(values)) for values in zip(*available))
            if available
            else None
        )
    return ranges


def mean_across_datasets(raw, model, key, index=None):
    values = []
    for dataset in DATASETS:
        result = raw[model, dataset]
        if result is None:
            continue
        value = result[key]
        values.append(value if index is None else value[index])
    return sum(values) / len(values) if values else None


def sorted_models(mean_rank, mean_similarity):
    return sorted(
        BENCHMARKS,
        key=lambda model: (
            mean_rank[model] is None,
            mean_rank[model] if mean_rank[model] is not None else float("inf"),
            -mean_similarity[model]
            if mean_similarity[model] is not None
            else float("inf"),
        ),
    )


def write_csv(path, raw, models, table_key, mean_metrics, mean_rank, mean_speed):
    speed_name = "mean_lines_s" if RECOGNITION_LEVEL == "line" else "mean_words_s"
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "origin",
                "mean_character_similarity",
                "mean_exact_match",
                "mean_cer",
                "mean_wer",
                *DATASETS,
                speed_name,
            ]
        )
        for model in models:
            writer.writerow(
                [
                    model,
                    BENCHMARKS[model]["origin"],
                    *(
                        f"{metric[model]:.6f}" if metric[model] is not None else "-"
                        for metric in mean_metrics
                    ),
                    *(
                        format_cell(
                            raw[model, dataset][table_key]
                            if raw[model, dataset] is not None
                            else None
                        )
                        for dataset in DATASETS
                    ),
                    f"{mean_speed[model]:.6f}" if mean_speed[model] is not None else "-",
                ]
            )


def markdown_table(raw, models, table_key, mean_metrics, mean_rank, mean_speed):
    best = best_by_dataset(raw, table_key)
    ranges = ranges_by_dataset(raw, table_key)
    best_means = []
    for index, metric in enumerate(mean_metrics):
        available = [value for value in metric.values() if value is not None]
        best_means.append(
            (max(available) if index < 2 else min(available)) if available else None
        )
    available_speed = [value for value in mean_speed.values() if value is not None]
    best_speed = max(available_speed) if available_speed else None
    mean_ranges = [
        (min(available), max(available))
        for metric in mean_metrics
        for available in [[value for value in metric.values() if value is not None]]
    ]
    speed_range = (min(available_speed), max(available_speed)) if available_speed else None
    speed_title = "Mean lines/s" if RECOGNITION_LEVEL == "line" else "Mean words/s"
    dataset_headers = [
        markdown_dataset_header(name, DATASETS[name]) for name in DATASETS
    ]
    lines = [
        "| Model | Origin | Mean Character Similarity | Mean Exact Match | Mean CER | Mean WER | "
        + " | ".join(dataset_headers)
        + f" | {speed_title} |",
        "| --- | --- | --- | --- | --- | --- | "
        + " | ".join("---" for _ in DATASETS)
        + " | --- |",
    ]
    for model in models:
        cells = [
            mark_training_overlap(
                format_cell(
                    raw[model, dataset][table_key]
                    if raw[model, dataset] is not None
                    else None,
                    best[dataset],
                    precision=2,
                    ranges=ranges[dataset],
                ),
                model,
                dataset,
            )
            for dataset in DATASETS
        ]
        speed = mean_speed[model]
        mean_cells = []
        for index, (metric, best_mean) in enumerate(zip(mean_metrics, best_means)):
            value = metric[model]
            cell = "-" if value is None else f"{value:.2f}"
            if value is not None:
                cell = heat_span(
                    cell,
                    value,
                    mean_ranges[index],
                    index < 2,
                    value == best_mean,
                )
            mean_cells.append(cell)
        speed_cell = "-" if speed is None else f"{speed:.2f}"
        if speed is not None:
            speed_cell = heat_span(
                speed_cell, speed, speed_range, True, speed == best_speed
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    f"[source]({BENCHMARKS[model]['origin']})",
                    *mean_cells,
                    *cells,
                    speed_cell,
                ]
            )
            + " |"
        )
    return lines


def write_summary():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    model_names = [
        model
        for model, config in BENCHMARKS.items()
        if RECOGNITION_LEVEL in config.get("levels", {"word", "line"})
    ]
    raw = {
        (model, dataset): read_metrics(model, dataset)
        for model in model_names
        for dataset in DATASETS
    }
    mean_metrics = [
        {
            model: mean_across_datasets(raw, model, "mean", index)
            for model in model_names
        }
        for index in range(4)
    ]
    mean_similarity = mean_metrics[0]
    mean_rank = mean_metric_ranks(
        mean_metrics,
        higher_is_better=[True, True, False, False],
    )
    mean_speed = {
        model: mean_across_datasets(raw, model, "throughput")
        for model in model_names
    }
    mean_models = sorted_models(mean_rank, mean_similarity)

    write_csv(
        SUMMARY_CSV,
        raw,
        mean_models,
        "mean",
        mean_metrics,
        mean_rank,
        mean_speed,
    )
    speed_unit = "lines" if RECOGNITION_LEVEL == "line" else "words"
    title = "Line recognition benchmark summary" if RECOGNITION_LEVEL == "line" else "Recognition benchmark summary"
    lines = [
        f"# {title}",
        "",
        "Each dataset cell: `Character Similarity / Exact Match / CER / WER`.",
        "`*` marks a dataset known to have been used to train that model.",
        f"`Mean {speed_unit}/s` is the arithmetic mean across available datasets on the device used by each model.",
        "",
        "## Aggregate metrics",
        "",
        "Character Similarity is `1 - edit_distance / max(reference_length, prediction_length)` averaged across samples.",
        "Exact Match is the share of fully correct samples. Character Similarity and Exact Match range from 0 to 1; higher is better.",
        "CER and WER are normalized edit errors; lower is better and values can exceed 1 when predictions contain many insertions.",
        "Rows are sorted by the mean rank across all four aggregate metrics.",
        "Heatmap: <span style=\"color:#4f8a5b\">better</span> / <span style=\"color:#8a7a52\">middle</span> / <span style=\"color:#b35c5c\">worse</span> within each metric column.",
        "",
        *markdown_table(
            raw,
            mean_models,
            "mean",
            mean_metrics,
            mean_rank,
            mean_speed,
        ),
    ]
    SUMMARY_MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Summary Markdown: {SUMMARY_MARKDOWN}")


if __name__ == "__main__":
    write_summary()
