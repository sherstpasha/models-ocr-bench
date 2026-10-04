"""Build the line-level detection result table."""

import csv
import json
from pathlib import Path

from configs.line_detection.benchmark_config import BENCHMARKS, DATASETS, RESULTS_ROOT
from utils.ranking import mean_metric_ranks
from utils.dataset_sources import markdown_dataset_header


def read_metrics(model_name, config, dataset_name):
    path = Path(config["output_dir"]) / f"{dataset_name}_{model_name}.json"
    if not path.is_file():
        return None
    gpu = json.loads(path.read_text(encoding="utf-8")).get("gpu") or {}
    bbox = gpu.get("accuracy_metrics") or {}
    polygon = gpu.get("polygon_iou_metrics") or {}
    values = (
        bbox.get("f1@0.5"),
        bbox.get("f1@0.5:0.95"),
        polygon.get("dice_f1"),
        polygon.get("hmean"),
        gpu.get("throughput_fps"),
    )
    return None if any(value is None for value in values) else values


def write_summary():
    datasets = list(DATASETS)
    raw = {
        (model, dataset): read_metrics(model, config, dataset)
        for model, config in BENCHMARKS.items()
        for dataset in datasets
    }
    means = {index: {} for index in range(4)}
    speeds = {}
    for model in BENCHMARKS:
        available = [raw[model, dataset] for dataset in datasets if raw[model, dataset]]
        for index in range(4):
            means[index][model] = sum(item[index] for item in available) / len(available) if available else None
        speeds[model] = sum(item[4] for item in available) / len(available) if available else None
    mean_rank = mean_metric_ranks([means[index] for index in range(4)])
    models = sorted(
        BENCHMARKS,
        key=lambda model: (
            mean_rank[model] is None,
            mean_rank[model] if mean_rank[model] is not None else float("inf"),
            -(means[0][model] if means[0][model] is not None else -1),
        ),
    )
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_ROOT / "summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "origin", "mean_f1@0.5", "mean_f1@0.5:0.95", "mean_dice_f1", "mean_polygon_hmean@0.5", "mean_metric_rank", *datasets, "mean_gpu_fps"])
        for model in models:
            writer.writerow([
                model,
                BENCHMARKS[model]["origin"],
                *("-" if means[index][model] is None else f"{means[index][model]:.6f}" for index in range(4)),
                "-" if mean_rank[model] is None else f"{mean_rank[model]:.6f}",
                *(
                    "-" if raw[model, dataset] is None else format_csv_cell(raw[model, dataset])
                    for dataset in datasets
                ),
                "-" if speeds[model] is None else f"{speeds[model]:.6f}",
            ])
    best_mean_rank = min(
        (value for value in mean_rank.values() if value is not None), default=None
    )
    best_means = [
        max((value for value in means[index].values() if value is not None), default=None)
        for index in range(4)
    ]
    dataset_headers = [
        markdown_dataset_header(name, DATASETS[name]) for name in datasets
    ]
    headers = ["Model", "Origin", "Mean F1@0.5", "Mean F1@0.5:0.95", "Mean Dice F1", "Mean Polygon H-mean@0.5", "Mean Metric Rank", *dataset_headers, "Mean GPU FPS"]
    lines = [
        "# Line-level detection benchmark",
        "",
        "Each dataset cell: `F1@0.5 / F1@0.5:0.95 / Dice F1 / Polygon H-mean@0.5`.",
        "`Mean Metric Rank` averages conventional ranks across the four mean metrics: rank 1 is best. Rows are sorted by it in ascending order.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for model in models:
        cells = [
            "-" if raw[model, dataset] is None else format_md_cell(raw[model, dataset])
            for dataset in datasets
        ]
        rank = mean_rank[model]
        rank_cell = "-" if rank is None else f"{rank:.2f}"
        if rank is not None and rank == best_mean_rank:
            rank_cell = f"**{rank_cell}**"
        mean_cells = []
        for index in range(4):
            value = means[index][model]
            cell = "-" if value is None else f"{value:.2f}"
            if value is not None and value == best_means[index]:
                cell = f"**{cell}**"
            mean_cells.append(cell)
        lines.append("| " + " | ".join([
            model,
            f"[source]({BENCHMARKS[model]['origin']})",
            *mean_cells,
            rank_cell,
            *cells,
            "-" if speeds[model] is None else f"{speeds[model]:.2f}",
        ]) + " |")
    md_path = RESULTS_ROOT / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary Markdown: {md_path}")


def format_csv_cell(values):
    return " / ".join(f"{value:.6f}" for value in values[:4])


def format_md_cell(values):
    return " / ".join(f"{value:.2f}" for value in values[:4])


if __name__ == "__main__":
    write_summary()
