import csv
import json
from pathlib import Path

from configs.detection.benchmark_config import BENCHMARKS, DATASETS, RESULTS_ROOT
from utils.datasets import result_is_compatible
from utils.ranking import mean_metric_ranks
from utils.dataset_sources import markdown_dataset_header


SUMMARY_CSV = RESULTS_ROOT / "summary.csv"
SUMMARY_MARKDOWN = RESULTS_ROOT / "summary.md"


def read_metrics(model_name, config, dataset_name):
    result_file = Path(config["output_dir"]) / f"{dataset_name}_{model_name}.json"
    if not result_file.is_file():
        return None
    with result_file.open("r", encoding="utf-8") as file:
        result = json.load(file)
    gpu = result.get("gpu")
    if not result_is_compatible(gpu, DATASETS[dataset_name]):
        return None
    metrics = gpu.get("accuracy_metrics", {})
    polygon = gpu.get("polygon_iou_metrics", {})
    f1_50 = metrics.get("f1@0.5")
    f1_range = metrics.get("f1@0.5:0.95")
    dice_f1 = polygon.get("dice_f1")
    polygon_hmean = polygon.get("hmean")
    throughput_fps = gpu.get("throughput_fps")
    if any(value is None for value in (f1_50, f1_range, dice_f1, polygon_hmean, throughput_fps)):
        return None
    return f1_50, f1_range, dice_f1, polygon_hmean, throughput_fps


def format_cell(metrics, best=None, precision=6):
    if metrics is None:
        return "-"
    metrics = metrics[:4]
    values = [f"{value:.{precision}f}" for value in metrics]
    if best is not None:
        values = [
            f"**{value}**" if metric == maximum else value
            for value, metric, maximum in zip(values, metrics, best)
        ]
    return " / ".join(values)


def mean_metric(model_name, dataset_names, raw_metrics, metric_index):
    values = [
        raw_metrics[model_name, dataset_name][metric_index]
        for dataset_name in dataset_names
        if raw_metrics[model_name, dataset_name] is not None
    ]
    return sum(values) / len(values) if values else None


def summary_data():
    dataset_names = list(DATASETS)
    raw_metrics = {
        (model_name, dataset_name): read_metrics(model_name, config, dataset_name)
        for model_name, config in BENCHMARKS.items()
        for dataset_name in dataset_names
    }
    mean_f1 = {
        model_name: mean_metric(model_name, dataset_names, raw_metrics, 0)
        for model_name in BENCHMARKS
    }
    mean_f1_range = {
        model_name: mean_metric(model_name, dataset_names, raw_metrics, 1)
        for model_name in BENCHMARKS
    }
    mean_dice = {
        model_name: mean_metric(model_name, dataset_names, raw_metrics, 2)
        for model_name in BENCHMARKS
    }
    mean_polygon = {
        model_name: mean_metric(model_name, dataset_names, raw_metrics, 3)
        for model_name in BENCHMARKS
    }
    mean_fps = {
        model_name: mean_metric(model_name, dataset_names, raw_metrics, 4)
        for model_name in BENCHMARKS
    }
    mean_rank = mean_metric_ranks(
        [mean_f1, mean_f1_range, mean_dice, mean_polygon]
    )
    model_names = sorted(
        BENCHMARKS,
        key=lambda name: (
            mean_rank[name] is None,
            mean_rank[name] if mean_rank[name] is not None else float("inf"),
            -(mean_f1[name] if mean_f1[name] is not None else float("-inf")),
        ),
    )
    return dataset_names, model_names, raw_metrics, mean_f1, mean_f1_range, mean_dice, mean_polygon, mean_rank, mean_fps


def best_metrics(dataset_names, raw_metrics):
    best = {}
    for dataset_name in dataset_names:
        available = [
            raw_metrics[model_name, dataset_name]
            for model_name in BENCHMARKS
            if raw_metrics[model_name, dataset_name] is not None
        ]
        best[dataset_name] = (
            tuple(max(values[index] for values in available) for index in range(4))
            if available
            else None
        )
    return best


def write_summary():
    dataset_names, model_names, raw_metrics, mean_f1, mean_f1_range, mean_dice, mean_polygon, mean_rank, mean_fps = summary_data()
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["model", "origin", "mean_f1@0.5", "mean_f1@0.5:0.95", "mean_dice_f1", "mean_polygon_hmean@0.5", "mean_metric_rank", *dataset_names, "mean_gpu_fps"]
        )
        for model_name in model_names:
            writer.writerow(
                [
                    model_name,
                    BENCHMARKS[model_name]["origin"],
                    (
                        f"{mean_f1[model_name]:.6f}"
                        if mean_f1[model_name] is not None
                        else "-"
                    ),
                    "-" if mean_f1_range[model_name] is None else f"{mean_f1_range[model_name]:.6f}",
                    "-" if mean_dice[model_name] is None else f"{mean_dice[model_name]:.6f}",
                    "-" if mean_polygon[model_name] is None else f"{mean_polygon[model_name]:.6f}",
                    "-" if mean_rank[model_name] is None else f"{mean_rank[model_name]:.6f}",
                    *(
                        format_cell(raw_metrics[model_name, dataset_name])
                        for dataset_name in dataset_names
                    ),
                    (
                        f"{mean_fps[model_name]:.6f}"
                        if mean_fps[model_name] is not None
                        else "-"
                    ),
                ]
            )

    best = best_metrics(dataset_names, raw_metrics)
    available_f1 = [value for value in mean_f1.values() if value is not None]
    available_fps = [value for value in mean_fps.values() if value is not None]
    best_mean_f1 = max(available_f1) if available_f1 else None
    best_mean_f1_range = max(
        (value for value in mean_f1_range.values() if value is not None), default=None
    )
    best_mean_dice = max(
        (value for value in mean_dice.values() if value is not None), default=None
    )
    best_mean_polygon = max(
        (value for value in mean_polygon.values() if value is not None), default=None
    )
    best_mean_fps = max(available_fps) if available_fps else None
    best_mean_rank = min(
        (value for value in mean_rank.values() if value is not None), default=None
    )
    dataset_headers = [
        markdown_dataset_header(name, DATASETS[name]) for name in dataset_names
    ]
    headers = ["Model", "Origin", "Mean F1@0.5", "Mean F1@0.5:0.95", "Mean Dice F1", "Mean Polygon H-mean@0.5", "Mean Metric Rank", *dataset_headers, "Mean GPU FPS"]
    lines = [
        "# Benchmark summary",
        "",
        "Each dataset cell: `F1@0.5 / F1@0.5:0.95 / Dice F1 / Polygon H-mean@0.5`.",
        "`Mean GPU FPS` is the arithmetic mean across available datasets.",
        "`Mean Metric Rank` averages conventional ranks across the four mean metrics: rank 1 is best.",
        "Rows are sorted by `Mean Metric Rank` in ascending order.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for model_name in model_names:
        cells = [
            format_cell(
                raw_metrics[model_name, dataset_name],
                best=best[dataset_name],
                precision=4,
            )
            for dataset_name in dataset_names
        ]
        f1 = mean_f1[model_name]
        fps = mean_fps[model_name]
        f1_cell = "-" if f1 is None else f"{f1:.4f}"
        fps_cell = "-" if fps is None else f"{fps:.4f}"
        rank = mean_rank[model_name]
        rank_cell = "-" if rank is None else f"{rank:.4f}"
        if f1 is not None and f1 == best_mean_f1:
            f1_cell = f"**{f1_cell}**"
        if fps is not None and fps == best_mean_fps:
            fps_cell = f"**{fps_cell}**"
        if rank is not None and rank == best_mean_rank:
            rank_cell = f"**{rank_cell}**"
        mean_cells = []
        for value, best_value in (
            (mean_f1_range[model_name], best_mean_f1_range),
            (mean_dice[model_name], best_mean_dice),
            (mean_polygon[model_name], best_mean_polygon),
        ):
            cell = "-" if value is None else f"{value:.4f}"
            if value is not None and value == best_value:
                cell = f"**{cell}**"
            mean_cells.append(cell)
        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    f"[source]({BENCHMARKS[model_name]['origin']})",
                    f1_cell,
                    *mean_cells,
                    rank_cell,
                    *cells,
                    fps_cell,
                ]
            )
            + " |"
        )
    SUMMARY_MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Summary Markdown: {SUMMARY_MARKDOWN}")


def main():
    write_summary()


if __name__ == "__main__":
    main()
