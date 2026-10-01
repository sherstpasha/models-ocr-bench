import csv
import json
from pathlib import Path

from configs.benchmark_config import BENCHMARKS, DATASETS, PROJECT_ROOT
from utils.datasets import result_is_compatible


SUMMARY_CSV = PROJECT_ROOT / "benchmark_results" / "summary.csv"
SUMMARY_MARKDOWN = PROJECT_ROOT / "benchmark_results" / "summary.md"


def read_f1(model_name, config, dataset_name):
    result_file = Path(config["output_dir"]) / f"{dataset_name}_{model_name}.json"
    if not result_file.is_file():
        return None
    with result_file.open("r", encoding="utf-8") as file:
        result = json.load(file)
    gpu = result.get("gpu")
    if not result_is_compatible(gpu, DATASETS[dataset_name]):
        return None
    metrics = gpu.get("accuracy_metrics", {})
    f1_50 = metrics.get("f1@0.5")
    f1_range = metrics.get("f1@0.5:0.95")
    if f1_50 is None or f1_range is None:
        return None
    return f1_50, f1_range


def format_cell(metrics, best=None):
    if metrics is None:
        return "-"
    values = [f"{value:.6f}" for value in metrics]
    if best is not None:
        values = [
            f"**{value}**" if metric == maximum else value
            for value, metric, maximum in zip(values, metrics, best)
        ]
    return " / ".join(values)


def summary_data():
    dataset_names = list(DATASETS)
    raw_metrics = {
        (model_name, dataset_name): read_f1(model_name, config, dataset_name)
        for model_name, config in BENCHMARKS.items()
        for dataset_name in dataset_names
    }
    return dataset_names, raw_metrics


def best_metrics(dataset_names, raw_metrics):
    best = {}
    for dataset_name in dataset_names:
        available = [
            raw_metrics[model_name, dataset_name]
            for model_name in BENCHMARKS
            if raw_metrics[model_name, dataset_name] is not None
        ]
        best[dataset_name] = (
            tuple(max(values[index] for values in available) for index in range(2))
            if available
            else None
        )
    return best


def write_summary():
    dataset_names, raw_metrics = summary_data()
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", *dataset_names])
        for model_name in BENCHMARKS:
            writer.writerow(
                [
                    model_name,
                    *(
                        format_cell(raw_metrics[model_name, dataset_name])
                        for dataset_name in dataset_names
                    ),
                ]
            )

    best = best_metrics(dataset_names, raw_metrics)
    headers = ["Model", *dataset_names]
    lines = [
        "# Benchmark summary",
        "",
        "Each cell: `F1@0.5 / F1@0.5:0.95`.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for model_name in BENCHMARKS:
        cells = [
            format_cell(
                raw_metrics[model_name, dataset_name],
                best=best[dataset_name],
            )
            for dataset_name in dataset_names
        ]
        lines.append("| " + " | ".join([model_name, *cells]) + " |")
    SUMMARY_MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Summary Markdown: {SUMMARY_MARKDOWN}")


def main():
    write_summary()


if __name__ == "__main__":
    main()
