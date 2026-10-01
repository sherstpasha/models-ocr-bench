import csv
import json
from pathlib import Path

from configs.benchmark_config import BENCHMARKS, DATASETS, PROJECT_ROOT


SUMMARY_CSV = PROJECT_ROOT / "benchmark_results" / "summary.csv"
SUMMARY_MARKDOWN = PROJECT_ROOT / "benchmark_results" / "summary.md"


def read_f1(model_name, config, dataset_name):
    result_file = Path(config["output_dir"]) / f"{dataset_name}_{model_name}.json"
    if not result_file.is_file():
        return None
    with result_file.open("r", encoding="utf-8") as file:
        result = json.load(file)
    metrics = (result.get("gpu") or {}).get("accuracy_metrics", {})
    f1_50 = metrics.get("f1@0.5")
    f1_range = metrics.get("f1@0.5:0.95")
    if f1_50 is None or f1_range is None:
        return None
    return f1_50, f1_range


def format_cell(metrics):
    if metrics is None:
        return "—"
    return f"{metrics[0]:.6f} / {metrics[1]:.6f}"


def summary_rows():
    dataset_names = list(DATASETS)
    rows = []
    for model_name, config in BENCHMARKS.items():
        row = {"model": model_name}
        for dataset_name in dataset_names:
            row[dataset_name] = format_cell(
                read_f1(model_name, config, dataset_name)
            )
        rows.append(row)
    return dataset_names, rows


def write_summary():
    dataset_names, rows = summary_rows()
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["model", *dataset_names])
        writer.writeheader()
        writer.writerows(rows)

    headers = ["Model", *dataset_names]
    lines = [
        "# Benchmark summary",
        "",
        "Each cell: `F1@0.5 / F1@0.5:0.95`.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([row["model"], *(row[name] for name in dataset_names)])
            + " |"
        )
    SUMMARY_MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Summary Markdown: {SUMMARY_MARKDOWN}")


def main():
    write_summary()


if __name__ == "__main__":
    main()
