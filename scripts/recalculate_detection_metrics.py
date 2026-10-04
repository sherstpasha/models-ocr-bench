"""Recalculate all word/line detection metrics from saved predictions only."""

import argparse
import json
from pathlib import Path

from configs.detection.benchmark_config import (
    BENCHMARKS as WORD_MODELS,
    DATASETS as WORD_DATASETS,
)
from configs.line_detection.benchmark_config import (
    BENCHMARKS as LINE_MODELS,
    DATASETS as LINE_DATASETS,
)
from scripts.detection.summarize_results import write_summary as write_word_summary
from scripts.line_detection.summarize_results import write_summary as write_line_summary
from utils.metrics import (
    evaluate_dataset,
    evaluate_polygon_dataset,
    load_coco_detection_ground_truth,
)
from utils.prediction_artifacts import prediction_artifact_path


def load_predictions(path):
    source = json.loads(Path(path).read_text(encoding="utf-8"))["predictions"]
    boxes = {
        image_name: [tuple(item["bbox"]) for item in objects]
        for image_name, objects in source.items()
    }
    polygons = {
        image_name: [item["polygon"] for item in objects]
        for image_name, objects in source.items()
    }
    return boxes, polygons


def recalculate_task(label, models, datasets, selected_model=None, selected_dataset=None):
    completed = 0
    missing = []
    for model_name, config in models.items():
        if selected_model and model_name != selected_model:
            continue
        for dataset_name, dataset in datasets.items():
            if selected_dataset and dataset_name != selected_dataset:
                continue
            result_file = Path(config["output_dir"]) / f"{dataset_name}_{model_name}.json"
            artifact = prediction_artifact_path(result_file)
            if not result_file.is_file() or not artifact.is_file():
                missing.append(f"{model_name}/{dataset_name}")
                continue

            print(f"[{label}] {model_name} / {dataset_name}", flush=True)
            predicted_boxes, predicted_polygons = load_predictions(artifact)
            ground_truth_boxes, ground_truth_polygons = load_coco_detection_ground_truth(
                dataset["annotations"]
            )
            bbox_metrics = evaluate_dataset(predicted_boxes, ground_truth_boxes)
            polygon_metrics = evaluate_polygon_dataset(
                predicted_polygons, ground_truth_polygons
            )

            result = json.loads(result_file.read_text(encoding="utf-8"))
            target = result.get("gpu") or result.get("cpu")
            if target is None:
                missing.append(f"{model_name}/{dataset_name}: no device result")
                continue
            target["accuracy_metrics"] = bbox_metrics
            target["polygon_iou_metrics"] = polygon_metrics
            result_file.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            completed += 1
    return completed, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", "word", "line"), default="all")
    parser.add_argument("--model", help="Recalculate only this configured model.")
    parser.add_argument("--dataset", help="Recalculate only this configured dataset.")
    args = parser.parse_args()

    completed = 0
    missing = []
    if args.task in {"all", "word"}:
        count, absent = recalculate_task(
            "word", WORD_MODELS, WORD_DATASETS, args.model, args.dataset
        )
        completed += count
        missing.extend(absent)
        write_word_summary()
    if args.task in {"all", "line"}:
        count, absent = recalculate_task(
            "line", LINE_MODELS, LINE_DATASETS, args.model, args.dataset
        )
        completed += count
        missing.extend(absent)
        write_line_summary()

    print(f"Recalculated result files: {completed}")
    if missing:
        print("Missing prediction artifacts:")
        for item in missing:
            print(f"- {item}")


if __name__ == "__main__":
    main()
