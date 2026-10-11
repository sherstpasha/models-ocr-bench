"""Tune anisotropic Doc-UFCN mask dilation on two benchmark datasets."""

import json
from pathlib import Path

import cv2
import numpy as np

from configs.line_detection.benchmark_config import BENCHMARKS, DATASETS
from scripts.line_detection.benchmark_doc_ufcn import expand_mask_polygon, load_ground_truth
from utils.metrics import evaluate_dataset, evaluate_polygon_dataset
from utils.prediction_artifacts import prediction_artifact_path


MODEL_NAME = "doc_ufcn_generic_historical_line"
DATASET_NAMES = ("school_notebooks_ru", "russian_old_orthography")
CONFIG = BENCHMARKS[MODEL_NAME]
TUNING_FILE = Path(CONFIG["output_dir"]) / "mask_expansion_tuning_two_datasets.json"


def load_predictions():
    combined = {}
    for dataset_name in DATASET_NAMES:
        result_file = Path(CONFIG["output_dir"]) / f"{dataset_name}_{MODEL_NAME}.json"
        payload = json.loads(prediction_artifact_path(result_file).read_text(encoding="utf-8"))
        combined.update({f"{dataset_name}/{name}": value for name, value in payload["predictions"].items()})
    return combined


def image_shapes():
    shapes = {}
    for dataset_name in DATASET_NAMES:
        for path in Path(DATASETS[dataset_name]["folder"]).rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                shapes[f"{dataset_name}/{path.name}"] = image.shape
    return shapes


def combined_ground_truth():
    boxes, polygons = {}, {}
    for dataset_name in DATASET_NAMES:
        dataset_boxes, dataset_polygons = load_ground_truth(DATASETS[dataset_name]["annotations"])
        boxes.update({f"{dataset_name}/{name}": value for name, value in dataset_boxes.items()})
        polygons.update({f"{dataset_name}/{name}": value for name, value in dataset_polygons.items()})
    return boxes, polygons


def expanded_predictions(records, shapes, expand_x, expand_y):
    boxes, polygons = {}, {}
    for name, objects in records.items():
        image_polygons = []
        image_boxes = []
        shape = shapes[name]
        for obj in objects:
            output_parts = []
            for part in obj["polygon"]:
                points = np.asarray(part, dtype=np.float32).reshape(-1, 2)
                output_parts.extend(expand_mask_polygon(points, expand_x, expand_y, shape))
            if not output_parts:
                continue
            points = np.concatenate([np.asarray(part) for part in output_parts])
            image_polygons.append(output_parts)
            image_boxes.append((float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())))
        boxes[name] = image_boxes
        polygons[name] = image_polygons
    return boxes, polygons


def fibonacci_numbers(limit):
    values = [1, 1]
    while values[-1] < limit:
        values.append(values[-1] + values[-2])
    return values


def fibonacci_maximum(function, low, high):
    """Return the best integer in a closed interval for a unimodal function."""
    if high <= low:
        return low
    fib = fibonacci_numbers(high - low + 1)
    n = len(fib) - 1
    left, right = low, high
    x1 = left + fib[n - 2] * (right - left) // fib[n]
    x2 = left + fib[n - 1] * (right - left) // fib[n]
    f1, f2 = function(x1), function(x2)
    while right - left > 3 and n > 2:
        n -= 1
        if f1 < f2:
            left, x1, f1 = x1, x2, f2
            x2 = left + fib[n - 1] * (right - left) // fib[n]
            f2 = function(x2)
        else:
            right, x2, f2 = x2, x1, f1
            x1 = left + fib[n - 2] * (right - left) // fib[n]
            f1 = function(x1)
    candidates = range(max(low, left - 1), min(high, right + 1) + 1)
    return max(candidates, key=function)


def coarse_bracketed_fibonacci_maximum(function, low, high, step=8):
    """Coarsely locate the best mode, then refine that bracket with Fibonacci."""
    grid = list(range(low, high + 1, step))
    if grid[-1] != high:
        grid.append(high)
    best_index = max(range(len(grid)), key=lambda index: function(grid[index]))
    bracket_low = grid[max(0, best_index - 1)]
    bracket_high = grid[min(len(grid) - 1, best_index + 1)]
    return fibonacci_maximum(function, bracket_low, bracket_high)


def main():
    records = load_predictions()
    shapes = image_shapes()
    gt_boxes, gt_polygons = combined_ground_truth()
    cache = {}

    def score(x, y):
        key = int(x), int(y)
        if key not in cache:
            boxes, polygons = expanded_predictions(records, shapes, *key)
            per_dataset = {}
            for dataset_name in DATASET_NAMES:
                prefix = f"{dataset_name}/"
                dataset_boxes = {name: value for name, value in boxes.items() if name.startswith(prefix)}
                dataset_polygons = {name: value for name, value in polygons.items() if name.startswith(prefix)}
                dataset_gt_boxes = {name: value for name, value in gt_boxes.items() if name.startswith(prefix)}
                dataset_gt_polygons = {name: value for name, value in gt_polygons.items() if name.startswith(prefix)}
                bbox = evaluate_dataset(dataset_boxes, dataset_gt_boxes)
                polygon = evaluate_polygon_dataset(dataset_polygons, dataset_gt_polygons)
                values = {
                    "f1@0.5": bbox["f1@0.5"],
                    "f1@0.5:0.95": bbox["f1@0.5:0.95"],
                    "dice_f1": polygon["dice_f1"],
                    "polygon_hmean": polygon["hmean"],
                }
                values["objective_mean"] = sum(values.values()) / len(values)
                per_dataset[dataset_name] = values
            metrics = {
                "objective_mean": sum(value["objective_mean"] for value in per_dataset.values()) / len(per_dataset),
                "per_dataset": per_dataset,
            }
            cache[key] = metrics
            print(f"x={key[0]:2d} y={key[1]:2d} objective={metrics['objective_mean']:.6f} {metrics}", flush=True)
        return cache[key]["objective_mean"]

    x = y = 0
    for _ in range(4):
        previous = x, y
        x = coarse_bracketed_fibonacci_maximum(lambda value: score(value, y), 0, 64)
        y = coarse_bracketed_fibonacci_maximum(lambda value: score(x, value), 0, 64)
        if (x, y) == previous:
            break
    result = {
        "datasets": list(DATASET_NAMES),
        "objective": "equal dataset mean of mean(f1@0.5, f1@0.5:0.95, dice_f1, polygon_hmean)",
        "search_ranges": {"x": [0, 64], "y": [0, 64]},
        "best": {"mask_expand_x": x, "mask_expand_y": y, **cache[x, y]},
        "evaluations": [
            {"mask_expand_x": key[0], "mask_expand_y": key[1], **value}
            for key, value in sorted(cache.items())
        ],
    }
    TUNING_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best expansion: x={x}, y={y}; saved {TUNING_FILE}", flush=True)


if __name__ == "__main__":
    main()
