"""Shared benchmark loop for optional third-party line detectors."""

import gc
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from tqdm import tqdm

from configs.line_detection.benchmark_config import DATASETS
from scripts.line_detection.prepare_datasets import PREPARATION_VERSION
from utils.datasets import prediction_key, result_is_compatible
from utils.metrics import evaluate_dataset, evaluate_polygon_dataset, load_coco_detection_ground_truth
from utils.prediction_artifacts import prediction_artifact_path, save_predictions


SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_files(folder):
    return sorted(str(path) for path in Path(folder).rglob("*") if path.is_file() and path.suffix.lower() in SUFFIXES)


def polygons_to_boxes(polygons):
    boxes = []
    valid = []
    for polygon in polygons:
        parts = polygon if polygon and isinstance(polygon[0][0], (list, tuple)) else [polygon]
        points = np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1, 2) for part in parts])
        if len(points) < 3:
            continue
        boxes.append((float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())))
        valid.append([np.asarray(part, dtype=np.float32).reshape(-1, 2).tolist() for part in parts])
    return boxes, valid


def run(model_name, config, create_model, detect, architecture):
    if not torch.cuda.is_available():
        raise RuntimeError(f"{architecture} benchmark requires CUDA")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model = None
    for dataset_name, dataset in DATASETS.items():
        print(f"\n### DATASET: {dataset_name}", flush=True)
        output_file = output_dir / f"{dataset_name}_{model_name}.json"
        if output_file.is_file():
            existing = json.loads(output_file.read_text(encoding="utf-8")).get("gpu") or {}
            if result_is_compatible(existing, dataset) and existing.get("ground_truth_version") == PREPARATION_VERSION and prediction_artifact_path(output_file).is_file():
                print(f"Skip: already completed ({output_file})", flush=True)
                continue
        images = image_files(dataset["folder"])
        if not images or not Path(dataset["annotations"]).is_file():
            print("Skip: missing images or annotations", flush=True)
            continue
        if model is None:
            print(f"Loading {architecture}...", flush=True)
            model = create_model(config)
            print(f"{architecture} is ready.", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        process = psutil.Process()
        ram_after_load = process.memory_info().rss / 1024**2
        gpu_after_load = torch.cuda.memory_allocated() / 1024**2
        for path in images[: config.get("warmup", 1)]:
            detect(model, path, config)
        torch.cuda.synchronize()
        predictions, polygon_predictions, timings = {}, {}, []
        ram_peak = ram_after_load
        for path in tqdm(images, desc=architecture, unit="image"):
            started = time.perf_counter()
            polygons = detect(model, path, config)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            boxes, polygons = polygons_to_boxes(polygons)
            key = prediction_key(path, dataset)
            predictions[key], polygon_predictions[key] = boxes, polygons
            ram_peak = max(ram_peak, process.memory_info().rss / 1024**2)
        values = np.asarray(timings)
        gpu_peak = torch.cuda.max_memory_allocated() / 1024**2
        stats = {
            "device": config["device"], "detection_level": "line", "architecture": architecture,
            "repository": config["origin"], "ground_truth_version": PREPARATION_VERSION,
            "num_images": len(images), "mean_time_ms": float(values.mean() * 1000),
            "median_time_ms": float(np.median(values) * 1000), "throughput_fps": float(len(images) / values.sum()),
            "ram_after_load_mb": ram_after_load, "ram_peak_mb": ram_peak, "ram_delta_mb": ram_peak - ram_after_load,
            "gpu_after_load_mb": gpu_after_load, "gpu_peak_mb": gpu_peak, "gpu_delta_mb": gpu_peak - gpu_after_load,
        }
        save_predictions(output_file, predictions, polygon_predictions)
        gt_boxes, gt_polygons = load_coco_detection_ground_truth(dataset["annotations"])
        stats["accuracy_metrics"] = evaluate_dataset(predictions, gt_boxes)
        stats["polygon_iou_metrics"] = evaluate_polygon_dataset(polygon_predictions, gt_polygons)
        output_file.write_text(json.dumps({"dataset": dataset_name, "level": "line", "cpu": None, "gpu": stats}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {output_file}", flush=True)
