"""Benchmark Surya's text-line detector."""

import gc
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image
from surya.detection import DetectionPredictor
from tqdm import tqdm

from configs.line_detection.benchmark_config import BENCHMARKS, DATASETS
from scripts.line_detection.prepare_datasets import PREPARATION_VERSION
from utils.datasets import prediction_key, result_is_compatible
from utils.metrics import (
    evaluate_dataset,
    evaluate_polygon_dataset,
    load_coco_detection_ground_truth,
)
from utils.prediction_artifacts import prediction_artifact_path, save_predictions


MODEL_NAME = "surya_text_line_detection"
CONFIG = BENCHMARKS[MODEL_NAME]


def get_image_files(folder):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        str(path)
        for path in Path(folder).rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def detect_lines(predictor, image_path):
    with Image.open(image_path) as image:
        results = predictor([image.convert("RGB")])
    if len(results) != 1:
        raise RuntimeError(f"Expected one Surya result for {image_path}")

    boxes = []
    polygons = []
    for detection in results[0].bboxes:
        points = np.asarray(detection.polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            continue
        polygons.append([points.tolist()])
        boxes.append(
            (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )
        )
    return boxes, polygons


def benchmark_gpu(predictor, images, dataset_config):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    ram_after_load = process.memory_info().rss / 1024**2
    gpu_after_load = torch.cuda.memory_allocated() / 1024**2

    for image_path in images[: CONFIG["warmup"]]:
        detect_lines(predictor, image_path)
    torch.cuda.synchronize()

    predictions = {}
    polygon_predictions = {}
    times = []
    ram_peak = ram_after_load
    for image_path in tqdm(images, desc="Surya detection", unit="image"):
        started = time.perf_counter()
        boxes, polygons = detect_lines(predictor, image_path)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        key = prediction_key(image_path, dataset_config)
        predictions[key] = boxes
        polygon_predictions[key] = polygons
        ram_peak = max(ram_peak, process.memory_info().rss / 1024**2)

    values = np.asarray(times)
    gpu_peak = torch.cuda.max_memory_allocated() / 1024**2
    return {
        "device": CONFIG["device"],
        "detection_level": "line",
        "architecture": "Surya text-line detector",
        "repository": CONFIG["origin"],
        "ground_truth_version": PREPARATION_VERSION,
        "num_images": len(images),
        "mean_time_ms": float(values.mean() * 1000),
        "median_time_ms": float(np.median(values) * 1000),
        "throughput_fps": float(len(images) / values.sum()),
        "ram_after_load_mb": ram_after_load,
        "ram_peak_mb": ram_peak,
        "ram_delta_mb": ram_peak - ram_after_load,
        "gpu_after_load_mb": gpu_after_load,
        "gpu_peak_mb": gpu_peak,
        "gpu_delta_mb": gpu_peak - gpu_after_load,
        "predictions": predictions,
        "polygon_predictions": polygon_predictions,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Surya line benchmark requires CUDA. "
            f"Installed torch: {torch.__version__}; CUDA runtime: {torch.version.cuda}."
        )
    predictor = None
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, dataset in DATASETS.items():
        print(f"\n### DATASET: {dataset_name}", flush=True)
        output_file = output_dir / f"{dataset_name}_{MODEL_NAME}.json"
        if output_file.is_file():
            existing = json.loads(output_file.read_text(encoding="utf-8")).get("gpu")
            if (
                result_is_compatible(existing, dataset)
                and existing.get("ground_truth_version") == PREPARATION_VERSION
                and prediction_artifact_path(output_file).is_file()
            ):
                print(f"Skip: already completed ({output_file})", flush=True)
                continue

        images = get_image_files(dataset["folder"])
        if not images or not dataset["annotations"].is_file():
            print("Skip: missing images or annotations", flush=True)
            continue
        if predictor is None:
            print("Loading Surya text-line detector...", flush=True)
            # Surya 0.22 defaults to a shared server process.  Keep the model
            # local so GPU/RAM accounting is comparable with other backends.
            predictor = DetectionPredictor.local(device=CONFIG["device"])
            print("Surya detector is ready.", flush=True)

        gpu_stats = benchmark_gpu(predictor, images, dataset)
        save_predictions(output_file, gpu_stats["predictions"], gpu_stats["polygon_predictions"])
        ground_truth_boxes, ground_truth_polygons = load_coco_detection_ground_truth(
            dataset["annotations"]
        )
        gpu_stats["accuracy_metrics"] = evaluate_dataset(
            gpu_stats.pop("predictions"), ground_truth_boxes
        )
        gpu_stats["polygon_iou_metrics"] = evaluate_polygon_dataset(
            gpu_stats.pop("polygon_predictions"), ground_truth_polygons
        )
        output_file.write_text(
            json.dumps(
                {"dataset": dataset_name, "level": "line", "cpu": None, "gpu": gpu_stats},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved: {output_file}", flush=True)


if __name__ == "__main__":
    main()
