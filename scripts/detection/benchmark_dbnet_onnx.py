"""Benchmark the Darayut Khmer/English DBNet as a word detector."""

import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil
import pyclipper
import torch
from huggingface_hub import hf_hub_download

from configs.detection.benchmark_config import BENCHMARKS, DATASETS
from utils.datasets import prediction_key, result_is_compatible
from utils.metrics import evaluate_dataset, evaluate_polygon_dataset, load_coco_detection_ground_truth
from utils.prediction_artifacts import prediction_artifact_path, save_predictions


MODEL_NAME = "dbnet_khmer_word"
CONFIG = BENCHMARKS[MODEL_NAME]
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_model(config):
    path = Path(config["model_dir"]) / config["filename"]
    if not path.is_file():
        path = Path(hf_hub_download(
            repo_id=config["repository"], filename=config["filename"],
            local_dir=config["model_dir"],
        ))
    return path


def create_model(config):
    return ort.InferenceSession(
        str(ensure_model(config)),
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
    )


def _score(probability, points):
    height, width = probability.shape
    points = points.astype(np.int32)
    x0, x1 = int(np.clip(points[:, 0].min(), 0, width - 1)), int(np.clip(points[:, 0].max(), 0, width - 1))
    y0, y1 = int(np.clip(points[:, 1].min(), 0, height - 1)), int(np.clip(points[:, 1].max(), 0, height - 1))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    local = points.copy()
    local[:, 0] -= x0
    local[:, 1] -= y0
    cv2.fillPoly(mask, [local], 1)
    return cv2.mean(probability[y0:y1 + 1, x0:x1 + 1], mask)[0]


def _unclip(points, ratio):
    x, y = points[:, 0], points[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perimeter = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum()
    if area <= 0 or perimeter <= 0:
        return None
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(points.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(area * ratio / perimeter)
    return np.asarray(expanded[0], dtype=np.float32) if len(expanded) == 1 else None


def detect(model, image_path, config):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    height, width = bgr.shape[:2]
    scale = config["short_size"] / min(height, width)
    rh, rw = max(int(round(height * scale / 32) * 32), 32), max(int(round(width * scale / 32) * 32), 32)
    rgb = cv2.cvtColor(cv2.resize(bgr, (rw, rh)), cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = ((tensor - MEAN) / STD).transpose(2, 0, 1)[None]
    output = model.run(None, {model.get_inputs()[0].name: tensor})[0]
    probability = output[0, 0] if output.ndim == 4 and output.shape[1] == 2 else np.squeeze(output)
    bitmap = (probability > config["threshold"]).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes, polygons = [], []
    for contour in contours[:1000]:
        points = contour.squeeze(1)
        if points.ndim != 2 or len(points) < 4 or _score(probability, points) < config["box_threshold"]:
            continue
        points = _unclip(points, config["unclip_ratio"])
        if points is None:
            continue
        rectangle = cv2.minAreaRect(points)
        if min(rectangle[1]) < 3:
            continue
        box = cv2.boxPoints(rectangle).astype(np.float32)
        box[:, 0] = np.clip(box[:, 0] / rw * width, 0, width)
        box[:, 1] = np.clip(box[:, 1] / rh * height, 0, height)
        boxes.append((float(box[:, 0].min()), float(box[:, 1].min()), float(box[:, 0].max()), float(box[:, 1].max())))
        polygons.append([box.tolist()])
    return boxes, polygons


def image_files(folder):
    return sorted(str(p) for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES)


def main():
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model = None
    for dataset_name, dataset in DATASETS.items():
        output_file = output_dir / f"{dataset_name}_{MODEL_NAME}.json"
        if output_file.is_file():
            old = json.loads(output_file.read_text(encoding="utf-8")).get("gpu") or {}
            if result_is_compatible(old, dataset) and prediction_artifact_path(output_file).is_file():
                continue
        images = image_files(dataset["folder"])
        if not images or not Path(dataset["annotations"]).is_file():
            continue
        if model is None:
            model = create_model(CONFIG)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        process = psutil.Process()
        ram0 = process.memory_info().rss / 1024**2
        for path in images[:CONFIG["warmup"]]:
            detect(model, path, CONFIG)
        torch.cuda.synchronize()
        predictions, polygon_predictions, timings = {}, {}, []
        ram_peak = ram0
        for path in images:
            started = time.perf_counter()
            boxes, polygons = detect(model, path, CONFIG)
            key = prediction_key(path, dataset)
            predictions[key], polygon_predictions[key] = boxes, polygons
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            ram_peak = max(ram_peak, process.memory_info().rss / 1024**2)
        values = np.asarray(timings)
        stats = {
            "device": CONFIG["device"], "architecture": "DBNet MobileNetV3 + FPN",
            "word_level_target": True, "repository": CONFIG["origin"], "num_images": len(images),
            "mean_time_ms": float(values.mean() * 1000), "median_time_ms": float(np.median(values) * 1000),
            "throughput_fps": float(len(images) / values.sum()), "ram_after_load_mb": ram0,
            "ram_peak_mb": ram_peak, "ram_delta_mb": ram_peak - ram0,
            "gpu_after_load_mb": 0.0, "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }
        save_predictions(output_file, predictions, polygon_predictions)
        gt_boxes, gt_polygons = load_coco_detection_ground_truth(dataset["annotations"])
        stats["accuracy_metrics"] = evaluate_dataset(predictions, gt_boxes)
        stats["polygon_iou_metrics"] = evaluate_polygon_dataset(polygon_predictions, gt_polygons)
        output_file.write_text(json.dumps({"dataset": dataset_name, "cpu": None, "gpu": stats}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {output_file}", flush=True)


if __name__ == "__main__":
    main()
