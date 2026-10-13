"""Benchmark Nemotron OCR v2 multilingual inside its Linux/CUDA container."""

import csv
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
from tqdm import tqdm

from utils.datasets import prediction_key, result_is_compatible
from utils.metrics import (
    accuracy_score,
    cer_score,
    character_similarity_score,
    evaluate_dataset,
    evaluate_polygon_dataset,
    load_coco_detection_ground_truth,
    normalize_text,
    wer_score,
)
from utils.prediction_artifacts import prediction_artifact_path, save_predictions


MODEL_NAME = "nemotron_ocr_v2_multilingual_line"
MODEL_DIR = Path("/workspace/bench/models/nemotron-ocr-v2/v2_multilingual")
ORIGIN = "https://huggingface.co/nvidia/nemotron-ocr-v2"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_files(folder):
    return sorted(
        path for path in Path(folder).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def gpu_memory_mb():
    return torch.cuda.memory_allocated() / 1024**2


def run_detection():
    from configs.line_detection.benchmark_config import DATASETS, RESULTS_ROOT
    from scripts.line_detection.prepare_datasets import PREPARATION_VERSION

    output_dir = RESULTS_ROOT / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    model = NemotronOCRV2(model_dir=MODEL_DIR, detector_only=True)
    process = psutil.Process()
    for dataset_name, dataset in DATASETS.items():
        output_file = output_dir / f"{dataset_name}_{MODEL_NAME}.json"
        if output_file.is_file():
            existing = json.loads(output_file.read_text(encoding="utf-8")).get("gpu") or {}
            if (result_is_compatible(existing, dataset)
                    and existing.get("ground_truth_version") == PREPARATION_VERSION
                    and prediction_artifact_path(output_file).is_file()):
                print(f"Skip: already completed ({output_file})", flush=True)
                continue
        images = image_files(dataset["folder"])
        if not images or not Path(dataset["annotations"]).is_file():
            print(f"Skip {dataset_name}: missing prepared data", flush=True)
            continue
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model(str(images[0]), merge_level="word")
        torch.cuda.synchronize()
        ram_after_load = process.memory_info().rss / 1024**2
        gpu_after_load = gpu_memory_mb()
        predictions, polygons_by_image, timings = {}, {}, []
        ram_peak = ram_after_load
        for path in tqdm(images, desc=f"Nemotron detection / {dataset_name}"):
            started = time.perf_counter()
            regions = model(str(path), merge_level="word")
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            polygons = [[region["quad"]] for region in regions]
            boxes = []
            for parts in polygons:
                points = np.asarray(parts[0], dtype=np.float32)
                boxes.append((float(points[:, 0].min()), float(points[:, 1].min()),
                              float(points[:, 0].max()), float(points[:, 1].max())))
            key = prediction_key(path, dataset)
            predictions[key] = boxes
            polygons_by_image[key] = polygons
            ram_peak = max(ram_peak, process.memory_info().rss / 1024**2)
        values = np.asarray(timings)
        save_predictions(output_file, predictions, polygons_by_image)
        gt_boxes, gt_polygons = load_coco_detection_ground_truth(dataset["annotations"])
        stats = {
            "device": "cuda:0",
            "detection_level": "line",
            "architecture": "Nemotron OCR v2 multilingual detector",
            "repository": ORIGIN,
            "ground_truth_version": PREPARATION_VERSION,
            "num_images": len(images),
            "mean_time_ms": float(values.mean() * 1000),
            "median_time_ms": float(np.median(values) * 1000),
            "throughput_fps": float(len(images) / values.sum()),
            "ram_after_load_mb": ram_after_load,
            "ram_peak_mb": ram_peak,
            "ram_delta_mb": ram_peak - ram_after_load,
            "gpu_after_load_mb": gpu_after_load,
            "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "accuracy_metrics": evaluate_dataset(predictions, gt_boxes),
            "polygon_iou_metrics": evaluate_polygon_dataset(polygons_by_image, gt_polygons),
        }
        output_file.write_text(json.dumps({"dataset": dataset_name, "level": "line", "cpu": None, "gpu": stats}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {output_file}", flush=True)


def load_recognition_samples(config):
    with Path(config["labels"]).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [{**row, "path": Path(config["images_dir"]) / row["image"]} for row in rows]


def joined_text(regions):
    ordered = sorted(regions, key=lambda item: (item.get("upper", 0), item.get("left", 0)))
    return " ".join(item.get("text", "").strip() for item in ordered if item.get("text", "").strip())


def run_recognition():
    os.environ["OCR_BENCH_RECOGNITION_LEVEL"] = "line"
    from configs.recognition.benchmark_config import DATASETS, RESULTS_ROOT

    output_dir = RESULTS_ROOT / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    model = NemotronOCRV2(model_dir=MODEL_DIR, skip_relational=True)
    process = psutil.Process()
    batch_size = 8
    for dataset_name, dataset in DATASETS.items():
        result_path = output_dir / f"{dataset_name}_{MODEL_NAME}.json"
        predictions_path = output_dir / f"{dataset_name}_{MODEL_NAME}.csv"
        if result_path.is_file():
            metrics = (json.loads(result_path.read_text(encoding="utf-8")).get("gpu") or {}).get("accuracy_metrics", {})
            if all(key in metrics for key in ("cer", "wer")):
                print(f"Skip: already completed ({result_path})", flush=True)
                continue
        samples = load_recognition_samples(dataset)
        if not samples or not all(sample["path"].is_file() for sample in samples):
            print(f"Skip {dataset_name}: missing prepared data", flush=True)
            continue
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model(str(samples[0]["path"]), merge_level="word")
        torch.cuda.synchronize()
        ram_after_load = process.memory_info().rss / 1024**2
        gpu_after_load = gpu_memory_mb()
        predictions, confidences = [], []
        started = time.perf_counter()
        for offset in tqdm(range(0, len(samples), batch_size), desc=f"Nemotron recognition / {dataset_name}"):
            batch = samples[offset:offset + batch_size]
            outputs = model([str(sample["path"]) for sample in batch], merge_level="word")
            for regions in outputs:
                predictions.append(joined_text(regions))
                confidences.append(float(np.mean([item.get("confidence", 0.0) for item in regions])) if regions else 0.0)
        torch.cuda.synchronize()
        total_time = time.perf_counter() - started
        references = [sample["text"] for sample in samples]
        norm_ref = [normalize_text(text) for text in references]
        norm_pred = [normalize_text(text) for text in predictions]
        metrics = {
            "cer": cer_score(norm_pred, norm_ref),
            "wer": wer_score(norm_pred, norm_ref),
            "exact_match": accuracy_score(norm_pred, norm_ref),
            "character_similarity": character_similarity_score(norm_pred, norm_ref),
            "evaluated_words": len(samples),
        }
        stats = {
            "device": "cuda",
            "num_words": len(samples),
            "batch_size": batch_size,
            "total_time_s": total_time,
            "mean_time_ms": total_time * 1000 / len(samples),
            "throughput_words_s": len(samples) / total_time,
            "ram_after_load_mb": ram_after_load,
            "ram_peak_mb": process.memory_info().rss / 1024**2,
            "gpu_after_load_mb": gpu_after_load,
            "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "mean_confidence": float(np.mean(confidences)),
            "accuracy_metrics": metrics,
        }
        result_path.write_text(json.dumps({"task": "recognition", "dataset": dataset_name, "model": MODEL_NAME, "cpu": None, "gpu": stats}, ensure_ascii=False, indent=2), encoding="utf-8")
        with predictions_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "prediction", "reference", "confidence"])
            for sample, prediction, reference, confidence in zip(samples, predictions, references, confidences):
                writer.writerow([sample["image"], prediction, reference, confidence])
        print(f"Saved: {result_path}", flush=True)


if __name__ == "__main__":
    if os.environ.get("OCR_BENCH_NEMOTRON_TASK") == "recognition":
        run_recognition()
    else:
        run_detection()
