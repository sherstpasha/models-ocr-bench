"""Benchmark the docTR word-level DB-ResNet50 detector."""

import gc
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image

from configs.detection.benchmark_config import BENCHMARKS, DATASETS
from utils.datasets import prediction_key, result_is_compatible
from utils.metrics import evaluate_dataset
from utils.prediction_artifacts import prediction_artifact_path, save_predictions


MODEL_NAME = "doctr_db_resnet50"
CONFIG = BENCHMARKS[MODEL_NAME]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_image_files(folder):
    return sorted(
        str(path)
        for path in Path(folder).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_ground_truth(coco_json):
    source = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    image_names = {image["id"]: image["file_name"] for image in source["images"]}
    ground_truths = {}
    for annotation in source["annotations"]:
        filename = image_names.get(annotation["image_id"])
        segmentation = annotation.get("segmentation")
        if not filename or not segmentation:
            continue
        parts = segmentation if isinstance(segmentation[0], list) else [segmentation]
        for part in parts:
            points = np.asarray(part, dtype=np.float32).reshape(-1, 2)
            ground_truths.setdefault(filename, []).append(
                (
                    float(points[:, 0].min()),
                    float(points[:, 1].min()),
                    float(points[:, 0].max()),
                    float(points[:, 1].max()),
                )
            )
    return ground_truths


def normalized_words_to_pixels(words, width, height):
    boxes = []
    for word in np.asarray(words, dtype=np.float32):
        if word.size < 4:
            continue
        x_min, y_min, x_max, y_max = map(float, word[:4])
        box = (
            x_min * width,
            y_min * height,
            x_max * width,
            y_max * height,
        )
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    return boxes


def detect_words(predictor, image_path):
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    height, width = image.shape[:2]
    output = predictor([image])
    if len(output) != 1 or "words" not in output[0]:
        raise RuntimeError(f"Unexpected docTR detection output for {image_path}")
    return normalized_words_to_pixels(output[0]["words"], width, height)


def create_predictor():
    from doctr.models import detection_predictor

    predictor = detection_predictor(
        CONFIG["architecture"],
        pretrained=True,
        assume_straight_pages=CONFIG["assume_straight_pages"],
        preserve_aspect_ratio=CONFIG["preserve_aspect_ratio"],
        symmetric_pad=CONFIG["symmetric_pad"],
    )
    return predictor.cuda().eval()


def benchmark_gpu(images, dataset_config):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    predictor = create_predictor()
    ram_after_load = psutil.Process().memory_info().rss / 1024**2
    gpu_after_load = torch.cuda.memory_allocated() / 1024**2

    for image_path in images[: CONFIG["warmup"]]:
        detect_words(predictor, image_path)
    torch.cuda.synchronize()

    predictions = {}
    times = []
    with torch.inference_mode():
        for image_path in images:
            started = time.perf_counter()
            boxes = detect_words(predictor, image_path)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            predictions[prediction_key(image_path, dataset_config)] = boxes

    values = np.asarray(times)
    ram_peak = psutil.Process().memory_info().rss / 1024**2
    gpu_peak = torch.cuda.max_memory_allocated() / 1024**2
    return {
        "device": "cuda",
        "architecture": CONFIG["architecture"],
        "word_level": True,
        "num_images": len(images),
        "mean_time_ms": float(values.mean() * 1000),
        "median_time_ms": float(np.median(values) * 1000),
        "throughput_fps": float(len(images) / values.sum()),
        "ram_after_load_mb": ram_after_load,
        "ram_peak_mb": ram_peak,
        "gpu_after_load_mb": gpu_after_load,
        "gpu_peak_mb": gpu_peak,
        "gpu_delta_mb": gpu_peak - gpu_after_load,
        "predictions": predictions,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("docTR detection benchmark requires CUDA")

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, dataset in DATASETS.items():
        print(f"\n### DATASET: {dataset_name}")
        output_file = output_dir / f"{dataset_name}_{MODEL_NAME}.json"
        if output_file.is_file():
            existing = json.loads(output_file.read_text(encoding="utf-8")).get("gpu")
            if result_is_compatible(existing, dataset) and prediction_artifact_path(output_file).is_file():
                print(f"Skip: already completed ({output_file})")
                continue

        images = get_image_files(dataset["folder"])
        if not images or not dataset["annotations"].is_file():
            print("Skip: missing images or annotations")
            continue

        ground_truths = load_ground_truth(dataset["annotations"])
        gpu_stats = benchmark_gpu(images, dataset)
        save_predictions(output_file, gpu_stats["predictions"])
        gpu_stats["accuracy_metrics"] = evaluate_dataset(
            gpu_stats.pop("predictions"), ground_truths
        )
        result = {"dataset": dataset_name, "cpu": None, "gpu": gpu_stats}
        output_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
