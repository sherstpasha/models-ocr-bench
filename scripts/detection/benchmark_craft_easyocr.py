import gc
import json
import time
from pathlib import Path
from typing import Dict

import easyocr
import numpy as np
import torch
from PIL import Image

from configs.detection.benchmark_config import BENCHMARKS, DATASETS
from utils.metrics import evaluate_dataset
from utils.datasets import prediction_key, result_is_compatible


MODEL_NAME = "craft_easyocr"
CONFIG = BENCHMARKS[MODEL_NAME]


def get_image_files(folder: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = []
    base = Path(folder)
    for ext in exts:
        files.extend(base.rglob(f"*{ext}"))
        files.extend(base.rglob(f"*{ext.upper()}"))
    return sorted(list(dict.fromkeys(map(str, files))))


def load_ground_truth(coco_json: str) -> Dict[str, list]:
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    ground_truths = {}
    for ann in coco["annotations"]:
        filename = id_to_name.get(ann["image_id"])
        segmentation = ann.get("segmentation")
        if not filename or not segmentation:
            continue
        segments = segmentation if isinstance(segmentation[0], list) else [segmentation]
        for segment in segments:
            points = np.asarray(segment, dtype=np.float32).reshape(-1, 2)
            ground_truths.setdefault(filename, []).append(
                (
                    float(points[:, 0].min()),
                    float(points[:, 1].min()),
                    float(points[:, 0].max()),
                    float(points[:, 1].max()),
                )
            )
    return ground_truths


def memory_usage():
    import psutil

    ram_mb = psutil.Process().memory_info().rss / 1024**2
    gpu_mb = torch.cuda.memory_allocated() / 1024**2
    return ram_mb, gpu_mb


def detect(reader, image):
    horizontal_lists, free_lists = reader.detect(
        image,
        min_size=0,
        text_threshold=CONFIG["text_threshold"],
        low_text=CONFIG["low_text"],
        link_threshold=CONFIG["link_threshold"],
        canvas_size=CONFIG["canvas_size"],
        mag_ratio=CONFIG["mag_ratio"],
        slope_ths=0.1,
        ycenter_ths=0.5,
        height_ths=0.5,
        width_ths=0.5,
        add_margin=0.1,
        optimal_num_chars=None,
    )

    boxes = []
    for x_min, x_max, y_min, y_max in horizontal_lists[0]:
        if x_max > x_min and y_max > y_min:
            boxes.append((float(x_min), float(y_min), float(x_max), float(y_max)))
    for polygon in free_lists[0]:
        points = np.asarray(polygon, dtype=np.float32)
        boxes.append(
            (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )
        )
    return boxes


def benchmark_gpu(images, collect_predictions=False, dataset_config=None):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    reader = easyocr.Reader(
        ["en", "ru"],
        gpu=True,
        model_storage_directory=str(CONFIG["model_dir"]),
        download_enabled=True,
        detector=True,
        recognizer=False,
    )
    ram0, gpu0 = memory_usage()

    for image_path in images[: CONFIG["warmup"]]:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        detect(reader, image)
    torch.cuda.synchronize()

    times = []
    predictions = {} if collect_predictions else None
    for image_path in images:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        start = time.perf_counter()
        boxes = detect(reader, image)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        if collect_predictions:
            predictions[prediction_key(image_path, dataset_config)] = boxes

    ram1, gpu1 = memory_usage()
    values = np.asarray(times)
    stats = {
        "device": "cuda",
        "num_images": len(images),
        "canvas_size": CONFIG["canvas_size"],
        "mean_time_ms": float(values.mean() * 1000),
        "median_time_ms": float(np.median(values) * 1000),
        "throughput_fps": float(len(images) / values.sum()),
        "ram_after_load_mb": ram0,
        "ram_peak_mb": ram1,
        "ram_delta_mb": ram1 - ram0,
        "gpu_after_load_mb": gpu0,
        "gpu_peak_mb": max(gpu1, torch.cuda.max_memory_allocated() / 1024**2),
    }
    stats["gpu_delta_mb"] = stats["gpu_peak_mb"] - gpu0
    if collect_predictions:
        stats["predictions"] = predictions
    return stats


def load_existing_gpu(output_file: Path):
    if not output_file.is_file():
        return None
    with output_file.open("r", encoding="utf-8") as f:
        return json.load(f).get("gpu")


def save_results(output_file: Path, dataset_name: str, gpu_stats):
    gpu_stats.pop("predictions", None)
    result = {"dataset": dataset_name, "cpu": None, "gpu": gpu_stats}
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only CRAFT benchmark requested, but CUDA is unavailable")

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, dataset in DATASETS.items():
        print(f"\n### DATASET: {dataset_name}")
        output_file = output_dir / f"{dataset_name}_{MODEL_NAME}.json"
        existing_gpu = load_existing_gpu(output_file)
        if result_is_compatible(existing_gpu, dataset):
            print(f"Skip: already completed ({output_file})")
            continue
        if not dataset["folder"].exists() or not dataset["annotations"].exists():
            print("Skip: missing folder or annotations")
            continue
        images = get_image_files(dataset["folder"])
        if not images:
            print("Skip: no images")
            continue

        ground_truths = load_ground_truth(dataset["annotations"])
        gpu_stats = benchmark_gpu(
            images, collect_predictions=True, dataset_config=dataset
        )
        gpu_stats["accuracy_metrics"] = evaluate_dataset(
            gpu_stats["predictions"], ground_truths
        )
        save_results(output_file, dataset_name, gpu_stats)
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
