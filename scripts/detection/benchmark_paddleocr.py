"""Benchmark PP-OCRv6 medium detection with word-level output."""

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil

from configs.detection.benchmark_config import BENCHMARKS, DATASETS
from utils.datasets import prediction_key, result_is_compatible
from utils.metrics import evaluate_dataset
from utils.prediction_artifacts import prediction_artifact_path, save_predictions


MODEL_NAME = "paddleocr_v6_medium_word"
CONFIG = BENCHMARKS[MODEL_NAME]

# PaddleX uses this directory for automatically downloaded official models.
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(CONFIG["model_dir"]))

# ModelScope (loaded by PaddleOCR) imports its CPU-only Torch dependency. On
# Windows Torch must load first, before Paddle adds its CUDA DLLs to the process.
import torch  # noqa: E402, F401
import paddle  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402


def get_image_files(folder):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        str(path)
        for path in Path(folder).rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
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


def cuda_synchronize():
    paddle.device.cuda.synchronize()


def gpu_memory_mb():
    return paddle.device.cuda.memory_allocated() / 1024**2


def result_payload(page):
    value = page.json if hasattr(page, "json") else page
    return value.get("res", value)


def box_to_xyxy(box):
    points = np.asarray(box, dtype=np.float32)
    if points.shape == (4,):
        x_min, y_min, x_max, y_max = map(float, points)
    else:
        points = points.reshape(-1, 2)
        x_min = float(points[:, 0].min())
        y_min = float(points[:, 1].min())
        x_max = float(points[:, 0].max())
        y_max = float(points[:, 1].max())
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def flatten_word_boxes(raw_boxes):
    boxes = []
    for line_boxes in raw_boxes:
        values = np.asarray(line_boxes, dtype=np.float32)
        if values.size == 0:
            continue
        if values.ndim == 1:
            values = values[np.newaxis, :]
        for value in values:
            box = box_to_xyxy(value)
            if box is not None:
                boxes.append(box)
    return boxes


def detect_words(ocr, image_path):
    pages = list(ocr.predict(str(image_path)))
    if len(pages) != 1:
        raise RuntimeError(f"Expected one PaddleOCR result for {image_path}")
    payload = result_payload(pages[0])
    raw_boxes = payload.get("text_word_boxes")
    if raw_boxes is None:
        if not payload.get("rec_texts"):
            return []
        raise RuntimeError(
            "PaddleOCR did not return text_word_boxes. Install PaddleOCR >= 3.7 "
            "and keep return_word_box=True."
        )
    return flatten_word_boxes(raw_boxes)


def create_pipeline():
    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)
    return PaddleOCR(
        text_detection_model_name=CONFIG["detection_model"],
        text_recognition_model_name=CONFIG["recognition_model"],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=CONFIG["return_word_box"],
        text_det_limit_side_len=CONFIG["text_det_limit_side_len"],
        text_det_limit_type=CONFIG["text_det_limit_type"],
        device=CONFIG["device"],
    )


def benchmark_gpu(images, dataset_config):
    gc.collect()
    paddle.device.cuda.empty_cache()
    paddle.device.cuda.reset_max_memory_allocated()

    ocr = create_pipeline()
    ram_after_load = psutil.Process().memory_info().rss / 1024**2
    gpu_after_load = gpu_memory_mb()

    for image_path in images[: CONFIG["warmup"]]:
        detect_words(ocr, image_path)
    cuda_synchronize()

    predictions = {}
    times = []
    for image_path in images:
        started = time.perf_counter()
        boxes = detect_words(ocr, image_path)
        cuda_synchronize()
        times.append(time.perf_counter() - started)
        predictions[prediction_key(image_path, dataset_config)] = boxes

    values = np.asarray(times)
    ram_peak = psutil.Process().memory_info().rss / 1024**2
    gpu_peak = paddle.device.cuda.max_memory_allocated() / 1024**2
    return {
        "device": CONFIG["device"],
        "detection_model": CONFIG["detection_model"],
        "recognition_model_for_word_split": CONFIG["recognition_model"],
        "return_word_box": True,
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
    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
        raise RuntimeError("PaddleOCR benchmark requires PaddlePaddle with CUDA")

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
