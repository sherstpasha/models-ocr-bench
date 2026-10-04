"""Benchmark PP-OCRv6 medium detector on text lines."""

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil

from configs.line_detection.benchmark_config import BENCHMARKS, DATASETS
from scripts.line_detection.prepare_datasets import PREPARATION_VERSION
from utils.datasets import prediction_key, result_is_compatible
from utils.metrics import evaluate_dataset, evaluate_polygon_dataset, load_coco_detection_ground_truth
from utils.prediction_artifacts import prediction_artifact_path, save_predictions


MODEL_NAME = "paddleocr_v6_medium_line"
CONFIG = BENCHMARKS[MODEL_NAME]
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(CONFIG["model_dir"]))

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
        bbox = annotation.get("bbox")
        if not filename or not bbox:
            continue
        x, y, width, height = map(float, bbox)
        ground_truths.setdefault(filename, []).append((x, y, x + width, y + height))
    return ground_truths


def result_payload(page):
    value = page.json if hasattr(page, "json") else page
    return value.get("res", value)


def box_to_xyxy(box):
    points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    if x_max <= x_min or y_max <= y_min:
        return None
    return float(x_min), float(y_min), float(x_max), float(y_max)


def detect_lines(ocr, image_path):
    pages = list(ocr.predict(str(image_path)))
    if len(pages) != 1:
        raise RuntimeError(f"Expected one PaddleOCR result for {image_path}")
    payload = result_payload(pages[0])
    raw_boxes = payload.get("dt_polys")
    if raw_boxes is None:
        raise RuntimeError("PaddleOCR result does not contain line-level dt_polys")
    boxes = []
    polygons = []
    for raw in raw_boxes:
        box = box_to_xyxy(raw)
        if box is not None:
            boxes.append(box)
            polygons.append([np.asarray(raw, dtype=np.float32).reshape(-1, 2).tolist()])
    return boxes, polygons


def create_pipeline():
    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)
    return PaddleOCR(
        text_detection_model_name=CONFIG["detection_model"],
        text_recognition_model_name=CONFIG["recognition_model"],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=False,
        text_det_limit_side_len=CONFIG["text_det_limit_side_len"],
        text_det_limit_type=CONFIG["text_det_limit_type"],
        device=CONFIG["device"],
    )


def gpu_memory_mb():
    return paddle.device.cuda.memory_allocated() / 1024**2


def benchmark_gpu(images, dataset_config):
    gc.collect()
    paddle.device.cuda.empty_cache()
    paddle.device.cuda.reset_max_memory_allocated()
    ocr = create_pipeline()
    ram_after_load = psutil.Process().memory_info().rss / 1024**2
    gpu_after_load = gpu_memory_mb()
    for image_path in images[: CONFIG["warmup"]]:
        detect_lines(ocr, image_path)
    paddle.device.cuda.synchronize()

    predictions = {}
    polygon_predictions = {}
    times = []
    for image_path in images:
        started = time.perf_counter()
        boxes, polygons = detect_lines(ocr, image_path)
        paddle.device.cuda.synchronize()
        times.append(time.perf_counter() - started)
        predictions[prediction_key(image_path, dataset_config)] = boxes
        polygon_predictions[prediction_key(image_path, dataset_config)] = polygons
    values = np.asarray(times)
    gpu_peak = paddle.device.cuda.max_memory_allocated() / 1024**2
    return {
        "device": CONFIG["device"],
        "detection_level": "line",
        "detection_model": CONFIG["detection_model"],
        "return_word_box": False,
        "num_images": len(images),
        "mean_time_ms": float(values.mean() * 1000),
        "median_time_ms": float(np.median(values) * 1000),
        "throughput_fps": float(len(images) / values.sum()),
        "ram_after_load_mb": ram_after_load,
        "ram_peak_mb": psutil.Process().memory_info().rss / 1024**2,
        "gpu_after_load_mb": gpu_after_load,
        "gpu_peak_mb": gpu_peak,
        "gpu_delta_mb": gpu_peak - gpu_after_load,
        "predictions": predictions,
        "polygon_predictions": polygon_predictions,
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
            if result_is_compatible(existing, dataset) and existing.get("ground_truth_version") == PREPARATION_VERSION and prediction_artifact_path(output_file).is_file():
                print(f"Skip: already completed ({output_file})")
                continue
        images = get_image_files(dataset["folder"])
        if not images or not dataset["annotations"].is_file():
            print("Skip: missing images or annotations")
            continue
        gpu_stats = benchmark_gpu(images, dataset)
        gpu_stats["ground_truth_version"] = PREPARATION_VERSION
        save_predictions(output_file, gpu_stats["predictions"], gpu_stats["polygon_predictions"])
        ground_truth_boxes, ground_truth_polygons = load_coco_detection_ground_truth(dataset["annotations"])
        gpu_stats["accuracy_metrics"] = evaluate_dataset(gpu_stats.pop("predictions"), ground_truth_boxes)
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
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
