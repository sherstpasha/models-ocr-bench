"""Benchmark the Kansallisarkisto RF-DETR 2XL text-line model."""

import gc
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from rfdetr import RFDETRSeg2XLarge
from tqdm import tqdm

from configs.line_detection.benchmark_config import BENCHMARKS, DATASETS
from scripts.line_detection.prepare_datasets import PREPARATION_VERSION
from utils.datasets import prediction_key, result_is_compatible
from utils.metrics import boxes_to_polygon_objects, evaluate_dataset, evaluate_polygon_dataset, load_coco_detection_ground_truth
from utils.prediction_artifacts import prediction_artifact_path, save_predictions


MODEL_NAME = "rfdetr_textline_textregion_2xl"
CONFIG = BENCHMARKS[MODEL_NAME]


def load_ground_truth(coco_json):
    source = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    image_names = {image["id"]: image["file_name"] for image in source["images"]}
    ground_truths = {}
    for annotation in source["annotations"]:
        filename = image_names.get(annotation["image_id"])
        bbox = annotation.get("bbox")
        if filename and bbox:
            x, y, width, height = map(float, bbox)
            ground_truths.setdefault(filename, []).append((x, y, x + width, y + height))
    return ground_truths


def get_image_files(folder):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        str(path)
        for path in Path(folder).rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def download_weights():
    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=CONFIG["repository"],
            filename=CONFIG["filename"],
            local_dir=CONFIG["model_dir"],
        )
    )


def create_model(weights):
    model = RFDETRSeg2XLarge(
        pretrain_weights=str(weights),
        device=CONFIG["device"],
    )
    # The published checkpoint was evaluated in FP16. Avoid compilation so the
    # benchmark has no large one-time tracing step and works with dynamic inputs.
    model.inference(compile=False, dtype=torch.float16, inplace=True)
    return model


def text_line_mask(detections):
    names = detections.data.get("class_name")
    if names is not None and len(names) == len(detections):
        return np.asarray(
            [str(name).lower().startswith("text_line") for name in names],
            dtype=bool,
        )
    if detections.class_id is None:
        raise RuntimeError("RF-DETR returned neither class_name nor class_id")
    # The model card defines text_region=1 and the four line classes=2..5.
    return np.isin(np.asarray(detections.class_id), [2, 3, 4, 5])


def detect_lines(model, image_path):
    with Image.open(image_path) as image:
        detections = model.predict(
            image.convert("RGB"),
            threshold=CONFIG["threshold"],
            include_source_image=False,
        )
    mask = text_line_mask(detections)
    return [tuple(map(float, box)) for box in np.asarray(detections.xyxy)[mask]]


def benchmark_gpu(model, images, dataset_config):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ram_after_load = psutil.Process().memory_info().rss / 1024**2
    gpu_after_load = torch.cuda.memory_allocated() / 1024**2
    for image_path in images[: CONFIG["warmup"]]:
        detect_lines(model, image_path)
    torch.cuda.synchronize()

    predictions = {}
    polygon_predictions = {}
    times = []
    for image_path in tqdm(images, desc="RF-DETR 2XL", unit="image"):
        started = time.perf_counter()
        boxes = detect_lines(model, image_path)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        predictions[prediction_key(image_path, dataset_config)] = boxes
        polygon_predictions[prediction_key(image_path, dataset_config)] = boxes_to_polygon_objects(boxes)
    values = np.asarray(times)
    gpu_peak = torch.cuda.max_memory_allocated() / 1024**2
    return {
        "device": CONFIG["device"],
        "detection_level": "line",
        "architecture": "RF-DETR Seg 2XLarge",
        "repository": CONFIG["repository"],
        "checkpoint": CONFIG["filename"],
        "threshold": CONFIG["threshold"],
        "line_classes": [
            "text_line_upright",
            "text_line_upside_down",
            "text_line_rotated_cw",
            "text_line_rotated_ccw",
        ],
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
    if not torch.cuda.is_available():
        raise RuntimeError("RF-DETR line benchmark requires CUDA")
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = download_weights()
    model = None
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
        if model is None:
            model = create_model(weights)
        gpu_stats = benchmark_gpu(model, images, dataset)
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
