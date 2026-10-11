"""Benchmark Teklia's generic historical Doc-UFCN line detector."""

import gc
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import yaml
from doc_ufcn.main import DocUFCN
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from configs.line_detection.benchmark_config import BENCHMARKS, DATASETS
from scripts.line_detection.prepare_datasets import PREPARATION_VERSION
from utils.datasets import prediction_key, result_is_compatible
from utils.metrics import evaluate_dataset, evaluate_polygon_dataset
from utils.prediction_artifacts import prediction_artifact_path, save_predictions


MODEL_NAMES = [
    name for name, config in BENCHMARKS.items()
    if config.get("backend") == "doc_ufcn" and config.get("run", False)
]


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
    boxes = {}
    polygons = {}
    for annotation in source["annotations"]:
        filename = image_names.get(annotation["image_id"])
        bbox = annotation.get("bbox")
        if filename and bbox:
            x, y, width, height = map(float, bbox)
            boxes.setdefault(filename, []).append((x, y, x + width, y + height))
            segmentation = annotation.get("segmentation") or []
            if segmentation and len(segmentation[0]) >= 6:
                polygon = [
                    np.asarray(part, dtype=np.float32).reshape(-1, 2).tolist()
                    for part in segmentation
                    if len(part) >= 6
                ]
            else:
                polygon = [[[x, y], [x + width, y], [x + width, y + height], [x, y + height]]]
            polygons.setdefault(filename, []).append(polygon)
    return boxes, polygons


def download_model_files(config):
    config["model_dir"].mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, filename in (
        ("weights", config["weights_filename"]),
        ("parameters", config["parameters_filename"]),
    ):
        target = config["model_dir"] / filename
        if target.is_file() and target.stat().st_size > 0:
            print(f"Using cached Doc-UFCN file: {target}", flush=True)
            paths[key] = target
            continue

        print(f"Resolving Doc-UFCN file: {filename}", flush=True)
        try:
            cached = Path(
                hf_hub_download(
                    repo_id=config["repository"],
                    filename=filename,
                    local_files_only=True,
                )
            )
            print(f"Found in Hugging Face cache: {cached}", flush=True)
        except Exception:
            print(f"Downloading Doc-UFCN file: {filename}", flush=True)
            cached = Path(
                hf_hub_download(
                    repo_id=config["repository"],
                    filename=filename,
                )
            )

        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(cached, temporary)
        temporary.replace(target)
        paths[key] = target
        print(f"Ready: {target}", flush=True)
    return paths


def create_model(paths, config):
    print("Reading Doc-UFCN parameters...", flush=True)
    payload = yaml.safe_load(paths["parameters"].read_text(encoding="utf-8"))
    parameters = payload["parameters"]
    print(
        f"Creating Doc-UFCN on {config['device']} "
        f"(input={parameters['input_size']} px)...",
        flush=True,
    )
    model = DocUFCN(
        len(parameters["classes"]),
        int(parameters["input_size"]),
        config["device"],
    )
    print("Loading Doc-UFCN checkpoint into GPU memory...", flush=True)
    model.load(
        str(paths["weights"]),
        list(parameters["mean"]),
        list(parameters["std"]),
        mode="eval",
    )
    print("Doc-UFCN model is ready.", flush=True)
    return model, parameters


def expand_mask_polygon(points, expand_x, expand_y, image_shape):
    """Dilate one polygon mask independently and return its outer contours."""
    if expand_x <= 0 and expand_y <= 0:
        return [points.tolist()]
    height, width = image_shape[:2]
    left = max(0, int(np.floor(points[:, 0].min())) - expand_x)
    top = max(0, int(np.floor(points[:, 1].min())) - expand_y)
    right = min(width - 1, int(np.ceil(points[:, 0].max())) + expand_x)
    bottom = min(height - 1, int(np.ceil(points[:, 1].max())) + expand_y)
    mask = np.zeros((bottom - top + 1, right - left + 1), dtype=np.uint8)
    local = np.rint(points - np.asarray([left, top])).astype(np.int32)
    cv2.fillPoly(mask, [local], 1)
    kernel = np.ones((2 * expand_y + 1, 2 * expand_x + 1), dtype=np.uint8)
    expanded = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    offset = np.asarray([left, top], dtype=np.float32)
    return [
        (contour.reshape(-1, 2).astype(np.float32) + offset).tolist()
        for contour in contours
        if len(contour) >= 3
    ]


def detect_lines(model, image_path, min_cc, config):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    polygons_by_class, _raw, _mask, _overlap = model.predict(image, min_cc=min_cc)
    boxes = []
    polygons = []
    for item in polygons_by_class.get(1, []):
        polygon = item.get("polygon")
        if polygon is None or len(polygon) < 3:
            continue
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        parts = expand_mask_polygon(
            points,
            int(config.get("mask_expand_x", 0)),
            int(config.get("mask_expand_y", 0)),
            image.shape,
        )
        if not parts:
            continue
        expanded_points = np.concatenate([np.asarray(part) for part in parts])
        polygons.append(parts)
        boxes.append(
            (
                float(expanded_points[:, 0].min()),
                float(expanded_points[:, 1].min()),
                float(expanded_points[:, 0].max()),
                float(expanded_points[:, 1].max()),
            )
        )
    return boxes, polygons


def benchmark_gpu(model, parameters, images, dataset_config, config):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    ram_after_load = process.memory_info().rss / 1024**2
    gpu_after_load = torch.cuda.memory_allocated() / 1024**2
    min_cc = int(parameters["min_cc"])
    for image_path in images[: config["warmup"]]:
        detect_lines(model, image_path, min_cc, config)
    torch.cuda.synchronize()

    predictions = {}
    polygon_predictions = {}
    times = []
    ram_peak = ram_after_load
    for image_path in tqdm(images, desc="Doc-UFCN", unit="image"):
        started = time.perf_counter()
        boxes, polygons = detect_lines(model, image_path, min_cc, config)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        predictions[prediction_key(image_path, dataset_config)] = boxes
        polygon_predictions[prediction_key(image_path, dataset_config)] = polygons
        ram_peak = max(ram_peak, process.memory_info().rss / 1024**2)
    values = np.asarray(times)
    gpu_peak = torch.cuda.max_memory_allocated() / 1024**2
    return {
        "device": config["device"],
        "detection_level": "line",
        "architecture": "Doc-UFCN",
        "repository": config["repository"],
        "input_size": int(parameters["input_size"]),
        "min_cc": min_cc,
        "mask_expand_x": int(config.get("mask_expand_x", 0)),
        "mask_expand_y": int(config.get("mask_expand_y", 0)),
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


def benchmark_model(model_name, config):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = download_model_files(config)
    model = parameters = None
    for dataset_name, dataset in DATASETS.items():
        print(f"\n### MODEL: {model_name} / DATASET: {dataset_name}", flush=True)
        output_file = output_dir / f"{dataset_name}_{model_name}.json"
        if output_file.is_file():
            existing = json.loads(output_file.read_text(encoding="utf-8")).get("gpu")
            expansion_matches = (
                existing.get("mask_expand_x", 0) == int(config.get("mask_expand_x", 0))
                and existing.get("mask_expand_y", 0) == int(config.get("mask_expand_y", 0))
            )
            if result_is_compatible(existing, dataset) and existing.get("ground_truth_version") == PREPARATION_VERSION and expansion_matches and prediction_artifact_path(output_file).is_file():
                print(f"Skip: already completed ({output_file})")
                continue
        images = get_image_files(dataset["folder"])
        if not images or not dataset["annotations"].is_file():
            print("Skip: missing images or annotations")
            continue
        if model is None:
            model, parameters = create_model(paths, config)
        print(f"Running {len(images)} images...", flush=True)
        gpu_stats = benchmark_gpu(model, parameters, images, dataset, config)
        gpu_stats["ground_truth_version"] = PREPARATION_VERSION
        save_predictions(output_file, gpu_stats["predictions"], gpu_stats["polygon_predictions"])
        ground_truth_boxes, ground_truth_polygons = load_ground_truth(dataset["annotations"])
        gpu_stats["accuracy_metrics"] = evaluate_dataset(gpu_stats.pop("predictions"), ground_truth_boxes)
        gpu_stats["polygon_iou_metrics"] = evaluate_polygon_dataset(gpu_stats.pop("polygon_predictions"), ground_truth_polygons)
        output_file.write_text(json.dumps({"dataset": dataset_name, "level": "line", "cpu": None, "gpu": gpu_stats}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {output_file}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Doc-UFCN line benchmark requires a CUDA build of PyTorch. "
            f"Installed torch: {torch.__version__}; CUDA runtime: {torch.version.cuda}. "
            "Reinstall requirements-doc-ufcn.txt with --upgrade --force-reinstall."
        )
    selected = os.environ.get("OCR_BENCH_ONLY_MODEL")
    for model_name in MODEL_NAMES:
        if selected not in (None, model_name):
            continue
        benchmark_model(model_name, BENCHMARKS[model_name])


if __name__ == "__main__":
    main()
