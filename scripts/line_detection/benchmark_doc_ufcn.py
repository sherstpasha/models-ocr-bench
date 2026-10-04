"""Benchmark Teklia's generic historical Doc-UFCN line detector."""

import gc
import json
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


MODEL_NAME = "doc_ufcn_generic_historical_line"
CONFIG = BENCHMARKS[MODEL_NAME]


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


def download_model_files():
    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, filename in (
        ("weights", CONFIG["weights_filename"]),
        ("parameters", CONFIG["parameters_filename"]),
    ):
        target = CONFIG["model_dir"] / filename
        if target.is_file() and target.stat().st_size > 0:
            print(f"Using cached Doc-UFCN file: {target}", flush=True)
            paths[key] = target
            continue

        print(f"Resolving Doc-UFCN file: {filename}", flush=True)
        try:
            cached = Path(
                hf_hub_download(
                    repo_id=CONFIG["repository"],
                    filename=filename,
                    local_files_only=True,
                )
            )
            print(f"Found in Hugging Face cache: {cached}", flush=True)
        except Exception:
            print(f"Downloading Doc-UFCN file: {filename}", flush=True)
            cached = Path(
                hf_hub_download(
                    repo_id=CONFIG["repository"],
                    filename=filename,
                )
            )

        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(cached, temporary)
        temporary.replace(target)
        paths[key] = target
        print(f"Ready: {target}", flush=True)
    return paths


def create_model(paths):
    print("Reading Doc-UFCN parameters...", flush=True)
    payload = yaml.safe_load(paths["parameters"].read_text(encoding="utf-8"))
    parameters = payload["parameters"]
    print(
        f"Creating Doc-UFCN on {CONFIG['device']} "
        f"(input={parameters['input_size']} px)...",
        flush=True,
    )
    model = DocUFCN(
        len(parameters["classes"]),
        int(parameters["input_size"]),
        CONFIG["device"],
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


def detect_lines(model, image_path, min_cc):
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


def benchmark_gpu(model, parameters, images, dataset_config):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    ram_after_load = process.memory_info().rss / 1024**2
    gpu_after_load = torch.cuda.memory_allocated() / 1024**2
    min_cc = int(parameters["min_cc"])
    for image_path in images[: CONFIG["warmup"]]:
        detect_lines(model, image_path, min_cc)
    torch.cuda.synchronize()

    predictions = {}
    polygon_predictions = {}
    times = []
    ram_peak = ram_after_load
    for image_path in tqdm(images, desc="Doc-UFCN", unit="image"):
        started = time.perf_counter()
        boxes, polygons = detect_lines(model, image_path, min_cc)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        predictions[prediction_key(image_path, dataset_config)] = boxes
        polygon_predictions[prediction_key(image_path, dataset_config)] = polygons
        ram_peak = max(ram_peak, process.memory_info().rss / 1024**2)
    values = np.asarray(times)
    gpu_peak = torch.cuda.max_memory_allocated() / 1024**2
    return {
        "device": CONFIG["device"],
        "detection_level": "line",
        "architecture": "Doc-UFCN",
        "repository": CONFIG["repository"],
        "input_size": int(parameters["input_size"]),
        "min_cc": min_cc,
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
            "Doc-UFCN line benchmark requires a CUDA build of PyTorch. "
            f"Installed torch: {torch.__version__}; CUDA runtime: {torch.version.cuda}. "
            "Reinstall requirements-doc-ufcn.txt with --upgrade --force-reinstall."
        )
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = download_model_files()
    model = parameters = None
    for dataset_name, dataset in DATASETS.items():
        print(f"\n### DATASET: {dataset_name}", flush=True)
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
            model, parameters = create_model(paths)
        print(f"Running {len(images)} images...", flush=True)
        gpu_stats = benchmark_gpu(model, parameters, images, dataset)
        gpu_stats["ground_truth_version"] = PREPARATION_VERSION
        save_predictions(
            output_file,
            gpu_stats["predictions"],
            gpu_stats["polygon_predictions"],
        )
        ground_truth_boxes, ground_truth_polygons = load_ground_truth(dataset["annotations"])
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
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
