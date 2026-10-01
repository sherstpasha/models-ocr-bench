import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import onnxruntime as ort

from utils.metrics import evaluate_dataset
from configs.benchmark_config import BENCHMARKS, DATASETS


CONFIG = BENCHMARKS["east_50_g1"]
OUTPUT_DIR = CONFIG["output_dir"]

if not CONFIG["cpu_only"]:
    # Load CUDA/cuDNN DLLs installed by the onnxruntime-gpu extras in .venv.
    ort.preload_dlls(directory="")

CUDA_ORT_AVAILABLE = "CUDAExecutionProvider" in ort.get_available_providers()


from manuscript.detectors import EAST


# =============================================================================
# HELPERS
# =============================================================================

def get_image_files(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = []
    base = Path(folder)
    for ext in exts:
        files.extend(base.glob(f"*{ext}"))
        files.extend(base.glob(f"*{ext.upper()}"))
    return sorted(list(dict.fromkeys(map(str, files))))


def load_ground_truth(annotation_file: str) -> Dict[str, List[tuple]]:
    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    ground_truths: Dict[str, List[tuple]] = {}

    for ann in coco["annotations"]:
        filename = id_to_name.get(ann["image_id"])
        if not filename:
            continue
        seg = ann.get("segmentation")
        if not seg:
            continue
        seg_parts = seg if isinstance(seg[0], list) else [seg]
        for seg_poly in seg_parts:
            if len(seg_poly) < 8:
                continue
            pts = np.array(seg_poly, dtype=np.float32).reshape(-1, 2)
            x_min = float(np.min(pts[:, 0]))
            y_min = float(np.min(pts[:, 1]))
            x_max = float(np.max(pts[:, 0]))
            y_max = float(np.max(pts[:, 1]))
            ground_truths.setdefault(filename, []).append((x_min, y_min, x_max, y_max))

    return ground_truths


def get_memory_usage() -> Dict[str, float]:
    import psutil

    ram_mb = psutil.Process().memory_info().rss / 1024 / 1024
    return {"ram_mb": ram_mb, "gpu_mb": None}


def extract_boxes(result: Any) -> List[tuple]:
    """Convert current and legacy manuscript-ocr results to xyxy boxes."""
    page = result.get("page") if isinstance(result, dict) else result
    if page is None or not hasattr(page, "blocks"):
        raise TypeError(f"Unsupported EAST result type: {type(result).__name__}")

    boxes = []
    for block in page.blocks:
        for line in getattr(block, "lines", []):
            text_spans = getattr(line, "text_spans", None)
            if text_spans is None:
                text_spans = getattr(line, "words", [])
            for text_span in text_spans:
                polygon = getattr(text_span, "polygon", None)
                if polygon:
                    xs = [point[0] for point in polygon]
                    ys = [point[1] for point in polygon]
                    boxes.append((min(xs), min(ys), max(xs), max(ys)))
                    continue

                bbox = getattr(text_span, "bbox", None)
                if bbox:
                    boxes.append(tuple(bbox))
                    continue

                geometry = getattr(text_span, "geometry", None)
                if geometry is not None and getattr(geometry, "bbox", None):
                    boxes.append(tuple(geometry.bbox))
    return boxes


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_device(image_files: List[str], device: str, collect_predictions: bool = False) -> Dict[str, Any]:
    gc.collect()

    detector_options = dict(
        device=device,
        target_size=CONFIG["target_size"],
        score_thresh=CONFIG["score_thresh"],
    )
    if CONFIG["weights"] is not None and CONFIG["preset"] is not None:
        raise ValueError("Set either EAST weights or preset, not both")
    model_source = CONFIG["weights"] or CONFIG["preset"]
    if model_source is not None:
        detector_options["weights"] = str(model_source)
    detector = EAST(**detector_options)
    mem_after_load = get_memory_usage()

    for img_path in image_files[: min(CONFIG["warmup_runs"], len(image_files))]:
        _ = detector.predict(img_path)

    if device == "cuda":
        active_providers = detector.onnx_session.get_providers()
        if "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError(
                "EAST requested CUDA but ONNX Runtime fell back to CPU. "
                f"Active providers: {active_providers}"
            )

    inference_times = []
    detection_counts = []
    predictions = {} if collect_predictions else None
    peak_memory = dict(mem_after_load)

    for img_path in image_files:
        start_time = time.time()
        result = detector.predict(img_path)
        inference_times.append(time.time() - start_time)

        boxes = extract_boxes(result)
        num_detections = len(boxes)

        detection_counts.append(num_detections)
        if collect_predictions:
            predictions[Path(img_path).name] = boxes

        current_mem = get_memory_usage()
        peak_memory["ram_mb"] = max(peak_memory["ram_mb"], current_mem["ram_mb"])
        if peak_memory["gpu_mb"] is not None and current_mem["gpu_mb"] is not None:
            peak_memory["gpu_mb"] = max(peak_memory["gpu_mb"], current_mem["gpu_mb"])

    times = np.array(inference_times)
    dets = np.array(detection_counts)

    stats = {
        "device": device,
        "num_images": len(image_files),
        "target_size": CONFIG["target_size"],
        "mean_time_ms": float(np.mean(times) * 1000),
        "median_time_ms": float(np.median(times) * 1000),
        "std_time_ms": float(np.std(times) * 1000),
        "min_time_ms": float(np.min(times) * 1000),
        "max_time_ms": float(np.max(times) * 1000),
        "total_time_s": float(np.sum(times)),
        "throughput_fps": float(len(image_files) / np.sum(times)),
        "mean_detections": float(np.mean(dets)) if len(dets) else 0.0,
        "total_detections": int(np.sum(dets)) if len(dets) else 0,
        "ram_after_load_mb": mem_after_load["ram_mb"],
        "ram_peak_mb": peak_memory["ram_mb"],
        "ram_delta_mb": peak_memory["ram_mb"] - mem_after_load["ram_mb"],
    }

    if device == "cuda":
        stats["gpu_after_load_mb"] = mem_after_load["gpu_mb"]
        stats["gpu_peak_mb"] = peak_memory["gpu_mb"]
        stats["gpu_delta_mb"] = (
            peak_memory["gpu_mb"] - mem_after_load["gpu_mb"]
            if peak_memory["gpu_mb"] is not None and mem_after_load["gpu_mb"] is not None
            else None
        )

    if collect_predictions:
        stats["predictions"] = predictions

    return stats


def save_results(cpu_stats: Dict[str, Any], gpu_stats: Dict[str, Any], output_file: Path, dataset_name: str) -> None:
    out = {"dataset_name": dataset_name, "cpu": cpu_stats, "gpu": gpu_stats}
    if cpu_stats and "predictions" in cpu_stats:
        del cpu_stats["predictions"]
    if gpu_stats and "predictions" in gpu_stats:
        del gpu_stats["predictions"]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def load_existing_results(output_file: Path):
    if not output_file.is_file():
        return None, None
    with output_file.open("r", encoding="utf-8") as f:
        result = json.load(f)
    return result.get("cpu"), result.get("gpu")


def main(output_dir=OUTPUT_DIR) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not CONFIG["cpu_only"] and not CUDA_ORT_AVAILABLE:
        providers = ", ".join(ort.get_available_providers())
        raise RuntimeError(
            "GPU benchmark requested, but CUDAExecutionProvider is unavailable. "
            f"Active ONNX Runtime providers: {providers}. Remove both onnxruntime "
            "packages and reinstall only onnxruntime-gpu."
        )

    for dataset_name, ds in DATASETS.items():
        print(f"\n### DATASET: {dataset_name}")
        output_file = Path(output_dir) / f"{dataset_name}_east_50_g1.json"
        cpu_stats, gpu_stats = load_existing_results(output_file)
        run_cpu = not CONFIG["gpu_only"]
        run_gpu = not CONFIG["cpu_only"]

        if (not run_cpu or cpu_stats) and (not run_gpu or gpu_stats):
            print(f"Skip: already completed ({output_file})")
            continue

        folder = ds["folder"]
        annotations = ds["annotations"]

        if not Path(folder).exists() or not Path(annotations).exists():
            print(f"Skip: missing path for {dataset_name}")
            continue

        image_files = get_image_files(folder)
        if not image_files:
            print(f"Skip: no images in {folder}")
            continue

        ground_truths = load_ground_truth(annotations)
        if run_cpu and not cpu_stats:
            print("Run missing device: cpu")
            cpu_stats = benchmark_device(image_files, "cpu", collect_predictions=True)
            cpu_stats["accuracy_metrics"] = evaluate_dataset(cpu_stats["predictions"], ground_truths)

        if run_gpu and not gpu_stats:
            print("Run missing device: cuda")
            gpu_stats = benchmark_device(image_files, "cuda", collect_predictions=True)
            gpu_stats["accuracy_metrics"] = evaluate_dataset(gpu_stats["predictions"], ground_truths)

        save_results(cpu_stats, gpu_stats, output_file, dataset_name)
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
