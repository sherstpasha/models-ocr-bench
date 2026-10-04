"""Benchmark OpenOCR's RepViT-DB text detector."""

import gc
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
from huggingface_hub import hf_hub_download
from openocr import OpenOCR

from configs.detection.benchmark_config import BENCHMARKS, DATASETS
from utils.datasets import prediction_key, result_is_compatible
from utils.metrics import evaluate_dataset
from utils.prediction_artifacts import prediction_artifact_path, save_predictions


MODEL_NAME = "openocr_repvit_db"
CONFIG = BENCHMARKS[MODEL_NAME]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Load the CUDA/cuDNN wheels installed by requirements-gpu.txt on Windows.
_DLL_DIRECTORY_HANDLES = []
if os.name == "nt":
    nvidia_root = Path(ort.__file__).resolve().parent.parent / "nvidia"
    nvidia_bin_dirs = [str(path.resolve()) for path in nvidia_root.glob("*/bin")]
    for bin_dir in nvidia_bin_dirs:
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(bin_dir))
    # cuDNN loads several sub-libraries lazily by filename, and those lookups do
    # not consistently honor add_dll_directory on Windows.
    os.environ["PATH"] = os.pathsep.join(nvidia_bin_dirs + [os.environ["PATH"]])
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls(directory="")


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


def to_xyxy(raw_boxes):
    boxes = []
    for raw_box in raw_boxes:
        points = np.asarray(raw_box, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            continue
        box = (
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        )
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    return boxes


def detect(detector, image_path):
    results = detector(
        image_path=str(image_path),
        det_input_size=CONFIG["det_input_size"],
    )
    if len(results) != 1 or "boxes" not in results[0]:
        raise RuntimeError(f"Unexpected OpenOCR output for {image_path}")
    active = detector.model.onnx_det_engine.onnx_session.get_providers()
    if not active or active[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            "OpenOCR inference fell back from CUDA to CPU. Active providers: "
            f"{active}. Reinstall dependencies from requirements-gpu.txt."
        )
    return to_xyxy(results[0]["boxes"])


def create_detector():
    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)
    model_path = hf_hub_download(
        repo_id=CONFIG["repository"],
        filename=CONFIG["filename"],
        local_dir=CONFIG["model_dir"],
    )
    # OpenOCR 0.1.5 wraps the provider list in an extra tuple when GPU is
    # requested, which makes ONNX Runtime silently fall back to CPU. Create the
    # public detector on CPU and replace only its inference session with a
    # correctly configured CUDA session.
    logging.getLogger("openocr_unified").setLevel(logging.ERROR)
    detector = OpenOCR(
        task="det",
        backend="onnx",
        onnx_det_model_path=str(model_path),
        use_gpu="false",
    )
    engine = detector.model.onnx_det_engine
    engine.onnx_session = ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    engine.input_name = engine.get_input_name(engine.onnx_session)
    engine.output_name = engine.get_output_name(engine.onnx_session)
    if engine.onnx_session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            "OpenOCR model did not activate CUDAExecutionProvider: "
            f"{engine.onnx_session.get_providers()}"
        )
    return detector


def benchmark_gpu(images, dataset_config):
    gc.collect()
    detector = create_detector()
    process = psutil.Process()
    ram_after_load = process.memory_info().rss / 1024**2

    for image_path in images[: CONFIG["warmup"]]:
        detect(detector, image_path)

    predictions = {}
    times = []
    ram_peak = ram_after_load
    for image_path in images:
        started = time.perf_counter()
        boxes = detect(detector, image_path)
        times.append(time.perf_counter() - started)
        ram_peak = max(ram_peak, process.memory_info().rss / 1024**2)
        predictions[prediction_key(image_path, dataset_config)] = boxes

    values = np.asarray(times)
    return {
        "device": "cuda:0",
        "provider": "CUDAExecutionProvider",
        "architecture": "RepViT-DB",
        "word_level_target": True,
        "num_images": len(images),
        "mean_time_ms": float(values.mean() * 1000),
        "median_time_ms": float(np.median(values) * 1000),
        "throughput_fps": float(len(images) / values.sum()),
        "ram_after_load_mb": ram_after_load,
        "ram_peak_mb": ram_peak,
        "ram_delta_mb": ram_peak - ram_after_load,
        "predictions": predictions,
    }


def main():
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "OpenOCR GPU benchmark requires onnxruntime-gpu with "
            "CUDAExecutionProvider"
        )

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
