"""Benchmark PaddleOCR's Cyrillic PP-OCRv5 recognizer on word crops."""

import csv
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
from huggingface_hub import snapshot_download

from configs.recognition.benchmark_config import BENCHMARKS, DATASETS
from scripts.recognition.prepare_handwritten_essay import prepare_dataset
from utils.metrics import (
    accuracy_score,
    cer_score,
    character_similarity_score,
    normalize_text,
    wer_score,
)


MODEL_NAMES = (
    "paddleocr_cyrillic_v5_mobile",
    "paddleocr_eslav_v5_mobile",
    "paddleocr_v6_medium_ru",
    "paddleocr_v6_tiny_ru",
)

os.environ.setdefault(
    "PADDLE_PDX_CACHE_HOME",
    str(BENCHMARKS["paddleocr_cyrillic_v5_mobile"]["model_dir"]),
)

# ModelScope, imported internally by PaddleOCR, imports Torch. On Windows its
# CPU DLLs must be loaded before Paddle adds CUDA DLLs to the process.
import torch  # noqa: E402, F401
import paddle  # noqa: E402
from paddleocr import TextRecognition  # noqa: E402


def load_samples(config):
    with config["labels"].open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return [{**row, "path": config["images_dir"] / row["image"]} for row in rows]


def completed_result(path):
    if not path.is_file():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    gpu = result.get("gpu") or {}
    metrics = gpu.get("accuracy_metrics", {})
    return (
        all(key in metrics for key in ("cer", "wer"))
        and gpu.get("throughput_words_s") is not None
    )


def result_payload(result):
    value = result.json if hasattr(result, "json") else result
    return value.get("res", value)


def predict(recognizer, paths, batch_size):
    results = recognizer.predict([str(path) for path in paths], batch_size=batch_size)
    payloads = [result_payload(result) for result in results]
    predictions = [str(payload.get("rec_text", "")) for payload in payloads]
    confidences = [float(payload.get("rec_score", 0.0)) for payload in payloads]
    return predictions, confidences


def ensure_model(config):
    repository = config.get("repository")
    if not repository:
        return None
    model_dir = Path(config["model_dir"])
    required = ("inference.json", "inference.pdiparams", "inference.yml")
    if not all((model_dir / name).is_file() for name in required):
        snapshot_download(repo_id=repository, local_dir=model_dir)
    return model_dir


def enable_author_aspect_ratio_padding(recognizer):
    predictor = getattr(
        recognizer, "paddlex_predictor", getattr(recognizer, "predictor", recognizer)
    )
    transforms = getattr(predictor, "pre_tfs", None)
    if transforms is None:
        transforms = getattr(recognizer, "pre_tfs", None)
    if transforms is None or "ReisizeNorm" not in transforms:
        raise RuntimeError("Cannot locate PaddleOCR ReisizeNorm preprocessing transform")
    resize_op = transforms["ReisizeNorm"]
    original_resize = resize_op.resize_norm_img
    resize_op.resize_norm_img = lambda img, ratio: original_resize(
        img,
        min(max(ratio, 1.5 * img.shape[1] / img.shape[0]), resize_op.max_imgW / 48),
    )


def run_model(model_name, config, dataset_name, dataset_config):
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{dataset_name}_{model_name}.json"
    predictions_path = output_dir / f"{dataset_name}_{model_name}.csv"
    if completed_result(result_path):
        print(f"Skip: already completed ({result_path})")
        return

    samples = load_samples(dataset_config)
    paths = [sample["path"] for sample in samples]
    if not samples or not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Prepared recognition dataset is incomplete: {dataset_name}")
    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
        raise RuntimeError("PaddleOCR recognition benchmark requires PaddlePaddle with CUDA")

    gc.collect()
    paddle.set_device(config["device"])
    paddle.device.cuda.empty_cache()
    paddle.device.cuda.reset_max_memory_allocated()
    custom_dir = ensure_model(config)
    arguments = {"model_name": config["model_name"], "device": config["device"]}
    if custom_dir is not None:
        arguments["model_dir"] = str(custom_dir)
    recognizer = TextRecognition(**arguments)
    if config.get("aspect_ratio_padding"):
        enable_author_aspect_ratio_padding(recognizer)
    predict(
        recognizer,
        paths[: config["warmup_size"]],
        config["batch_size"],
    )
    paddle.device.synchronize()

    ram_after_load = psutil.Process().memory_info().rss / 1024**2
    gpu_after_load = paddle.device.cuda.memory_allocated() / 1024**2
    started = time.perf_counter()
    predictions, confidences = predict(recognizer, paths, config["batch_size"])
    paddle.device.synchronize()
    total_time = time.perf_counter() - started

    references = [sample["text"] for sample in samples]
    normalized_references = [normalize_text(text) for text in references]
    normalized_predictions = [normalize_text(text) for text in predictions]
    metrics = {
        "cer": cer_score(normalized_predictions, normalized_references),
        "wer": wer_score(normalized_predictions, normalized_references),
        "exact_match": accuracy_score(normalized_predictions, normalized_references),
        "character_similarity": character_similarity_score(
            normalized_predictions, normalized_references
        ),
        "evaluated_words": len(samples),
    }
    gpu_peak = paddle.device.cuda.max_memory_allocated() / 1024**2
    gpu_stats = {
        "device": config["device"],
        "model_name": config["model_name"],
        "num_words": len(samples),
        "batch_size": config["batch_size"],
        "total_time_s": total_time,
        "mean_time_ms": total_time * 1000 / len(samples),
        "throughput_words_s": len(samples) / total_time,
        "ram_after_load_mb": ram_after_load,
        "ram_peak_mb": psutil.Process().memory_info().rss / 1024**2,
        "gpu_after_load_mb": gpu_after_load,
        "gpu_peak_mb": gpu_peak,
        "gpu_delta_mb": gpu_peak - gpu_after_load,
        "mean_confidence": float(np.mean(confidences)),
        "accuracy_metrics": metrics,
    }
    result = {
        "task": "recognition",
        "dataset": dataset_name,
        "model": model_name,
        "cpu": None,
        "gpu": gpu_stats,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with predictions_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "prediction", "reference", "confidence"])
        for sample, prediction, reference, confidence in zip(
            samples, predictions, references, confidences
        ):
            writer.writerow([sample["image"], prediction, reference, confidence])
    print(f"Saved: {result_path}")


def main():
    for dataset_name, dataset_config in DATASETS.items():
        print(f"\n### DATASET: {dataset_name}")
        prepare_dataset(dataset_name, dataset_config)
        for model_name in MODEL_NAMES:
            if os.environ.get("OCR_BENCH_ONLY_MODEL") not in (None, model_name):
                continue
            print(f"\n## MODEL: {model_name}")
            run_model(model_name, BENCHMARKS[model_name], dataset_name, dataset_config)


if __name__ == "__main__":
    main()
