import csv
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil

from configs.recognition.benchmark_config import BENCHMARKS, DATASETS
from manuscript.recognizers import TRBA
from scripts.recognition.prepare_handwritten_essay import prepare_dataset
from utils.metrics import (
    accuracy_score,
    cer_score,
    character_similarity_score,
    normalize_text,
    wer_score,
)


def load_samples(config):
    with config["labels"].open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return [
        {
            **row,
            "path": config["images_dir"] / row["image"],
        }
        for row in rows
    ]


def completed_result(path):
    if not path.is_file():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = (result.get("gpu") or {}).get("accuracy_metrics", {})
    return all(key in metrics for key in ("cer", "wer"))


def run_model(model_name, model_config, dataset_name, dataset_config):
    output_dir = model_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{dataset_name}_{model_name}.json"
    predictions_path = output_dir / f"{dataset_name}_{model_name}.csv"
    if completed_result(result_path):
        print(f"Skip: already completed ({result_path})")
        return

    samples = load_samples(dataset_config)
    image_paths = [sample["path"] for sample in samples]
    references = [sample["text"] for sample in samples]
    if not samples or not all(path.is_file() for path in image_paths):
        raise FileNotFoundError(f"Prepared recognition dataset is incomplete: {dataset_name}")

    gc.collect()
    recognizer = TRBA(
        weights=model_config["weights"],
        device=model_config["device"],
        batch_size=model_config["batch_size"],
    )
    warmup = image_paths[: model_config["warmup_size"]]
    recognizer._predict_word_images(warmup, batch_size=model_config["batch_size"])
    providers = recognizer.onnx_session.get_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"TRBA requested CUDA but active providers are: {providers}")

    ram_after_load = psutil.Process().memory_info().rss / 1024**2
    started = time.perf_counter()
    raw_predictions = recognizer._predict_word_images(
        image_paths, batch_size=model_config["batch_size"]
    )
    total_time = time.perf_counter() - started
    ram_peak = psutil.Process().memory_info().rss / 1024**2
    predictions = [item.get("text", "") for item in raw_predictions]
    confidences = [float(item.get("confidence", 0.0)) for item in raw_predictions]

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
    gpu_stats = {
        "device": "cuda",
        "active_providers": providers,
        "num_words": len(samples),
        "batch_size": model_config["batch_size"],
        "total_time_s": total_time,
        "mean_time_ms": total_time * 1000 / len(samples),
        "throughput_words_s": len(samples) / total_time,
        "ram_after_load_mb": ram_after_load,
        "ram_peak_mb": ram_peak,
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
        # AzbukaBoard/get_metrics.py expects image and prediction in columns 0 and 1.
        writer.writerow(["image", "prediction", "reference", "confidence"])
        for sample, reference, prediction, confidence in zip(
            samples, references, predictions, confidences
        ):
            writer.writerow([sample["image"], prediction, reference, confidence])
    print(f"Saved: {result_path}")


def main():
    for dataset_name, dataset_config in DATASETS.items():
        print(f"\n### DATASET: {dataset_name}")
        prepare_dataset(dataset_name, dataset_config)
        for model_name, model_config in BENCHMARKS.items():
            if os.environ.get("OCR_BENCH_ONLY_MODEL") not in (None, model_name):
                continue
            if not model_config.get("run", False) or model_config.get("backend") != "trba":
                continue
            print(f"\n## MODEL: {model_name}")
            run_model(model_name, model_config, dataset_name, dataset_config)


if __name__ == "__main__":
    main()
