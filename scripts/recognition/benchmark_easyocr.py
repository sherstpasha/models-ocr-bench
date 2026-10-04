import csv
import gc
import json
import os
import time

import cv2
import easyocr
import numpy as np
import psutil
import torch

from configs.recognition.benchmark_config import BENCHMARKS, DATASETS
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
    return [{**row, "path": config["images_dir"] / row["image"]} for row in rows]


def completed_result(path):
    if not path.is_file():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = (result.get("gpu") or {}).get("accuracy_metrics", {})
    return all(key in metrics for key in ("cer", "wer"))


def recognize(reader, image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return "", 0.0
    height, width = image.shape
    output = reader.recognize(
        image,
        horizontal_list=[[0, width, 0, height]],
        free_list=[],
        detail=1,
    )
    if not output:
        return "", 0.0
    return str(output[0][1]), float(output[0][2])


def run_model(model_name, model_config, dataset_name, dataset_config):
    output_dir = model_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{dataset_name}_{model_name}.json"
    predictions_path = output_dir / f"{dataset_name}_{model_name}.csv"
    if completed_result(result_path):
        print(f"Skip: already completed ({result_path})")
        return

    samples = load_samples(dataset_config)
    if not samples or not all(sample["path"].is_file() for sample in samples):
        raise FileNotFoundError(f"Prepared recognition dataset is incomplete: {dataset_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("EasyOCR recognition benchmark requires CUDA")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    reader = easyocr.Reader(
        ["ru"],
        gpu=True,
        detector=False,
        recognizer=True,
        recog_network=model_config["recognition_network"],
        model_storage_directory=str(model_config["model_dir"]),
        download_enabled=True,
        verbose=False,
    )
    for sample in samples[: model_config["warmup_size"]]:
        recognize(reader, sample["path"])
    torch.cuda.synchronize()

    ram_after_load = psutil.Process().memory_info().rss / 1024**2
    predictions = []
    confidences = []
    started = time.perf_counter()
    for sample in samples:
        prediction, confidence = recognize(reader, sample["path"])
        predictions.append(prediction)
        confidences.append(confidence)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - started
    ram_peak = psutil.Process().memory_info().rss / 1024**2

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
    gpu_stats = {
        "device": "cuda",
        "num_words": len(samples),
        "total_time_s": total_time,
        "mean_time_ms": total_time * 1000 / len(samples),
        "throughput_words_s": len(samples) / total_time,
        "ram_after_load_mb": ram_after_load,
        "ram_peak_mb": ram_peak,
        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
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
        for model_name, model_config in BENCHMARKS.items():
            if os.environ.get("OCR_BENCH_ONLY_MODEL") not in (None, model_name):
                continue
            if (
                not model_config.get("run", False)
                or model_config.get("backend") != "easyocr"
            ):
                continue
            print(f"\n## MODEL: {model_name}")
            run_model(model_name, model_config, dataset_name, dataset_config)


if __name__ == "__main__":
    main()
