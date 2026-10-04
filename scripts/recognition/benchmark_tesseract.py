"""Benchmark official Tesseract tessdata_best models on word crops."""

import csv
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import psutil
from tqdm.auto import tqdm

from configs.recognition.benchmark_config import BENCHMARKS, DATASETS, PROJECT_ROOT
from scripts.recognition.prepare_handwritten_essay import prepare_dataset
from utils.metrics import (
    accuracy_score,
    cer_score,
    character_similarity_score,
    normalize_text,
    wer_score,
)


MODELS = {
    name: config
    for name, config in BENCHMARKS.items()
    if config.get("backend") == "tesseract"
}


def find_tesseract():
    candidates = [
        PROJECT_ROOT / "tools" / "tesseract" / "tesseract.exe",
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    ]
    command = shutil.which("tesseract")
    if command:
        candidates.insert(0, Path(command))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Tesseract 5 is not installed. Run: "
        "winget install --id tesseract-ocr.tesseract --exact"
    )


def ensure_traineddata(config):
    target = config["traineddata_path"]
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {target.name}...", flush=True)
    urllib.request.urlretrieve(config["traineddata_url"], target)
    return target


def load_samples(config):
    with config["labels"].open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return [{**row, "path": config["images_dir"] / row["image"]} for row in rows]


def completed_result(path):
    if not path.is_file():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    cpu = result.get("cpu") or {}
    metrics = cpu.get("accuracy_metrics", {})
    return (
        result.get("benchmark_version") == 2
        and
        all(key in metrics for key in ("cer", "wer", "exact_match"))
        and cpu.get("throughput_words_s") is not None
    )


def parse_tsv(payload, batch_size):
    words = [[] for _ in range(batch_size)]
    scores = [[] for _ in range(batch_size)]
    reader = csv.DictReader(payload.splitlines(), delimiter="\t")
    for row in reader:
        if row.get("level") != "5" or not row.get("text", "").strip():
            continue
        page = int(row["page_num"]) - 1
        if 0 <= page < batch_size:
            words[page].append(row["text"].strip())
            confidence = float(row.get("conf", -1))
            if confidence >= 0:
                scores[page].append(confidence / 100.0)
    predictions = [" ".join(items) for items in words]
    confidences = [float(np.mean(items)) if items else 0.0 for items in scores]
    return predictions, confidences


def recognize_batch(executable, config, samples, temporary_dir):
    list_path = temporary_dir / "images.txt"
    list_path.write_text(
        "\n".join(str(sample["path"].resolve()) for sample in samples) + "\n",
        encoding="utf-8",
    )
    command = [
        str(executable),
        str(list_path),
        "stdout",
        "--tessdata-dir",
        str(config["traineddata_path"].parent),
        "-l",
        config["language"],
        "--oem",
        "1",
        "--psm",
        str(config["psm"]),
        "-c",
        "tessedit_create_tsv=1",
    ]
    process = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return parse_tsv(process.stdout, len(samples))


def predict(executable, config, samples, description):
    predictions = []
    confidences = []
    batch_size = config["batch_size"]
    starts = tqdm(
        range(0, len(samples), batch_size),
        total=(len(samples) + batch_size - 1) // batch_size,
        desc=description,
        unit="batch",
    )
    with tempfile.TemporaryDirectory(prefix="tesseract-benchmark-") as directory:
        temporary_dir = Path(directory)
        for start in starts:
            batch_predictions, batch_confidences = recognize_batch(
                executable,
                config,
                samples[start : start + batch_size],
                temporary_dir,
            )
            predictions.extend(batch_predictions)
            confidences.extend(batch_confidences)
    return predictions, confidences


def run_model(executable, model_name, config, dataset_name, dataset_config):
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{dataset_name}_{model_name}.json"
    predictions_path = output_dir / f"{dataset_name}_{model_name}.csv"
    if completed_result(result_path):
        print(f"Skip: already completed ({result_path})")
        return

    ensure_traineddata(config)
    samples = load_samples(dataset_config)
    if not samples or not all(sample["path"].is_file() for sample in samples):
        raise FileNotFoundError(f"Prepared recognition dataset is incomplete: {dataset_name}")

    started = time.perf_counter()
    predictions, confidences = predict(
        executable,
        config,
        samples,
        f"{model_name} / {dataset_name}",
    )
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
    cpu_stats = {
        "device": "cpu",
        "num_words": len(samples),
        "batch_size": config["batch_size"],
        "total_time_s": total_time,
        "mean_time_ms": total_time * 1000 / len(samples),
        "throughput_words_s": len(samples) / total_time,
        "ram_peak_mb": psutil.Process().memory_info().rss / 1024**2,
        "mean_confidence": float(np.mean(confidences)),
        "accuracy_metrics": metrics,
    }
    result = {
        "benchmark_version": 2,
        "task": "recognition",
        "dataset": dataset_name,
        "model": model_name,
        "origin": config["origin"],
        "engine": subprocess.check_output(
            [str(executable), "--version"], text=True, encoding="utf-8", errors="replace"
        ).splitlines()[0],
        "cpu": cpu_stats,
        "gpu": None,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "prediction", "reference", "confidence"])
        for sample, prediction, reference, confidence in zip(
            samples, predictions, references, confidences
        ):
            writer.writerow([sample["image"], prediction, reference, confidence])
    print(f"Saved: {result_path}")


def main():
    executable = find_tesseract()
    for dataset_name, dataset_config in DATASETS.items():
        print(f"\n### DATASET: {dataset_name}")
        prepare_dataset(dataset_name, dataset_config)
        for model_name, config in MODELS.items():
            if os.environ.get("OCR_BENCH_ONLY_MODEL") not in (None, model_name):
                continue
            print(f"\n## MODEL: {model_name}")
            run_model(executable, model_name, config, dataset_name, dataset_config)


if __name__ == "__main__":
    main()
