"""Benchmark Russian docTR recognition ONNX checkpoints on prepared crops."""

import csv
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

from configs.recognition.benchmark_config import BENCHMARKS, DATASETS
from scripts.recognition.prepare_handwritten_essay import prepare_dataset
from utils.metrics import (
    accuracy_score,
    cer_score,
    character_similarity_score,
    normalize_text,
    wer_score,
)


def ensure_model(config):
    config["model_dir"].mkdir(parents=True, exist_ok=True)
    return {
        filename: Path(hf_hub_download(
            repo_id=config["repository"],
            filename=filename,
            local_dir=config["model_dir"],
        ))
        for filename in ("model.onnx", "config.json")
    }


def create_session(config):
    files = ensure_model(config)
    model_config = json.loads(files["config.json"].read_text(encoding="utf-8"))
    available = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if config["device"] == "cuda" and "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(files["model.onnx"]), providers=providers)
    return session, model_config


def prepare_image(path, config):
    _, height, width = config["input_shape"]
    with Image.open(path) as source:
        image = source.convert("RGB")
        scale = min(height / image.height, width / image.width)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
    array = np.zeros((height, width, 3), dtype=np.float32)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    array[: pixels.shape[0], : pixels.shape[1]] = pixels
    mean = np.asarray(config["mean"], dtype=np.float32)
    std = np.asarray(config["std"], dtype=np.float32)
    return ((array - mean) / std).transpose(2, 0, 1)


def decode(logits, config):
    sequences = logits.argmax(axis=-1)
    vocabulary = config["vocab"]
    texts = []
    if config["decoding"] == "ctc_greedy":
        blank = config["blank_index"]
        for sequence in sequences:
            texts.append("".join(
                vocabulary[index]
                for position, index in enumerate(sequence)
                if index != blank and (position == 0 or index != sequence[position - 1])
            ))
    else:
        eos = config["eos_index"]
        for sequence in sequences:
            stop = np.flatnonzero(sequence == eos)
            length = int(stop[0]) if len(stop) else len(sequence)
            texts.append("".join(vocabulary[index] for index in sequence[:length]))
    return texts


def predict_batches(session, model_config, samples, batch_size, description=None):
    predictions = []
    starts = range(0, len(samples), batch_size)
    if description:
        starts = tqdm(
            starts,
            total=(len(samples) + batch_size - 1) // batch_size,
            desc=description,
            unit="batch",
        )
    for start in starts:
        batch = np.stack([
            prepare_image(sample["path"], model_config)
            for sample in samples[start : start + batch_size]
        ])
        logits = session.run(
            [model_config["output_name"]],
            {model_config["input_name"]: batch},
        )[0]
        predictions.extend(decode(logits, model_config))
    return predictions


def completed_result(path):
    if not path.is_file():
        return False
    metrics = (json.loads(path.read_text(encoding="utf-8")).get("gpu") or {}).get(
        "accuracy_metrics", {}
    )
    return all(key in metrics for key in ("cer", "wer", "exact_match"))


def run_model(model_name, config, dataset_name, dataset):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{dataset_name}_{model_name}.json"
    predictions_path = result_path.with_suffix(".csv")
    if completed_result(result_path):
        print(f"Skip: already completed ({result_path})")
        return
    with dataset["labels"].open(encoding="utf-8", newline="") as file:
        samples = [
            {**row, "path": dataset["images_dir"] / row["image"]}
            for row in csv.DictReader(file)
        ]
    session, model_config = create_session(config)
    predict_batches(session, model_config, samples[: config["warmup_size"]], config["batch_size"])
    process = psutil.Process()
    ram_after_load = process.memory_info().rss / 1024**2
    started = time.perf_counter()
    predictions = predict_batches(
        session,
        model_config,
        samples,
        config["batch_size"],
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
    stats = {
        "device": config["device"],
        "providers": session.get_providers(),
        "num_words": len(samples),
        "batch_size": config["batch_size"],
        "total_time_s": total_time,
        "mean_time_ms": total_time * 1000 / len(samples),
        "throughput_words_s": len(samples) / total_time,
        "ram_after_load_mb": ram_after_load,
        "ram_peak_mb": process.memory_info().rss / 1024**2,
        "accuracy_metrics": metrics,
    }
    result_path.write_text(json.dumps({
        "task": "recognition",
        "dataset": dataset_name,
        "model": model_name,
        "repository": config["repository"],
        "cpu": None,
        "gpu": stats if config["device"] == "cuda" else None,
        "cpu": stats if config["device"] == "cpu" else None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "prediction", "reference"])
        for sample, prediction, reference in zip(samples, predictions, references):
            writer.writerow([sample["image"], prediction, reference])
    del session
    gc.collect()
    print(f"Saved: {result_path}")


def main():
    selected = os.environ.get("OCR_BENCH_ONLY_MODEL")
    for model_name, config in BENCHMARKS.items():
        if config.get("backend") != "doctr_recognition_onnx" or not config.get("run", False):
            continue
        if selected not in (None, model_name):
            continue
        for dataset_name, dataset in DATASETS.items():
            print(f"\n### DATASET: {dataset_name}\n\n## MODEL: {model_name}")
            prepare_dataset(dataset_name, dataset)
            run_model(model_name, config, dataset_name, dataset)


if __name__ == "__main__":
    main()
