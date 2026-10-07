"""Benchmark the Church Slavonic Puigcerver CRNN-CTC checkpoint."""

import csv
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
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


class PuigcerverCRNN(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        layers = []
        input_channels = 1
        for filters, pool_size in zip(config["cnn_filters"], config["cnn_poolsize"]):
            layers.extend([
                nn.Conv2d(input_channels, filters, 3, padding=1),
                nn.BatchNorm2d(filters),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            if pool_size:
                layers.append(nn.MaxPool2d(pool_size, pool_size))
            input_channels = filters
        self.cnn = nn.Sequential(*layers)
        pooled_height = config["img_height"]
        for pool_size in config["cnn_poolsize"]:
            if pool_size:
                pooled_height //= pool_size
        self.rnn = nn.LSTM(
            input_channels * pooled_height,
            config["rnn_hidden"],
            num_layers=config["rnn_layers"],
            dropout=config["dropout"],
            bidirectional=True,
        )
        self.lin_dropout = nn.Dropout(config["dropout"])
        self.fc = nn.Linear(config["rnn_hidden"] * 2, num_classes)

    def forward(self, images):
        features = self.cnn(images)
        sequence = features.permute(3, 0, 1, 2).flatten(2)
        sequence, _ = self.rnn(sequence)
        return self.fc(self.lin_dropout(sequence))


def download_files(config):
    config["model_dir"].mkdir(parents=True, exist_ok=True)
    return {
        key: Path(hf_hub_download(
            repo_id=config["repository"],
            filename=config[key],
            local_dir=config["model_dir"],
        ))
        for key in ("checkpoint_filename", "config_filename", "symbols_filename")
    }


def load_model(config):
    files = download_files(config)
    model_config = json.loads(files["config_filename"].read_text(encoding="utf-8"))
    checkpoint = torch.load(
        files["checkpoint_filename"], map_location="cpu", weights_only=True
    )
    index_to_character = {
        int(index): character for index, character in checkpoint["idx2char"].items()
    }
    model = PuigcerverCRNN(model_config, len(index_to_character))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config["device"]).eval()
    return model, index_to_character, int(model_config["img_height"])


def prepare_image(path, height):
    with Image.open(path) as image:
        image = image.convert("L")
        width = max(1, round(image.width * height / image.height))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).unsqueeze(0)


def decode(logits, index_to_character):
    predictions = logits.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    texts = []
    for sequence in predictions:
        characters = []
        previous = None
        for index in sequence:
            if index != 0 and index != previous:
                character = index_to_character.get(index, "")
                characters.append(" " if character == "<space>" else character)
            previous = index
        texts.append("".join(characters))
    return texts


def predict_batches(model, samples, height, index_to_character, batch_size, description=None):
    predictions = []
    starts = range(0, len(samples), batch_size)
    if description:
        starts = tqdm(starts, total=(len(samples) + batch_size - 1) // batch_size,
                      desc=description, unit="batch")
    for start in starts:
        images = [prepare_image(sample["path"], height) for sample in samples[start:start + batch_size]]
        max_width = max(image.shape[-1] for image in images)
        batch = torch.ones((len(images), 1, height, max_width), dtype=torch.float32)
        for index, image in enumerate(images):
            batch[index, :, :, :image.shape[-1]] = image
        with torch.inference_mode():
            logits = model(batch.to(next(model.parameters()).device, non_blocking=True))
        predictions.extend(decode(logits, index_to_character))
    return predictions


def completed_result(path):
    if not path.is_file():
        return False
    metrics = (json.loads(path.read_text(encoding="utf-8")).get("gpu") or {}).get("accuracy_metrics", {})
    return all(key in metrics for key in ("cer", "wer"))


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
    model, characters, height = load_model(config)
    predict_batches(model, samples[:config["warmup_size"]], height, characters, config["batch_size"])
    torch.cuda.synchronize()
    process = psutil.Process()
    ram_after_load = process.memory_info().rss / 1024**2
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictions = predict_batches(
        model, samples, height, characters, config["batch_size"],
        f"{model_name} / {dataset_name}",
    )
    torch.cuda.synchronize()
    total_time = time.perf_counter() - started
    references = [sample["text"] for sample in samples]
    normalized_references = [normalize_text(text) for text in references]
    normalized_predictions = [normalize_text(text) for text in predictions]
    metrics = {
        "cer": cer_score(normalized_predictions, normalized_references),
        "wer": wer_score(normalized_predictions, normalized_references),
        "exact_match": accuracy_score(normalized_predictions, normalized_references),
        "character_similarity": character_similarity_score(normalized_predictions, normalized_references),
        "evaluated_words": len(samples),
    }
    gpu = {
        "device": config["device"], "num_words": len(samples),
        "batch_size": config["batch_size"], "total_time_s": total_time,
        "mean_time_ms": total_time * 1000 / len(samples),
        "throughput_words_s": len(samples) / total_time,
        "ram_after_load_mb": ram_after_load,
        "ram_peak_mb": process.memory_info().rss / 1024**2,
        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "accuracy_metrics": metrics,
    }
    result_path.write_text(json.dumps({
        "task": "recognition", "dataset": dataset_name, "model": model_name,
        "repository": config["repository"], "cpu": None, "gpu": gpu,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "prediction", "reference"])
        for sample, prediction, reference in zip(samples, predictions, references):
            writer.writerow([sample["image"], prediction, reference])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved: {result_path}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CRNN-CTC recognition benchmark requires CUDA")
    selected = os.environ.get("OCR_BENCH_ONLY_MODEL")
    for model_name, config in BENCHMARKS.items():
        if config.get("backend") != "crnn_ctc" or not config.get("run", False):
            continue
        if selected not in (None, model_name):
            continue
        for dataset_name, dataset in DATASETS.items():
            print(f"\n### DATASET: {dataset_name}\n\n## MODEL: {model_name}")
            prepare_dataset(dataset_name, dataset)
            run_model(model_name, config, dataset_name, dataset)


if __name__ == "__main__":
    main()
