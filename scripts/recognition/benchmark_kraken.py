"""Benchmark Kraken PP-OCRv6 recognition on prepared word or line crops."""

import argparse
import csv
import gc
import json
import time

import numpy as np
import psutil
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm.auto import tqdm

from configs.recognition.benchmark_config import BENCHMARKS, DATASETS, RECOGNITION_LEVEL
from scripts.recognition.prepare_handwritten_essay import prepare_dataset
from utils.metrics import (
    accuracy_score,
    cer_score,
    character_similarity_score,
    normalize_text,
    wer_score,
)


MODEL_NAMES = [
    name for name, config in BENCHMARKS.items() if config.get("backend") == "kraken"
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite completed results.",
    )
    return parser.parse_args()


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
        all(key in metrics for key in ("cer", "wer", "exact_match"))
        and gpu.get("throughput_words_s") is not None
    )


def ensure_model(config):
    path = config["model_path"]
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=config["repository"],
            filename=config["filename"],
            local_dir=path.parent,
        )
    return path


def patch_kraken_cuda_lengths():
    """Work around Kraken 7.1 leaving CTC sequence lengths on the CPU."""
    import kraken.lib.ppocr.network as network

    def lengths_and_mask(seq_lens, width_in, width_out, device):
        scaled = (
            seq_lens.to(device=device, dtype=torch.float32)
            * (width_out / float(width_in))
        ).floor().long()
        output_lengths = scaled.clamp(min=1, max=width_out)
        positions = torch.arange(width_out, device=device)
        return output_lengths, positions[None, :] < output_lengths[:, None]

    network._lengths_and_mask = torch.compiler.disable(lengths_and_mask)


def create_recognizer(config):
    if not torch.cuda.is_available():
        raise RuntimeError("Kraken recognition benchmark requires CUDA")
    torch.set_float32_matmul_precision("high")
    patch_kraken_cuda_lengths()
    from kraken.configs import RecognitionInferenceConfig
    from kraken.tasks import RecognitionTaskModel

    model = RecognitionTaskModel.load_model(ensure_model(config))
    inference_config = RecognitionInferenceConfig(
        accelerator="gpu",
        device=[0],
        batch_size=config["batch_size"],
        num_line_workers=0,
        precision="32-true",
    )
    model.net.prepare_for_inference(inference_config)
    return model.net


def make_page(samples):
    from kraken.containers import BaselineLine, Segmentation

    images = []
    for sample in samples:
        with Image.open(sample["path"]) as source:
            images.append(source.convert("RGB"))
    gap = 3
    page_width = max(image.width for image in images)
    page_height = sum(image.height for image in images) + gap * (len(images) - 1)
    page = Image.new("RGB", (page_width, page_height), "white")
    lines = []
    top = 0
    for index, image in enumerate(images):
        page.paste(image, (0, top))
        right = max(1, image.width - 1)
        bottom = max(top + 1, top + image.height - 1)
        lines.append(
            BaselineLine(
                id=str(index),
                baseline=[(0, bottom), (right, bottom)],
                boundary=[(0, top), (right, top), (right, bottom), (0, bottom)],
            )
        )
        top += image.height + gap
    segmentation = Segmentation(
        type="baselines",
        imagename="",
        text_direction="horizontal-lr",
        script_detection=False,
        lines=lines,
    )
    return page, segmentation


def predict_batches(recognizer, samples, batch_size, description=None):
    predictions = []
    confidences = []
    starts = range(0, len(samples), batch_size)
    if description:
        starts = tqdm(
            starts,
            total=(len(samples) + batch_size - 1) // batch_size,
            desc=description,
            unit="batch",
        )
    for batch_index, start in enumerate(starts):
        batch = samples[start : start + batch_size]
        page, segmentation = make_page(batch)
        records = list(recognizer.predict(page, segmentation))
        if len(records) != len(batch):
            raise RuntimeError(
                f"Kraken returned {len(records)} records for {len(batch)} crops"
            )
        for record in records:
            predictions.append(record.prediction)
            confidence = list(record.confidences or [])
            confidences.append(float(np.mean(confidence)) if confidence else 0.0)
        del records, segmentation, page, batch
        if RECOGNITION_LEVEL == "line" and (batch_index + 1) % 64 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    return predictions, confidences


def run_model(
    model_name, config, recognizer, dataset_name, dataset_config, force=False
):
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{dataset_name}_{model_name}.json"
    predictions_path = output_dir / f"{dataset_name}_{model_name}.csv"
    if not force and completed_result(result_path):
        print(f"Skip: already completed ({result_path})")
        return

    samples = load_samples(dataset_config)
    if not samples or not all(sample["path"].is_file() for sample in samples):
        raise FileNotFoundError(f"Prepared recognition dataset is incomplete: {dataset_name}")

    predict_batches(
        recognizer,
        samples[: config["warmup_size"]],
        config["batch_size"],
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ram_after_load = psutil.Process().memory_info().rss / 1024**2
    started = time.perf_counter()
    predictions, confidences = predict_batches(
        recognizer,
        samples,
        config["batch_size"],
        description=f"{model_name} / {dataset_name}",
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
        "character_similarity": character_similarity_score(
            normalized_predictions, normalized_references
        ),
        "evaluated_words": len(samples),
    }
    gpu_stats = {
        "device": config["device"],
        "num_words": len(samples),
        "batch_size": config["batch_size"],
        "total_time_s": total_time,
        "mean_time_ms": total_time * 1000 / len(samples),
        "throughput_words_s": len(samples) / total_time,
        "ram_after_load_mb": ram_after_load,
        "ram_peak_mb": psutil.Process().memory_info().rss / 1024**2,
        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "mean_confidence": float(np.mean(confidences)),
        "accuracy_metrics": metrics,
    }
    result = {
        "task": "recognition",
        "dataset": dataset_name,
        "model": model_name,
        "repository": config["repository"],
        "evaluation_input": (
            "ground-truth line crops"
            if RECOGNITION_LEVEL == "line"
            else "word crops treated as single text lines"
        ),
        "cpu": None,
        "gpu": gpu_stats,
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
    args = parse_args()
    for dataset_name, dataset_config in DATASETS.items():
        print(f"\n### DATASET: {dataset_name}")
        prepare_dataset(dataset_name, dataset_config)
    for model_name in MODEL_NAMES:
        config = BENCHMARKS[model_name]
        pending = []
        for dataset_name, dataset_config in DATASETS.items():
            result_path = config["output_dir"] / f"{dataset_name}_{model_name}.json"
            if args.force or not completed_result(result_path):
                pending.append((dataset_name, dataset_config))
            else:
                print(f"Skip: already completed ({result_path})")
        if not pending:
            continue
        gc.collect()
        torch.cuda.empty_cache()
        recognizer = create_recognizer(config)
        for dataset_name, dataset_config in pending:
            print(f"\n## MODEL: {model_name} / {dataset_name}")
            run_model(
                model_name,
                config,
                recognizer,
                dataset_name,
                dataset_config,
                force=args.force,
            )
        del recognizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
