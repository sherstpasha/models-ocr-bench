"""Benchmark the TurkicOCR SVTRv2-B ONNX Cyrillic recognizer."""

import csv
import gc
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil
from tqdm.auto import tqdm

from configs.recognition.benchmark_config import BENCHMARKS, DATASETS
from scripts.recognition.prepare_handwritten_essay import prepare_dataset
from utils.metrics import (
    accuracy_score,
    cer_score,
    character_similarity_score,
    normalize_text,
    wer_score,
)


MODEL_NAME = "turkicocr_svtrv2_b"
CONFIG = BENCHMARKS[MODEL_NAME]

# Keep every CUDA wheel directory registered for the lifetime of the process.
# cuDNN loads some sub-libraries lazily on Windows and otherwise ONNX Runtime
# reports CUDAExecutionProvider as available but silently creates a CPU session.
_DLL_DIRECTORY_HANDLES = []
if os.name == "nt":
    nvidia_root = Path(ort.__file__).resolve().parent.parent / "nvidia"
    nvidia_bin_dirs = [str(path.resolve()) for path in nvidia_root.glob("*/bin")]
    for bin_dir in nvidia_bin_dirs:
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(bin_dir))
    os.environ["PATH"] = os.pathsep.join(nvidia_bin_dirs + [os.environ["PATH"]])
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls(directory="")


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


def load_characters(path):
    characters = path.read_text(encoding="utf-8").splitlines()
    # OpenOCR CTCLabelDecode prepends the blank class and appends a space when
    # use_space_char=True. The published graph has 140 output classes.
    return [None, *characters, " "]


def image_width(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    height, width = image.shape[:2]
    return min(CONFIG["max_width"], max(1, round(CONFIG["image_height"] * width / height)))


def prepare_batch(samples):
    height = CONFIG["image_height"]
    prepared = []
    widths = []
    for sample in samples:
        image = cv2.imread(str(sample["path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {sample['path']}")
        source_height, source_width = image.shape[:2]
        width = min(CONFIG["max_width"], max(1, round(height * source_width / source_height)))
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 127.5 - 1.0
        prepared.append(normalized.transpose(2, 0, 1))
        widths.append(width)

    padded_width = min(CONFIG["max_width"], math.ceil(max(widths) / 8) * 8)
    batch = np.zeros((len(samples), 3, height, padded_width), dtype=np.float32)
    for index, image in enumerate(prepared):
        batch[index, :, :, : image.shape[2]] = image
    return batch


def decode(logits, characters):
    indices = logits.argmax(axis=2)
    scores = logits.max(axis=2)
    texts = []
    confidences = []
    for sequence, sequence_scores in zip(indices, scores):
        result = []
        selected_scores = []
        previous = None
        for index, score in zip(sequence.tolist(), sequence_scores.tolist()):
            if index != 0 and index != previous and index < len(characters):
                result.append(characters[index])
                selected_scores.append(float(score))
            previous = index
        texts.append("".join(result))
        confidences.append(float(np.mean(selected_scores)) if selected_scores else 0.0)
    return texts, confidences


def predict_batches(session, samples, characters, batch_size, description=None):
    # Similar widths are grouped to avoid padding every short word to 640 px.
    indexed = sorted(enumerate(samples), key=lambda pair: image_width(pair[1]["path"]))
    predictions = [""] * len(samples)
    confidences = [0.0] * len(samples)
    starts = range(0, len(indexed), batch_size)
    if description:
        starts = tqdm(
            starts,
            total=(len(indexed) + batch_size - 1) // batch_size,
            desc=description,
            unit="batch",
        )
    input_name = session.get_inputs()[0].name
    for start in starts:
        items = indexed[start : start + batch_size]
        batch = prepare_batch([sample for _, sample in items])
        logits = session.run(None, {input_name: batch})[0]
        texts, scores = decode(logits, characters)
        for (original_index, _), text, score in zip(items, texts, scores):
            predictions[original_index] = text
            confidences[original_index] = score
    return predictions, confidences


def run_model(dataset_name, dataset_config):
    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{dataset_name}_{MODEL_NAME}.json"
    predictions_path = output_dir / f"{dataset_name}_{MODEL_NAME}.csv"
    if completed_result(result_path):
        print(f"Skip: already completed ({result_path})")
        return

    samples = load_samples(dataset_config)
    if not samples or not all(sample["path"].is_file() for sample in samples):
        raise FileNotFoundError(f"Prepared recognition dataset is incomplete: {dataset_name}")
    if not CONFIG["model_path"].is_file() or not CONFIG["charset_path"].is_file():
        raise FileNotFoundError("TurkicOCR ONNX model or charset is missing")

    gc.collect()
    session = ort.InferenceSession(
        str(CONFIG["model_path"]),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"TurkicOCR requested CUDA but active providers are: {session.get_providers()}")
    characters = load_characters(CONFIG["charset_path"])
    if len(characters) != session.get_outputs()[0].shape[-1]:
        raise RuntimeError("TurkicOCR charset does not match the ONNX output classes")

    predict_batches(
        session,
        samples[: CONFIG["warmup_size"]],
        characters,
        CONFIG["batch_size"],
    )
    ram_after_load = psutil.Process().memory_info().rss / 1024**2
    started = time.perf_counter()
    predictions, confidences = predict_batches(
        session,
        samples,
        characters,
        CONFIG["batch_size"],
        description=f"{MODEL_NAME} / {dataset_name}",
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
    gpu_stats = {
        "device": "cuda",
        "active_providers": session.get_providers(),
        "num_words": len(samples),
        "batch_size": CONFIG["batch_size"],
        "total_time_s": total_time,
        "mean_time_ms": total_time * 1000 / len(samples),
        "throughput_words_s": len(samples) / total_time,
        "ram_after_load_mb": ram_after_load,
        "ram_peak_mb": psutil.Process().memory_info().rss / 1024**2,
        "mean_confidence": float(np.mean(confidences)),
        "accuracy_metrics": metrics,
    }
    result = {
        "task": "recognition",
        "dataset": dataset_name,
        "model": MODEL_NAME,
        "repository": CONFIG["repository"],
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
    for dataset_name, dataset_config in DATASETS.items():
        print(f"\n### DATASET: {dataset_name}")
        prepare_dataset(dataset_name, dataset_config)
        run_model(dataset_name, dataset_config)


if __name__ == "__main__":
    main()
