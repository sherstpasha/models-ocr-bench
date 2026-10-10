import csv
import gc
import json
import os
import re
import time

import psutil
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import (
    BitImageProcessor,
    PreTrainedTokenizerFast,
    RobertaTokenizerFast,
    TrOCRProcessor,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
)
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


class TrOCRProcessorCustom(TrOCRProcessor):
    """Compatibility wrapper required by Kansallisarkisto TrOCR checkpoints."""

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.chat_template = None


def postprocess_prediction(text, mode=None):
    if mode != "character_spaced":
        return text
    # This checkpoint emits spaces between characters and a wider whitespace
    # run between words: "в о т  т а к" -> "вот так".
    text = re.sub(
        r"\s+", lambda match: " " if len(match.group()) >= 2 else "", text
    ).strip()
    text = re.sub(r"\s+([,.;:!?…%)\]»\-])", r"\1", text)
    return re.sub(r"([(\"«\[])\s+", r"\1", text)


def load_custom_processor(repository, cache_dir):
    """Load new Transformers processor metadata with Transformers 4.x."""
    processor_path = hf_hub_download(
        repository, "processor_config.json", cache_dir=cache_dir
    )
    tokenizer_path = hf_hub_download(
        repository, "tokenizer.json", cache_dir=cache_dir
    )
    processor_config = json.loads(
        open(processor_path, encoding="utf-8").read()
    )["image_processor"]
    processor_config.pop("image_processor_type", None)
    image_processor = BitImageProcessor(**processor_config)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="<s>",
        cls_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    return TrOCRProcessorCustom(image_processor, tokenizer)


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


def predict_batches(
    processor,
    model,
    samples,
    batch_size,
    description=None,
    preserve_aspect_height=None,
    generation_max_length=None,
    initial_predictions=None,
    checkpoint_callback=None,
):
    predictions = list(initial_predictions or [])
    starts = range(len(predictions), len(samples), batch_size)
    if description:
        starts = tqdm(
            starts,
            total=(len(samples) + batch_size - 1) // batch_size,
            initial=(len(predictions) + batch_size - 1) // batch_size,
            desc=description,
            unit="batch",
        )
    for start in starts:
        batch = samples[start : start + batch_size]
        images = []
        for sample in batch:
            with Image.open(sample["path"]) as image:
                image = image.convert("RGB")
                if preserve_aspect_height:
                    width = max(
                        1,
                        round(image.width * preserve_aspect_height / image.height),
                    )
                    image = image.resize(
                        (width, preserve_aspect_height), Image.Resampling.LANCZOS
                    )
                images.append(image)
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to("cuda", non_blocking=True)
        with torch.inference_mode():
            generation_kwargs = {}
            if generation_max_length:
                generation_kwargs["max_length"] = generation_max_length
            generated_ids = model.generate(pixel_values, **generation_kwargs)
        predictions.extend(
            text.strip()
            for text in processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        )
        if checkpoint_callback and (
            len(predictions) % 100 == 0 or len(predictions) == len(samples)
        ):
            checkpoint_callback(predictions)
    return predictions


def run_model(model_name, model_config, dataset_name, dataset_config):
    output_dir = model_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{dataset_name}_{model_name}.json"
    predictions_path = output_dir / f"{dataset_name}_{model_name}.csv"
    checkpoint_path = output_dir / f"{dataset_name}_{model_name}.checkpoint.json"
    if completed_result(result_path):
        print(f"Skip: already completed ({result_path})")
        return

    samples = load_samples(dataset_config)
    if not samples or not all(sample["path"].is_file() for sample in samples):
        raise FileNotFoundError(f"Prepared recognition dataset is incomplete: {dataset_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("TrOCR recognition benchmark requires CUDA")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cache_dir = model_config["model_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    if model_config.get("custom_processor", False):
        processor = load_custom_processor(model_config["repository"], cache_dir)
    elif model_config.get("tokenizer_json_processor", False):
        # Some checkpoints publish tokenizer.json but omit the vocab/merges
        # files required by the slow Roberta tokenizer selected by
        # TrOCRProcessor.from_pretrained.
        image_processor = ViTImageProcessor.from_pretrained(
            model_config["repository"], cache_dir=cache_dir
        )
        tokenizer = RobertaTokenizerFast.from_pretrained(
            model_config["repository"], cache_dir=cache_dir
        )
        processor = TrOCRProcessor(
            image_processor=image_processor, tokenizer=tokenizer
        )
    else:
        processor = TrOCRProcessor.from_pretrained(
            model_config.get("processor_repository", model_config["repository"]),
            cache_dir=cache_dir,
            subfolder=model_config.get("processor_subfolder", ""),
            use_fast=False,
        )
    model = VisionEncoderDecoderModel.from_pretrained(
        model_config["repository"], cache_dir=cache_dir
    )
    model.to("cuda")
    model.eval()

    predict_batches(
        processor,
        model,
        samples[: model_config["warmup_size"]],
        model_config["batch_size"],
        preserve_aspect_height=model_config.get("preserve_aspect_height"),
        generation_max_length=model_config.get("generation_max_length"),
    )
    torch.cuda.synchronize()
    ram_after_load = psutil.Process().memory_info().rss / 1024**2
    checkpoint = {"raw_predictions": [], "elapsed_s": 0.0}
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if len(checkpoint.get("raw_predictions", [])) > len(samples):
            checkpoint = {"raw_predictions": [], "elapsed_s": 0.0}
    resumed_predictions = checkpoint.get("raw_predictions", [])
    elapsed_before = float(checkpoint.get("elapsed_s", 0.0))
    started = time.perf_counter()

    def save_checkpoint(current_predictions):
        payload = {
            "raw_predictions": current_predictions,
            "elapsed_s": elapsed_before + time.perf_counter() - started,
        }
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        for attempt in range(5):
            try:
                temporary.replace(checkpoint_path)
                break
            except PermissionError:
                if attempt == 4:
                    # Windows indexing/antivirus can transiently hold the old
                    # checkpoint. A direct write is less atomic but preserves
                    # hours of inference instead of aborting the benchmark.
                    checkpoint_path.write_text(
                        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                    )
                    temporary.unlink(missing_ok=True)
                    break
                time.sleep(0.2 * (attempt + 1))

    raw_predictions = predict_batches(
        processor,
        model,
        samples,
        model_config["batch_size"],
        description=f"{model_name} / {dataset_name}",
        preserve_aspect_height=model_config.get("preserve_aspect_height"),
        generation_max_length=model_config.get("generation_max_length"),
        initial_predictions=resumed_predictions,
        checkpoint_callback=save_checkpoint,
    )
    predictions = [
        postprocess_prediction(text, model_config.get("prediction_postprocess"))
        for text in raw_predictions
    ]
    torch.cuda.synchronize()
    total_time = elapsed_before + time.perf_counter() - started
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
        "batch_size": model_config["batch_size"],
        "total_time_s": total_time,
        "mean_time_ms": total_time * 1000 / len(samples),
        "throughput_words_s": len(samples) / total_time,
        "ram_after_load_mb": ram_after_load,
        "ram_peak_mb": ram_peak,
        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "accuracy_metrics": metrics,
    }
    result = {
        "task": "recognition",
        "dataset": dataset_name,
        "model": model_name,
        "repository": model_config["repository"],
        "cpu": None,
        "gpu": gpu_stats,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with predictions_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "prediction", "reference", "raw_prediction"])
        for sample, prediction, reference, raw_prediction in zip(
            samples, predictions, references, raw_predictions
        ):
            writer.writerow(
                [sample["image"], prediction, reference, raw_prediction]
            )
    print(f"Saved: {result_path}")
    checkpoint_path.unlink(missing_ok=True)


def main():
    skip_high_power = os.environ.get("OCR_BENCH_SKIP_HIGH_POWER") == "1"
    for dataset_name, dataset_config in DATASETS.items():
        print(f"\n### DATASET: {dataset_name}")
        prepare_dataset(dataset_name, dataset_config)
        for model_name, model_config in BENCHMARKS.items():
            if os.environ.get("OCR_BENCH_ONLY_MODEL") not in (None, model_name):
                continue
            if (
                not model_config.get("run", False)
                or model_config.get("backend") != "trocr"
            ):
                continue
            if skip_high_power and model_config.get("high_power", False):
                print(
                    f"\n## MODEL: {model_name}\n"
                    "Skip high-power model in safe all-in-one mode."
                )
                continue
            print(f"\n## MODEL: {model_name}")
            run_model(model_name, model_config, dataset_name, dataset_config)


if __name__ == "__main__":
    main()
