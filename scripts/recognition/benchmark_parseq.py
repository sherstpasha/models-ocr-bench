"""Benchmark Hukyl PARSeq checkpoints on word and line crops."""

import csv
import gc
import json
import os
import time
from pathlib import Path

import psutil
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules import transformer
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


class DecoderLayer(nn.Module):
    """Pre-LN, two-stream transformer decoder used by PARSeq."""

    def __init__(self, width, heads, feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(width, feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm_q = nn.LayerNorm(width)
        self.norm_c = nn.LayerNorm(width)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def _stream(self, target, target_norm, keys, memory, mask, padding_mask):
        value = self.self_attn(
            target_norm,
            keys,
            keys,
            attn_mask=mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        target = target + self.dropout1(value)
        value = self.cross_attn(
            self.norm1(target), memory, memory, need_weights=False
        )[0]
        target = target + self.dropout2(value)
        value = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(target)))))
        return target + self.dropout3(value)

    def forward(
        self, query, content, memory, query_mask=None, content_mask=None,
        content_key_padding_mask=None, update_content=True,
    ):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self._stream(
            query, query_norm, content_norm, memory,
            query_mask, content_key_padding_mask,
        )
        if update_content:
            content = self._stream(
                content, content_norm, content_norm, memory,
                content_mask, content_key_padding_mask,
            )
        return query, content


class Decoder(nn.Module):
    def __init__(self, layer, depth, width):
        super().__init__()
        self.layers = transformer._get_clones(layer, depth)
        self.norm = nn.LayerNorm(width)

    def forward(
        self, query, content, memory, query_mask=None, content_mask=None,
        content_key_padding_mask=None,
    ):
        for index, layer in enumerate(self.layers):
            query, content = layer(
                query, content, memory, query_mask, content_mask,
                content_key_padding_mask, index != len(self.layers) - 1,
            )
        return self.norm(query)


class TokenEmbedding(nn.Module):
    def __init__(self, tokens, width):
        super().__init__()
        self.embedding = nn.Embedding(tokens, width)
        self.scale = width**0.5

    def forward(self, tokens):
        return self.scale * self.embedding(tokens)


class PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size, width):
        super().__init__()
        self.proj = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)

    def forward(self, images):
        return self.proj(images).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.scale = (width // heads) ** -0.5
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)

    def forward(self, values):
        batch, length, width = values.shape
        qkv = self.qkv(values).reshape(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        output = attention.softmax(-1) @ value
        return self.proj(output.transpose(1, 2).reshape(batch, length, width))


class Mlp(nn.Module):
    def __init__(self, width, ratio):
        super().__init__()
        self.fc1 = nn.Linear(width, width * ratio)
        self.fc2 = nn.Linear(width * ratio, width)

    def forward(self, values):
        return self.fc2(F.gelu(self.fc1(values)))


class EncoderBlock(nn.Module):
    def __init__(self, width, heads, ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = Attention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = Mlp(width, ratio)

    def forward(self, values):
        values = values + self.attn(self.norm1(values))
        return values + self.mlp(self.norm2(values))


class Encoder(nn.Module):
    """Weight-compatible inference subset of timm VisionTransformer."""

    def __init__(self, image_size, patch_size, width, depth, heads, ratio):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, width)
        patch_count = (image_size[0] // patch_size[0]) * (
            image_size[1] // patch_size[1]
        )
        self.pos_embed = nn.Parameter(torch.empty(1, patch_count, width))
        self.blocks = nn.Sequential(*[
            EncoderBlock(width, heads, ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width)

    def forward(self, images):
        values = self.patch_embed(images) + self.pos_embed
        return self.norm(self.blocks(values))


class PARSeq(nn.Module):
    def __init__(self, config, charset):
        super().__init__()
        width = config["embed_dim"]
        self.eos_id = 0
        self.bos_id = len(charset) + 1
        self.pad_id = len(charset) + 2
        self.charset = charset
        self.max_label_length = config["max_label_length"]
        self.refine_iters = config.get("refine_iters", 1)
        self.encoder = Encoder(
            (config["img_height"], config["img_width"]),
            tuple(config["patch_size"]),
            width,
            config["enc_depth"],
            config["enc_num_heads"],
            config["enc_mlp_ratio"],
        )
        layer = DecoderLayer(
            width,
            config["dec_num_heads"],
            width * config["dec_mlp_ratio"],
            config["dropout"],
        )
        self.decoder = Decoder(layer, config["dec_depth"], width)
        num_tokens = len(charset) + 3
        self.head = nn.Linear(width, num_tokens - 2)
        self.text_embed = TokenEmbedding(num_tokens, width)
        self.pos_queries = nn.Parameter(
            torch.empty(1, self.max_label_length + 1, width)
        )
        self.dropout = nn.Dropout(config["dropout"])

    def encode(self, images):
        return self.encoder(images)

    def decode(
        self, target, memory, target_mask=None, padding_mask=None,
        target_query=None, query_mask=None,
    ):
        batch, length = target.shape
        null_context = self.text_embed(target[:, :1])
        embedded = self.pos_queries[:, : length - 1] + self.text_embed(target[:, 1:])
        content = self.dropout(torch.cat([null_context, embedded], dim=1))
        if target_query is None:
            target_query = self.pos_queries[:, :length].expand(batch, -1, -1)
        return self.decoder(
            self.dropout(target_query), content, memory,
            query_mask, target_mask, padding_mask,
        )

    def forward(self, images):
        batch = images.shape[0]
        steps = self.max_label_length + 1
        memory = self.encode(images)
        positions = self.pos_queries[:, :steps].expand(batch, -1, -1)
        causal = torch.triu(
            torch.ones((steps, steps), dtype=torch.bool, device=images.device), 1
        )
        target = torch.full(
            (batch, steps), self.pad_id, dtype=torch.long, device=images.device
        )
        target[:, 0] = self.bos_id
        logits = []
        for index in range(steps):
            end = index + 1
            output = self.decode(
                target[:, :end], memory, causal[:end, :end],
                target_query=positions[:, index:end],
                query_mask=causal[index:end, :end],
            )
            current = self.head(output)
            logits.append(current)
            if end < steps:
                target[:, end] = current[:, 0].argmax(-1)
                if (target == self.eos_id).any(-1).all():
                    break
        logits = torch.cat(logits, dim=1)
        if self.refine_iters:
            length = logits.shape[1]
            causal = causal[:length, :length]
            query_mask = causal.clone()
            query_mask[torch.triu(
                torch.ones_like(query_mask), diagonal=2
            )] = False
            bos = torch.full(
                (batch, 1), self.bos_id, dtype=torch.long, device=images.device
            )
            for _ in range(self.refine_iters):
                target = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                padding = (target == self.eos_id).int().cumsum(-1) > 0
                output = self.decode(
                    target, memory, causal, padding,
                    positions[:, :length], query_mask,
                )
                logits = self.head(output)
        return logits

    def decode_text(self, logits):
        results = []
        for sequence in logits.argmax(-1).tolist():
            chars = []
            for token in sequence:
                if token == self.eos_id:
                    break
                if 1 <= token <= len(self.charset):
                    chars.append(self.charset[token - 1])
            results.append("".join(chars))
        return results


def load_samples(config):
    with config["labels"].open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return [{**row, "path": config["images_dir"] / row["image"]} for row in rows]


def completed_result(path):
    if not path.is_file():
        return False
    metrics = (json.loads(path.read_text(encoding="utf-8")).get("gpu") or {}).get(
        "accuracy_metrics", {}
    )
    return all(key in metrics for key in ("cer", "wer"))


def load_model(config):
    checkpoint = hf_hub_download(
        config["repository"], config["filename"], cache_dir=config["model_dir"]
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = PARSeq(payload["config"], payload["charset"])
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(config["device"]).eval(), payload["config"]


def image_batch(samples, height, width, device):
    tensors = []
    for sample in samples:
        with Image.open(sample["path"]) as image:
            # The model card specifies unconditional 48x512 resizing and
            # INTER_AREA for downscaling. PIL BOX is area-based resampling.
            image = image.convert("RGB").resize((width, height), Image.Resampling.BOX)
            data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            data = data.view(height, width, 3).permute(2, 0, 1).float().div_(255)
            tensors.append(data.sub_(0.5).div_(0.5))
    return torch.stack(tensors).to(device, non_blocking=True)


def predict_batches(
    model, samples, batch_size, shape, description=None,
    initial_predictions=None, checkpoint_callback=None,
):
    predictions = list(initial_predictions or [])
    starts = range(len(predictions), len(samples), batch_size)
    if description:
        starts = tqdm(starts, desc=description, unit="batch")
    for start in starts:
        batch = image_batch(
            samples[start:start + batch_size], shape[0], shape[1], "cuda"
        )
        with torch.inference_mode():
            predictions.extend(model.decode_text(model(batch)))
        if checkpoint_callback and (
            len(predictions) % 100 == 0 or len(predictions) == len(samples)
        ):
            checkpoint_callback(predictions)
    return predictions


def run_model(model_name, model_config, dataset_name, dataset_config):
    output_dir = model_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{dataset_name}_{model_name}.json"
    csv_path = output_dir / f"{dataset_name}_{model_name}.csv"
    checkpoint_path = output_dir / f"{dataset_name}_{model_name}.checkpoint.json"
    if completed_result(result_path):
        print(f"Skip: already completed ({result_path})")
        return
    samples = load_samples(dataset_config)
    if not samples or not all(sample["path"].is_file() for sample in samples):
        raise FileNotFoundError(f"Prepared recognition dataset is incomplete: {dataset_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("PARSeq recognition benchmark requires CUDA")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, config = load_model(model_config)
    shape = (config["img_height"], config["img_width"])
    predict_batches(
        model, samples[:model_config["warmup_size"]],
        model_config["batch_size"], shape,
    )
    torch.cuda.synchronize()
    ram_after_load = psutil.Process().memory_info().rss / 1024**2
    checkpoint = {"predictions": [], "elapsed_s": 0.0}
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    resumed = checkpoint.get("predictions", [])
    elapsed_before = float(checkpoint.get("elapsed_s", 0.0))
    started = time.perf_counter()

    def save_checkpoint(predictions):
        payload = {
            "predictions": predictions,
            "elapsed_s": elapsed_before + time.perf_counter() - started,
        }
        temporary = checkpoint_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(checkpoint_path)

    predictions = predict_batches(
        model, samples, model_config["batch_size"], shape,
        f"{model_name} / {dataset_name}", resumed, save_checkpoint,
    )
    torch.cuda.synchronize()
    total_time = elapsed_before + time.perf_counter() - started
    references = [sample["text"] for sample in samples]
    normalized_predictions = [normalize_text(text) for text in predictions]
    normalized_references = [normalize_text(text) for text in references]
    metrics = {
        "cer": cer_score(normalized_predictions, normalized_references),
        "wer": wer_score(normalized_predictions, normalized_references),
        "exact_match": accuracy_score(normalized_predictions, normalized_references),
        "character_similarity": character_similarity_score(
            normalized_predictions, normalized_references
        ),
        "evaluated_words": len(samples),
    }
    gpu = {
        "device": "cuda",
        "num_words": len(samples),
        "batch_size": model_config["batch_size"],
        "total_time_s": total_time,
        "mean_time_ms": total_time * 1000 / len(samples),
        "throughput_words_s": len(samples) / total_time,
        "ram_after_load_mb": ram_after_load,
        "ram_peak_mb": psutil.Process().memory_info().rss / 1024**2,
        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "accuracy_metrics": metrics,
    }
    result = {
        "task": "recognition",
        "dataset": dataset_name,
        "model": model_name,
        "repository": model_config["repository"],
        "cpu": None,
        "gpu": gpu,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "prediction", "reference"])
        for sample, prediction, reference in zip(samples, predictions, references):
            writer.writerow([sample["image"], prediction, reference])
    checkpoint_path.unlink(missing_ok=True)
    print(f"Saved: {result_path}")


def main():
    for dataset_name, dataset_config in DATASETS.items():
        prepare_dataset(dataset_name, dataset_config)
        for model_name, model_config in BENCHMARKS.items():
            if os.environ.get("OCR_BENCH_ONLY_MODEL") not in (None, model_name):
                continue
            if not model_config.get("run") or model_config.get("backend") != "parseq":
                continue
            print(f"\n## MODEL: {model_name} / {dataset_name}")
            run_model(model_name, model_config, dataset_name, dataset_config)


if __name__ == "__main__":
    main()
