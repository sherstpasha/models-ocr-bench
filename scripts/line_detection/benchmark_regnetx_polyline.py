"""Benchmark the RegNetX-8GF Stage-0 polyline baseline detector."""

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

from configs.line_detection.benchmark_config import BENCHMARKS
from scripts.line_detection.benchmark_external_common import run


MODEL_NAME = "regnetx_8gf_polyline_baseline_stage0_line"
CONFIG = BENCHMARKS[MODEL_NAME]


def _load_official_code(config):
    source_dir = Path(config["source_dir"])
    if not (source_dir / "pipeline.py").is_file():
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=config["repository"], repo_type="space",
            local_dir=source_dir, ignore_patterns=["*.safetensors"],
        )
    source = str(source_dir)
    if source not in sys.path:
        sys.path.insert(0, source)
    from pipeline import load_model
    from regnetx_infer.engine.infer import infer_image
    return load_model, infer_image


def ensure_weights(config):
    path = Path(config["model_dir"]) / config["filename"]
    if not path.is_file():
        path = Path(hf_hub_download(
            repo_id=config["repository"], filename=config["filename"],
            local_dir=config["model_dir"],
        ))
    return path


def create_model(config):
    load_model, infer_image = _load_official_code(config)
    model = load_model(str(ensure_weights(config)), image_size=config["image_size"])
    return model.to(config["device"]), infer_image


def detect(model_and_runner, image_path, config):
    model, infer_image = model_and_runner
    with Image.open(image_path) as image:
        lines = infer_image(
            model, image.convert("RGB"),
            conf_thresh=config["confidence"], image_size=config["image_size"],
            letterbox=True, polygonize=True, fix_intersections=False,
            device=config["device"],
        )
    return [[line["boundary"]] for line in lines if len(line.get("boundary") or []) >= 3]


def main():
    selected = os.environ.get("OCR_BENCH_ONLY_MODEL")
    if selected not in (None, MODEL_NAME):
        return
    run(MODEL_NAME, CONFIG, create_model, detect, "RegNetX-8GF Stage-0 polyline baseline detector")


if __name__ == "__main__":
    main()
