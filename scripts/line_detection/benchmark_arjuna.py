"""Benchmark the Arjuna OCR PP-OCRv6 line detector."""

import os
import sys
import types
from pathlib import Path

import cv2
from huggingface_hub import snapshot_download

from configs.line_detection.benchmark_config import BENCHMARKS
from scripts.line_detection.benchmark_external_common import run


MODEL_NAME = "arjuna_ocr_kn_en_line"
CONFIG = BENCHMARKS[MODEL_NAME]


def ensure_model(config):
    root = Path(config["model_dir"])
    required = [
        root / config["filename"],
        root / "kanen_infer" / "pipeline_base.py",
        root / "kanen_infer" / "config.py",
        root / "kanen_infer" / "providers.py",
    ]
    if not all(path.is_file() for path in required):
        snapshot_download(
            repo_id=config["repository"],
            local_dir=root,
            allow_patterns=["kanen_infer/*.py", "models/det/*", "kanen.yaml"],
        )
    return root


def create_model(config):
    root = ensure_model(config)
    # The upstream package targets Python 3.12 and its top-level import loads
    # optional formatting modules that use 3.12-only f-string syntax. Load the
    # detector submodules directly so the benchmark remains usable on 3.10.
    package = types.ModuleType("kanen_infer")
    package.__path__ = [str(root / "kanen_infer")]
    sys.modules["kanen_infer"] = package
    from kanen_infer.config import load_config
    from kanen_infer.pipeline_base import Detector

    cfg = load_config(overrides={"provider": {"policy": "cuda"}})
    return Detector(cfg=cfg)


def detect(model, image_path, config):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    return [[box.tolist()] for box, _score in model(image)]


def main():
    selected = os.environ.get("OCR_BENCH_ONLY_MODEL")
    if selected not in (None, MODEL_NAME):
        return
    run(MODEL_NAME, CONFIG, create_model, detect, "Arjuna OCR PP-OCRv6 Small line detector")


if __name__ == "__main__":
    main()
