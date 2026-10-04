"""Benchmark Kraken's bundled BLLA text-line segmenter."""

from PIL import Image
from kraken import blla

from configs.line_detection.benchmark_config import BENCHMARKS
from scripts.line_detection.benchmark_external_common import run

MODEL_NAME = "kraken_blla_default"
CONFIG = BENCHMARKS[MODEL_NAME]


def create_model(_config):
    # The stock BLLA model is bundled with Kraken and loaded lazily once.
    from importlib import resources
    from kraken.lib.vgsl import TorchVGSLModel
    return TorchVGSLModel.load_model(resources.files("kraken").joinpath("blla.mlmodel"))


def detect(model, image_path, config):
    with Image.open(image_path) as image:
        result = blla.segment(image.convert("RGB"), model=model, device=config["device"], autocast=True)
    return [[[[float(x), float(y)] for x, y in line.boundary]] for line in result.lines if line.boundary]


if __name__ == "__main__":
    run(MODEL_NAME, CONFIG, create_model, detect, "Kraken BLLA")
