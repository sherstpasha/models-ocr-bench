"""Benchmark NDLOCR-Lite's published DEIMv2 layout detector as a line detector."""

import os
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from configs.line_detection.benchmark_config import BENCHMARKS
from scripts.line_detection.benchmark_external_common import run


MODEL_NAME = "ndlocr_lite_deimv2_line"
CONFIG = BENCHMARKS[MODEL_NAME]
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
INV_STD = np.reciprocal(np.asarray([0.229, 0.224, 0.225], dtype=np.float32))


def ensure_model(config):
    path = Path(config["model_dir"]) / config["filename"]
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    bundled = (
        Path(__file__).resolve().parents[2] / "models" / "sources" / "ndlocr-lite"
        / "src" / "model" / config["filename"]
    )
    if bundled.is_file():
        shutil.copy2(bundled, path)
    else:
        temporary = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(config["weights_url"], temporary)
        temporary.replace(path)
    return path


def create_model(config):
    providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    session = ort.InferenceSession(str(ensure_model(config)), providers=providers)
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"NDLOCR-Lite requires CUDAExecutionProvider, got {session.get_providers()}")
    return session


def detect(model, image_path, config):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    side = max(height, width)
    padded = np.zeros((side, side, 3), dtype=np.uint8)
    padded[:height, :width] = image
    input_height, input_width = model.get_inputs()[0].shape[2:]
    resized = Image.fromarray(padded).resize((input_width, input_height))
    tensor = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = ((tensor - MEAN) * INV_STD).transpose(2, 0, 1)[None]
    size = np.asarray([[input_height, input_width]], dtype=np.int64)
    inputs = model.get_inputs()
    labels, boxes, scores, _ = model.run(None, {inputs[0].name: tensor, inputs[1].name: size})
    labels = np.asarray(labels).reshape(-1).astype(np.int64) - 1
    boxes = np.asarray(boxes).reshape(-1, 4)
    scores = np.asarray(scores).reshape(-1)
    scale = np.asarray([side / input_width, side / input_height] * 2, dtype=np.float32)
    allowed = set(config["line_class_ids"])
    polygons = []
    for class_id, box, score in zip(labels, boxes, scores):
        if class_id not in allowed or score <= config["confidence"]:
            continue
        x1, y1, x2, y2 = box * scale
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 <= x1 or y2 <= y1:
            continue
        polygons.append([[[float(x1), float(y1)], [float(x2), float(y1)],
                          [float(x2), float(y2)], [float(x1), float(y2)]]])
    return polygons


def main():
    selected = os.environ.get("OCR_BENCH_ONLY_MODEL")
    if selected not in (None, MODEL_NAME):
        return
    run(MODEL_NAME, CONFIG, create_model, detect, "NDLOCR-Lite DEIMv2 line detector")


if __name__ == "__main__":
    main()
