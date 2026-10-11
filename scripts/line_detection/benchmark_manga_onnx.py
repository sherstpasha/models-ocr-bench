"""Benchmark the PP-OCRv6 Manga v0.2 ONNX line detector."""

import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
from huggingface_hub import hf_hub_download

from configs.line_detection.benchmark_config import BENCHMARKS
from scripts.line_detection.benchmark_external_common import run


MODEL_NAME = "ppocrv6_manga_det_v02_line"
CONFIG = BENCHMARKS[MODEL_NAME]
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_model(config):
    path = Path(config["model_dir"]) / config["filename"]
    if not path.is_file():
        path = Path(hf_hub_download(
            repo_id=config["repository"], filename=config["filename"],
            local_dir=config["model_dir"],
        ))
    return path


def create_model(config):
    providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    return ort.InferenceSession(str(ensure_model(config)), providers=providers)


def box_score(probability, points):
    height, width = probability.shape
    points = points.astype(np.int32)
    x0 = int(np.clip(np.floor(points[:, 0].min()), 0, width - 1))
    x1 = int(np.clip(np.ceil(points[:, 0].max()), 0, width - 1))
    y0 = int(np.clip(np.floor(points[:, 1].min()), 0, height - 1))
    y1 = int(np.clip(np.ceil(points[:, 1].max()), 0, height - 1))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    shifted = points.copy()
    shifted[:, 0] -= x0
    shifted[:, 1] -= y0
    cv2.fillPoly(mask, [shifted], 1)
    return cv2.mean(probability[y0:y1 + 1, x0:x1 + 1], mask)[0]


def unclip(points, ratio):
    x, y = points[:, 0], points[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perimeter = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum()
    if area <= 0 or perimeter <= 0:
        return None
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(points.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    paths = offset.Execute(area * ratio / perimeter)
    return np.asarray(paths[0], dtype=np.float32) if len(paths) == 1 else None


def detect(model, image_path, config):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    size = config["input_size"]
    if max(height, width) / min(height, width) > 2:
        scale = min(1.0, (0.75 * size * size / (height * width)) ** 0.5)
    else:
        scale = size / max(height, width)
    resized_height = max(int(round(height * scale / 32) * 32), 32)
    resized_width = max(int(round(width * scale / 32) * 32), 32)
    tensor = cv2.resize(image, (resized_width, resized_height)).astype(np.float32) / 255.0
    tensor = ((tensor - MEAN) / STD).transpose(2, 0, 1)[None]
    input_name = model.get_inputs()[0].name
    probability = model.run(None, {input_name: tensor})[0][0, 0]
    binary = (probability > config["threshold"]).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        points = contour.squeeze(1)
        if points.ndim != 2 or len(points) < 4:
            continue
        if box_score(probability, points) < config["box_threshold"]:
            continue
        points = unclip(points, config["unclip_ratio"])
        if points is None or not len(points):
            continue
        rectangle = cv2.minAreaRect(points)
        if min(rectangle[1]) < 3:
            continue
        box = cv2.boxPoints(rectangle).astype(np.float32)
        box[:, 0] = np.clip(np.round(box[:, 0] / resized_width * width), 0, width)
        box[:, 1] = np.clip(np.round(box[:, 1] / resized_height * height), 0, height)
        polygons.append([box.tolist()])
    return polygons


def main():
    selected = os.environ.get("OCR_BENCH_ONLY_MODEL")
    if selected not in (None, MODEL_NAME):
        return
    run(MODEL_NAME, CONFIG, create_model, detect, "PP-OCRv6 Manga v0.2 line detector")


if __name__ == "__main__":
    main()
