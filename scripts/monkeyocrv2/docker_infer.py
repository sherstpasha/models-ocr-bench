"""Run one DPText-DETR checkpoint over one mounted image directory."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.modeling import build_model

from adet.checkpoint import AdetCheckpointer
from adet.config import get_cfg
from adet.data.detection_utils import build_augmentation
from detectron2.data import transforms as T


SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def create_model(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.MONKEY_V2_VITAE.MODEL_DIR = args.backbone
    cfg.MODEL.MONKEY_V2_VITAE.WEIGHTS = str(Path(args.backbone) / "model.safetensors")
    cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = args.threshold
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    model = build_model(cfg)
    model.eval()
    AdetCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    augmentations = T.AugmentationList(build_augmentation(cfg, is_train=False))
    return model, augmentations


def predict(model, augmentations, path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    augmented = T.AugInput(image)
    augmentations(augmented)
    tensor = torch.as_tensor(
        np.ascontiguousarray(augmented.image.transpose(2, 0, 1))
    ).float()
    with torch.inference_mode():
        instances = model([{
            "image": tensor,
            "height": height,
            "width": width,
        }])[0]["instances"].to("cpu")
    polygons = instances.polygons.numpy().reshape(-1, 16, 2)
    return [[polygon.astype(float).tolist()] for polygon in polygons]


def main():
    args = arguments()
    root = Path(args.images)
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"No images under {root}")
    model, augmentations = create_model(args)
    for path in paths[: args.warmup]:
        predict(model, augmentations, path)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gpu_after_load = torch.cuda.memory_allocated() / 1024**2
    predictions = {}
    timings = []
    for index, path in enumerate(paths, start=1):
        started = time.perf_counter()
        polygons = predict(model, augmentations, path)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        predictions[path.relative_to(root).as_posix()] = polygons
        if index % 25 == 0 or index == len(paths):
            print(f"{index}/{len(paths)}", flush=True)
    elapsed = float(sum(timings))
    values = np.asarray(timings, dtype=np.float64)
    result = {
        "predictions": predictions,
        "stats": {
            "num_images": len(paths),
            "mean_time_ms": float(values.mean() * 1000),
            "median_time_ms": float(np.median(values) * 1000),
            "throughput_fps": float(len(paths) / elapsed),
            "gpu_after_load_mb": gpu_after_load,
            "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        },
    }
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
