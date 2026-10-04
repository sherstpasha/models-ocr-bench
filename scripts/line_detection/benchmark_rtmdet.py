"""Benchmark Riksarkivet's RTMDet text-line instance segmenter."""

import cv2
import numpy as np
from huggingface_hub import snapshot_download

from configs.line_detection.benchmark_config import BENCHMARKS
from scripts.line_detection.benchmark_external_common import run

MODEL_NAME = "riksarkivet_rtmdet_lines"
CONFIG = BENCHMARKS[MODEL_NAME]


def create_model(config):
    from mmdet.apis import DetInferencer
    snapshot_download(repo_id=config["repository"], local_dir=config["model_dir"])
    return DetInferencer(
        model=str(config["model_dir"] / "config.py"),
        weights=str(config["model_dir"] / "model.pth"),
        device=config["device"],
    )


def _mask_polygon(mask):
    if isinstance(mask, dict):
        from pycocotools import mask as mask_utils
        mask = mask_utils.decode(mask)
    array = np.asarray(mask)
    if array.ndim == 1:
        points = array.reshape(-1, 2)
        return points.tolist() if len(points) >= 3 else None
    if array.ndim == 2 and array.shape[1] == 2:
        return array.tolist() if len(array) >= 3 else None
    contours, _ = cv2.findContours(array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)[:, 0, :]
    return contour.tolist() if len(contour) >= 3 else None


def detect(model, image_path, config):
    result = model(
        image_path,
        pred_score_thr=config["score_threshold"],
        return_vis=False,
        no_save_pred=True,
    )["predictions"][0]
    scores = result.get("scores", [1.0] * len(result.get("masks", [])))
    polygons = []
    for score, mask in zip(scores, result.get("masks", [])):
        if float(score) < config["score_threshold"]:
            continue
        # MMDetection may serialize a multipart mask as a list of flat polygons.
        parts = mask if isinstance(mask, list) and mask and isinstance(mask[0], list) else [mask]
        converted = [_mask_polygon(part) for part in parts]
        converted = [part for part in converted if part]
        if converted:
            polygons.append(converted)
    return polygons


if __name__ == "__main__":
    run(MODEL_NAME, CONFIG, create_model, detect, "Riksarkivet RTMDet Lines")
