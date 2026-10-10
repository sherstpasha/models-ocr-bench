"""Benchmark Ultralytics text-line instance-segmentation models."""

import os

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from configs.line_detection.benchmark_config import BENCHMARKS
from scripts.line_detection.benchmark_external_common import run


ULTRALYTICS_BENCHMARKS = {
    name: config
    for name, config in BENCHMARKS.items()
    if config.get("backend") == "ultralytics" and config.get("run", False)
}


def create_model(config):
    config["model_dir"].mkdir(parents=True, exist_ok=True)
    checkpoint = hf_hub_download(
        repo_id=config["repository"],
        filename=config["filename"],
        local_dir=config["model_dir"],
    )
    return YOLO(checkpoint)


def detect(model, image_path, config):
    result = model.predict(
        image_path,
        imgsz=config["imgsz"],
        conf=config["confidence"],
        device=config["device"],
        retina_masks=True,
        verbose=False,
    )[0]
    allowed_names = {name.casefold() for name in config.get("class_names", [])}
    class_ids = result.boxes.cls.int().tolist()
    polygons = []
    if result.masks is not None:
        for class_id, points in zip(class_ids, result.masks.xy):
            class_name = str(result.names[class_id])
            if allowed_names and class_name.casefold() not in allowed_names:
                continue
            if len(points) >= 3:
                polygons.append([points.tolist()])
        return polygons
    # Detection-only YOLO checkpoints expose boxes instead of instance masks.
    # Convert xyxy boxes to polygons so they use the same line-evaluation path.
    for class_id, box in zip(class_ids, result.boxes.xyxy.tolist()):
        class_name = str(result.names[class_id])
        if allowed_names and class_name.casefold() not in allowed_names:
            continue
        x1, y1, x2, y2 = box
        polygons.append([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])
    return polygons


if __name__ == "__main__":
    selected = os.environ.get("OCR_BENCH_ONLY_MODEL")
    for model_name, config in ULTRALYTICS_BENCHMARKS.items():
        if selected not in (None, model_name):
            continue
        run(
            model_name,
            config,
            create_model,
            detect,
            config.get("architecture", model_name),
        )
