"""Benchmark Ultralytics text-line instance-segmentation models."""

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from configs.line_detection.benchmark_config import BENCHMARKS
from scripts.line_detection.benchmark_external_common import run


MODEL_NAME = "court_records_textline_yolov8x_seg"
CONFIG = BENCHMARKS[MODEL_NAME]


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
    if result.masks is None:
        return []
    return [[points.tolist()] for points in result.masks.xy if len(points) >= 3]


if __name__ == "__main__":
    run(MODEL_NAME, CONFIG, create_model, detect, "Court Records YOLOv8x-Seg")
