"""Host-side benchmark loop for the Dockerized MonkeyOCRv2 detector."""

import json
import subprocess
from pathlib import Path, PurePosixPath

import numpy as np

from scripts.line_detection.prepare_datasets import PREPARATION_VERSION
from utils.datasets import prediction_key, result_is_compatible
from utils.metrics import (
    evaluate_dataset,
    evaluate_polygon_dataset,
    load_coco_detection_ground_truth,
)
from utils.prediction_artifacts import prediction_artifact_path, save_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE = "models-ocr-bench-monkeyocrv2:1.0.0"


def ensure_runtime(config):
    models_dir = PROJECT_ROOT / "models"
    weights = models_dir / Path(config["docker_weights"]).name
    backbone = models_dir / "monkeyocrv2_as"
    required_backbone = (
        "config.json",
        "configuration_monkeyocrv2_vitae.py",
        "model.safetensors",
        "modeling_monkeyocrv2_vitae_vision.py",
        "preprocessor_config.json",
    )
    if not weights.is_file():
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="HB16888/MonkeyOCRv2_det_dptext",
            filename=weights.name,
            local_dir=models_dir,
        )
    if not all((backbone / name).is_file() for name in required_backbone):
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="zenosai/MonkeyOCRv2-AS",
            local_dir=backbone,
            allow_patterns=list(required_backbone),
        )
    if subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode:
        subprocess.run(
            ["docker", "build", "-f", "scripts/monkeyocrv2/Dockerfile", "-t", IMAGE, "."],
            cwd=PROJECT_ROOT,
            check=True,
        )


def docker_path(path):
    return str(Path(path).resolve())


def polygons_to_boxes(polygons):
    boxes = []
    for polygon in polygons:
        points = np.concatenate([
            np.asarray(part, dtype=np.float32).reshape(-1, 2)
            for part in polygon
        ])
        boxes.append([
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        ])
    return boxes


def run(model_name, config, datasets, level):
    ensure_runtime(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_dir = PROJECT_ROOT / "scripts" / "monkeyocrv2"
    models_dir = PROJECT_ROOT / "models"
    for dataset_name, dataset in datasets.items():
        output_file = output_dir / f"{dataset_name}_{model_name}.json"
        if output_file.is_file():
            old = json.loads(output_file.read_text(encoding="utf-8")).get("gpu") or {}
            compatible = result_is_compatible(old, dataset)
            if level == "line":
                compatible = compatible and old.get("ground_truth_version") == PREPARATION_VERSION
            if compatible and prediction_artifact_path(output_file).is_file():
                print(f"Skip: already completed ({output_file})", flush=True)
                continue
        if not Path(dataset["folder"]).is_dir() or not Path(dataset["annotations"]).is_file():
            print(f"Skip {dataset_name}: missing images or annotations", flush=True)
            continue
        raw_file = output_dir / f".{dataset_name}_{model_name}_raw.json"
        command = [
            "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
            "--entrypoint", "python",
            "-v", f"{docker_path(dataset['folder'])}:/input:ro",
            "-v", f"{docker_path(models_dir)}:/models:ro",
            "-v", f"{docker_path(runner_dir)}:/runner:ro",
            "-v", f"{docker_path(output_dir)}:/output",
            IMAGE, "/runner/docker_infer.py",
            "--config", config["docker_config"],
            "--weights", config["docker_weights"],
            "--backbone", "/models/monkeyocrv2_as",
            "--images", "/input",
            "--output", f"/output/{raw_file.name}",
            "--threshold", str(config["score_threshold"]),
            "--warmup", str(config.get("warmup", 1)),
        ]
        print(f"\n### {model_name}: {dataset_name}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        raw = json.loads(raw_file.read_text(encoding="utf-8"))
        polygon_predictions = {}
        predictions = {}
        for relative, polygons in raw["predictions"].items():
            host_path = Path(dataset["folder"]).joinpath(*PurePosixPath(relative).parts)
            key = prediction_key(host_path, dataset)
            polygon_predictions[key] = polygons
            predictions[key] = polygons_to_boxes(polygons)
        save_predictions(output_file, predictions, polygon_predictions)
        gt_boxes, gt_polygons = load_coco_detection_ground_truth(dataset["annotations"])
        stats = {
            "device": "cuda:0",
            "architecture": "MonkeyOCRv2 DPText-DETR",
            "repository": config["origin"],
            "num_images": raw["stats"]["num_images"],
            "mean_time_ms": raw["stats"]["mean_time_ms"],
            "median_time_ms": raw["stats"]["median_time_ms"],
            "throughput_fps": raw["stats"]["throughput_fps"],
            "gpu_after_load_mb": raw["stats"]["gpu_after_load_mb"],
            "gpu_peak_mb": raw["stats"]["gpu_peak_mb"],
            "gpu_delta_mb": raw["stats"]["gpu_peak_mb"] - raw["stats"]["gpu_after_load_mb"],
            "accuracy_metrics": evaluate_dataset(predictions, gt_boxes),
            "polygon_iou_metrics": evaluate_polygon_dataset(polygon_predictions, gt_polygons),
        }
        if level == "line":
            stats["detection_level"] = "line"
            stats["ground_truth_version"] = PREPARATION_VERSION
        else:
            stats["word_level_target"] = True
        output_file.write_text(json.dumps({
            "dataset": dataset_name,
            "level": level,
            "cpu": None,
            "gpu": stats,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raw_file.unlink(missing_ok=True)
        print(f"Saved: {output_file}", flush=True)
