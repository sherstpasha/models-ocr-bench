import gc
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from utils.metrics import evaluate_dataset, has_standard_f1_metrics
from configs.benchmark_config import BENCHMARKS, DATASETS


YOLO_BENCHMARKS = {
    name: config
    for name, config in BENCHMARKS.items()
    if config.get("backend") == "ultralytics" and config.get("run", False)
}


def get_image_files(folder: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = []
    base = Path(folder)
    for ext in exts:
        files += base.rglob(f"*{ext}")
        files += base.rglob(f"*{ext.upper()}")
    return sorted(list(dict.fromkeys(map(str, files))))


def load_ground_truth(coco_json: str) -> Dict[str, list]:
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    gt = {}

    for ann in coco["annotations"]:
        filename = id_to_name.get(ann["image_id"])
        if not filename or "segmentation" not in ann:
            continue
        segs = ann["segmentation"]
        segs = segs if isinstance(segs[0], list) else [segs]
        for seg in segs:
            pts = np.array(seg).reshape(-1, 2)
            box = (
                float(pts[:, 0].min()),
                float(pts[:, 1].min()),
                float(pts[:, 0].max()),
                float(pts[:, 1].max()),
            )
            gt.setdefault(filename, []).append(box)
    return gt


def memory_usage():
    import psutil

    ram = psutil.Process().memory_info().rss / 1024**2
    vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else None
    return ram, vram


def benchmark_device(config, images, device: str, collect_predictions: bool = False):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model_source = hf_hub_download(
        repo_id=config["repository"],
        filename=config["filename"],
        local_dir=config["model_dir"],
    )
    model = YOLO(str(model_source))
    model.to(device)
    ram0, vram0 = memory_usage()

    for img in images[: config["warmup"]]:
        model.predict(
            img,
            imgsz=config["imgsz"],
            conf=config["conf"],
            device=device,
            verbose=False,
        )
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    preds = {} if collect_predictions else None

    for img in images:
        t0 = time.time()
        results = model.predict(
            img,
            imgsz=config["imgsz"],
            conf=config["conf"],
            device=device,
            verbose=False,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

        if collect_predictions:
            boxes = []
            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    boxes.append(tuple(xyxy[:4]))
            preds[Path(img).name] = boxes

    ram1, vram1 = memory_usage()
    stats = {
        "device": device,
        "num_images": len(images),
        "mean_time_ms": float(np.mean(times) * 1000),
        "median_time_ms": float(np.median(times) * 1000),
        "throughput_fps": float(len(images) / np.sum(times)),
        "ram_after_load_mb": ram0,
        "ram_peak_mb": ram1,
        "ram_delta_mb": ram1 - ram0,
    }
    if device == "cuda":
        stats["gpu_after_load_mb"] = vram0
        stats["gpu_peak_mb"] = vram1
        stats["gpu_delta_mb"] = vram1 - vram0
    if collect_predictions:
        stats["predictions"] = preds
    return stats


def save_results(output_path: Path, dataset_name: str, cpu: dict, gpu: dict):
    out = {"dataset": dataset_name, "cpu": cpu, "gpu": gpu}
    if cpu and "predictions" in cpu:
        del cpu["predictions"]
    if gpu and "predictions" in gpu:
        del gpu["predictions"]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def load_existing_results(output_file: Path):
    if not output_file.is_file():
        return None, None
    with output_file.open("r", encoding="utf-8") as f:
        result = json.load(f)
    return result.get("cpu"), result.get("gpu")


def benchmark_model(model_name, config):
    output_dir = config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for dataset_name, ds in DATASETS.items():
        print(f"\n### DATASET: {dataset_name}")
        output_file = Path(output_dir) / f"{dataset_name}_{model_name}.json"
        cpu_stats, gpu_stats = load_existing_results(output_file)
        run_cpu = not config["gpu_only"]
        run_gpu = not config["cpu_only"]
        if run_cpu and not has_standard_f1_metrics(cpu_stats):
            cpu_stats = None
        if run_gpu and not has_standard_f1_metrics(gpu_stats):
            gpu_stats = None

        if (not run_cpu or cpu_stats) and (not run_gpu or gpu_stats):
            print(f"Skip: already completed ({output_file})")
            continue

        if not Path(ds["folder"]).exists() or not Path(ds["annotations"]).exists():
            print("Skip: missing folder or annotations")
            continue

        images = get_image_files(ds["folder"])
        if not images:
            print("Skip: no images")
            continue

        ground_truths = load_ground_truth(ds["annotations"])
        if run_cpu and not cpu_stats:
            print("Run missing device: cpu")
            cpu_stats = benchmark_device(config, images, "cpu", collect_predictions=True)
            cpu_stats["accuracy_metrics"] = evaluate_dataset(cpu_stats["predictions"], ground_truths)

        if run_gpu and not gpu_stats:
            print("Run missing device: cuda")
            gpu_stats = benchmark_device(config, images, "cuda", collect_predictions=True)
            gpu_stats["accuracy_metrics"] = evaluate_dataset(gpu_stats["predictions"], ground_truths)

        save_results(output_file, dataset_name, cpu_stats, gpu_stats)
        print(f"Saved: {output_file}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only benchmarks requested, but CUDA is unavailable in PyTorch")
    for model_name, config in YOLO_BENCHMARKS.items():
        print(f"\n## MODEL: {model_name}")
        benchmark_model(model_name, config)


if __name__ == "__main__":
    main()
