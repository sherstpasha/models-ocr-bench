"""Benchmark the manuscript-ocr Mask2Former text-line detector."""

import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil
from tqdm import tqdm

from configs.line_detection.benchmark_config import BENCHMARKS, DATASETS
from scripts.line_detection.prepare_datasets import PREPARATION_VERSION
from utils.datasets import prediction_key, result_is_compatible
from utils.metrics import evaluate_dataset, evaluate_polygon_dataset, load_coco_detection_ground_truth
from utils.prediction_artifacts import prediction_artifact_path, save_predictions


MODEL_NAME = "mask2former_line_v0_prev"
CONFIG = BENCHMARKS[MODEL_NAME]

# Load CUDA/cuDNN DLLs installed by the onnxruntime-gpu extras before the
# manuscript detector creates its session.
ort.preload_dlls(directory="")

from manuscript.detectors import Mask2Former  # noqa: E402


class BenchmarkMask2Former(Mask2Former):
    """Avoid the very slow ORT extended graph optimization for this ONNX graph."""

    def _preprocess(self, image):
        pixels, left, top, resized_size = super()._preprocess(image)
        height, width = image.shape[:2]
        self._benchmark_geometry = (width, height, *resized_size)
        return pixels, left, top, resized_size

    def _postprocess(self, class_logits, mask_logits, geometry):
        """Keep masks at model resolution instead of expanding every query to 30+ MP."""
        _height, _width, left, top, resized_width, resized_height = geometry
        class_scores = self._softmax(class_logits.astype(np.float32))[:, :-1].max(axis=-1)
        candidates = []
        original_width, original_height, _, _ = self._benchmark_geometry
        area_scale = (original_width / resized_width) * (original_height / resized_height)
        for query, class_score in enumerate(class_scores):
            logits = cv2.resize(
                mask_logits[query].astype(np.float32),
                (self.mask_interpolation_size,) * 2,
                interpolation=cv2.INTER_LINEAR,
            )
            binary = logits > self.mask_logit_threshold
            if not binary.any():
                continue
            score = float(class_score * self._sigmoid(logits)[binary].mean())
            if score < self.score_threshold:
                continue
            model_mask = cv2.resize(
                binary.astype(np.uint8),
                (self.image_size,) * 2,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            cropped = model_mask[top:top + resized_height, left:left + resized_width]
            if int(cropped.sum() * area_scale) >= self.min_mask_area:
                candidates.append((cropped, score))
        candidates.sort(key=lambda item: item[1], reverse=True)
        kept = []
        for mask, score in candidates:
            if all(self._mask_iou(mask, old) <= self.nms_iou_threshold for old, _ in kept):
                kept.append((mask, score))
        return kept

    def _mask_polygon(self, mask):
        polygon = Mask2Former._mask_polygon(mask)
        if polygon is None:
            return None
        original_width, original_height, resized_width, resized_height = self._benchmark_geometry
        scale_x = original_width / resized_width
        scale_y = original_height / resized_height
        return [(x * scale_x, y * scale_y) for x, y in polygon]

    def _initialize_session(self):
        if self.onnx_session is not None:
            return
        # Keep the manuscript preloader: cuDNN 9 is split into sublibraries on
        # Windows, and merely finding CUDAExecutionProvider does not guarantee
        # that the first convolution can load all of them.
        self._prepare_runtime_dependencies()
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.onnx_session = ort.InferenceSession(
            self.weights,
            sess_options=options,
            providers=self.runtime_providers(),
        )
        pixel_input = next(
            item
            for item in self.onnx_session.get_inputs()
            if item.name == self.input_pixel_values
        )
        self.pixel_dtype = (
            np.float16 if pixel_input.type == "tensor(float16)" else np.float32
        )
        self._log_device_info(self.onnx_session)


def get_image_files(folder):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        str(path)
        for path in Path(folder).rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def load_ground_truth(coco_json):
    source = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    image_names = {image["id"]: image["file_name"] for image in source["images"]}
    ground_truths = {}
    for annotation in source["annotations"]:
        filename = image_names.get(annotation["image_id"])
        bbox = annotation.get("bbox")
        if filename and bbox:
            x, y, width, height = map(float, bbox)
            ground_truths.setdefault(filename, []).append((x, y, x + width, y + height))
    return ground_truths


def extract_detections(page):
    boxes = []
    polygons = []
    for block in page.blocks:
        for line in block.lines:
            for span in line.text_spans:
                polygon = getattr(span, "polygon", None)
                if not polygon:
                    continue
                points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                polygons.append([points.tolist()])
                boxes.append(
                    (
                        float(points[:, 0].min()),
                        float(points[:, 1].min()),
                        float(points[:, 0].max()),
                        float(points[:, 1].max()),
                    )
                )
    return boxes, polygons


def create_detector():
    cache_dir = Path.home() / ".manuscript" / "models" / CONFIG["weights"]
    cached_weights = cache_dir / f"{CONFIG['weights']}.onnx"
    cached_config = cache_dir / f"{CONFIG['weights']}.json"
    options = {"weights": CONFIG["weights"]}
    if cached_weights.is_file() and cached_config.is_file():
        # Skip repeated SHA-256 verification of the 144 MB external tensor file.
        options = {"weights": cached_weights, "config": cached_config}
    detector = BenchmarkMask2Former(
        **options, device=CONFIG["device"]
    )
    detector._initialize_session()
    providers = detector.onnx_session.get_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "Mask2Former requested CUDA but ONNX Runtime fell back to CPU. "
            f"Active providers: {providers}"
        )
    return detector


def benchmark_gpu(detector, images, dataset_config):
    gc.collect()
    process = psutil.Process()
    ram_after_load = process.memory_info().rss / 1024**2
    for image_path in images[: CONFIG["warmup"]]:
        detector.predict(image_path)

    predictions = {}
    polygon_predictions = {}
    times = []
    ram_peak = ram_after_load
    for image_path in tqdm(images, desc="Mask2Former", unit="image"):
        started = time.perf_counter()
        boxes, polygons = extract_detections(detector.predict(image_path))
        times.append(time.perf_counter() - started)
        predictions[prediction_key(image_path, dataset_config)] = boxes
        polygon_predictions[prediction_key(image_path, dataset_config)] = polygons
        ram_peak = max(ram_peak, process.memory_info().rss / 1024**2)

    values = np.asarray(times)
    return {
        "device": CONFIG["device"],
        "providers": detector.onnx_session.get_providers(),
        "detection_level": "line",
        "architecture": "Mask2Former",
        "weights": CONFIG["weights"],
        "image_size": detector.image_size,
        "score_threshold": detector.score_threshold,
        "mask_logit_threshold": detector.mask_logit_threshold,
        "nms_iou_threshold": detector.nms_iou_threshold,
        "num_images": len(images),
        "mean_time_ms": float(values.mean() * 1000),
        "median_time_ms": float(np.median(values) * 1000),
        "throughput_fps": float(len(images) / values.sum()),
        "ram_after_load_mb": ram_after_load,
        "ram_peak_mb": ram_peak,
        "ram_delta_mb": ram_peak - ram_after_load,
        "predictions": predictions,
        "polygon_predictions": polygon_predictions,
    }


def main():
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("Mask2Former line benchmark requires CUDAExecutionProvider")
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = None
    for dataset_name, dataset in DATASETS.items():
        print(f"\n### DATASET: {dataset_name}")
        output_file = output_dir / f"{dataset_name}_{MODEL_NAME}.json"
        if output_file.is_file():
            existing = json.loads(output_file.read_text(encoding="utf-8")).get("gpu")
            if result_is_compatible(existing, dataset) and existing.get("ground_truth_version") == PREPARATION_VERSION and prediction_artifact_path(output_file).is_file():
                print(f"Skip: already completed ({output_file})")
                continue
        images = get_image_files(dataset["folder"])
        if not images or not dataset["annotations"].is_file():
            print("Skip: missing images or annotations")
            continue
        if detector is None:
            detector = create_detector()
        gpu_stats = benchmark_gpu(detector, images, dataset)
        gpu_stats["ground_truth_version"] = PREPARATION_VERSION
        save_predictions(output_file, gpu_stats["predictions"], gpu_stats["polygon_predictions"])
        ground_truth_boxes, ground_truth_polygons = load_coco_detection_ground_truth(dataset["annotations"])
        gpu_stats["accuracy_metrics"] = evaluate_dataset(gpu_stats.pop("predictions"), ground_truth_boxes)
        gpu_stats["polygon_iou_metrics"] = evaluate_polygon_dataset(
            gpu_stats.pop("polygon_predictions"), ground_truth_polygons
        )
        output_file.write_text(
            json.dumps(
                {"dataset": dataset_name, "level": "line", "cpu": None, "gpu": gpu_stats},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
