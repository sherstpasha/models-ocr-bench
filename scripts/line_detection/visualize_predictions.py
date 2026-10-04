"""Render deterministic GT/prediction comparisons from saved line predictions."""

import json
import random
from pathlib import Path

import cv2
import numpy as np

from configs.line_detection.benchmark_config import BENCHMARKS, DATASETS
from utils.datasets import prediction_key
from utils.prediction_artifacts import prediction_artifact_path


SEED = 42
EXAMPLES_PER_DATASET = 3
PANEL_MAX_WIDTH = 1200


def ground_truth_by_image(path):
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {image["id"]: image["file_name"] for image in source["images"]}
    result = {}
    for annotation in source["annotations"]:
        name = names.get(annotation["image_id"])
        if name:
            result.setdefault(name, []).append(annotation.get("segmentation", []))
    return result


def image_paths(config):
    return {
        prediction_key(path, config): path
        for path in Path(config["folder"]).rglob("*")
        if path.is_file()
    }


def draw_objects(image, objects, color, thickness):
    rendered = image.copy()
    for parts in objects:
        for part in parts:
            points = np.rint(np.asarray(part, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
            if len(points) >= 3:
                cv2.polylines(rendered, [points], True, color, thickness, cv2.LINE_AA)
    return rendered


def add_title(image, title, color):
    cv2.rectangle(image, (0, 0), (image.shape[1], 70), (255, 255, 255), -1)
    cv2.putText(image, title, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)


def comparison(image_path, ground_truth, predictions):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    scale = min(1.0, PANEL_MAX_WIDTH / image.shape[1])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def scaled(objects):
        return [
            [(np.asarray(part, dtype=np.float32) * scale).tolist() for part in parts]
            for parts in objects
        ]

    left = draw_objects(image, scaled(ground_truth), (0, 180, 0), 3)
    right = draw_objects(image, scaled(predictions), (0, 0, 255), 3)
    add_title(left, f"GT: {len(ground_truth)} lines", (0, 150, 0))
    add_title(right, f"Prediction: {len(predictions)} lines", (0, 0, 220))
    divider = np.full((image.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.hstack([left, divider, right])


def main():
    for model_name, config in BENCHMARKS.items():
        for dataset_name, dataset in DATASETS.items():
            result_file = Path(config["output_dir"]) / f"{dataset_name}_{model_name}.json"
            artifact = prediction_artifact_path(result_file)
            if not artifact.is_file():
                print(f"Skip {model_name}/{dataset_name}: missing {artifact.name}")
                continue
            predictions = json.loads(artifact.read_text(encoding="utf-8"))["predictions"]
            ground_truths = ground_truth_by_image(dataset["annotations"])
            paths = image_paths(dataset)
            names = sorted(set(predictions) & set(ground_truths) & set(paths))
            selected = random.Random(f"{SEED}:{dataset_name}").sample(
                names, min(EXAMPLES_PER_DATASET, len(names))
            )
            output_dir = Path(config["output_dir"]) / "visualizations" / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)
            for index, name in enumerate(selected, start=1):
                predicted_polygons = [item["polygon"] for item in predictions[name]]
                rendered = comparison(paths[name], ground_truths[name], predicted_polygons)
                destination = output_dir / f"{index}_{Path(name).stem}.jpg"
                cv2.imwrite(str(destination), rendered, [cv2.IMWRITE_JPEG_QUALITY, 92])
                print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
