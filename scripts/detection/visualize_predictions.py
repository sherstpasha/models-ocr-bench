"""Render deterministic GT/prediction comparisons for word detection."""

import json
import random
from pathlib import Path

import cv2

from configs.detection.benchmark_config import BENCHMARKS, DATASETS
from scripts.line_detection.visualize_predictions import comparison, ground_truth_by_image
from utils.datasets import prediction_key
from utils.prediction_artifacts import prediction_artifact_path


SEED = 42
EXAMPLES_PER_DATASET = 3


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
            paths = {
                prediction_key(path, dataset): path
                for path in Path(dataset["folder"]).rglob("*")
                if path.is_file()
            }
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
