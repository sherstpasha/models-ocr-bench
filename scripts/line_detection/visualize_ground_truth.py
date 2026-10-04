"""Render the first prepared line-level GT example for each dataset."""

import json
from pathlib import Path

import cv2
import numpy as np

from configs.line_detection.benchmark_config import DATASETS, RESULTS_ROOT
from utils.datasets import prediction_key


def main():
    output_dir = RESULTS_ROOT / "ground_truth_examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, config in DATASETS.items():
        source = json.loads(Path(config["annotations"]).read_text(encoding="utf-8"))
        for example_index, image in enumerate(source["images"][:3], start=1):
            candidates = [
                candidate
                for candidate in Path(config["folder"]).rglob("*")
                if candidate.is_file()
                and prediction_key(candidate, config) == image["file_name"]
            ]
            if not candidates:
                raise FileNotFoundError(image["file_name"])
            canvas = cv2.imread(str(candidates[0]), cv2.IMREAD_COLOR)
            for annotation in source["annotations"]:
                if annotation["image_id"] != image["id"]:
                    continue
                for part in annotation.get("segmentation", []):
                    polygon = np.rint(np.asarray(part).reshape(-1, 2)).astype(np.int32)
                    cv2.polylines(canvas, [polygon], True, (0, 255, 0), 3, cv2.LINE_AA)
            destination = output_dir / f"{name}_gt_{example_index}.jpg"
            cv2.imwrite(str(destination), canvas)
            print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
