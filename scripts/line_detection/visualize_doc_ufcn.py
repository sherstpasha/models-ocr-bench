"""Render one Doc-UFCN prediction example for every line dataset."""

from pathlib import Path

import cv2
import numpy as np
import torch

from configs.line_detection.benchmark_config import DATASETS
from scripts.line_detection.benchmark_doc_ufcn import (
    CONFIG,
    create_model,
    download_model_files,
    get_image_files,
)


def render_example(model, parameters, image_path, output_path):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    polygons_by_class, _raw, _mask, _overlap = model.predict(
        image_rgb, min_cc=int(parameters["min_cc"])
    )

    rendered = image_bgr.copy()
    polygons = polygons_by_class.get(1, [])
    for item in polygons:
        polygon = np.asarray(item.get("polygon", []), dtype=np.int32).reshape(-1, 2)
        if len(polygon) >= 3:
            cv2.polylines(rendered, [polygon], True, (0, 0, 255), 3, cv2.LINE_AA)

    label = f"Doc-UFCN predictions: {len(polygons)}"
    cv2.rectangle(rendered, (12, 12), (540, 62), (255, 255, 255), -1)
    cv2.putText(
        rendered, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (0, 0, 255), 2, cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), rendered):
        raise RuntimeError(f"Cannot save visualization: {output_path}")
    print(f"Saved: {output_path} ({len(polygons)} predictions)", flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Doc-UFCN visualization requires CUDA")
    paths = download_model_files()
    model, parameters = create_model(paths)
    output_dir = Path(CONFIG["output_dir"]) / "examples"
    for dataset_name, dataset in DATASETS.items():
        images = get_image_files(dataset["folder"])
        if not images:
            print(f"Skip {dataset_name}: no images", flush=True)
            continue
        suffix = Path(images[0]).suffix.lower()
        render_example(
            model,
            parameters,
            images[0],
            output_dir / f"{dataset_name}_doc_ufcn{suffix}",
        )


if __name__ == "__main__":
    main()
