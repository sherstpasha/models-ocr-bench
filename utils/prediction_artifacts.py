"""Persist reusable per-image detection predictions outside metric JSON files."""

import json
from pathlib import Path


def box_polygon(box):
    x1, y1, x2, y2 = map(float, box)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def prediction_artifact_path(output_file):
    output_file = Path(output_file)
    return output_file.with_name(f"{output_file.stem}_predictions.json")


def save_predictions(output_file, predictions, polygon_predictions=None):
    """Save boxes and optional multipart polygons in a stable JSON schema."""
    records = {}
    for image_name, boxes in predictions.items():
        polygons = (polygon_predictions or {}).get(image_name, [])
        objects = []
        for index, box in enumerate(boxes):
            polygon = polygons[index] if index < len(polygons) else [box_polygon(box)]
            objects.append(
                {
                    "bbox": [float(value) for value in box],
                    "polygon": polygon,
                }
            )
        records[image_name] = objects

    destination = prediction_artifact_path(output_file)
    destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "coordinate_format": "xy",
                "predictions": records,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Predictions: {destination}", flush=True)
    return destination
