import unicodedata
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:  # Minimal Doc-UFCN environment uses OpenCV instead.
    Image = ImageDraw = None


Box = Tuple[float, float, float, float]
STANDARD_IOU_THRESHOLDS = tuple(round(0.5 + index * 0.05, 2) for index in range(10))


def load_coco_detection_ground_truth(coco_json):
    source = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    image_names = {image["id"]: image["file_name"] for image in source["images"]}
    boxes = {}
    polygons = {}
    for annotation in source["annotations"]:
        filename = image_names.get(annotation["image_id"])
        bbox = annotation.get("bbox")
        if not filename or not bbox:
            continue
        x, y, width, height = map(float, bbox)
        boxes.setdefault(filename, []).append((x, y, x + width, y + height))
        parts = [
            np.asarray(part, dtype=np.float32).reshape(-1, 2).tolist()
            for part in annotation.get("segmentation", [])
            if len(part) >= 6
        ]
        if not parts:
            parts = [[[x, y], [x + width, y], [x + width, y + height], [x, y + height]]]
        polygons.setdefault(filename, []).append(parts)
    return boxes, polygons


def boxes_to_polygon_objects(boxes):
    return [
        [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
        for x1, y1, x2, y2 in boxes
    ]


def _polygon_parts(value):
    array = np.asarray(value, dtype=object)
    if not len(array):
        return []
    first = value[0]
    if len(first) == 2 and np.isscalar(first[0]):
        value = [value]
    return [np.asarray(part, dtype=np.float32).reshape(-1, 2) for part in value if len(part) >= 3]


def _paint_polygons(mask, parts, offset):
    contours = [np.rint(part - offset).astype(np.int32) for part in parts if len(part) >= 3]
    if not contours:
        return
    if Image is not None:
        image = Image.fromarray(mask)
        draw = ImageDraw.Draw(image)
        for contour in contours:
            draw.polygon([tuple(point) for point in contour], fill=1)
        mask[:] = np.asarray(image)
        return
    import cv2

    cv2.fillPoly(mask, contours, 1)


def _parts_bbox(parts):
    points = np.concatenate(parts)
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def _polygon_iou_parts(first, second, first_box, second_box):
    if box_iou(first_box, second_box) == 0.0:
        return 0.0
    left = int(np.floor(min(first_box[0], second_box[0])))
    top = int(np.floor(min(first_box[1], second_box[1])))
    right = int(np.ceil(max(first_box[2], second_box[2])))
    bottom = int(np.ceil(max(first_box[3], second_box[3])))
    shape = (bottom - top + 1, right - left + 1)
    first_mask = np.zeros(shape, dtype=np.uint8)
    second_mask = np.zeros(shape, dtype=np.uint8)
    offset = np.asarray([left, top], dtype=np.float32)
    _paint_polygons(first_mask, first, offset)
    _paint_polygons(second_mask, second, offset)
    intersection = np.count_nonzero(first_mask & second_mask)
    union = np.count_nonzero(first_mask | second_mask)
    return float(intersection / union) if union else 0.0


def polygon_iou(first, second) -> float:
    """Return pixel-accurate IoU for simple or multipart polygon objects."""
    first = _polygon_parts(first)
    second = _polygon_parts(second)
    if not first or not second:
        return 0.0
    return _polygon_iou_parts(first, second, _parts_bbox(first), _parts_bbox(second))


def evaluate_polygon_dataset(predictions, ground_truths, iou_threshold=0.5):
    """Evaluate one-to-one polygon detections and return precision/recall/H-mean."""
    image_names = set(predictions) | set(ground_truths)
    prepared_predictions = {
        image_name: [
            (parts, _parts_bbox(parts))
            for value in predictions.get(image_name, [])
            if (parts := _polygon_parts(value))
        ]
        for image_name in image_names
    }
    prepared_ground_truths = {
        image_name: [
            (parts, _parts_bbox(parts))
            for value in ground_truths.get(image_name, [])
            if (parts := _polygon_parts(value))
        ]
        for image_name in image_names
    }
    candidates = {}
    for image_name in image_names:
        candidates[image_name] = sorted(
            (
                (_polygon_iou_parts(pred, gt, pred_box, gt_box), pred_index, gt_index)
                for pred_index, (pred, pred_box) in enumerate(prepared_predictions[image_name])
                for gt_index, (gt, gt_box) in enumerate(prepared_ground_truths[image_name])
                if box_iou(pred_box, gt_box) > 0.0
            ),
            reverse=True,
        )
    metrics = _evaluate_at_threshold(
        predictions, ground_truths, candidates, image_names, iou_threshold
    )
    dice_intersection = 0
    dice_predicted = 0
    dice_expected = 0
    for image_name in image_names:
        predicted_parts = [part for obj, _box in prepared_predictions[image_name] for part in obj]
        expected_parts = [part for obj, _box in prepared_ground_truths[image_name] for part in obj]
        all_parts = predicted_parts + expected_parts
        if not all_parts:
            continue
        points = np.concatenate(all_parts)
        left, top = np.floor(points.min(axis=0)).astype(int)
        right, bottom = np.ceil(points.max(axis=0)).astype(int)
        shape = (bottom - top + 1, right - left + 1)
        predicted_mask = np.zeros(shape, dtype=np.uint8)
        expected_mask = np.zeros(shape, dtype=np.uint8)
        offset = np.asarray([left, top], dtype=np.float32)
        _paint_polygons(predicted_mask, predicted_parts, offset)
        _paint_polygons(expected_mask, expected_parts, offset)
        dice_intersection += int(np.count_nonzero(predicted_mask & expected_mask))
        dice_predicted += int(np.count_nonzero(predicted_mask))
        dice_expected += int(np.count_nonzero(expected_mask))
    dice_denominator = dice_predicted + dice_expected
    return {
        "iou_threshold": iou_threshold,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "hmean": metrics["f1"],
        "dice_f1": 2 * dice_intersection / dice_denominator if dice_denominator else 0.0,
        "mean_matched_iou": metrics["mean_matched_iou"],
        "true_positives": metrics["true_positives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "evaluated_images": metrics["evaluated_images"],
    }


def box_iou(first: Box, second: Box) -> float:
    """Return intersection-over-union for two ``(x1, y1, x2, y2)`` boxes."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_dataset(
    predictions: Dict[str, List[Box]],
    ground_truths: Dict[str, List[Box]],
    iou_threshold: float = 0.5,
) -> Dict[str, object]:
    """Evaluate detections and report F1 at IoU 0.5 and IoU 0.5:0.95."""
    image_names = set(predictions) | set(ground_truths)
    candidates_by_image = {}
    for image_name in image_names:
        predicted = predictions.get(image_name, [])
        expected = ground_truths.get(image_name, [])
        candidates_by_image[image_name] = sorted(
            (
                (box_iou(pred_box, gt_box), pred_index, gt_index)
                for pred_index, pred_box in enumerate(predicted)
                for gt_index, gt_box in enumerate(expected)
            ),
            reverse=True,
        )

    thresholds = list(STANDARD_IOU_THRESHOLDS)
    if iou_threshold not in thresholds:
        thresholds.append(iou_threshold)
    metrics_by_threshold = {
        threshold: _evaluate_at_threshold(
            predictions,
            ground_truths,
            candidates_by_image,
            image_names,
            threshold,
        )
        for threshold in thresholds
    }
    result = metrics_by_threshold[iou_threshold].copy()
    result["f1@0.5"] = metrics_by_threshold[0.5]["f1"]
    result["f1@0.5:0.95"] = sum(
        metrics_by_threshold[threshold]["f1"]
        for threshold in STANDARD_IOU_THRESHOLDS
    ) / len(STANDARD_IOU_THRESHOLDS)
    result["f1_by_iou"] = {
        f"{threshold:.2f}": metrics_by_threshold[threshold]["f1"]
        for threshold in STANDARD_IOU_THRESHOLDS
    }
    return result


def _evaluate_at_threshold(
    predictions,
    ground_truths,
    candidates_by_image,
    image_names,
    iou_threshold,
):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_ious: List[float] = []

    for image_name in image_names:
        predicted = predictions.get(image_name, [])
        expected = ground_truths.get(image_name, [])
        used_predictions = set()
        used_ground_truths = set()
        for iou, pred_index, gt_index in candidates_by_image[image_name]:
            if iou < iou_threshold:
                break
            if pred_index in used_predictions or gt_index in used_ground_truths:
                continue
            used_predictions.add(pred_index)
            used_ground_truths.add(gt_index)
            matched_ious.append(iou)

        true_positives += len(used_predictions)
        false_positives += len(predicted) - len(used_predictions)
        false_negatives += len(expected) - len(used_ground_truths)

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou_threshold": iou_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "evaluated_images": len(image_names),
    }


def has_standard_f1_metrics(stats) -> bool:
    """Return whether saved benchmark stats contain the current F1 metrics."""
    if not stats:
        return False
    metrics = stats.get("accuracy_metrics", {})
    return "f1@0.5" in metrics and "f1@0.5:0.95" in metrics


def normalize_text(text: str, lowercase: bool = True, normalize_unicode: str = "NFC") -> str:
    text = "" if text is None else str(text)
    if normalize_unicode:
        text = unicodedata.normalize(normalize_unicode, text)
    if lowercase:
        text = text.lower()
    return text.strip()


def levenshtein_distance(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    if len(s1) < len(s2):
        s1, s2 = s2, s1

    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, start=1):
        current = [i]
        for j, c2 in enumerate(s2, start=1):
            insertions = previous[j] + 1
            deletions = current[j - 1] + 1
            substitutions = previous[j - 1] + (c1 != c2)
            current.append(min(insertions, deletions, substitutions))
        previous = current

    return previous[-1]


def levenshtein_distance_tokens(seq1: Sequence[str], seq2: Sequence[str]) -> int:
    if seq1 == seq2:
        return 0
    if len(seq1) == 0:
        return len(seq2)
    if len(seq2) == 0:
        return len(seq1)

    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    previous = list(range(len(seq2) + 1))
    for i, t1 in enumerate(seq1, start=1):
        current = [i]
        for j, t2 in enumerate(seq2, start=1):
            insertions = previous[j] + 1
            deletions = current[j - 1] + 1
            substitutions = previous[j - 1] + (t1 != t2)
            current.append(min(insertions, deletions, substitutions))
        previous = current

    return previous[-1]


def cer_score(predictions: List[str], references: List[str]) -> float:
    total_distance = 0
    total_chars = 0
    for pred, ref in zip(predictions, references):
        total_distance += levenshtein_distance(pred, ref)
        total_chars += len(ref)
    return float(total_distance / total_chars) if total_chars > 0 else 0.0


def wer_score(predictions: List[str], references: List[str]) -> float:
    total_distance = 0
    total_words = 0
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        total_distance += levenshtein_distance_tokens(pred_words, ref_words)
        total_words += len(ref_words)
    return float(total_distance / total_words) if total_words > 0 else 0.0


def accuracy_score(predictions: List[str], references: List[str]) -> float:
    if not references:
        return 0.0
    correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
    return float(correct / len(references))


def character_similarity_score(
    predictions: List[str], references: List[str]
) -> float:
    """Return mean normalized edit similarity in the inclusive [0, 1] range."""
    if not references:
        return 0.0
    similarities = []
    for prediction, reference in zip(predictions, references):
        denominator = max(len(prediction), len(reference))
        if denominator == 0:
            similarities.append(1.0)
            continue
        distance = levenshtein_distance(prediction, reference)
        similarities.append(1.0 - distance / denominator)
    return float(sum(similarities) / len(similarities))


def evaluate_recognition(
    predictions: Dict[str, str],
    ground_truths: Dict[str, str],
    lowercase: bool = True,
    normalize_unicode: str = "NFC",
) -> Dict[str, object]:
    matched_images = [img for img in ground_truths.keys() if img in predictions]
    missing_predictions = [img for img in ground_truths.keys() if img not in predictions]

    y_true = []
    y_pred = []
    for image_name in matched_images:
        y_true.append(normalize_text(ground_truths[image_name], lowercase, normalize_unicode))
        y_pred.append(normalize_text(predictions[image_name], lowercase, normalize_unicode))

    return {
        "cer": cer_score(y_pred, y_true),
        "wer": wer_score(y_pred, y_true),
        "accuracy": accuracy_score(y_pred, y_true),
        "character_similarity": character_similarity_score(y_pred, y_true),
        "matched_images": matched_images,
        "missing_predictions": missing_predictions,
    }
