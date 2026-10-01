import unicodedata
from typing import Dict, List, Sequence, Tuple


Box = Tuple[float, float, float, float]
STANDARD_IOU_THRESHOLDS = tuple(round(0.5 + index * 0.05, 2) for index in range(10))


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
        "matched_images": matched_images,
        "missing_predictions": missing_predictions,
    }
