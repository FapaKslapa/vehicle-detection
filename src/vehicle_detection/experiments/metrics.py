from __future__ import annotations

import contextlib
import io
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MAX_DETECTIONS = (1, 10, 100)
MATCH_IOU = 0.5
AP, AP50, AP_SMALL, AP_MEDIUM, AP_LARGE = 0, 1, 3, 4, 5
AR_SMALL, AR_MEDIUM, AR_LARGE = 9, 10, 11
NOT_DEFINED = -1


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    ap: float
    ap50: float
    ap_small: float
    ap_medium: float
    ap_large: float
    ar_small: float
    ar_medium: float
    ar_large: float
    per_class_ap: Mapping[int, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PrecisionRecall:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class MatchMetrics:
    precision: float
    recall: float
    f1: float
    per_class: Mapping[int, PrecisionRecall]


@dataclass(frozen=True, slots=True)
class CountingMetrics:
    mae: float
    rmse: float
    per_class_mae: Mapping[int, float]


def detection_metrics(
    gt: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> DetectionMetrics:
    if not gt["annotations"]:
        return DetectionMetrics(*([math.nan] * 8))
    if not results:
        return DetectionMetrics(*([0.0] * 8))
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = {**gt}
        truth.createIndex()
        detections = truth.loadRes([dict(result) for result in results])
        evaluator = COCOeval(truth, detections, iouType="bbox")
        evaluator.params.maxDets = list(MAX_DETECTIONS)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats

    def pick(index: int) -> float:
        value = float(stats[index])
        return math.nan if value == NOT_DEFINED else value

    return DetectionMetrics(
        ap=pick(AP),
        ap50=pick(AP50),
        ap_small=pick(AP_SMALL),
        ap_medium=pick(AP_MEDIUM),
        ap_large=pick(AP_LARGE),
        ar_small=pick(AR_SMALL),
        ar_medium=pick(AR_MEDIUM),
        ar_large=pick(AR_LARGE),
        per_class_ap=_per_class_ap(evaluator),
    )


def match_metrics(
    gt: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    threshold: float,
    iou: float = MATCH_IOU,
) -> MatchMetrics:
    category_ids = [category["id"] for category in gt["categories"]]
    truth = _boxes_by_key(gt["annotations"])
    predicted = _boxes_by_key(result for result in results if result["score"] >= threshold)
    true_positives = Counter[int]()
    false_positives = Counter[int]()
    false_negatives = Counter[int]()
    for key in truth.keys() | predicted.keys():
        category = key[1]
        boxes = [box for _, box in truth.get(key, [])]
        ranked = sorted(predicted.get(key, []), key=lambda item: -item[0])
        taken: set[int] = set()
        for _, box in ranked:
            best, best_iou = None, iou
            for index, candidate in enumerate(boxes):
                if index in taken:
                    continue
                overlap = _iou(box, candidate)
                if overlap >= best_iou:
                    best, best_iou = index, overlap
            if best is None:
                false_positives[category] += 1
            else:
                taken.add(best)
                true_positives[category] += 1
        false_negatives[category] += len(boxes) - len(taken)
    per_class = {
        category: _precision_recall(
            true_positives[category], false_positives[category], false_negatives[category]
        )
        for category in category_ids
    }
    overall = _precision_recall(
        sum(true_positives.values()),
        sum(false_positives.values()),
        sum(false_negatives.values()),
    )
    return MatchMetrics(overall.precision, overall.recall, overall.f1, per_class)


def counting_metrics(
    gt: Mapping[str, Any], results: Sequence[Mapping[str, Any]], threshold: float
) -> CountingMetrics:
    image_ids = [image["id"] for image in gt["images"]]
    category_ids = [category["id"] for category in gt["categories"]]
    truth = _counts(gt["annotations"])
    predicted = _counts(result for result in results if result["score"] >= threshold)
    total_errors = []
    class_errors: dict[int, list[float]] = {category: [] for category in category_ids}
    for image_id in image_ids:
        true_counts = truth.get(image_id, Counter())
        predicted_counts = predicted.get(image_id, Counter())
        total_errors.append(sum(predicted_counts.values()) - sum(true_counts.values()))
        for category in category_ids:
            class_errors[category].append(abs(predicted_counts[category] - true_counts[category]))
    if not total_errors:
        raise ValueError("nessuna immagine nel ground truth per il conteggio")
    return CountingMetrics(
        mae=sum(abs(error) for error in total_errors) / len(total_errors),
        rmse=math.sqrt(sum(error**2 for error in total_errors) / len(total_errors)),
        per_class_mae={
            category: sum(errors) / len(errors) for category, errors in class_errors.items()
        },
    )


def best_threshold(
    gt: Mapping[str, Any], results: Sequence[Mapping[str, Any]], grid: Sequence[float]
) -> tuple[float, CountingMetrics]:
    if not grid:
        raise ValueError("servono soglie da provare")
    best: tuple[float, CountingMetrics] | None = None
    for threshold in sorted(grid):
        metrics = counting_metrics(gt, results, threshold)
        if best is None or metrics.mae < best[1].mae:
            best = (threshold, metrics)
    assert best is not None
    return best


def _per_class_ap(evaluator: COCOeval) -> dict[int, float]:
    precision = evaluator.eval["precision"]
    scores: dict[int, float] = {}
    for index, category in enumerate(evaluator.params.catIds):
        values = precision[:, :, index, 0, -1]
        values = values[values > -1]
        scores[category] = float(values.mean()) if values.size else math.nan
    return scores


def _precision_recall(
    true_positives: int, false_positives: int, false_negatives: int
) -> PrecisionRecall:
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return PrecisionRecall(precision, recall, f1)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _boxes_by_key(items: Any) -> dict[tuple[int, int], list[tuple[float, tuple[float, ...]]]]:
    grouped: dict[tuple[int, int], list[tuple[float, tuple[float, ...]]]] = {}
    for item in items:
        x, y, width, height = item["bbox"]
        grouped.setdefault((item["image_id"], item["category_id"]), []).append(
            (item.get("score", 1.0), (x, y, x + width, y + height))
        )
    return grouped


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - intersection
    )
    return intersection / union if union else 0.0


def _counts(items: Any) -> dict[int, Counter[int]]:
    counts: dict[int, Counter[int]] = {}
    for item in items:
        counts.setdefault(item["image_id"], Counter())[item["category_id"]] += 1
    return counts
