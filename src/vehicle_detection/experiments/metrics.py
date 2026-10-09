from __future__ import annotations

import contextlib
import io
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MAX_DETECTIONS = (1, 10, 100)
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
    )


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


def _counts(items: Any) -> dict[int, Counter[int]]:
    counts: dict[int, Counter[int]] = {}
    for item in items:
        counts.setdefault(item["image_id"], Counter())[item["category_id"]] += 1
    return counts
