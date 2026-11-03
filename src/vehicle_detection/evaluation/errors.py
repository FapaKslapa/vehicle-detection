"""Error analysis of detections against COCO ground truth.

Every prediction at or above the score threshold gets exactly one outcome, and so does
every ground-truth box. The outcomes follow TIDE (Bolya et al., ECCV 2020) with two
choices made explicit:

- Matching is the greedy, class-aware IoU >= 0.5 matching of
  ``evaluation.metrics.match_counts``, so ``correct`` predictions are exactly its true
  positives and the precision, recall and F1 reported by ``compare`` can be split into
  these outcomes without any disagreement.
- A prediction that lands (IoU >= 0.5) on a vehicle already found under another class is
  a ``cross_class_duplicate``, not a classification error. NMS runs per class, so the
  same vehicle can survive as both "car" and "sedan"; this is what inflates counts.

Prediction outcomes, checked in this order for unmatched predictions:
  duplicate              same class, IoU >= 0.5 with a vehicle already matched
  cross_class_duplicate  other class, IoU >= 0.5 with a vehicle already matched
  classification         other class, IoU >= 0.5 with a vehicle nobody matched
  localization           same class, 0.1 <= IoU < 0.5
  both                   other class, 0.1 <= IoU < 0.5
  background             IoU < 0.1 with every vehicle

Ground-truth outcomes:
  detected          matched by a prediction of its own class
  wrong_class       a prediction of another class covers it with IoU >= 0.5
  poorly_localized  some prediction overlaps it with 0.1 <= IoU < 0.5
  missed            nothing overlaps it with IoU >= 0.1

The confusion matrix uses a separate class-agnostic matching (the class is ignored when
pairing boxes), which is the standard way to ask "which class did the model give to the
vehicles it found".
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pandas as pd

FOREGROUND_IOU = 0.5
BACKGROUND_IOU = 0.1
NO_BOX = "background"
AGNOSTIC_CATEGORY = {"id": 1, "name": "vehicle"}

Box = tuple[float, float, float, float]


class PredictionOutcome(StrEnum):
    CORRECT = "correct"
    DUPLICATE = "duplicate"
    CROSS_CLASS_DUPLICATE = "cross_class_duplicate"
    CLASSIFICATION = "classification"
    LOCALIZATION = "localization"
    BOTH = "both"
    BACKGROUND = "background"


class TruthOutcome(StrEnum):
    DETECTED = "detected"
    WRONG_CLASS = "wrong_class"
    POORLY_LOCALIZED = "poorly_localized"
    MISSED = "missed"


@dataclass(frozen=True, slots=True)
class ErrorLabels:
    predictions: pd.DataFrame
    truths: pd.DataFrame

    def prediction_counts(self) -> dict[str, int]:
        counts = self.predictions["outcome"].value_counts()
        return {outcome.value: int(counts.get(outcome.value, 0)) for outcome in PredictionOutcome}

    def truth_counts(self) -> dict[str, int]:
        counts = self.truths["outcome"].value_counts()
        return {outcome.value: int(counts.get(outcome.value, 0)) for outcome in TruthOutcome}


@dataclass(frozen=True, slots=True)
class LocalisationSummary:
    precision: float
    recall: float
    f1: float
    class_accuracy: float


@dataclass(frozen=True, slots=True)
class _Truth:
    position: int
    box: Box
    category_id: int


@dataclass(frozen=True, slots=True)
class _Prediction:
    position: int
    box: Box
    category_id: int
    score: float


PREDICTION_COLUMNS = (
    "result_index",
    "image_id",
    "category_id",
    "score",
    "outcome",
    "annotation_index",
    "iou",
)
TRUTH_COLUMNS = ("annotation_index", "image_id", "category_id", "outcome", "best_iou")


def label_errors(
    gt: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    threshold: float,
    foreground: float = FOREGROUND_IOU,
    background: float = BACKGROUND_IOU,
) -> ErrorLabels:
    if not 0 < background < foreground <= 1:
        raise ValueError(f"need 0 < background ({background}) < foreground ({foreground}) <= 1")
    truths = _truths_by_image(gt["annotations"])
    predictions = _ranked_predictions(results, threshold)
    prediction_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    for image_id in sorted(truths.keys() | predictions.keys()):
        image_truths = truths.get(image_id, [])
        image_predictions = predictions.get(image_id, [])
        overlaps = [[iou(p.box, t.box) for t in image_truths] for p in image_predictions]
        matched = _greedy_match(image_predictions, image_truths, overlaps, foreground, True)
        taken = set(matched.values())
        for p_index, prediction in enumerate(image_predictions):
            outcome, t_index, value = _prediction_outcome(
                p_index, prediction, image_truths, overlaps, matched, taken, foreground, background
            )
            prediction_rows.append(
                {
                    "result_index": prediction.position,
                    "image_id": image_id,
                    "category_id": prediction.category_id,
                    "score": prediction.score,
                    "outcome": outcome.value,
                    "annotation_index": (
                        None if t_index is None else image_truths[t_index].position
                    ),
                    "iou": value,
                }
            )
        matched_by_truth = {t_index: p_index for p_index, t_index in matched.items()}
        for t_index, truth in enumerate(image_truths):
            outcome, value = _truth_outcome(
                t_index,
                truth,
                image_predictions,
                overlaps,
                matched_by_truth,
                foreground,
                background,
            )
            truth_rows.append(
                {
                    "annotation_index": truth.position,
                    "image_id": image_id,
                    "category_id": truth.category_id,
                    "outcome": outcome.value,
                    "best_iou": value,
                }
            )
    prediction_frame = pd.DataFrame(prediction_rows, columns=list(PREDICTION_COLUMNS))
    prediction_frame["annotation_index"] = prediction_frame["annotation_index"].astype("Int64")
    truth_frame = pd.DataFrame(truth_rows, columns=list(TRUTH_COLUMNS))
    return ErrorLabels(
        predictions=prediction_frame.sort_values("result_index", ignore_index=True),
        truths=truth_frame.sort_values("annotation_index", ignore_index=True),
    )


def agnostic_matches(
    gt: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    threshold: float,
    iou_threshold: float = FOREGROUND_IOU,
) -> dict[int, int]:
    """Map annotation index -> result index, pairing boxes regardless of their class."""
    truths = _truths_by_image(gt["annotations"])
    predictions = _ranked_predictions(results, threshold)
    pairs: dict[int, int] = {}
    for image_id, image_predictions in predictions.items():
        image_truths = truths.get(image_id, [])
        overlaps = [[iou(p.box, t.box) for t in image_truths] for p in image_predictions]
        matched = _greedy_match(image_predictions, image_truths, overlaps, iou_threshold, False)
        for p_index, t_index in matched.items():
            pairs[image_truths[t_index].position] = image_predictions[p_index].position
    return pairs


def confusion_matrix(
    gt: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    threshold: float,
    iou_threshold: float = FOREGROUND_IOU,
) -> pd.DataFrame:
    """Rows are true classes, columns predicted classes; ``background`` is "no box"."""
    names = category_names(gt)
    labels = [*names.values(), NO_BOX]
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype="int64")
    pairs = agnostic_matches(gt, results, threshold, iou_threshold)
    for position, annotation in enumerate(gt["annotations"]):
        result = pairs.get(position)
        column = NO_BOX if result is None else names[results[result]["category_id"]]
        matrix.loc[names[annotation["category_id"]], column] += 1
    used = set(pairs.values())
    for position, result in enumerate(results):
        if result["score"] >= threshold and position not in used:
            matrix.loc[NO_BOX, names[result["category_id"]]] += 1
    matrix.index.name = "truth"
    matrix.columns.name = "predicted"
    return matrix


def localisation_summary(matrix: pd.DataFrame) -> LocalisationSummary:
    real = [label for label in matrix.index if label != NO_BOX]
    found = int(matrix.loc[real, real].to_numpy().sum())
    truths = int(matrix.loc[real].to_numpy().sum())
    predictions = int(matrix[real].to_numpy().sum())
    correct = int(sum(matrix.loc[label, label] for label in real))
    precision = _ratio(found, predictions)
    recall = _ratio(found, truths)
    return LocalisationSummary(
        precision=precision,
        recall=recall,
        f1=_ratio(2 * precision * recall, precision + recall),
        class_accuracy=_ratio(correct, found),
    )


def class_agnostic(
    gt: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Collapse every class into one "vehicle" class, for class-agnostic AP."""
    category_id = AGNOSTIC_CATEGORY["id"]
    return (
        {
            **gt,
            "categories": [dict(AGNOSTIC_CATEGORY)],
            "annotations": [{**a, "category_id": category_id} for a in gt["annotations"]],
        },
        [{**r, "category_id": category_id} for r in results],
    )


def category_names(gt: Mapping[str, Any]) -> dict[int, str]:
    ordered = sorted(gt["categories"], key=lambda category: category["id"])
    names = {category["id"]: category["name"] for category in ordered}
    if NO_BOX in names.values():
        raise ValueError(f"a category is called '{NO_BOX}', which is reserved")
    return names


def iou(first: Box, second: Box) -> float:
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


def xyxy(bbox: Sequence[float]) -> Box:
    x, y, width, height = bbox
    return (x, y, x + width, y + height)


def _truths_by_image(annotations: Iterable[Mapping[str, Any]]) -> dict[int, list[_Truth]]:
    grouped: dict[int, list[_Truth]] = defaultdict(list)
    for position, annotation in enumerate(annotations):
        grouped[annotation["image_id"]].append(
            _Truth(position, xyxy(annotation["bbox"]), annotation["category_id"])
        )
    return grouped


def _ranked_predictions(
    results: Iterable[Mapping[str, Any]], threshold: float
) -> dict[int, list[_Prediction]]:
    grouped: dict[int, list[_Prediction]] = defaultdict(list)
    for position, result in enumerate(results):
        if result["score"] >= threshold:
            grouped[result["image_id"]].append(
                _Prediction(
                    position, xyxy(result["bbox"]), result["category_id"], float(result["score"])
                )
            )
    # stable sort: equal scores keep the input order, as in evaluation.metrics.match_counts
    return {image_id: sorted(items, key=lambda p: -p.score) for image_id, items in grouped.items()}


def _greedy_match(
    predictions: Sequence[_Prediction],
    truths: Sequence[_Truth],
    overlaps: Sequence[Sequence[float]],
    threshold: float,
    same_class: bool,
) -> dict[int, int]:
    matched: dict[int, int] = {}
    taken: set[int] = set()
    for p_index, prediction in enumerate(predictions):
        best, best_iou = None, threshold
        for t_index, truth in enumerate(truths):
            if t_index in taken or (same_class and truth.category_id != prediction.category_id):
                continue
            # ">=" keeps the last of equal candidates, like evaluation.metrics.match_counts
            if overlaps[p_index][t_index] >= best_iou:
                best, best_iou = t_index, overlaps[p_index][t_index]
        if best is not None:
            matched[p_index] = best
            taken.add(best)
    return matched


def _best(
    p_index: int,
    truths: Sequence[_Truth],
    overlaps: Sequence[Sequence[float]],
    keep: Callable[[_Truth], bool],
) -> tuple[float, int | None]:
    candidates = [
        (overlaps[p_index][t_index], t_index) for t_index, truth in enumerate(truths) if keep(truth)
    ]
    return max(candidates, default=(0.0, None))


def _prediction_outcome(
    p_index: int,
    prediction: _Prediction,
    truths: Sequence[_Truth],
    overlaps: Sequence[Sequence[float]],
    matched: Mapping[int, int],
    taken: set[int],
    foreground: float,
    background: float,
) -> tuple[PredictionOutcome, int | None, float]:
    if p_index in matched:
        t_index = matched[p_index]
        return PredictionOutcome.CORRECT, t_index, overlaps[p_index][t_index]
    same_iou, same = _best(
        p_index, truths, overlaps, lambda t: t.category_id == prediction.category_id
    )
    other_iou, other = _best(
        p_index, truths, overlaps, lambda t: t.category_id != prediction.category_id
    )
    if same_iou >= foreground:
        return PredictionOutcome.DUPLICATE, same, same_iou
    if other_iou >= foreground:
        if other in taken:
            return PredictionOutcome.CROSS_CLASS_DUPLICATE, other, other_iou
        return PredictionOutcome.CLASSIFICATION, other, other_iou
    if same_iou >= background:
        return PredictionOutcome.LOCALIZATION, same, same_iou
    if other_iou >= background:
        return PredictionOutcome.BOTH, other, other_iou
    best_iou, best = max((same_iou, same), (other_iou, other), key=lambda item: item[0])
    return PredictionOutcome.BACKGROUND, best if best_iou > 0 else None, best_iou


def _truth_outcome(
    t_index: int,
    truth: _Truth,
    predictions: Sequence[_Prediction],
    overlaps: Sequence[Sequence[float]],
    matched_by_truth: Mapping[int, int],
    foreground: float,
    background: float,
) -> tuple[TruthOutcome, float]:
    if t_index in matched_by_truth:
        return TruthOutcome.DETECTED, overlaps[matched_by_truth[t_index]][t_index]
    other_best = max(
        (
            overlaps[p_index][t_index]
            for p_index, prediction in enumerate(predictions)
            if prediction.category_id != truth.category_id
        ),
        default=0.0,
    )
    any_best = max((row[t_index] for row in overlaps), default=0.0)
    if other_best >= foreground:
        return TruthOutcome.WRONG_CLASS, other_best
    if any_best >= background:
        return TruthOutcome.POORLY_LOCALIZED, any_best
    return TruthOutcome.MISSED, any_best


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
