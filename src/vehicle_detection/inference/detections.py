from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sahi.prediction import ObjectPrediction

Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    box: Box
    score: float
    category_id: int

    def translated(self, dx: float, dy: float) -> Detection:
        x_min, y_min, x_max, y_max = self.box
        return Detection(
            (x_min + dx, y_min + dy, x_max + dx, y_max + dy), self.score, self.category_id
        )

    def scaled(self, factor: float) -> Detection:
        x_min, y_min, x_max, y_max = self.box
        return Detection(
            (x_min * factor, y_min * factor, x_max * factor, y_max * factor),
            self.score,
            self.category_id,
        )


def detections_from_predictions(predictions: Iterable[ObjectPrediction]) -> list[Detection]:
    detections = []
    for prediction in predictions:
        x_min, y_min, x_max, y_max = prediction.bbox.get_shifted_box().to_xyxy()
        detections.append(
            Detection(
                (float(x_min), float(y_min), float(x_max), float(y_max)),
                float(prediction.score.value),
                int(prediction.category.id),
            )
        )
    return detections


def to_coco_results(image_id: int, detections: Iterable[Detection]) -> list[dict[str, Any]]:
    results = []
    for detection in detections:
        x_min, y_min, x_max, y_max = detection.box
        results.append(
            {
                "image_id": image_id,
                "category_id": detection.category_id,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "score": detection.score,
            }
        )
    return results


def suppress_contained(detections: Iterable[Detection], threshold: float) -> list[Detection]:
    ordered = sorted(detections, key=lambda d: (-_area(d.box), -d.score))
    kept: list[Detection] = []
    for candidate in ordered:
        limit = threshold * _area(candidate.box)
        if any(
            other.category_id == candidate.category_id
            and _intersection(other.box, candidate.box) >= limit
            for other in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a: Box, b: Box) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, width) * max(0.0, height)
