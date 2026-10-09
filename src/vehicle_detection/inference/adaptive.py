from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Protocol

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from vehicle_detection.inference.detections import Box, Detection


class AdaptiveConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    coarse_size: int = Field(default=512, ge=64)
    padding: int = Field(default=32, ge=0)


class ImagePredictor(Protocol):
    def predict(self, image: np.ndarray) -> list[Detection]: ...


def regions_from_detections(
    detections: Sequence[Detection], image_size: tuple[int, int], padding: int
) -> list[Box]:
    width, height = image_size
    padded = [
        (
            max(0.0, x_min - padding),
            max(0.0, y_min - padding),
            min(float(width), x_max + padding),
            min(float(height), y_max + padding),
        )
        for x_min, y_min, x_max, y_max in (detection.box for detection in detections)
    ]
    return sorted(_merge([box for box in padded if box[2] > box[0] and box[3] > box[1]]))


def _merge(boxes: list[Box]) -> list[Box]:
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        result: list[Box] = []
        for box in merged:
            for index, other in enumerate(result):
                if _intersects(box, other):
                    result[index] = _union(box, other)
                    changed = True
                    break
            else:
                result.append(box)
        merged = result
    return merged


def _intersects(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


class AdaptivePredictor:
    def __init__(
        self,
        coarse: Callable[[np.ndarray], list[Detection]],
        fine: ImagePredictor,
        config: AdaptiveConfig,
    ) -> None:
        self._coarse = coarse
        self._fine = fine
        self._config = config

    def predict(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        scale = min(1.0, self._config.coarse_size / max(height, width))
        coarse_image = image if scale == 1.0 else self._downscale(image, scale)
        coarse_detections = [d.scaled(1.0 / scale) for d in self._coarse(coarse_image)]
        regions = regions_from_detections(coarse_detections, (width, height), self._config.padding)
        detections: list[Detection] = []
        for x_min, y_min, x_max, y_max in regions:
            left, top = math.floor(x_min), math.floor(y_min)
            right, bottom = math.ceil(x_max), math.ceil(y_max)
            crop = image[top:bottom, left:right]
            detections.extend(d.translated(left, top) for d in self._fine.predict(crop))
        detections.sort(key=lambda detection: detection.score, reverse=True)
        return detections

    @staticmethod
    def _downscale(image: np.ndarray, scale: float) -> np.ndarray:
        height, width = image.shape[:2]
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
