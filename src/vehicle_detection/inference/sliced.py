from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sahi.auto_model import AutoDetectionModel
from sahi.models.base import DetectionModel
from sahi.predict import get_sliced_prediction

from vehicle_detection.data.tiling_config import TilingConfig
from vehicle_detection.inference.detections import (
    Detection,
    detections_from_predictions,
    suppress_contained,
)
from vehicle_detection.inference.images import image_source


class PostprocessConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    postprocess_type: Literal["GREEDYNMM", "NMM"] = "GREEDYNMM"
    match_metric: Literal["IOU", "IOS"] = "IOS"
    match_threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    perform_standard_pred: bool = True
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    max_detections: int = Field(default=1000, ge=1)
    containment_threshold: float = Field(default=0.9, gt=0.5, le=1.0)


class SlicedPredictor:
    def __init__(
        self,
        detection_model: DetectionModel,
        tiling: TilingConfig,
        postprocess: PostprocessConfig | None = None,
    ) -> None:
        self._model = detection_model
        self._tiling = tiling
        self._postprocess = postprocess or PostprocessConfig()

    @classmethod
    def from_pretrained(
        cls,
        model_type: str,
        model_path: str | Path,
        device: str,
        tiling: TilingConfig,
        postprocess: PostprocessConfig | None = None,
        category_mapping: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> SlicedPredictor:
        postprocess = postprocess or PostprocessConfig()
        model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=str(model_path),
            device=device,
            confidence_threshold=postprocess.confidence_threshold,
            category_mapping=category_mapping,
            **kwargs,
        )
        return cls(model, tiling, postprocess)

    def predict(self, image: str | Path | np.ndarray) -> list[Detection]:
        source = image_source(image)
        result = get_sliced_prediction(
            source,
            self._model,
            slice_height=self._tiling.slice_size,
            slice_width=self._tiling.slice_size,
            overlap_height_ratio=self._tiling.overlap_ratio,
            overlap_width_ratio=self._tiling.overlap_ratio,
            perform_standard_pred=self._postprocess.perform_standard_pred,
            postprocess_type=self._postprocess.postprocess_type,
            postprocess_match_metric=self._postprocess.match_metric,
            postprocess_match_threshold=self._postprocess.match_threshold,
            auto_slice_resolution=False,
            force_postprocess_type=True,
            verbose=0,
        )
        detections = suppress_contained(
            detections_from_predictions(result.object_prediction_list),
            self._postprocess.containment_threshold,
        )
        detections.sort(key=lambda detection: detection.score, reverse=True)
        return detections[: self._postprocess.max_detections]
