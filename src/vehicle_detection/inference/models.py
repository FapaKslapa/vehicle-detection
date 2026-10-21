from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from sahi.auto_model import AutoDetectionModel
from sahi.models.base import DetectionModel
from sahi.models.torchvision import TorchVisionDetectionModel
from sahi.predict import get_prediction
from sahi.utils.torchvision import MODEL_NAME_TO_CONSTRUCTOR

from vehicle_detection.inference.adaptive import ImagePredictor
from vehicle_detection.inference.detections import Detection, detections_from_predictions
from vehicle_detection.inference.images import image_source

ModelKind = Literal["ultralytics", "rtdetr", "torchvision"]
FIRST_LABEL: Mapping[str, int] = {"ultralytics": 0, "rtdetr": 0, "torchvision": 1}


@dataclass(frozen=True, slots=True)
class LabelSpace:
    names: Mapping[str, str]
    label_to_category: Mapping[int, int]


def label_space(categories: Sequence[Mapping[str, Any]], first_label: int) -> LabelSpace:
    if not categories:
        raise ValueError("categorie COCO assenti")
    ids = [category["id"] for category in categories]
    if len(set(ids)) != len(ids):
        raise ValueError("id di categoria duplicati nel COCO")
    ordered = sorted(categories, key=lambda category: category["id"])
    return LabelSpace(
        names={str(first_label + i): category["name"] for i, category in enumerate(ordered)},
        label_to_category={first_label + i: category["id"] for i, category in enumerate(ordered)},
    )


def torchvision_detection_model(
    module: torch.nn.Module, space: LabelSpace, confidence_threshold: float, device: str
) -> DetectionModel:
    model = TorchVisionDetectionModel(
        category_mapping=dict(space.names),
        confidence_threshold=confidence_threshold,
        device=device,
        load_at_init=False,
    )
    model.set_model(module)
    return model


def build_detection_model(
    kind: str,
    weights: str | Path,
    space: LabelSpace,
    confidence_threshold: float,
    device: str,
    torchvision_name: str = "fasterrcnn_resnet50_fpn",
    input_size: int | None = None,
) -> DetectionModel:
    if kind not in FIRST_LABEL:
        raise ValueError(
            f"tipo di modello non supportato: {kind} (validi: {', '.join(FIRST_LABEL)})"
        )
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"pesi non trovati: {weights}")
    if kind == "torchvision":
        module = MODEL_NAME_TO_CONSTRUCTOR[torchvision_name](
            num_classes=len(space.names) + 1, weights=None, weights_backbone=None
        )
        module.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        return torchvision_detection_model(module, space, confidence_threshold, device)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type=kind,
        model_path=str(weights),
        device=device,
        confidence_threshold=confidence_threshold,
        category_mapping=dict(space.names),
    )
    set_input_size(detection_model, input_size)
    return detection_model


def set_input_size(detection_model: DetectionModel, input_size: int | None) -> None:
    if input_size is None:
        return
    overrides = getattr(getattr(detection_model, "model", None), "overrides", None)
    if overrides is None:
        raise ValueError("input_size è supportato solo dai modelli Ultralytics")
    overrides["imgsz"] = input_size


class FullImagePredictor:
    def __init__(self, detection_model: DetectionModel, max_detections: int = 1000) -> None:
        self._model = detection_model
        self._max_detections = max_detections

    def predict(self, image: str | Path | np.ndarray) -> list[Detection]:
        result = get_prediction(image_source(image), self._model, verbose=0)
        detections = detections_from_predictions(result.object_prediction_list)
        detections.sort(key=lambda detection: detection.score, reverse=True)
        return detections[: self._max_detections]


class RemappedPredictor:
    def __init__(self, predictor: ImagePredictor, label_to_category: Mapping[int, int]) -> None:
        self._predictor = predictor
        self._label_to_category = label_to_category

    def predict(self, image: Any) -> list[Detection]:
        return [
            Detection(d.box, d.score, self._label_to_category[d.category_id])
            for d in self._predictor.predict(image)
            if d.category_id in self._label_to_category
        ]
