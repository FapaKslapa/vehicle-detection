from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

BRIGHT_THRESHOLD = 200
MIN_COMPONENT_AREA = 4
FULL_SCORE = 0.9
TRUNCATED_SCORE = 0.6


@dataclass(frozen=True)
class ImageSpec:
    name: str
    width: int
    height: int
    split: str
    session: str
    boxes: tuple[tuple[int, int, int, int], ...] = ()


@dataclass(frozen=True)
class SyntheticDataset:
    coco_path: Path
    images_dir: Path
    manifest_path: Path


def render(spec: ImageSpec) -> np.ndarray:
    canvas = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
    for x, y, w, h in spec.boxes:
        canvas[y : y + h, x : x + w] = 255
    return canvas


def write_dataset(root: Path, specs: Sequence[ImageSpec]) -> SyntheticDataset:
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    images: list[dict] = []
    annotations: list[dict] = []
    rows = ["image_id,session_id,location,split"]
    for image_id, spec in enumerate(specs, start=1):
        cv2.imwrite(str(images_dir / f"{spec.name}.jpg"), render(spec))
        images.append(
            {
                "id": image_id,
                "file_name": f"{spec.name}.jpg",
                "width": spec.width,
                "height": spec.height,
            }
        )
        for x, y, w, h in spec.boxes:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
        rows.append(f"{spec.name},{spec.session},site,{spec.split}")
    coco = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "car"}]}
    coco_path = root / "instances.json"
    coco_path.write_text(json.dumps(coco))
    manifest_path = root / "manifest.csv"
    manifest_path.write_text("\n".join(rows) + "\n")
    return SyntheticDataset(coco_path, images_dir, manifest_path)


class BrightSquareDetector(DetectionModel):
    full_score = FULL_SCORE
    truncated_score = TRUNCATED_SCORE

    def load_model(self) -> None:
        self.model = object()

    def perform_inference(self, image: np.ndarray) -> None:
        self._original_predictions = [image]

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int | float]] | None = None,
        full_shape_list: list[list[int | float]] | None = None,
    ) -> None:
        shifts = fix_shift_amount_list(shift_amount_list)
        shapes = fix_full_shape_list(full_shape_list) or [None]
        per_image = []
        for image, shift, shape in zip(self._original_predictions, shifts, shapes, strict=False):
            height, width = image.shape[:2]
            mask = (image[..., 0] > BRIGHT_THRESHOLD).astype(np.uint8)
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            predictions = []
            for x, y, w, h, area in stats[1:]:
                if area < MIN_COMPONENT_AREA:
                    continue
                truncated = x == 0 or y == 0 or x + w == width or y + h == height
                predictions.append(
                    ObjectPrediction(
                        bbox=[int(x), int(y), int(x + w), int(y + h)],
                        category_id=0,
                        category_name="car",
                        score=self.truncated_score if truncated else self.full_score,
                        shift_amount=shift,
                        full_shape=shape,
                    )
                )
            per_image.append(predictions)
        self._object_prediction_list_per_image = per_image


def make_detector(
    full_score: float = FULL_SCORE, truncated_score: float = TRUNCATED_SCORE
) -> BrightSquareDetector:
    detector = BrightSquareDetector(category_mapping={"0": "car"}, confidence_threshold=0.1)
    detector.full_score = full_score
    detector.truncated_score = truncated_score
    return detector
