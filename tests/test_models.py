import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision

from synthetic import make_detector
from vehicle_detection.data.tiling_config import TilingConfig
from vehicle_detection.inference.detections import Detection
from vehicle_detection.inference.models import (
    FullImagePredictor,
    RemappedPredictor,
    build_detection_model,
    label_space,
    torchvision_detection_model,
)
from vehicle_detection.inference.sliced import SlicedPredictor

CATEGORIES = [{"id": 1, "name": "car"}, {"id": 2, "name": "hatchback"}, {"id": 5, "name": "suv"}]


def test_label_space_for_ultralytics_starts_at_zero():
    space = label_space(CATEGORIES, first_label=0)
    assert space.names == {"0": "car", "1": "hatchback", "2": "suv"}
    assert space.label_to_category == {0: 1, 1: 2, 2: 5}


def test_label_space_for_torchvision_reserves_background():
    space = label_space(CATEGORIES, first_label=1)
    assert space.names == {"1": "car", "2": "hatchback", "3": "suv"}
    assert space.label_to_category == {1: 1, 2: 2, 3: 5}


def test_label_space_sorts_categories_by_id():
    space = label_space(list(reversed(CATEGORIES)), first_label=0)
    assert space.label_to_category == {0: 1, 1: 2, 2: 5}


def test_label_space_rejects_empty_and_duplicate_categories():
    with pytest.raises(ValueError, match="categorie"):
        label_space([], first_label=0)
    with pytest.raises(ValueError, match="duplicat"):
        label_space([{"id": 1, "name": "a"}, {"id": 1, "name": "b"}], first_label=0)


def test_unknown_kind_rejected(tmp_path):
    space = label_space(CATEGORIES, first_label=0)
    with pytest.raises(ValueError, match="tipo di modello"):
        build_detection_model("onnx", tmp_path / "w.pt", space, 0.25, "cpu")


def test_missing_weights_raise_italian_error(tmp_path):
    space = label_space(CATEGORIES, first_label=0)
    with pytest.raises(FileNotFoundError, match="pesi non trovati"):
        build_detection_model("ultralytics", tmp_path / "w.pt", space, 0.25, "cpu")


def test_remapped_predictor_translates_labels_and_drops_unknown():
    class Fixed:
        def predict(self, image):
            return [
                Detection((0, 0, 1, 1), 0.9, 0),
                Detection((0, 0, 1, 1), 0.8, 2),
                Detection((0, 0, 1, 1), 0.7, 9),
            ]

    remapped = RemappedPredictor(Fixed(), {0: 1, 2: 5})
    assert [d.category_id for d in remapped.predict(np.zeros((4, 4, 3), np.uint8))] == [1, 5]


def square_image() -> np.ndarray:
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[100:140, 50:90] = 255
    image[200:260, 150:200] = 255
    return image


def test_full_image_predictor_returns_boxes_in_image_coordinates():
    predictor = FullImagePredictor(make_detector(), max_detections=10)
    boxes = sorted(d.box for d in predictor.predict(square_image()))
    assert boxes == [(50.0, 100.0, 90.0, 140.0), (150.0, 200.0, 200.0, 260.0)]


def test_full_image_predictor_caps_and_orders_detections():
    predictor = FullImagePredictor(make_detector(), max_detections=1)
    detections = predictor.predict(square_image())
    assert len(detections) == 1


def test_full_image_predictor_accepts_a_path(tmp_path):
    import cv2

    path = tmp_path / "x.png"
    cv2.imwrite(str(path), square_image())
    assert len(FullImagePredictor(make_detector(), max_detections=10).predict(path)) == 2


def test_full_image_predictor_missing_file_raises_italian_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="immagine non trovata"):
        FullImagePredictor(make_detector(), max_detections=10).predict(tmp_path / "no.png")


def test_torchvision_labels_are_never_background_and_map_to_coco_ids():
    torch.manual_seed(0)
    module = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, weights_backbone=None, num_classes=4, min_size=320, max_size=320
    )
    space = label_space(CATEGORIES, first_label=1)
    model = torchvision_detection_model(module, space, confidence_threshold=0.0, device="cpu")
    image = np.random.default_rng(0).integers(0, 255, (320, 320, 3), dtype=np.uint8)
    raw = FullImagePredictor(model, max_detections=500).predict(image)
    assert raw
    assert {d.category_id for d in raw} <= {1, 2, 3}
    mapped = RemappedPredictor(
        FullImagePredictor(model, max_detections=500), space.label_to_category
    )
    assert {d.category_id for d in mapped.predict(image)} <= {1, 2, 5}


def test_torchvision_model_works_inside_sliced_predictor():
    torch.manual_seed(0)
    module = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, weights_backbone=None, num_classes=4, min_size=256, max_size=256
    )
    space = label_space(CATEGORIES, first_label=1)
    model = torchvision_detection_model(module, space, confidence_threshold=0.0, device="cpu")
    predictor = SlicedPredictor(model, TilingConfig(slice_size=256, overlap_ratio=0.2))
    image = np.random.default_rng(1).integers(0, 255, (512, 512, 3), dtype=np.uint8)
    detections = predictor.predict(image)
    assert detections
    assert all(0 <= d.box[0] <= d.box[2] <= 512 for d in detections)


def test_torchvision_weights_roundtrip_through_build_detection_model(tmp_path):
    torch.manual_seed(0)
    module = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, weights_backbone=None, num_classes=4
    )
    weights = tmp_path / "frcnn.pt"
    torch.save(module.state_dict(), weights)
    space = label_space(CATEGORIES, first_label=1)
    model = build_detection_model(
        "torchvision",
        weights,
        space,
        0.25,
        "cpu",
        torchvision_name="fasterrcnn_mobilenet_v3_large_fpn",
    )
    assert model.num_categories == 3


YOLO_WEIGHTS = Path(os.environ.get("VD_TEST_YOLO_WEIGHTS", ""))


@pytest.mark.skipif(
    not YOLO_WEIGHTS.is_file(), reason="imposta VD_TEST_YOLO_WEIGHTS su un .pt locale"
)
def test_ultralytics_adapter_loads_local_weights():
    space = label_space([{"id": i + 1, "name": f"c{i}"} for i in range(80)], first_label=0)
    model = build_detection_model("ultralytics", YOLO_WEIGHTS, space, 0.25, "cpu")
    detections = FullImagePredictor(model, max_detections=50).predict(square_image())
    assert all(isinstance(d, Detection) for d in detections)
