import numpy as np
import pytest

from synthetic import ImageSpec, make_detector, render
from vehicle_detection.data.tiling_config import TilingConfig
from vehicle_detection.inference.adaptive import (
    AdaptiveConfig,
    AdaptivePredictor,
    regions_from_detections,
)
from vehicle_detection.inference.detections import Detection
from vehicle_detection.inference.sliced import SlicedPredictor


def det(x_min, y_min, x_max, y_max):
    return Detection((float(x_min), float(y_min), float(x_max), float(y_max)), 0.9, 0)


def test_no_detections_no_regions():
    assert regions_from_detections([], (1024, 1024), 32) == []


def test_padding_expands_and_clamps_to_the_image():
    regions = regions_from_detections([det(10, 10, 50, 50)], (100, 100), 32)
    assert regions == [(0.0, 0.0, 82.0, 82.0)]


def test_overlapping_regions_are_merged():
    boxes = [det(100, 100, 140, 140), det(150, 100, 190, 140)]
    regions = regions_from_detections(boxes, (1024, 1024), 20)
    assert regions == [(80.0, 80.0, 210.0, 160.0)]


def test_distant_regions_stay_separate_and_sorted():
    boxes = [det(800, 800, 840, 840), det(100, 100, 140, 140)]
    regions = regions_from_detections(boxes, (1024, 1024), 10)
    assert regions == [(90.0, 90.0, 150.0, 150.0), (790.0, 790.0, 850.0, 850.0)]


def test_chain_of_overlaps_collapses_to_one_region():
    boxes = [det(0, 0, 40, 40), det(60, 0, 100, 40), det(120, 0, 160, 40)]
    regions = regions_from_detections(boxes, (1024, 1024), 15)
    assert regions == [(0.0, 0.0, 175.0, 55.0)]


def bright_bbox(image: np.ndarray) -> list[Detection]:
    ys, xs = np.nonzero(image[..., 0] > 128)
    if xs.size == 0:
        return []
    return [det(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)]


class RecordingFine:
    def __init__(self, inner):
        self.inner = inner
        self.shapes = []

    def predict(self, image):
        self.shapes.append(image.shape[:2])
        return self.inner.predict(image)


def adaptive(coarse, fine, **config):
    return AdaptivePredictor(coarse, fine, AdaptiveConfig(**config))


def test_end_to_end_maps_detections_back_to_full_image_coordinates():
    seen = []

    def coarse(image):
        seen.append(image.shape[:2])
        return bright_bbox(image)

    image = render(ImageSpec("x", 1024, 1024, "train", "s", ((700, 300, 40, 40),)))
    fine = RecordingFine(SlicedPredictor(make_detector(), TilingConfig()))
    detections = adaptive(coarse, fine, coarse_size=256, padding=32).predict(image)
    assert seen == [(256, 256)]
    assert len(fine.shapes) == 1
    assert fine.shapes[0][0] < 1024 and fine.shapes[0][1] < 1024
    assert len(detections) == 1
    assert detections[0].box == pytest.approx((700, 300, 740, 340), abs=3.0)


def test_empty_coarse_pass_skips_the_fine_pass():
    fine = RecordingFine(SlicedPredictor(make_detector(), TilingConfig()))
    image = render(ImageSpec("x", 1024, 1024, "train", "s"))
    assert adaptive(bright_bbox, fine).predict(image) == []
    assert fine.shapes == []


def test_small_image_is_not_upscaled_for_the_coarse_pass():
    seen = []

    def coarse(image):
        seen.append(image.shape[:2])
        return []

    image = render(ImageSpec("x", 200, 300, "train", "s"))
    fine = RecordingFine(SlicedPredictor(make_detector(), TilingConfig()))
    adaptive(coarse, fine, coarse_size=512).predict(image)
    assert seen == [(300, 200)]


def test_config_validation():
    with pytest.raises(ValueError):
        AdaptiveConfig(coarse_size=8)
    with pytest.raises(ValueError):
        AdaptiveConfig(padding=-1)


def test_degenerate_and_off_image_boxes_make_no_region():
    boxes = [det(10, 10, 10, 10), det(-5, -5, -1, -1), det(2000, 2000, 2100, 2100)]
    assert regions_from_detections(boxes, (1024, 1024), 0) == []


def test_off_image_coarse_box_does_not_run_the_fine_pass():
    fine = RecordingFine(SlicedPredictor(make_detector(), TilingConfig()))
    image = render(ImageSpec("x", 600, 700, "train", "s"))
    predictor = adaptive(lambda _: [det(-5, -5, -1, -1)], fine, padding=0)
    assert predictor.predict(image) == []
    assert fine.shapes == []


class ManyDetections:
    def predict(self, image):
        return [det(i, 0, i + 5, 5) for i in range(10)]


def test_adaptive_predictor_respects_max_detections():
    image = render(ImageSpec("x", 1024, 1024, "train", "s", ((700, 300, 40, 40),)))
    predictor = AdaptivePredictor(
        bright_bbox, ManyDetections(), AdaptiveConfig(coarse_size=256), max_detections=4
    )
    assert len(predictor.predict(image)) == 4
