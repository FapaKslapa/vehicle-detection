import pytest
from sahi.prediction import ObjectPrediction

from vehicle_detection.inference.detections import (
    Detection,
    detections_from_predictions,
    suppress_contained,
    to_coco_results,
)


def test_translated_and_scaled():
    detection = Detection((10.0, 20.0, 50.0, 80.0), 0.8, 2)
    assert detection.translated(5, -10) == Detection((15.0, 10.0, 55.0, 70.0), 0.8, 2)
    assert detection.scaled(2.0) == Detection((20.0, 40.0, 100.0, 160.0), 0.8, 2)


def test_detection_is_immutable():
    detection = Detection((0.0, 0.0, 1.0, 1.0), 0.5, 0)
    with pytest.raises(AttributeError):
        detection.score = 0.9


def test_from_sahi_predictions():
    prediction = ObjectPrediction(
        bbox=[10, 20, 50, 80], category_id=2, category_name="truck", score=0.8
    )
    assert detections_from_predictions([prediction]) == [
        Detection((10.0, 20.0, 50.0, 80.0), 0.8, 2)
    ]


def test_from_sahi_predictions_applies_shift():
    prediction = ObjectPrediction(
        bbox=[10, 20, 50, 80],
        category_id=1,
        category_name="car",
        score=0.6,
        shift_amount=[100, 200],
        full_shape=[1024, 1024],
    )
    assert detections_from_predictions([prediction])[0].box == (110.0, 220.0, 150.0, 280.0)


def test_to_coco_results():
    results = to_coco_results(7, [Detection((10.0, 20.0, 50.0, 80.0), 0.8, 2)])
    assert results == [
        {"image_id": 7, "category_id": 2, "bbox": [10.0, 20.0, 40.0, 60.0], "score": 0.8}
    ]


def test_to_coco_results_empty():
    assert to_coco_results(1, []) == []


def box(x_min, y_min, x_max, y_max, score=0.9, category_id=0):
    return Detection((float(x_min), float(y_min), float(x_max), float(y_max)), score, category_id)


def test_contained_box_is_suppressed():
    kept = suppress_contained([box(512, 100, 540, 140), box(500, 100, 540, 140)], 0.9)
    assert kept == [box(500, 100, 540, 140)]


def test_suppression_ignores_input_order_and_scores():
    big, small = box(0, 0, 100, 100, 0.2), box(10, 10, 60, 60, 0.9)
    assert suppress_contained([small, big], 0.9) == [big]
    assert suppress_contained([big, small], 0.9) == [big]


def test_partial_overlap_is_kept():
    boxes = [box(0, 0, 100, 100), box(60, 0, 160, 100)]
    assert sorted(suppress_contained(boxes, 0.9), key=lambda d: d.box) == boxes


def test_different_categories_are_kept():
    boxes = [box(0, 0, 100, 100, category_id=0), box(10, 10, 60, 60, category_id=1)]
    assert len(suppress_contained(boxes, 0.9)) == 2


def test_identical_boxes_collapse_to_the_best_score():
    kept = suppress_contained([box(0, 0, 10, 10, 0.5), box(0, 0, 10, 10, 0.8)], 0.9)
    assert kept == [box(0, 0, 10, 10, 0.8)]


def test_suppression_of_nothing():
    assert suppress_contained([], 0.9) == []
