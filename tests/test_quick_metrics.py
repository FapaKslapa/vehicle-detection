import math

import pytest

from vehicle_detection.experiments.metrics import (
    best_threshold,
    counting_metrics,
    detection_metrics,
)

CATEGORIES = [{"id": 1, "name": "car"}, {"id": 2, "name": "truck"}]


def gt_box(box_id, image_id, category_id, x, y, w, h):
    return {
        "id": box_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, w, h],
        "area": w * h,
        "iscrowd": 0,
    }


def make_gt(annotations, image_ids=(1, 2)):
    return {
        "images": [
            {"id": i, "file_name": f"{i}.jpg", "width": 512, "height": 512} for i in image_ids
        ],
        "annotations": annotations,
        "categories": CATEGORIES,
    }


def result(image_id, category_id, x, y, w, h, score=0.9):
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, w, h],
        "score": score,
    }


def test_perfect_predictions_give_ap_one():
    gt = make_gt([gt_box(1, 1, 1, 10, 10, 50, 50), gt_box(2, 2, 2, 100, 100, 120, 120)])
    results = [result(1, 1, 10, 10, 50, 50), result(2, 2, 100, 100, 120, 120)]
    metrics = detection_metrics(gt, results)
    assert metrics.ap == pytest.approx(1.0)
    assert metrics.ap50 == pytest.approx(1.0)


def test_no_predictions_give_zero_ap_without_crashing():
    gt = make_gt([gt_box(1, 1, 1, 10, 10, 50, 50)])
    metrics = detection_metrics(gt, [])
    assert metrics.ap == 0.0
    assert metrics.ap_small == 0.0 or math.isnan(metrics.ap_small)


def test_no_ground_truth_gives_nan_instead_of_crashing():
    metrics = detection_metrics(make_gt([]), [])
    assert math.isnan(metrics.ap)


def test_size_breakdown_isolates_small_objects():
    gt = make_gt([gt_box(1, 1, 1, 10, 10, 20, 20), gt_box(2, 1, 1, 200, 200, 150, 150)])
    results = [result(1, 1, 200, 200, 150, 150)]
    metrics = detection_metrics(gt, results)
    assert metrics.ap_small == pytest.approx(0.0)
    assert metrics.ap_large == pytest.approx(1.0)
    assert math.isnan(metrics.ap_medium)


def test_counting_exact_counts_have_zero_error():
    gt = make_gt([gt_box(1, 1, 1, 0, 0, 10, 10), gt_box(2, 2, 2, 0, 0, 10, 10)])
    results = [result(1, 1, 0, 0, 10, 10), result(2, 2, 0, 0, 10, 10)]
    metrics = counting_metrics(gt, results, threshold=0.5)
    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.per_class_mae == {1: 0.0, 2: 0.0}


def test_counting_handles_images_without_gt_or_predictions():
    gt = make_gt([gt_box(1, 1, 1, 0, 0, 10, 10)])
    metrics = counting_metrics(gt, [], threshold=0.5)
    assert metrics.mae == pytest.approx(0.5)
    assert metrics.rmse == pytest.approx(math.sqrt(0.5))


def test_counting_threshold_filters_low_scores():
    gt = make_gt([gt_box(1, 1, 1, 0, 0, 10, 10)], image_ids=(1,))
    results = [result(1, 1, 0, 0, 10, 10, score=0.3), result(1, 1, 50, 50, 10, 10, score=0.8)]
    assert counting_metrics(gt, results, threshold=0.5).mae == 0.0
    assert counting_metrics(gt, results, threshold=0.1).mae == 1.0


def test_per_class_mae():
    gt = make_gt([gt_box(1, 1, 1, 0, 0, 10, 10), gt_box(2, 1, 2, 0, 0, 10, 10)], image_ids=(1,))
    results = [result(1, 1, 0, 0, 10, 10), result(1, 1, 0, 0, 10, 10)]
    metrics = counting_metrics(gt, results, threshold=0.5)
    assert metrics.per_class_mae == {1: 1.0, 2: 1.0}
    assert metrics.mae == 0.0


def test_best_threshold_picks_lowest_mae():
    gt = make_gt([gt_box(1, 1, 1, 0, 0, 10, 10)], image_ids=(1,))
    results = [result(1, 1, 0, 0, 10, 10, score=0.9), result(1, 1, 50, 0, 10, 10, score=0.4)]
    threshold, metrics = best_threshold(gt, results, [0.1, 0.5, 0.95])
    assert threshold == 0.5
    assert metrics.mae == 0.0


def test_threshold_tie_prefers_lower():
    gt = make_gt([gt_box(1, 1, 1, 0, 0, 10, 10)], image_ids=(1,))
    results = [result(1, 1, 0, 0, 10, 10, score=0.9)]
    threshold, _ = best_threshold(gt, results, [0.7, 0.2, 0.5])
    assert threshold == 0.2


def test_best_threshold_rejects_empty_grid():
    with pytest.raises(ValueError, match="soglie"):
        best_threshold(make_gt([]), [], [])
