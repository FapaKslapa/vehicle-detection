import math
import random

import pytest

from vehicle_detection.counting.counts import (
    TOTAL,
    category_slugs,
    count_summary,
    count_table,
    predicted_counts,
)
from vehicle_detection.experiments.metrics import counting_metrics

CATEGORIES = [{"id": 1, "name": "car"}, {"id": 2, "name": "pickup truck"}]


def make_gt(annotations, image_ids=(1, 2)):
    return {
        "images": [
            {"id": i, "file_name": f"{i}.jpg", "width": 1024, "height": 1024} for i in image_ids
        ],
        "annotations": annotations,
        "categories": CATEGORIES,
    }


def box(image_id, category_id, score=None):
    item = {"image_id": image_id, "category_id": category_id, "bbox": [0, 0, 10, 10]}
    if score is not None:
        item["score"] = score
    return item


def summary_row(summary, target):
    return summary.set_index("target").loc[target]


def test_exact_counts_have_no_error():
    gt = make_gt([box(1, 1), box(1, 2), box(2, 1)])
    results = [box(1, 1, 0.9), box(1, 2, 0.9), box(2, 1, 0.9)]
    summary = count_summary(count_table(gt, results, 0.5), CATEGORIES)
    assert summary["mae"].tolist() == [0.0, 0.0, 0.0]
    assert summary["exact_share"].tolist() == [1.0, 1.0, 1.0]


def test_class_names_become_column_safe():
    assert category_slugs(CATEGORIES) == {1: "car", 2: "pickup_truck"}


def test_clashing_class_names_are_refused():
    with pytest.raises(ValueError):
        category_slugs([{"id": 1, "name": "Pickup truck"}, {"id": 2, "name": "pickup-truck"}])


def test_the_table_has_true_predicted_and_error_columns_per_class():
    gt = make_gt([box(1, 1), box(1, 1)])
    table = count_table(gt, [box(1, 1, 0.9), box(1, 2, 0.9), box(1, 2, 0.1)], 0.5)
    first = table.set_index("image_id").loc[1]
    assert first["true_total"] == 2
    assert first["predicted_total"] == 2
    assert first["error_car"] == -1
    assert first["error_pickup_truck"] == 1
    assert table.set_index("image_id").loc[2, "true_total"] == 0


def test_over_counting_gives_a_positive_bias():
    gt = make_gt([box(1, 1), box(2, 1)])
    results = [box(1, 1, 0.9), box(1, 1, 0.9), box(2, 1, 0.9), box(2, 1, 0.9)]
    total = summary_row(count_summary(count_table(gt, results, 0.5), CATEGORIES), TOTAL)
    assert total["bias"] == pytest.approx(1.0)
    assert total["mae"] == pytest.approx(1.0)
    assert total["relative_mae"] == pytest.approx(1.0)


def test_relative_mae_is_not_defined_without_vehicles():
    gt = make_gt([])
    total = summary_row(count_summary(count_table(gt, [box(1, 1, 0.9)], 0.5), CATEGORIES), TOTAL)
    assert math.isnan(total["relative_mae"])


@pytest.mark.parametrize("seed", range(5))
def test_total_and_per_class_errors_match_the_metrics_module(seed):
    rng = random.Random(seed)
    images = range(1, 21)
    gt = make_gt(
        [box(i, rng.choice((1, 2))) for i in images for _ in range(rng.randint(0, 6))], images
    )
    results = [
        box(i, rng.choice((1, 2)), round(rng.random(), 2))
        for i in images
        for _ in range(rng.randint(0, 8))
    ]
    for threshold in (0.2, 0.5, 0.9):
        summary = count_summary(count_table(gt, results, threshold), CATEGORIES).set_index("target")
        reference = counting_metrics(gt, results, threshold)
        assert summary.loc[TOTAL, "mae"] == pytest.approx(reference.mae)
        assert summary.loc[TOTAL, "rmse"] == pytest.approx(reference.rmse)
        assert summary.loc["car", "mae"] == pytest.approx(reference.per_class_mae[1])
        assert summary.loc["pickup_truck", "mae"] == pytest.approx(reference.per_class_mae[2])


def test_predicted_counts_need_no_ground_truth():
    images = [{"id": 7, "file_name": "new.jpg"}, {"id": 8, "file_name": "empty.jpg"}]
    results = [box(7, 1, 0.9), box(7, 2, 0.9), box(7, 2, 0.95), box(7, 1, 0.1)]
    table = predicted_counts(images, results, CATEGORIES, 0.5).set_index("file_name")
    assert table.loc["new.jpg", TOTAL] == 3
    assert table.loc["new.jpg", "pickup_truck"] == 2
    assert table.loc["empty.jpg", TOTAL] == 0


def test_an_empty_table_cannot_be_summarised():
    with pytest.raises(ValueError):
        count_summary(count_table(make_gt([], image_ids=()), [], 0.5), CATEGORIES)
