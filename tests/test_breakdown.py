import pandas as pd
import pytest

from vehicle_detection.evaluation.breakdown import (
    bin_labels,
    bin_of,
    outcome_table,
    recall_breakdown,
    truth_attributes,
)
from vehicle_detection.evaluation.errors import agnostic_matches, category_names, label_errors

CATEGORIES = [{"id": 1, "name": "car"}, {"id": 2, "name": "sedan"}]


def truth(image_id, category_id, x, y, side):
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, side, side],
        "area": side * side,
        "iscrowd": 0,
    }


def make_gt(annotations):
    return {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 1024, "height": 1024},
            {"id": 2, "file_name": "b.jpg", "width": 1024, "height": 1024},
        ],
        "annotations": annotations,
        "categories": CATEGORIES,
    }


MANIFEST = pd.DataFrame(
    {
        "image_id": ["a", "b"],
        "session_id": ["scene_1", "scene_2"],
        "location": ["loc0", "loc1"],
        "split": ["val", "val"],
    }
)


def test_bin_labels_cover_below_between_and_above():
    assert bin_labels((60, 77, 100)) == ["<60", "60-77", "77-100", ">=100"]
    assert bin_labels((3, 6, 10), integer=True) == ["<3", "3-5", "6-9", ">=10"]


def test_bin_edges_must_increase():
    with pytest.raises(ValueError):
        bin_labels((77, 60))


def test_a_value_on_an_edge_falls_in_the_upper_bin():
    labels = bin_labels((60, 77, 100))
    assert bin_of(59.9, (60, 77, 100), labels) == "<60"
    assert bin_of(60, (60, 77, 100), labels) == "60-77"
    assert bin_of(100, (60, 77, 100), labels) == ">=100"


def test_attributes_describe_size_edge_density_and_scene():
    gt = make_gt(
        [
            truth(1, 1, 0, 300, 50),  # touches the left edge, small
            truth(1, 2, 500, 500, 80),
            truth(1, 1, 700, 700, 120),
            truth(2, 2, 984, 10, 40),  # touches the right edge
        ]
    )
    attributes = truth_attributes(gt, MANIFEST).set_index("annotation_index")
    assert list(attributes["size_bin"].astype(str)) == ["<60", "77-100", ">=100", "<60"]
    assert list(attributes["at_edge"]) == [True, False, False, True]
    assert list(attributes["vehicles_in_image"]) == [3, 3, 3, 1]
    assert list(attributes["density_bin"].astype(str)) == ["3-5", "3-5", "3-5", "<3"]
    assert list(attributes["scene"]) == ["scene_1", "scene_1", "scene_1", "scene_2"]
    assert list(attributes["location"]) == ["loc0", "loc0", "loc0", "loc1"]


def test_without_a_manifest_scene_and_location_are_unknown():
    attributes = truth_attributes(make_gt([truth(1, 1, 100, 100, 50)]))
    assert attributes.loc[0, "scene"] == "unknown"


def test_recall_separates_finding_a_vehicle_from_naming_it():
    gt = make_gt(
        [
            truth(1, 1, 100, 100, 50),  # found as car
            truth(1, 2, 300, 300, 50),  # found, but called car
            truth(1, 2, 600, 600, 50),  # missed
        ]
    )
    results = [
        {"image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [300, 300, 50, 50], "score": 0.9},
    ]
    labels = label_errors(gt, results, 0.5)
    table = outcome_table(
        truth_attributes(gt, MANIFEST),
        labels.truths,
        agnostic_matches(gt, results, 0.5),
        results,
        category_names(gt),
    )
    breakdown = recall_breakdown(table, attributes=("category",)).set_index("value")
    assert breakdown.loc["car", "recall_any_class"] == pytest.approx(1.0)
    assert breakdown.loc["sedan", "recall_any_class"] == pytest.approx(0.5)
    assert breakdown.loc["sedan", "recall_correct_class"] == pytest.approx(0.0)
    assert breakdown.loc["sedan", "class_accuracy_when_found"] == pytest.approx(0.0)
    assert breakdown.loc["sedan", "missed"] == 1
    assert breakdown.loc["sedan", "wrong_class"] == 1
    assert list(table["predicted_category"].fillna("-")) == ["car", "car", "-"]


def test_every_box_needs_an_outcome():
    gt = make_gt([truth(1, 1, 100, 100, 50)])
    with pytest.raises(ValueError):
        outcome_table(
            truth_attributes(gt),
            pd.DataFrame(columns=["annotation_index", "outcome", "best_iou"]),
            {},
            [],
            category_names(gt),
        )


def test_size_and_edge_crosses_the_size_bin_with_the_image_edge():
    gt = make_gt([truth(1, 1, 0, 300, 50), truth(1, 2, 500, 500, 80), truth(2, 1, 100, 100, 50)])
    attributes = truth_attributes(gt, MANIFEST).set_index("annotation_index")
    assert list(attributes["size_and_edge"].astype(str)) == [
        "<60 edge",
        "77-100 inside",
        "<60 inside",
    ]
    assert list(attributes["size_and_edge"].cat.categories) == [
        "<60 inside",
        "<60 edge",
        "60-77 inside",
        "60-77 edge",
        "77-100 inside",
        "77-100 edge",
        ">=100 inside",
        ">=100 edge",
    ]


def test_recall_by_size_and_edge_is_part_of_the_breakdown():
    gt = make_gt([truth(1, 1, 0, 300, 50), truth(1, 2, 500, 500, 50)])
    results = [{"image_id": 1, "category_id": 2, "bbox": [500, 500, 50, 50], "score": 0.9}]
    labels = label_errors(gt, results, 0.5)
    table = outcome_table(
        truth_attributes(gt, MANIFEST),
        labels.truths,
        agnostic_matches(gt, results, 0.5),
        results,
        category_names(gt),
    )
    breakdown = recall_breakdown(table).set_index(["attribute", "value"])
    assert breakdown.loc[("size_and_edge", "<60 inside"), "recall_any_class"] == 1.0
    assert breakdown.loc[("size_and_edge", "<60 edge"), "recall_any_class"] == 0.0
