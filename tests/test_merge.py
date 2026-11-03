import pandas as pd
import pytest

from vehicle_detection.evaluation.errors import NO_BOX, confusion_matrix, localisation_summary
from vehicle_detection.evaluation.merge import (
    DEFAULT_GROUP_TEXT,
    ClassGroup,
    merge_categories,
    merge_matrix,
    merge_summary,
    parse_group,
)

CATEGORIES = [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "hatchback"},
    {"id": 3, "name": "pickup truck"},
    {"id": 4, "name": "sedan"},
    {"id": 5, "name": "truck"},
]
CARS = ClassGroup("passenger car", ("car", "hatchback", "sedan", "suv"))


def truth(category_id, x):
    return {"image_id": 1, "category_id": category_id, "bbox": [x, 100, 40, 40], "iscrowd": 0}


def predicted(category_id, x, score=0.9):
    return {"image_id": 1, "category_id": category_id, "bbox": [x, 100, 40, 40], "score": score}


def make_gt(annotations):
    return {
        "images": [{"id": 1, "file_name": "1.jpg", "width": 1024, "height": 1024}],
        "annotations": annotations,
        "categories": CATEGORIES,
    }


def scene():
    gt = make_gt(
        [
            truth(1, 0),  # car, found as car
            truth(4, 100),  # sedan, found as car: a mix-up inside the group
            truth(2, 200),  # hatchback, found as sedan: inside the group
            truth(3, 300),  # pickup truck, found as truck: outside the group
            truth(5, 400),  # truck, found as sedan: crosses the group boundary
            truth(4, 500),  # sedan, missed
        ]
    )
    results = [
        predicted(1, 0),
        predicted(1, 100),
        predicted(4, 200),
        predicted(5, 300),
        predicted(4, 400),
    ]
    return gt, results


def test_the_default_group_is_the_four_passenger_car_types():
    assert parse_group(DEFAULT_GROUP_TEXT) == CARS


def test_none_turns_the_merge_off():
    assert parse_group("none") is None
    assert parse_group(None) is None


@pytest.mark.parametrize("text", ["passenger car", "x=car", "=car,sedan", "x=car,car"])
def test_a_badly_written_group_is_refused(text):
    with pytest.raises(ValueError):
        parse_group(text)


def test_a_group_needs_two_classes_that_exist():
    assert CARS.present(["car", "sedan", "truck"]) == ("car", "sedan")
    assert CARS.applies_to(["car", "sedan"])
    assert not CARS.applies_to(["car", "truck"])


def test_merge_categories_relabels_members_and_keeps_the_rest():
    gt, results = scene()
    merged_gt, merged_results = merge_categories(gt, results, CARS)
    assert merged_gt["categories"] == [
        {"id": 1, "name": "passenger car"},
        {"id": 3, "name": "pickup truck"},
        {"id": 5, "name": "truck"},
    ]
    assert [a["category_id"] for a in merged_gt["annotations"]] == [1, 1, 1, 3, 5, 1]
    assert [r["category_id"] for r in merged_results] == [1, 1, 1, 5, 1]
    assert [a["category_id"] for a in gt["annotations"]] == [1, 4, 2, 3, 5, 4]


def test_merging_needs_two_member_classes():
    gt = make_gt([])
    gt["categories"] = [{"id": 1, "name": "car"}, {"id": 2, "name": "truck"}]
    with pytest.raises(ValueError, match="fewer than two"):
        merge_categories(gt, [], CARS)


def test_the_merged_name_cannot_belong_to_another_class():
    gt = make_gt([])
    gt["categories"] = [*CATEGORIES, {"id": 9, "name": "passenger car"}]
    with pytest.raises(ValueError, match="already the name"):
        merge_categories(gt, [], CARS)


def test_merge_matrix_adds_up_rows_and_columns_in_order():
    gt, results = scene()
    matrix = confusion_matrix(gt, results, 0.5)
    merged = merge_matrix(matrix, CARS)
    assert list(merged.index) == ["passenger car", "pickup truck", "truck", NO_BOX]
    assert list(merged.columns) == list(merged.index)
    assert merged.loc["passenger car", "passenger car"] == 3
    assert merged.loc["passenger car", NO_BOX] == 1
    assert merged.loc["truck", "passenger car"] == 1
    assert int(merged.to_numpy().sum()) == int(matrix.to_numpy().sum())


def test_the_merged_matrix_matches_merging_the_classes_first():
    gt, results = scene()
    direct = confusion_matrix(*merge_categories(gt, results, CARS), 0.5)
    merged = merge_matrix(confusion_matrix(gt, results, 0.5), CARS)
    pd.testing.assert_frame_equal(direct, merged, check_names=False)


def test_merge_summary_separates_car_type_mix_ups_from_other_errors():
    gt, results = scene()
    summary = merge_summary(confusion_matrix(gt, results, 0.5), CARS)
    # 5 vehicles found; only the first car got its exact class
    assert summary["class_accuracy_when_found"] == pytest.approx(1 / 5)
    # merged: car, sedan->car and hatchback->sedan are now right
    assert summary["class_accuracy_merged"] == pytest.approx(3 / 5)
    assert summary["class_errors"] == 4
    assert summary["errors_within_group"] == 2
    assert summary["share_of_class_errors_within_group"] == pytest.approx(2 / 4)
    assert summary["group_vehicles_found"] == 3
    assert summary["group_exact_type_accuracy"] == pytest.approx(1 / 3)
    assert summary["group_any_member_accuracy"] == pytest.approx(1.0)
    assert summary["members"] == "car, hatchback, sedan"


def test_merged_accuracy_agrees_with_the_localisation_summary():
    gt, results = scene()
    merged = merge_matrix(confusion_matrix(gt, results, 0.5), CARS)
    summary = merge_summary(confusion_matrix(gt, results, 0.5), CARS)
    assert summary["class_accuracy_merged"] == localisation_summary(merged).class_accuracy
