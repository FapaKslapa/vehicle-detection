import random

import pandas as pd
import pytest

from vehicle_detection.evaluation.errors import (
    NO_BOX,
    PredictionOutcome,
    TruthOutcome,
    agnostic_matches,
    class_agnostic,
    confusion_matrix,
    label_errors,
    localisation_summary,
)
from vehicle_detection.experiments.metrics import match_counts

CATEGORIES = [{"id": 1, "name": "car"}, {"id": 2, "name": "sedan"}, {"id": 3, "name": "truck"}]
CAR, SEDAN, TRUCK = 1, 2, 3


def truth(box_id, image_id, category_id, x, y, w=40, h=40):
    return {
        "id": box_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, w, h],
        "area": w * h,
        "iscrowd": 0,
    }


def make_gt(annotations, image_ids=(1,)):
    return {
        "images": [
            {"id": i, "file_name": f"{i}.jpg", "width": 1024, "height": 1024} for i in image_ids
        ],
        "annotations": annotations,
        "categories": CATEGORIES,
    }


def predicted(image_id, category_id, x, y, w=40, h=40, score=0.9):
    return {"image_id": image_id, "category_id": category_id, "bbox": [x, y, w, h], "score": score}


def only_prediction(labels):
    assert len(labels.predictions) == 1
    return labels.predictions.iloc[0]


def only_truth(labels):
    assert len(labels.truths) == 1
    return labels.truths.iloc[0]


def test_a_perfect_prediction_is_correct_and_detected():
    labels = label_errors(make_gt([truth(1, 1, CAR, 100, 100)]), [predicted(1, CAR, 100, 100)], 0.5)
    assert only_prediction(labels)["outcome"] == PredictionOutcome.CORRECT
    assert only_truth(labels)["outcome"] == TruthOutcome.DETECTED
    assert only_prediction(labels)["iou"] == pytest.approx(1.0)


def test_the_right_box_with_the_wrong_class_is_a_classification_error():
    labels = label_errors(
        make_gt([truth(1, 1, SEDAN, 100, 100)]), [predicted(1, CAR, 100, 100)], 0.5
    )
    assert only_prediction(labels)["outcome"] == PredictionOutcome.CLASSIFICATION
    assert only_truth(labels)["outcome"] == TruthOutcome.WRONG_CLASS


def test_a_second_box_of_the_same_class_on_a_found_vehicle_is_a_duplicate():
    labels = label_errors(
        make_gt([truth(1, 1, CAR, 100, 100)]),
        [predicted(1, CAR, 100, 100, score=0.9), predicted(1, CAR, 102, 101, score=0.8)],
        0.5,
    )
    assert list(labels.predictions["outcome"]) == [
        PredictionOutcome.CORRECT,
        PredictionOutcome.DUPLICATE,
    ]
    assert only_truth(labels)["outcome"] == TruthOutcome.DETECTED


def test_a_box_of_another_class_on_a_found_vehicle_is_a_cross_class_duplicate():
    # the lower-scoring box has the right class: the vehicle is still found, and the
    # extra "car" box is what makes the count too high
    labels = label_errors(
        make_gt([truth(1, 1, SEDAN, 100, 100)]),
        [predicted(1, CAR, 100, 100, score=0.9), predicted(1, SEDAN, 101, 100, score=0.7)],
        0.5,
    )
    assert list(labels.predictions["outcome"]) == [
        PredictionOutcome.CROSS_CLASS_DUPLICATE,
        PredictionOutcome.CORRECT,
    ]
    assert only_truth(labels)["outcome"] == TruthOutcome.DETECTED


def test_a_loose_box_of_the_right_class_is_a_localization_error():
    labels = label_errors(make_gt([truth(1, 1, CAR, 100, 100)]), [predicted(1, CAR, 120, 120)], 0.5)
    assert only_prediction(labels)["outcome"] == PredictionOutcome.LOCALIZATION
    assert only_truth(labels)["outcome"] == TruthOutcome.POORLY_LOCALIZED


def test_a_loose_box_of_the_wrong_class_is_both():
    labels = label_errors(
        make_gt([truth(1, 1, TRUCK, 100, 100)]), [predicted(1, CAR, 120, 120)], 0.5
    )
    assert only_prediction(labels)["outcome"] == PredictionOutcome.BOTH


def test_a_box_far_from_every_vehicle_is_background_and_the_vehicle_is_missed():
    labels = label_errors(make_gt([truth(1, 1, CAR, 100, 100)]), [predicted(1, CAR, 600, 600)], 0.5)
    assert only_prediction(labels)["outcome"] == PredictionOutcome.BACKGROUND
    assert pd.isna(only_prediction(labels)["annotation_index"])
    assert only_truth(labels)["outcome"] == TruthOutcome.MISSED


def test_predictions_below_the_threshold_are_ignored():
    labels = label_errors(
        make_gt([truth(1, 1, CAR, 100, 100)]), [predicted(1, CAR, 100, 100, score=0.2)], 0.5
    )
    assert labels.predictions.empty
    assert only_truth(labels)["outcome"] == TruthOutcome.MISSED


def test_images_without_vehicles_still_label_their_predictions():
    labels = label_errors(make_gt([], image_ids=(1, 2)), [predicted(2, CAR, 10, 10)], 0.5)
    assert only_prediction(labels)["outcome"] == PredictionOutcome.BACKGROUND
    assert labels.truths.empty


def test_iou_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        label_errors(make_gt([]), [], 0.5, foreground=0.1, background=0.5)


def random_scene(seed, images=12, classes=(CAR, SEDAN, TRUCK)):
    rng = random.Random(seed)
    annotations, results = [], []
    for image_id in range(1, images + 1):
        for _ in range(rng.randint(0, 8)):
            x, y = rng.uniform(0, 900), rng.uniform(0, 900)
            category = rng.choice(classes)
            annotations.append(truth(len(annotations) + 1, image_id, category, x, y))
            for _ in range(rng.randint(0, 3)):
                results.append(
                    predicted(
                        image_id,
                        rng.choice(classes) if rng.random() < 0.3 else category,
                        x + rng.uniform(-25, 25),
                        y + rng.uniform(-25, 25),
                        score=round(rng.random(), 2),
                    )
                )
        for _ in range(rng.randint(0, 2)):
            results.append(
                predicted(image_id, rng.choice(classes), rng.uniform(0, 900), rng.uniform(0, 900))
            )
    return make_gt(annotations, image_ids=range(1, images + 1)), results


@pytest.mark.parametrize("seed", range(5))
def test_correct_predictions_are_exactly_the_true_positives_of_the_metrics(seed):
    gt, results = random_scene(seed)
    for threshold in (0.1, 0.5, 0.8):
        labels = label_errors(gt, results, threshold)
        counts = match_counts(gt, results, threshold)
        true_positives = sum(tp for tp, _, _ in counts.values())
        false_positives = sum(fp for _, fp, _ in counts.values())
        false_negatives = sum(fn for _, _, fn in counts.values())
        assert labels.prediction_counts()[PredictionOutcome.CORRECT] == true_positives
        assert len(labels.predictions) - true_positives == false_positives
        assert labels.truth_counts()[TruthOutcome.DETECTED] == true_positives
        assert len(labels.truths) - true_positives == false_negatives


@pytest.mark.parametrize("seed", range(5))
def test_every_box_gets_exactly_one_outcome(seed):
    gt, results = random_scene(seed)
    labels = label_errors(gt, results, 0.3)
    assert sorted(labels.truths["annotation_index"]) == list(range(len(gt["annotations"])))
    kept = [i for i, r in enumerate(results) if r["score"] >= 0.3]
    assert sorted(labels.predictions["result_index"]) == kept
    assert sum(labels.prediction_counts().values()) == len(kept)
    assert sum(labels.truth_counts().values()) == len(gt["annotations"])


def test_the_confusion_matrix_pairs_boxes_regardless_of_class():
    gt = make_gt([truth(1, 1, SEDAN, 100, 100), truth(2, 1, CAR, 300, 300)])
    results = [
        predicted(1, CAR, 100, 100),  # sedan found, called car
        predicted(1, CAR, 700, 700),  # nothing there
    ]
    matrix = confusion_matrix(gt, results, 0.5)
    assert matrix.loc["sedan", "car"] == 1
    assert matrix.loc["car", NO_BOX] == 1
    assert matrix.loc[NO_BOX, "car"] == 1
    assert int(matrix.to_numpy().sum()) == 3


@pytest.mark.parametrize("seed", range(5))
def test_confusion_rows_add_up_to_the_vehicles_and_columns_to_the_predictions(seed):
    gt, results = random_scene(seed)
    matrix = confusion_matrix(gt, results, 0.4)
    names = {c["id"]: c["name"] for c in CATEGORIES}
    for category_id, name in names.items():
        vehicles = sum(a["category_id"] == category_id for a in gt["annotations"])
        boxes = sum(r["category_id"] == category_id and r["score"] >= 0.4 for r in results)
        assert matrix.loc[name].sum() == vehicles
        assert matrix[name].sum() == boxes
    assert matrix.loc[NO_BOX, NO_BOX] == 0


def test_localisation_summary_separates_finding_from_naming():
    gt = make_gt([truth(1, 1, SEDAN, 100, 100), truth(2, 1, CAR, 300, 300)])
    results = [predicted(1, CAR, 100, 100), predicted(1, CAR, 300, 300)]
    summary = localisation_summary(confusion_matrix(gt, results, 0.5))
    assert summary.recall == pytest.approx(1.0)
    assert summary.precision == pytest.approx(1.0)
    assert summary.class_accuracy == pytest.approx(0.5)


def test_agnostic_matches_point_from_annotation_to_result():
    gt = make_gt([truth(1, 1, SEDAN, 100, 100), truth(2, 1, CAR, 300, 300)])
    results = [predicted(1, CAR, 700, 700), predicted(1, TRUCK, 300, 300)]
    assert agnostic_matches(gt, results, 0.5) == {1: 1}


def test_class_agnostic_collapses_every_class_into_one():
    gt = make_gt([truth(1, 1, SEDAN, 100, 100)])
    collapsed_gt, collapsed = class_agnostic(gt, [predicted(1, CAR, 100, 100)])
    assert [c["id"] for c in collapsed_gt["categories"]] == [1]
    assert collapsed_gt["annotations"][0]["category_id"] == 1
    assert collapsed[0]["category_id"] == 1
    assert gt["annotations"][0]["category_id"] == SEDAN


def test_a_category_called_background_is_refused():
    gt = make_gt([])
    gt["categories"] = [{"id": 1, "name": NO_BOX}]
    with pytest.raises(ValueError):
        confusion_matrix(gt, [], 0.5)
