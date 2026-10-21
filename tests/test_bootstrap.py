import numpy as np
import pytest

from vehicle_detection.experiments.bootstrap import (
    AveragePrecision,
    bootstrap,
    count_statistics,
    metric_values,
    resample_weights,
)
from vehicle_detection.experiments.metrics import detection_metrics, match_metrics

CATEGORIES = [{"id": 1, "name": "car"}, {"id": 2, "name": "truck"}]


def make_gt(boxes_by_image):
    images, annotations = [], []
    for image_id, boxes in boxes_by_image.items():
        images.append({"id": image_id, "file_name": f"{image_id}.jpg", "width": 512, "height": 512})
        for category_id, x, y, w, h in boxes:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
    return {"images": images, "annotations": annotations, "categories": CATEGORIES}


def detection(image_id, category_id, x, y, w, h, score=0.9):
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, w, h],
        "score": score,
    }


@pytest.fixture
def scene():
    boxes = {image_id: [(1, 10, 10, 50, 50), (2, 200, 200, 80, 80)] for image_id in range(1, 11)}
    gt = make_gt(boxes)
    perfect = [detection(i, c, x, y, w, h) for i, b in boxes.items() for c, x, y, w, h in b]
    half = [d for d in perfect if d["image_id"] % 2 == 0]
    return gt, perfect, half


def test_average_precision_matches_pycocotools_on_the_full_sample(scene):
    gt, perfect, half = scene
    ones = np.ones(len(gt["images"]), dtype=np.int64)
    assert AveragePrecision(gt, perfect)(ones) == pytest.approx(detection_metrics(gt, perfect).ap)
    assert AveragePrecision(gt, half)(ones) == pytest.approx(detection_metrics(gt, half).ap)


def test_average_precision_with_noisy_scores_matches_pycocotools():
    boxes = {i: [(1, 10 * i, 10, 40, 40)] for i in range(1, 9)}
    gt = make_gt(boxes)
    results = [detection(i, 1, 10 * i, 10, 40, 40, score=0.1 * i) for i in range(1, 9)]
    results += [detection(i, 1, 300, 300, 30, 30, score=0.05 * i) for i in range(1, 9)]
    ones = np.ones(len(gt["images"]), dtype=np.int64)
    assert AveragePrecision(gt, results)(ones) == pytest.approx(detection_metrics(gt, results).ap)


def test_a_zero_weight_removes_the_image_from_the_average_precision(scene):
    gt, _, half = scene
    weights = np.ones(len(gt["images"]), dtype=np.int64)
    weights[0] = 0
    reduced = make_gt({i: [(1, 10, 10, 50, 50), (2, 200, 200, 80, 80)] for i in range(2, 11)})
    kept = [d for d in half if d["image_id"] != 1]
    assert AveragePrecision(gt, half)(weights) == pytest.approx(detection_metrics(reduced, kept).ap)


def test_doubling_every_weight_keeps_the_average_precision(scene):
    gt, _, half = scene
    ap = AveragePrecision(gt, half)
    ones = np.ones(len(gt["images"]), dtype=np.int64)
    assert ap(2 * ones) == pytest.approx(ap(ones))


def test_count_statistics_give_the_same_f1_and_mae_as_the_metrics(scene):
    gt, _, half = scene
    image_ids = [image["id"] for image in gt["images"]]
    counts = count_statistics(gt, half, 0.5, image_ids)
    values = metric_values(AveragePrecision(gt, half), counts, np.ones(len(image_ids), dtype=int))
    assert values["f1"] == pytest.approx(match_metrics(gt, half, 0.5).f1)
    assert values["mae"] == pytest.approx(1.0)


def test_resample_weights_by_image_always_sum_to_the_sample_size():
    weights = resample_weights(12, 50, seed=3)
    assert weights.shape == (50, 12)
    assert (weights.sum(axis=1) == 12).all()


def test_resample_weights_by_scene_keep_the_images_of_a_scene_together():
    groups = ["a", "a", "b", "b", "b", "c"]
    weights = resample_weights(6, 40, seed=1, groups=groups)
    assert (weights[:, 0] == weights[:, 1]).all()
    assert (weights[:, 2] == weights[:, 3]).all()
    assert (weights[:, 3] == weights[:, 4]).all()


def test_resampling_is_reproducible_for_a_seed():
    first = resample_weights(10, 20, seed=7)
    assert (first == resample_weights(10, 20, seed=7)).all()
    assert not (first == resample_weights(10, 20, seed=8)).all()


def test_a_clearly_worse_variant_has_a_difference_that_excludes_zero(scene):
    gt, perfect, half = scene
    intervals, differences = bootstrap(
        gt,
        {"perfect": perfect, "half": half},
        {"perfect": 0.5, "half": 0.5},
        reference="perfect",
        resamples=60,
    )
    row = differences[(differences["variant"] == "half") & (differences["metric"] == "f1")].iloc[0]
    assert row["difference"] < 0
    assert row["excludes_zero"]
    estimate = intervals[(intervals["variant"] == "perfect") & (intervals["metric"] == "ap")]
    assert estimate.iloc[0]["estimate"] == pytest.approx(1.0)
    assert estimate.iloc[0]["low"] <= estimate.iloc[0]["estimate"] <= estimate.iloc[0]["high"]


def test_the_reference_variant_must_exist(scene):
    gt, perfect, _ = scene
    with pytest.raises(ValueError, match="riferimento"):
        bootstrap(gt, {"perfect": perfect}, {"perfect": 0.5}, reference="missing", resamples=5)


def test_resamples_must_be_positive(scene):
    gt, perfect, _ = scene
    with pytest.raises(ValueError, match="resamples"):
        bootstrap(gt, {"perfect": perfect}, {"perfect": 0.5}, reference="perfect", resamples=0)


def test_groups_must_cover_every_image(scene):
    gt, perfect, _ = scene
    with pytest.raises(ValueError, match="gruppo"):
        bootstrap(
            gt,
            {"perfect": perfect},
            {"perfect": 0.5},
            reference="perfect",
            resamples=5,
            groups=["a"],
        )
