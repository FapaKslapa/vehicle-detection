import numpy as np
import pytest
import torch

from vehicle_detection.inference.models import (
    FullImagePredictor,
    RemappedPredictor,
    build_detection_model,
    label_space,
)
from vehicle_detection.training import faster_rcnn
from vehicle_detection.training.faster_rcnn import (
    category_labels,
    load_config,
    save_weights,
    set_seed,
    setup_faster_rcnn,
)

CPU = torch.device("cpu")


def build(num_classes=8, **kwargs):
    return setup_faster_rcnn(num_classes, weights=None, weights_backbone=None, device=CPU, **kwargs)


def test_head_has_one_output_per_class_plus_background():
    model = build(num_classes=8)
    assert model.roi_heads.box_predictor.cls_score.out_features == 9
    assert model.roi_heads.box_predictor.bbox_pred.out_features == 9 * 4


@pytest.mark.parametrize("num_classes", [1, 3])
def test_head_follows_the_requested_class_count(num_classes):
    model = build(num_classes=num_classes)
    assert model.roi_heads.box_predictor.cls_score.out_features == num_classes + 1


def test_model_returns_boxes_scores_and_labels_in_eval_mode():
    model = build().eval()
    with torch.no_grad():
        output = model([torch.rand(3, 96, 96)])
    assert set(output[0]) == {"boxes", "scores", "labels"}
    assert output[0]["boxes"].shape[1] == 4


def test_default_device_comes_from_the_device_module(monkeypatch):
    monkeypatch.setattr(faster_rcnn, "best_device", lambda: CPU)
    model = setup_faster_rcnn(2, weights=None, weights_backbone=None)
    assert next(model.parameters()).device == CPU


def test_config_exposes_the_weights_setting():
    config = load_config()
    assert config["model"]["weights"] == "DEFAULT"
    assert "weights_backbone" in config["model"]

def test_training_samples_scales_and_resizes_boxes_without_mutating_targets():
    model = build(min_size=[64, 96], max_size=256).train()
    image = torch.rand(3, 40, 80)
    boxes = torch.tensor([[10.0, 5.0, 30.0, 20.0]])
    target = {"boxes": boxes.clone(), "labels": torch.tensor([1])}
    set_seed(7)
    seen = set()
    for _ in range(20):
        images, targets = model.transform([image], [target])
        height, width = images.image_sizes[0]
        seen.add(height)
        assert width == height * 2
        torch.testing.assert_close(targets[0]["boxes"], boxes * (height / 40))
        assert torch.equal(targets[0]["labels"], target["labels"])
    assert seen == {64, 96}
    torch.testing.assert_close(target["boxes"], boxes)


def test_multiscale_eval_uses_last_size_and_caps_long_edge():
    model = build(min_size=(64, 96), max_size=128).eval()
    for _ in range(3):
        images, _ = model.transform([torch.rand(3, 40, 80)])
        assert images.image_sizes == [(64, 128)]
    images, _ = model.transform([torch.rand(3, 40, 40)])
    assert images.image_sizes == [(96, 96)]


def test_multiscale_sampling_is_repeatable_with_seed():
    model = build(min_size=(64, 96), max_size=256).train()
    image = torch.rand(3, 40, 80)

    def draw():
        set_seed(11)
        return [model.transform([image])[0].image_sizes[0] for _ in range(10)]

    assert draw() == draw()


def test_single_scale_remains_supported():
    model = build(min_size=64, max_size=128)
    assert model.transform.min_size == (64,)


@pytest.mark.parametrize(
    ("min_size", "max_size"),
    [
        ([], 128),
        ([0, 64], 128),
        ([-1], 128),
        ([64.5], 128),
        ([True], 128),
        ([64], 32),
        ([64], 128.5),
    ],
)
def test_invalid_resize_settings_are_rejected(min_size, max_size):
    with pytest.raises(ValueError):
        build(min_size=min_size, max_size=max_size)


CATEGORIES = [{"id": 1, "name": "car"}, {"id": 2, "name": "hatchback"}, {"id": 5, "name": "suv"}]


def head_weights(model):
    return model.roi_heads.box_predictor.cls_score.weight.detach().clone()


def test_same_seed_gives_identical_models():
    first = build(num_classes=3, seed=11)
    second = build(num_classes=3, seed=11)
    assert torch.equal(head_weights(first), head_weights(second))


def test_different_seeds_give_different_models():
    assert not torch.equal(
        head_weights(build(num_classes=3, seed=1)), head_weights(build(num_classes=3, seed=2))
    )


def test_set_seed_makes_torch_and_random_repeatable():
    import random

    def draw():
        set_seed(5)
        return torch.rand(3), random.random()

    first, second = draw(), draw()
    assert torch.equal(first[0], second[0])
    assert first[1] == second[1]


def test_category_labels_follow_the_inference_label_contract():
    labels = category_labels(CATEGORIES)
    assert labels == {1: 1, 2: 2, 5: 3}
    space = label_space(CATEGORIES, first_label=1)
    assert {label: cid for cid, label in labels.items()} == dict(space.label_to_category)


def test_category_labels_sort_by_coco_id_and_never_use_background():
    labels = category_labels(list(reversed(CATEGORIES)))
    assert labels == {1: 1, 2: 2, 5: 3}
    assert 0 not in labels.values()


def test_saved_weights_load_through_the_inference_adapter(tmp_path):
    model = build(num_classes=len(CATEGORIES), seed=0)
    weights = save_weights(model, tmp_path / "nested" / "frcnn.pt")
    assert weights.is_file()
    space = label_space(CATEGORIES, first_label=1)
    detection_model = build_detection_model("torchvision", weights, space, 0.0, "cpu")
    assert detection_model.num_categories == len(CATEGORIES)
    image = np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    predictor = RemappedPredictor(
        FullImagePredictor(detection_model, max_detections=100), space.label_to_category
    )
    detections = predictor.predict(image)
    assert detections
    assert {d.category_id for d in detections} <= {1, 2, 5}
