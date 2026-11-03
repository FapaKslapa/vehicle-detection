import math

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from vehicle_detection.training.faster_rcnn import setup_faster_rcnn
from vehicle_detection.training.faster_rcnn_train import (
    CocoDataset,
    collate,
    train_epoch,
    validate,
)


def dataset(tmp_path):
    Image.new("RGB", (80, 40)).save(tmp_path / "frame.jpg")
    coco = {
        "images": [{"id": 7, "file_name": "frame.jpg", "width": 80, "height": 40}],
        "categories": [{"id": 5, "name": "car"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 7,
                "category_id": 5,
                "bbox": [10, 5, 20, 15],
                "area": 300,
                "iscrowd": 0,
            }
        ],
    }
    return CocoDataset(coco, tmp_path)


def test_original_image_and_pixel_boxes_are_preserved(tmp_path):
    image, target = dataset(tmp_path)[0]
    assert image.shape == (3, 40, 80)
    assert image.dtype == torch.float32
    assert 0 <= image.min() <= image.max() <= 1
    assert target["labels"].tolist() == [1]
    assert target["boxes"].tolist() == [[10, 5, 30, 20]]


def test_negative_images_have_valid_empty_targets(tmp_path):
    data = dataset(tmp_path)
    data.annotations.clear()
    _, target = data[0]
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)


@pytest.mark.parametrize("short_edge", [64, 96])
def test_real_faster_rcnn_losses_backpropagate_and_update_parameters(tmp_path, short_edge):
    model = setup_faster_rcnn(
        1,
        weights=None,
        weights_backbone=None,
        device=torch.device("cpu"),
        seed=7,
        min_size=short_edge,
        max_size=192,
    ).train()
    image, target = dataset(tmp_path)[0]
    classifier = model.roi_heads.box_predictor.cls_score.weight
    backbone = model.backbone.body.conv1.weight
    before = classifier.detach().clone()
    backbone_before = backbone.detach().clone()
    losses = model([image], [target])
    assert set(losses) == {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
    assert all(torch.isfinite(loss).all() for loss in losses.values())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad(set_to_none=True)
    sum(losses.values()).backward()
    for parameter in (classifier, backbone):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    optimizer.step()
    assert not torch.equal(classifier.detach(), before)
    assert not torch.equal(backbone.detach(), backbone_before)


def test_train_epoch_and_validation_use_real_faster_rcnn(tmp_path):
    model = setup_faster_rcnn(
        1,
        weights=None,
        weights_backbone=None,
        device=torch.device("cpu"),
        seed=7,
        min_size=[64, 96],
        max_size=192,
    )
    loader = DataLoader(dataset(tmp_path), collate_fn=collate)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    before = model.roi_heads.box_predictor.cls_score.weight.detach().clone()
    loss = train_epoch(model, loader, optimizer, torch.device("cpu"))
    assert math.isfinite(loss) and loss > 0
    assert not torch.equal(model.roi_heads.box_predictor.cls_score.weight, before)
    ap = validate(model, loader, torch.device("cpu"))
    assert math.isfinite(ap) and 0 <= ap <= 1


def test_real_faster_rcnn_workflow_saves_weights_and_resumes(tmp_path):
    import json

    from omegaconf import OmegaConf

    from vehicle_detection.training import faster_rcnn_train as workflow

    data = dataset(tmp_path)
    coco = data.coco
    coco["images"].append({"id": 8, "file_name": "val.jpg"})
    coco["annotations"].append({**coco["annotations"][0], "id": 2, "image_id": 8})
    Image.new("RGB", (80, 40)).save(tmp_path / "val.jpg")
    coco_path = tmp_path / "coco.json"
    coco_path.write_text(json.dumps(coco))
    manifest = tmp_path / "split.csv"
    manifest.write_text("image_id,session_id,location,split\nframe,s1,loc,train\nval,s2,loc,val\n")

    cfg = OmegaConf.create(
        {
            "model": {"weights": None, "min_size": [64, 96], "max_size": 128},
            "data": {
                "coco_path": str(coco_path),
                "images_dir": str(tmp_path),
                "manifest_path": str(manifest),
            },
            "training": {
                "epochs": 1,
                "batch_size": 1,
                "workers": 0,
                "lr": 0.001,
                "weight_decay": 0.0,
                "seed": 0,
                "device": "cpu",
                "output_dir": str(tmp_path / "first"),
                "resume": None,
            },
        }
    )
    best = workflow.run(cfg)
    state = torch.load(best, weights_only=True)
    classifier_key = "roi_heads.box_predictor.cls_score.weight"
    assert classifier_key in state
    assert torch.isfinite(state[classifier_key]).all()
    assert (best.parent / "metrics.jsonl").is_file()
    checkpoint = best.parent / "checkpoint.pt"
    cfg.training.resume = str(checkpoint)
    cfg.training.output_dir = str(tmp_path / "resumed")
    cfg.training.epochs = 2
    resumed = workflow.run(cfg)
    resumed_state = torch.load(resumed, weights_only=True)
    assert not torch.equal(resumed_state[classifier_key], state[classifier_key])
    saved = torch.load(resumed.parent / "checkpoint.pt", weights_only=True)
    assert saved["epoch"] == 1
    assert saved["provenance"]["config"]["model"]["min_size"] == [64, 96]
