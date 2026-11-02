import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

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


def test_training_updates_parameters_and_rejects_nonfinite_loss(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, images, targets):
            assert self.training
            assert targets[0]["boxes"].shape == (1, 4)
            return {"loss": self.parameter.square()}

    model = Model()
    loader = DataLoader(dataset(tmp_path), collate_fn=collate)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assert train_epoch(model, loader, optimizer, torch.device("cpu")) == 1
    assert model.parameter.item() == pytest.approx(0.8)
    with torch.no_grad():
        model.parameter.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        train_epoch(model, loader, optimizer, torch.device("cpu"))


def test_validation_maps_labels_back_to_coco_ids(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, images):
            assert not self.training
            return [
                {
                    "boxes": torch.tensor([[10.0, 5.0, 30.0, 20.0]]),
                    "scores": torch.tensor([0.99]),
                    "labels": torch.tensor([1]),
                }
            ]

    loader = DataLoader(dataset(tmp_path), collate_fn=collate)
    assert validate(Model(), loader, torch.device("cpu")) == pytest.approx(1.0)


def test_workflow_saves_weights_and_resumes(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

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

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.tensor(1.0))
            self.roi_heads = SimpleNamespace(score_thresh=0.05)

        def forward(self, images, targets):
            return {"loss": self.parameter.square()}

    monkeypatch.setattr(workflow, "setup_faster_rcnn", lambda *args, **kwargs: Model())
    monkeypatch.setattr(workflow, "validate", lambda *args: 0.5)
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
                "lr": 0.1,
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
    assert state["parameter"].item() == pytest.approx(0.8)
    checkpoint = best.parent / "checkpoint.pt"
    cfg.training.resume = str(checkpoint)
    cfg.training.output_dir = str(tmp_path / "resumed")
    cfg.training.epochs = 2
    resumed = workflow.run(cfg)
    assert torch.load(resumed, weights_only=True)["parameter"] < state["parameter"]
    saved = torch.load(resumed.parent / "checkpoint.pt", weights_only=True)
    assert saved["epoch"] == 1
    assert saved["provenance"]["config"]["model"]["min_size"] == [64, 96]
