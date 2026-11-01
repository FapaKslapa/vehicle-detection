"""Train Faster R-CNN on manifest-separated COCO images without pre-resizing."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from vehicle_detection.data.coco_split import load_coco, split_coco_by_manifest
from vehicle_detection.data.manifest import load_manifest
from vehicle_detection.device import best_device
from vehicle_detection.experiments.metrics import detection_metrics
from vehicle_detection.training.faster_rcnn import (
    CONFIG_PATH,
    category_labels,
    save_weights,
    set_seed,
    setup_faster_rcnn,
)


class CocoDataset(Dataset):
    def __init__(self, coco, images_dir):
        self.coco = coco
        self.images_dir = Path(images_dir).resolve()
        self.labels = category_labels(coco["categories"])
        self.annotations = defaultdict(list)
        for annotation in coco["annotations"]:
            self.annotations[annotation["image_id"]].append(annotation)

    def __len__(self):
        return len(self.coco["images"])

    def __getitem__(self, index):
        record = self.coco["images"][index]
        path = (self.images_dir / record["file_name"]).resolve()
        if not path.is_relative_to(self.images_dir):
            raise ValueError(f"Image escapes images_dir: {path}")
        with Image.open(path) as source:
            image = pil_to_tensor(source.convert("RGB")).float() / 255
        height, width = image.shape[-2:]
        boxes, labels = [], []
        for annotation in self.annotations[record["id"]]:
            if annotation.get("iscrowd", 0):
                continue
            x, y, w, h = annotation["bbox"]
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                raise ValueError(f"Invalid bounding box: {annotation['id']}")
            box = [max(0, x), max(0, y), min(width, x + w), min(height, y + h)]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            boxes.append(box)
            labels.append(self.labels[annotation["category_id"]])
        return image, {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(record["id"], dtype=torch.int64),
        }


def collate(batch):
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    count = 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in t.items()} for t in targets]
        losses = model(images, targets)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {losses}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.detach().item() * len(images)
        count += len(images)
    if not count:
        raise ValueError("Training split is empty")
    return total / count


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    category_of = {label: cid for cid, label in loader.dataset.labels.items()}
    results = []
    for images, targets in loader:
        predictions = model([image.to(device) for image in images])
        for target, prediction in zip(targets, predictions, strict=True):
            for box, score, label in zip(
                prediction["boxes"].cpu().tolist(),
                prediction["scores"].cpu().tolist(),
                prediction["labels"].cpu().tolist(),
                strict=True,
            ):
                x1, y1, x2, y2 = box
                results.append(
                    {
                        "image_id": target["image_id"].item(),
                        "category_id": category_of[label],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )
    return detection_metrics(loader.dataset.coco, results).ap


def run(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    settings = config["training"]
    if settings["epochs"] < 1 or settings["batch_size"] < 1 or settings["workers"] < 0:
        raise ValueError("epochs/batch_size must be positive; workers must be nonnegative")
    if settings["lr"] <= 0 or settings["weight_decay"] < 0:
        raise ValueError("lr must be positive; weight_decay must be nonnegative")
    set_seed(settings["seed"])
    manifest = Path(config["data"]["manifest_path"])
    subsets = split_coco_by_manifest(
        load_coco(config["data"]["coco_path"]), load_manifest(manifest)
    )
    for split in ("train", "val"):
        if split not in subsets or not subsets[split]["images"]:
            raise ValueError(f"Missing or empty {split} split")
    if not any(not a.get("iscrowd", 0) for a in subsets["val"]["annotations"]):
        raise ValueError("Validation requires non-crowd annotations for AP")
    datasets = {
        split: CocoDataset(subsets[split], config["data"]["images_dir"])
        for split in ("train", "val")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=settings["batch_size"],
            shuffle=split == "train",
            num_workers=settings["workers"],
            collate_fn=collate,
        )
        for split, dataset in datasets.items()
    }
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "config": config,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "categories": subsets["train"]["categories"],
        "provisional": manifest.stem.startswith("provisional"),
    }
    (output / "run.json").write_text(json.dumps(provenance, indent=2))
    device = torch.device(settings["device"]) if settings["device"] else best_device()
    model_config = dict(config["model"])
    if settings["resume"]:
        model_config.update(weights=None, weights_backbone=None)
    model = setup_faster_rcnn(len(datasets["train"].labels), device=device, **model_config)
    # Keep the validation prediction floor low for the AP precision-recall curve.
    model.roi_heads.score_thresh = 0.01
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=settings["lr"],
        momentum=0.9,
        weight_decay=settings["weight_decay"],
    )
    start, best = 0, -1.0
    if settings["resume"]:
        checkpoint = torch.load(settings["resume"], map_location="cpu", weights_only=True)
        previous = checkpoint["provenance"]
        if any(previous[k] != provenance[k] for k in ("manifest_sha256", "categories")):
            raise ValueError("Resume dataset differs from the checkpoint")
        if previous["config"]["model"] != config["model"]:
            raise ValueError("Resume model settings differ from the checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["epoch"] + 1
        # Select best weights among the epochs of this new output directory.
    if start >= settings["epochs"]:
        raise ValueError("epochs must exceed the completed checkpoint epoch")
    for epoch in range(start, settings["epochs"]):
        # Epoch-specific seeds also make resumed data order and scale draws repeatable.
        set_seed(settings["seed"] + epoch)
        loss = train_epoch(model, loaders["train"], optimizer, device)
        ap = validate(model, loaders["val"], device)
        if not math.isfinite(ap):
            raise ValueError("Validation AP is undefined")
        if ap > best:
            best = ap
            save_weights(model, output / "best.pt")
        save_weights(model, output / "last.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "provenance": provenance,
            },
            output / "checkpoint.pt",
        )
        row = {"epoch": epoch + 1, "train_loss": loss, "val_ap": ap}
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(row, flush=True)
    return output / "best.pt"


@hydra.main(version_base=None, config_path=str(CONFIG_PATH.parent), config_name="faster_rcnn")
def main(cfg: DictConfig):
    print(run(cfg))


if __name__ == "__main__":
    main()
