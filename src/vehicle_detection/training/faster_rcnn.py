from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from vehicle_detection.device import best_device
from vehicle_detection.inference.models import FIRST_LABEL, label_space

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "train" / "faster_rcnn.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def category_labels(categories: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    space = label_space(categories, FIRST_LABEL["torchvision"])
    return {category_id: label for label, category_id in space.label_to_category.items()}


def save_weights(model: nn.Module, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def setup_faster_rcnn(
    num_classes: int,
    weights: str | None = "DEFAULT",
    weights_backbone: str | None = None,
    device: torch.device | None = None,
    seed: int | None = None, 
    min_size: int | Sequence[int] = 800, 
    max_size: int = 1333,
) -> nn.Module:
    sizes = (min_size,) if isinstance(min_size, int) else tuple(min_size)
    if not sizes or any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("min_size must contain positive integers")
    if type(max_size) is not int or max_size < max(sizes):
        raise ValueError("max_size must be a positive integer >= every min_size")
    
    if seed is not None:
        set_seed(seed)
    model = fasterrcnn_resnet50_fpn(weights=weights,
                                    weights_backbone=weights_backbone,
                                    min_size=min_size,
                                    max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model.to(device or best_device())
