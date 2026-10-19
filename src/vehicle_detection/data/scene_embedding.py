from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image

from vehicle_detection.messages import preview

DEFAULT_SIZE = 224
DEFAULT_BATCH_SIZE = 64
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Embedder(Protocol):
    def __call__(self, images: np.ndarray) -> np.ndarray: ...


def embed_with_rotations(
    images_dir: str | Path,
    file_names: Sequence[str],
    embedder: Embedder,
    angles: Sequence[int],
    size: int = DEFAULT_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    images_dir = Path(images_dir)
    _check_sources(images_dir, file_names)
    base_parts: list[np.ndarray] = []
    rotated_parts: list[np.ndarray] = []
    for start in range(0, len(file_names), batch_size):
        batch = [_load(images_dir / name, size) for name in file_names[start : start + batch_size]]
        base_parts.append(_embed(embedder, batch))
        rotations = [
            _embed(
                embedder,
                [image.rotate(angle, resample=Image.Resampling.BILINEAR) for image in batch],
            )
            for angle in angles
        ]
        rotated_parts.append(np.stack(rotations, axis=1) if rotations else _empty(base_parts[-1]))
    return np.concatenate(base_parts), np.concatenate(rotated_parts)


class DinoV2Embedder:
    def __init__(self, device: torch.device, model_name: str = "dinov2_vits14") -> None:
        self._device = device
        self._model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
        self._mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        tensor = ((tensor - self._mean) / self._std).to(self._device)
        with torch.no_grad():
            return self._model(tensor).cpu().numpy().astype(np.float32)


def _load(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def _embed(embedder: Embedder, images: Sequence[Image.Image]) -> np.ndarray:
    return np.asarray(embedder(np.stack([np.asarray(image) for image in images])))


def _empty(reference: np.ndarray) -> np.ndarray:
    return np.zeros((reference.shape[0], 0, reference.shape[1]), dtype=reference.dtype)


def _check_sources(images_dir: Path, file_names: Sequence[str]) -> None:
    missing = sorted(name for name in file_names if not (images_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"immagini non trovate in {images_dir}: {preview(missing)}")
