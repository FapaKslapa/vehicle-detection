from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

RGB_CHANNELS = 3
RGBA_CHANNELS = 4


def image_source(image: str | Path | np.ndarray) -> str | np.ndarray:
    if isinstance(image, np.ndarray):
        return as_rgb(image)
    path = Path(image)
    if not path.is_file():
        raise FileNotFoundError(f"immagine non trovata: {path}")
    return str(path)


def as_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError(_unsupported(image))
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == RGBA_CHANNELS:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.ndim == 3 and image.shape[2] == RGB_CHANNELS:
        return image
    raise ValueError(_unsupported(image))


def _unsupported(image: np.ndarray) -> str:
    return (
        f"array immagine non supportato: shape={image.shape}, dtype={image.dtype} "
        "(attesi uint8 con 1, 3 o 4 canali)"
    )
