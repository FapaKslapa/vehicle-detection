from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from vehicle_detection.data.manifest import ManifestError

REQUIRED_KEYS = ("images", "annotations", "categories")
PREVIEW_LIMIT = 5


def load_coco(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"annotazioni COCO non trovate: {path}")
    coco = json.loads(path.read_text())
    missing = [key for key in REQUIRED_KEYS if key not in coco]
    if missing:
        raise ValueError(f"chiavi COCO mancanti in {path}: {', '.join(missing)}")
    return coco


def split_coco_by_manifest(
    coco: Mapping[str, Any], manifest: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    split_of = dict(zip(manifest["image_id"], manifest["split"], strict=True))
    coco_stems = {Path(image["file_name"]).stem for image in coco["images"]}
    unlisted = sorted(coco_stems - split_of.keys())
    if unlisted:
        raise ManifestError(f"immagini COCO assenti dal manifest: {_preview(unlisted)}")
    unknown = sorted(split_of.keys() - coco_stems)
    if unknown:
        raise ManifestError(f"immagini del manifest assenti dal COCO: {_preview(unknown)}")
    subsets: dict[str, dict[str, Any]] = {}
    split_of_image_id: dict[int, str] = {}
    for image in coco["images"]:
        split = split_of[Path(image["file_name"]).stem]
        subset = subsets.setdefault(
            split, {"images": [], "annotations": [], "categories": coco["categories"]}
        )
        subset["images"].append(image)
        split_of_image_id[image["id"]] = split
    for annotation in coco["annotations"]:
        split = split_of_image_id.get(annotation["image_id"])
        if split is not None:
            subsets[split]["annotations"].append(annotation)
    return subsets


def _preview(items: Sequence[str]) -> str:
    shown = ", ".join(items[:PREVIEW_LIMIT])
    hidden = len(items) - PREVIEW_LIMIT
    return f"{shown} e altri {hidden}" if hidden > 0 else shown
