from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sahi.slicing import slice_coco

from vehicle_detection.data.coco_split import load_coco, split_coco_by_manifest
from vehicle_detection.data.manifest import VALID_SPLITS, ManifestError, load_manifest
from vehicle_detection.data.tiling_config import TilingConfig
from vehicle_detection.messages import preview

TILE_NAME = re.compile(
    r"^(?P<stem>.+)_(?P<index>\d+)_(?P<x_min>\d+)_(?P<y_min>\d+)"
    r"_(?P<x_max>\d+)_(?P<y_max>\d+)\.(?:png|jpg)$"
)
ANNOTATION_FILE = "annotations"
STAGING_PREFIX = ".staging-"
BOX_VALUES = 4


@dataclass(frozen=True, slots=True)
class TileOrigin:
    stem: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass(frozen=True, slots=True)
class SplitReport:
    source_images: int
    tiles: int
    annotations: int
    empty_tiles: int


@dataclass(frozen=True, slots=True)
class TileReport:
    splits: Mapping[str, SplitReport]


def parse_tile_name(file_name: str) -> TileOrigin:
    match = TILE_NAME.match(file_name)
    if match is None:
        raise ValueError(f"nome tile non riconosciuto: {file_name}")
    return TileOrigin(
        stem=match["stem"],
        x_min=int(match["x_min"]),
        y_min=int(match["y_min"]),
        x_max=int(match["x_max"]),
        y_max=int(match["y_max"]),
    )


def build_tile_dataset(
    coco_path: str | Path,
    images_dir: str | Path,
    manifest_path: str | Path,
    out_dir: str | Path,
    config: TilingConfig,
    overwrite: bool = False,
) -> TileReport:
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"cartella immagini non trovata: {images_dir}")
    manifest = load_manifest(manifest_path)
    subsets = split_coco_by_manifest(load_coco(coco_path), manifest)
    _check_sources(images_dir, subsets)
    _check_boxes(subsets)
    _check_destinations(out_dir, subsets, overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=out_dir))
    try:
        reports = {
            split: _slice_split(subset, images_dir, staging / split, config)
            for split, subset in sorted(subsets.items())
        }
        _publish(staging, out_dir, reports)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return TileReport(splits=reports)


def _check_sources(images_dir: Path, subsets: Mapping[str, Any]) -> None:
    missing = sorted(
        image["file_name"]
        for subset in subsets.values()
        for image in subset["images"]
        if not (images_dir / image["file_name"]).is_file()
    )
    if missing:
        raise FileNotFoundError(f"immagini non trovate in {images_dir}: {preview(missing)}")


def _check_boxes(subsets: Mapping[str, Any]) -> None:
    degenerate = sorted(
        annotation["id"]
        for subset in subsets.values()
        for annotation in subset["annotations"]
        if len(annotation["bbox"]) != BOX_VALUES
        or annotation["bbox"][2] <= 0
        or annotation["bbox"][3] <= 0
    )
    if degenerate:
        raise ValueError(
            f"annotazioni con box degenerato: {preview([f'id {i}' for i in degenerate])} "
            "(prepara il COCO con normalise-coco)"
        )


def _publish(staging: Path, out_dir: Path, reports: Mapping[str, SplitReport]) -> None:
    for split in sorted(VALID_SPLITS - reports.keys()):
        if (out_dir / split).is_dir():
            shutil.rmtree(out_dir / split)
    for split in reports:
        target = out_dir / split
        if target.exists():
            shutil.rmtree(target)
        (staging / split).rename(target)


def _check_destinations(out_dir: Path, subsets: Mapping[str, Any], overwrite: bool) -> None:
    populated = [
        out_dir / split
        for split in sorted(VALID_SPLITS | subsets.keys())
        if (out_dir / split).is_dir() and any((out_dir / split).iterdir())
    ]
    if populated and not overwrite:
        names = ", ".join(str(path) for path in populated)
        raise FileExistsError(f"cartelle tile già popolate: {names} (usa overwrite)")


def _slice_split(
    subset: Mapping[str, Any], images_dir: Path, split_dir: Path, config: TilingConfig
) -> SplitReport:
    split_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "source.json"
        source.write_text(json.dumps(subset))
        tiled, _ = slice_coco(
            coco_annotation_file_path=str(source),
            image_dir=str(images_dir),
            output_coco_annotation_file_name=ANNOTATION_FILE,
            output_dir=str(split_dir),
            ignore_negative_samples=config.ignore_negative_samples,
            slice_height=config.slice_size,
            slice_width=config.slice_size,
            overlap_height_ratio=config.overlap_ratio,
            overlap_width_ratio=config.overlap_ratio,
            min_area_ratio=config.min_area_ratio,
            out_ext=config.image_extension,
            verbose=False,
        )
    expected = {Path(image["file_name"]).stem for image in subset["images"]}
    found = {parse_tile_name(image["file_name"]).stem for image in tiled["images"]}
    foreign = sorted(found - expected)
    if foreign:
        raise ManifestError(f"tile fuori dallo split {split_dir.name}: {', '.join(foreign)}")
    annotated = {annotation["image_id"] for annotation in tiled["annotations"]}
    return SplitReport(
        source_images=len(subset["images"]),
        tiles=len(tiled["images"]),
        annotations=len(tiled["annotations"]),
        empty_tiles=len(tiled["images"]) - len(annotated),
    )
