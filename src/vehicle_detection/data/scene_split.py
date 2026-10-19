from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vehicle_detection.data.coco_split import load_coco
from vehicle_detection.data.manifest import assert_disjoint_sessions
from vehicle_detection.data.scene_embedding import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SIZE,
    DinoV2Embedder,
    Embedder,
    embed_with_rotations,
)
from vehicle_detection.data.scenes import (
    SPLITS,
    LeakageReport,
    SceneConfig,
    assign_locations,
    assign_splits,
    cluster_scenes,
    leakage_report,
    random_split_crossing_share,
    rotation_invariant_similarity,
)
from vehicle_detection.device import best_device

DEFAULT_ANGLES = tuple(range(30, 360, 30))
CURVE_THRESHOLDS = (0.85, 0.9, 0.95, 0.97)


@dataclass(frozen=True, slots=True)
class SceneSplitReport:
    groups: int
    group_sizes: list[int]
    split_images: dict[str, int]
    split_groups: dict[str, int]
    location_share_per_split: dict[str, dict[str, float]]
    class_share_per_split: dict[str, dict[str, float]]
    leakage: LeakageReport
    leakage_curve: dict[str, float]
    random_baseline_share: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_scene_split(
    coco_path: str | Path,
    images_dir: str | Path,
    out_path: str | Path,
    embedder: Embedder,
    config: SceneConfig,
    angles: Sequence[int] = DEFAULT_ANGLES,
    size: int = DEFAULT_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_path: str | Path | None = None,
) -> SceneSplitReport:
    coco = load_coco(coco_path)
    images = sorted(coco["images"], key=lambda image: image["file_name"])
    file_names = [image["file_name"] for image in images]
    base, rotated = _embeddings(
        Path(images_dir), file_names, embedder, tuple(angles), size, batch_size, cache_path
    )
    similarity = rotation_invariant_similarity(base, rotated)
    labels = cluster_scenes(similarity, config)
    locations = assign_locations(base, labels, config.seed)
    categories = sorted(category["id"] for category in coco["categories"])
    counts = _class_counts(images, coco["annotations"], categories)
    assignment = assign_splits(labels, locations, counts, config)
    splits = np.array([assignment[int(label)] for label in labels])
    manifest = pd.DataFrame(
        {
            "image_id": [Path(name).stem for name in file_names],
            "session_id": [f"scene_{int(label):03d}" for label in labels],
            "location": [locations[int(label)] for label in labels],
            "split": splits,
        }
    )
    assert_disjoint_sessions(manifest)
    report = _report(labels, splits, manifest, counts, categories, similarity, config)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    out_path.with_suffix(".report.json").write_text(json.dumps(report.to_dict(), indent=2))
    return report


def _embeddings(
    images_dir: Path,
    file_names: Sequence[str],
    embedder: Embedder,
    angles: tuple[int, ...],
    size: int,
    batch_size: int,
    cache_path: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    key = hashlib.sha256(json.dumps([list(file_names), list(angles), size]).encode()).hexdigest()
    if cache_path is not None and Path(cache_path).is_file():
        with np.load(cache_path) as cached:
            if str(cached["key"]) == key:
                return cached["base"], cached["rotated"]
    base, rotated = embed_with_rotations(images_dir, file_names, embedder, angles, size, batch_size)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, base=base, rotated=rotated, key=np.array(key))
    return base, rotated


def _class_counts(
    images: Sequence[dict[str, Any]], annotations: Sequence[dict[str, Any]], categories: list[int]
) -> np.ndarray:
    row_of = {image["id"]: index for index, image in enumerate(images)}
    column_of = {category: index for index, category in enumerate(categories)}
    counts = np.zeros((len(images), len(categories)))
    for annotation in annotations:
        if annotation["image_id"] in row_of:
            counts[row_of[annotation["image_id"]], column_of[annotation["category_id"]]] += 1
    return counts


def _report(
    labels: np.ndarray,
    splits: np.ndarray,
    manifest: pd.DataFrame,
    counts: np.ndarray,
    categories: list[int],
    similarity: np.ndarray,
    config: SceneConfig,
) -> SceneSplitReport:
    sizes = sorted((int(size) for size in np.bincount(labels)), reverse=True)
    totals = counts.sum(axis=0)
    class_share = {
        split: {
            str(category): float(counts[splits == split, index].sum() / totals[index])
            for index, category in enumerate(categories)
            if totals[index] > 0
        }
        for split in SPLITS
    }
    location_share = {
        split: manifest[manifest.split == split].location.value_counts(normalize=True).to_dict()
        for split in SPLITS
    }
    return SceneSplitReport(
        groups=len(sizes),
        group_sizes=sizes,
        split_images={split: int((splits == split).sum()) for split in SPLITS},
        split_groups={
            split: int(manifest[manifest.split == split].session_id.nunique()) for split in SPLITS
        },
        location_share_per_split=location_share,
        class_share_per_split=class_share,
        leakage=leakage_report(similarity, splits, config.near_duplicate_similarity),
        leakage_curve={
            str(threshold): leakage_report(similarity, splits, threshold).crossing_share
            for threshold in CURVE_THRESHOLDS
        },
        random_baseline_share=random_split_crossing_share(config.train, config.val),
    )


def main(argv: Sequence[str] | None = None, embedder: Embedder | None = None) -> None:
    parser = argparse.ArgumentParser(description="Split per scena con controllo del leakage")
    parser.add_argument("--coco", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", default="splits/v1_candidate.csv")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--angles", type=int, nargs="+", default=list(DEFAULT_ANGLES))
    parser.add_argument("--neighbors", type=int, default=SceneConfig().neighbors)
    parser.add_argument("--resolution", type=float, default=SceneConfig().resolution)
    parser.add_argument("--min-group-size", type=int, default=SceneConfig().min_group_size)
    parser.add_argument("--max-group-share", type=float, default=SceneConfig().max_group_share)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    config = SceneConfig(
        neighbors=args.neighbors,
        resolution=args.resolution,
        min_group_size=args.min_group_size,
        max_group_share=args.max_group_share,
        seed=args.seed,
    )
    report = build_scene_split(
        args.coco,
        args.images,
        args.out,
        embedder or DinoV2Embedder(best_device()),
        config,
        angles=args.angles,
        size=args.size,
        batch_size=args.batch_size,
        cache_path=args.cache,
    )
    print(
        f"{report.groups} gruppi, immagini per split {report.split_images}, "
        f"coppie quasi identiche tra split diversi: {report.leakage.crossing_share:.1%} "
        f"(split casuale: {report.random_baseline_share:.1%})"
    )
    print(f"scritto {args.out} (candidato, da verificare prima di congelarlo come v1)")


if __name__ == "__main__":
    main()
