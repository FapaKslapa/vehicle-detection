from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vehicle_detection.data.coco_split import load_coco

LOCATION_COUNT = 2
KMEANS_ITERATIONS = 25
PREVIEW_LIMIT = 5


class ProvisionalConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    link_threshold: float = Field(default=8.0, gt=0)
    thumb_size: int = Field(default=48, ge=8, le=256)
    train: float = Field(default=0.7, gt=0, lt=1)
    val: float = Field(default=0.15, gt=0, lt=1)
    seed: int = 0

    @model_validator(mode="after")
    def _ratios_leave_room_for_test(self) -> ProvisionalConfig:
        if self.train + self.val >= 1:
            raise ValueError("train + val deve essere inferiore a 1")
        return self


def detect_sessions(
    images_dir: Path, file_names: Sequence[str], config: ProvisionalConfig
) -> list[str]:
    thumbnails = _thumbnails(Path(images_dir), file_names, config.thumb_size)
    return _session_ids(thumbnails, config.link_threshold)


def build_provisional_manifest(
    coco_path: str | Path,
    images_dir: str | Path,
    out_path: str | Path,
    config: ProvisionalConfig,
) -> pd.DataFrame:
    images_dir = Path(images_dir)
    file_names = sorted(image["file_name"] for image in load_coco(coco_path)["images"])
    thumbnails = _thumbnails(images_dir, file_names, config.thumb_size)
    sessions = _session_ids(thumbnails, config.link_threshold)
    locations = _locations(thumbnails, sessions, config.seed)
    splits = _assign_splits(sessions, config)
    frame = pd.DataFrame(
        {
            "image_id": [Path(name).stem for name in file_names],
            "session_id": sessions,
            "location": [locations[session] for session in sessions],
            "split": [splits[session] for session in sessions],
        }
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def _thumbnails(images_dir: Path, file_names: Sequence[str], size: int) -> np.ndarray:
    missing = sorted(name for name in file_names if not (images_dir / name).is_file())
    if missing:
        shown = ", ".join(missing[:PREVIEW_LIMIT])
        hidden = len(missing) - PREVIEW_LIMIT
        suffix = f" e altri {hidden}" if hidden > 0 else ""
        raise FileNotFoundError(f"immagini non trovate in {images_dir}: {shown}{suffix}")
    rows = []
    for name in file_names:
        with Image.open(images_dir / name) as image:
            gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        rows.append(np.asarray(gray, dtype=np.float32).ravel())
    return np.stack(rows) if rows else np.empty((0, size * size), dtype=np.float32)


def _session_ids(thumbnails: np.ndarray, threshold: float) -> list[str]:
    count = len(thumbnails)
    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for index in range(count - 1):
        distances = np.median(np.abs(thumbnails[index + 1 :] - thumbnails[index]), axis=1)
        for offset in np.flatnonzero(distances < threshold):
            low, high = find(index), find(index + 1 + int(offset))
            if low != high:
                parent[max(low, high)] = min(low, high)
    roots = [find(index) for index in range(count)]
    labels = {root: f"s{rank:04d}" for rank, root in enumerate(sorted(set(roots)))}
    return [labels[root] for root in roots]


def _locations(thumbnails: np.ndarray, sessions: Sequence[str], seed: int) -> dict[str, str]:
    session_ids = sorted(set(sessions))
    index_of = {session: position for position, session in enumerate(session_ids)}
    members = np.array([index_of[session] for session in sessions])
    means = np.stack(
        [thumbnails[members == position].mean(axis=0) for position in range(len(session_ids))]
    )
    if len(session_ids) < LOCATION_COUNT:
        return {session: "loc0" for session in session_ids}
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(means)))
    second = int(np.argmax(np.linalg.norm(means - means[first], axis=1)))
    centers = means[[first, second]].copy()
    assignment = np.zeros(len(means), dtype=int)
    for _ in range(KMEANS_ITERATIONS):
        distances = np.linalg.norm(means[:, None, :] - centers[None, :, :], axis=2)
        assignment = distances.argmin(axis=1)
        for cluster in range(LOCATION_COUNT):
            if (assignment == cluster).any():
                centers[cluster] = means[assignment == cluster].mean(axis=0)
    return {session: f"loc{assignment[index_of[session]]}" for session in session_ids}


def _assign_splits(sessions: Sequence[str], config: ProvisionalConfig) -> dict[str, str]:
    sizes = pd.Series(sessions).value_counts()
    order = list(sizes.index)
    np.random.default_rng(config.seed).shuffle(order)
    order.sort(key=lambda session: -sizes[session])
    total = len(sessions)
    targets = {
        "train": config.train * total,
        "val": config.val * total,
        "test": (1 - config.train - config.val) * total,
    }
    filled = dict.fromkeys(targets, 0)
    assignment: dict[str, str] = {}
    for session in order:
        split = max(targets, key=lambda name: (targets[name] - filled[name]) / targets[name])
        assignment[session] = split
        filled[split] += int(sizes[session])
    return assignment


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manifest provvisorio per sessioni ricostruite")
    parser.add_argument("--coco", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", default="splits/provisional.csv")
    parser.add_argument("--link-threshold", type=float, default=ProvisionalConfig().link_threshold)
    parser.add_argument("--thumb-size", type=int, default=ProvisionalConfig().thumb_size)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    config = ProvisionalConfig(
        link_threshold=args.link_threshold, thumb_size=args.thumb_size, seed=args.seed
    )
    frame = build_provisional_manifest(args.coco, args.images, args.out, config)
    summary = frame.groupby("split").agg(
        immagini=("image_id", "size"), sessioni=("session_id", "nunique")
    )
    print(summary.to_string())
    print(f"scritto {args.out} (provvisorio, solo per prove tecniche)")


if __name__ == "__main__":
    main()
