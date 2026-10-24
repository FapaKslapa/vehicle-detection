"""Count vehicles in a folder of new, unlabelled images: in total and per class.

This is the counting output of the final system. For every image it runs the detector
(any ``kind`` and ``mode`` that ``inference.run`` supports), removes boxes that overlap a
stronger box of any class (class-agnostic NMS, see ``evaluation.dedupe``), keeps the boxes
at or above the counting threshold, and counts them. It writes to ``out_dir``:

  counts.csv       one row per image: total and one column per class
  detections.json  the counted boxes, with the image list and categories
  run.json         the resolved config, a hash of the weights and the timing

The defaults (``configs/count.yaml``) are the setting chosen on val for E1-full-1024:
threshold 0.35 after class-agnostic NMS at IoU 0.5. Another model needs its own threshold
from ``just dedupe`` on its validation run.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from vehicle_detection.counting.counts import TOTAL, category_slugs, predicted_counts
from vehicle_detection.data.coco_split import load_coco
from vehicle_detection.device import best_device
from vehicle_detection.evaluation.dedupe import suppress_across_classes
from vehicle_detection.inference.adaptive import ImagePredictor
from vehicle_detection.inference.benchmark import CostReport, run_measured
from vehicle_detection.inference.detections import to_coco_results
from vehicle_detection.inference.models import build_detection_model
from vehicle_detection.inference.run import CONFIG_DIR, build_predictor

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
HASH_CHUNK = 2**20


@dataclass(frozen=True, slots=True)
class CountRun:
    images: list[dict[str, Any]]
    detections: list[dict[str, Any]]
    counts: pd.DataFrame
    costs: CostReport

    def totals(self) -> dict[str, int]:
        columns = [
            column for column in self.counts.columns if column not in ("image_id", "file_name")
        ]
        return {column: int(self.counts[column].sum()) for column in columns}


def list_images(images_dir: str | Path, extensions: Sequence[str] = IMAGE_EXTENSIONS) -> list[Path]:
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images folder not found: {images_dir}")
    wanted = {extension.lower() for extension in extensions}
    paths = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in wanted
    )
    if not paths:
        raise FileNotFoundError(f"no images ({', '.join(sorted(wanted))}) in {images_dir}")
    return paths


def count_images(
    paths: Sequence[Path],
    predictor: ImagePredictor,
    categories: Sequence[Mapping[str, Any]],
    threshold: float,
    dedupe_iou: float | None,
    device: torch.device | None = None,
    warmup: int = 1,
) -> CountRun:
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1: {threshold}")
    if dedupe_iou is not None and not 0 < dedupe_iou <= 1:
        raise ValueError(f"dedupe_iou must be in (0, 1] or null: {dedupe_iou}")
    category_slugs(categories)
    sizes: dict[Path, tuple[int, int]] = {}

    def load(path: Path) -> np.ndarray:
        array = load_rgb(path)
        sizes[path] = (int(array.shape[1]), int(array.shape[0]))
        return array

    per_image, costs = run_measured(
        predictor.predict, load, list(paths), device or torch.device("cpu"), warmup
    )
    images = [
        {"id": image_id, "file_name": path.name, "width": sizes[path][0], "height": sizes[path][1]}
        for image_id, path in enumerate(paths, start=1)
    ]
    results = [
        result
        for image, detections in zip(images, per_image, strict=True)
        for result in to_coco_results(image["id"], detections)
    ]
    if dedupe_iou is not None:
        results = suppress_across_classes(results, dedupe_iou)
    counted = [result for result in results if result["score"] >= threshold]
    counts = predicted_counts(images, counted, categories, threshold)
    return CountRun(images=images, detections=counted, counts=counts, costs=costs)


def write_count(
    outcome: CountRun,
    categories: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outcome.counts.to_csv(out_dir / "counts.csv", index=False)
    (out_dir / "detections.json").write_text(
        json.dumps(
            {
                "images": outcome.images,
                "categories": [dict(category) for category in categories],
                "detections": outcome.detections,
            }
        )
    )
    record = {**(provenance or {}), "costs": asdict(outcome.costs), "totals": outcome.totals()}
    (out_dir / "run.json").write_text(json.dumps(record, indent=2))


def describe(cfg: DictConfig) -> dict[str, Any]:
    weights = Path(cfg.weights)
    return {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "weights_sha256": _sha256(weights) if weights.is_file() else None,
    }


def run(cfg: DictConfig, model_factory: Callable[..., Any] = build_detection_model) -> CountRun:
    floor = cfg.postprocess.confidence_threshold
    if cfg.threshold < floor:
        raise ValueError(
            f"threshold {cfg.threshold} is below the detector's confidence floor {floor}: "
            "boxes between the two would never reach the counting step"
        )
    if cfg.dedupe_iou is not None and not 0 < cfg.dedupe_iou <= 1:
        raise ValueError(f"dedupe_iou must be in (0, 1] or null: {cfg.dedupe_iou}")
    paths = list_images(cfg.images_dir, tuple(cfg.extensions))
    categories = load_coco(cfg.categories_from)["categories"]
    predictor = build_predictor(cfg, categories, model_factory)
    device = best_device() if cfg.device == "auto" else torch.device(cfg.device)
    outcome = count_images(
        paths,
        predictor,
        categories,
        cfg.threshold,
        cfg.dedupe_iou,
        device=device,
        warmup=cfg.warmup,
    )
    write_count(outcome, categories, cfg.out_dir, describe(cfg))
    return outcome


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="count")
def main(cfg: DictConfig) -> None:
    outcome = run(cfg)
    totals = outcome.totals()
    images = len(outcome.images)
    per_class = ", ".join(f"{name} {count}" for name, count in totals.items() if name != TOTAL)
    print(
        f"{images} images, {totals[TOTAL]} vehicles ({totals[TOTAL] / images:.2f} per image): "
        f"{per_class}"
    )
    print(
        f"{outcome.costs.mean_seconds:.3f} s/img; counts in {Path(cfg.out_dir) / 'counts.csv'}",
        file=sys.stderr,
    )


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
