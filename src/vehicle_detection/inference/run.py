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
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from vehicle_detection.data.coco_split import load_coco, split_coco_by_manifest
from vehicle_detection.data.manifest import load_manifest
from vehicle_detection.data.tiling_config import TilingConfig
from vehicle_detection.device import best_device
from vehicle_detection.inference.adaptive import AdaptiveConfig, AdaptivePredictor, ImagePredictor
from vehicle_detection.inference.benchmark import CostReport, run_measured
from vehicle_detection.inference.detections import to_coco_results
from vehicle_detection.inference.models import (
    FIRST_LABEL,
    FullImagePredictor,
    RemappedPredictor,
    build_detection_model,
    label_space,
)
from vehicle_detection.inference.sliced import PostprocessConfig, SlicedPredictor
from vehicle_detection.messages import preview

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
MODES = ("full", "sliced", "adaptive")
PROVISIONAL_PREFIX = "provisional"
PROVISIONAL_WARNING = "ATTENZIONE: split provvisorio, i risultati non sono validi per il report"


@dataclass(frozen=True, slots=True)
class PredictionRun:
    results: list[dict[str, Any]]
    costs: CostReport

    @property
    def image_count(self) -> int:
        return self.costs.images


def predict_split(
    coco_path: str | Path,
    images_dir: str | Path,
    manifest_path: str | Path,
    split: str,
    predictor: ImagePredictor,
    device: torch.device | None = None,
    limit: int | None = None,
    warmup: int = 1,
) -> PredictionRun:
    images_dir = Path(images_dir)
    images = select_split_subset(coco_path, manifest_path, split, limit)["images"]
    _check_sources(images_dir, images)
    per_image, costs = run_measured(
        predictor.predict,
        lambda image: _load_rgb(images_dir / image["file_name"]),
        images,
        device or torch.device("cpu"),
        warmup,
    )
    results = [
        result
        for image, detections in zip(images, per_image, strict=True)
        for result in to_coco_results(image["id"], detections)
    ]
    return PredictionRun(results=results, costs=costs)


def select_split_subset(
    coco_path: str | Path, manifest_path: str | Path, split: str, limit: int | None = None
) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError(f"limit deve essere positivo: {limit}")
    subsets = split_coco_by_manifest(load_coco(coco_path), load_manifest(manifest_path))
    if split not in subsets:
        raise ValueError(
            f"split non presente nel manifest: {split} (presenti: {', '.join(sorted(subsets))})"
        )
    subset = subsets[split]
    images = sorted(subset["images"], key=lambda image: image["file_name"])[:limit]
    kept = {image["id"] for image in images}
    return {
        **subset,
        "images": images,
        "annotations": [a for a in subset["annotations"] if a["image_id"] in kept],
    }


def describe_run(cfg: DictConfig) -> dict[str, Any]:
    manifest = Path(cfg.manifest_path)
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "provisional": manifest.name.startswith(PROVISIONAL_PREFIX),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


def write_run(
    run: PredictionRun, out_dir: str | Path, provenance: Mapping[str, Any] | None = None
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if provenance is not None:
        (out_dir / "run.json").write_text(json.dumps(provenance, indent=2))
    (out_dir / "results.json").write_text(json.dumps(run.results))
    (out_dir / "cost.json").write_text(json.dumps(asdict(run.costs)))


def build_predictor(
    cfg: DictConfig,
    categories: Sequence[dict[str, Any]],
    model_factory: Callable[..., Any] = build_detection_model,
) -> ImagePredictor:
    if cfg.mode not in MODES:
        raise ValueError(f"modalità non supportata: {cfg.mode} (valide: {', '.join(MODES)})")
    if cfg.kind not in FIRST_LABEL:
        valid = ", ".join(FIRST_LABEL)
        raise ValueError(f"tipo di modello non supportato: {cfg.kind} (validi: {valid})")
    space = label_space(categories, FIRST_LABEL[cfg.kind])
    tiling = TilingConfig.model_validate(OmegaConf.to_container(cfg.tiling, resolve=True))
    postprocess = PostprocessConfig.model_validate(
        OmegaConf.to_container(cfg.postprocess, resolve=True)
    )
    model = model_factory(
        cfg.kind,
        cfg.weights,
        space,
        postprocess.confidence_threshold,
        _device(cfg).type if cfg.device == "auto" else cfg.device,
        torchvision_name=cfg.torchvision_name,
        input_size=cfg.input_size,
    )
    full = FullImagePredictor(model, postprocess.max_detections)
    sliced = SlicedPredictor(model, tiling, postprocess)
    if cfg.mode == "full":
        base: ImagePredictor = full
    elif cfg.mode == "sliced":
        base = sliced
    else:
        adaptive = AdaptiveConfig.model_validate(OmegaConf.to_container(cfg.adaptive, resolve=True))
        base = AdaptivePredictor(full.predict, sliced, adaptive, postprocess.max_detections)
    return RemappedPredictor(base, space.label_to_category)


def run(
    cfg: DictConfig, model_factory: Callable[..., Any] = build_detection_model
) -> PredictionRun:
    categories = load_coco(cfg.coco_path)["categories"]
    select_split_subset(cfg.coco_path, cfg.manifest_path, cfg.split, cfg.limit)
    predictor = build_predictor(cfg, categories, model_factory)
    outcome = predict_split(
        cfg.coco_path,
        cfg.images_dir,
        cfg.manifest_path,
        cfg.split,
        predictor,
        device=_device(cfg),
        limit=cfg.limit,
        warmup=cfg.warmup,
    )
    write_run(outcome, cfg.out_dir, describe_run(cfg))
    return outcome


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="predict")
def main(cfg: DictConfig) -> None:
    if describe_run(cfg)["provisional"]:
        print(PROVISIONAL_WARNING, file=sys.stderr)
    outcome = run(cfg)
    print(
        f"{cfg.mode} su {cfg.split}: {outcome.image_count} immagini, "
        f"{len(outcome.results)} detection, {outcome.costs.mean_seconds:.3f} s/img "
        f"(p95 {outcome.costs.p95_seconds:.3f} s)"
    )


def _device(cfg: DictConfig) -> torch.device:
    return best_device() if cfg.device == "auto" else torch.device(cfg.device)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _check_sources(images_dir: Path, images: Sequence[dict[str, Any]]) -> None:
    missing = sorted(
        image["file_name"] for image in images if not (images_dir / image["file_name"]).is_file()
    )
    if missing:
        raise FileNotFoundError(f"immagini non trovate in {images_dir}: {preview(missing)}")


if __name__ == "__main__":
    main()
