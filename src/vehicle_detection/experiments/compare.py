from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from vehicle_detection.data.coco_split import load_coco
from vehicle_detection.device import best_device
from vehicle_detection.evaluation.metrics import (
    CountingMetrics,
    DetectionMetrics,
    MatchMetrics,
    best_threshold,
    counting_metrics,
    detection_metrics,
    match_metrics,
)
from vehicle_detection.inference.adaptive import ImagePredictor
from vehicle_detection.inference.models import build_detection_model
from vehicle_detection.inference.run import (
    PredictionRun,
    build_predictor,
    describe_run,
    predict_split,
    select_split_subset,
)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
DEFAULT_THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))
COLUMNS = (
    "variant",
    "images",
    "detections",
    "ap",
    "ap50",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar_small",
    "ar_medium",
    "ar_large",
    "precision",
    "recall",
    "f1",
    "threshold",
    "threshold_source",
    "mae",
    "rmse",
    "seconds_per_image",
    "p95_seconds",
    "peak_memory_mb",
)
PER_CLASS_COLUMNS = ("variant", "category", "ap", "precision", "recall", "f1", "mae")
NOT_AVAILABLE = "n/a"
PROVISIONAL_BANNER = "SPLIT PROVVISORIO: non valido per il report"


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    predictor: ImagePredictor
    threshold: float | None = None


def compare(
    coco_path: str | Path,
    images_dir: str | Path,
    manifest_path: str | Path,
    split: str,
    variants: Sequence[Variant],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    threshold: float | None = None,
    limit: int | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    if not variants:
        raise ValueError("serve almeno una variante da confrontare")
    if split == "test" and any(
        threshold is None and variant.threshold is None for variant in variants
    ):
        raise ValueError("la soglia non va scelta sul test: passa threshold= scelto sul val")
    names = [variant.name for variant in variants]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"nomi di variante duplicati: {', '.join(duplicated)}")
    truth = select_split_subset(coco_path, manifest_path, split, limit)
    names_by_id = {category["id"]: category["name"] for category in truth["categories"]}
    rows = []
    per_class_rows = []
    predictions: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        outcome = predict_split(
            coco_path,
            images_dir,
            manifest_path,
            split,
            variant.predictor,
            device=device,
            limit=limit,
        )
        predictions[variant.name] = outcome.results
        detection = detection_metrics(truth, outcome.results)
        fixed = variant.threshold if variant.threshold is not None else threshold
        if fixed is None:
            chosen, counting = best_threshold(truth, outcome.results, thresholds)
        else:
            chosen, counting = fixed, counting_metrics(truth, outcome.results, fixed)
        matching = match_metrics(truth, outcome.results, chosen)
        per_class_rows.extend(
            _per_class_rows(variant.name, names_by_id, detection, matching, counting)
        )
        rows.append(
            _variant_row(variant.name, outcome, detection, matching, counting, chosen, fixed)
        )
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    frame.attrs["per_class"] = pd.DataFrame(per_class_rows, columns=list(PER_CLASS_COLUMNS))
    frame.attrs["predictions"] = predictions
    return frame


def _per_class_rows(
    name: str,
    names_by_id: Mapping[int, str],
    detection: DetectionMetrics,
    matching: MatchMetrics,
    counting: CountingMetrics,
) -> list[dict[str, Any]]:
    return [
        {
            "variant": name,
            "category": category_name,
            "ap": detection.per_class_ap.get(category, math.nan),
            "precision": matching.per_class[category].precision,
            "recall": matching.per_class[category].recall,
            "f1": matching.per_class[category].f1,
            "mae": counting.per_class_mae[category],
        }
        for category, category_name in names_by_id.items()
    ]


def _variant_row(
    name: str,
    outcome: PredictionRun,
    detection: DetectionMetrics,
    matching: MatchMetrics,
    counting: CountingMetrics,
    chosen: float,
    fixed: float | None,
) -> dict[str, Any]:
    peak_memory = outcome.costs.peak_memory_mb
    return {
        "variant": name,
        "images": outcome.image_count,
        "detections": len(outcome.results),
        "ap": detection.ap,
        "ap50": detection.ap50,
        "ap_small": detection.ap_small,
        "ap_medium": detection.ap_medium,
        "ap_large": detection.ap_large,
        "ar_small": detection.ar_small,
        "ar_medium": detection.ar_medium,
        "ar_large": detection.ar_large,
        "precision": matching.precision,
        "recall": matching.recall,
        "f1": matching.f1,
        "threshold": chosen,
        "threshold_source": "fixed" if fixed is not None else "tuned_on_split",
        "mae": counting.mae,
        "rmse": counting.rmse,
        "seconds_per_image": outcome.costs.mean_seconds,
        "p95_seconds": outcome.costs.p95_seconds,
        "peak_memory_mb": math.nan if peak_memory is None else peak_memory,
    }


def to_markdown(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    lines = [header, separator]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_format(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    frame: pd.DataFrame, out_dir: str | Path, provenance: Mapping[str, Any] | None = None
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "comparison.csv", index=False)
    predictions = frame.attrs.get("predictions")
    if predictions:
        predictions_dir = out_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        for name, results in predictions.items():
            (predictions_dir / f"{name}.json").write_text(json.dumps(results))
    per_class = frame.attrs.get("per_class")
    if per_class is not None:
        per_class.to_csv(out_dir / "per_class.csv", index=False)
        (out_dir / "per_class.md").write_text(to_markdown(per_class) + "\n")
    table = to_markdown(frame)
    if provenance is not None:
        (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
        if provenance.get("provisional"):
            table = f"{PROVISIONAL_BANNER}\n\n{table}"
    (out_dir / "comparison.md").write_text(table + "\n")


def run(cfg: DictConfig, model_factory: Callable[..., Any] = build_detection_model) -> pd.DataFrame:
    _check_thresholds(cfg)
    categories = load_coco(cfg.coco_path)["categories"]
    base = OmegaConf.masked_copy(cfg, [key for key in cfg if key != "variants"])
    variants = [
        Variant(
            item.name,
            build_predictor(OmegaConf.merge(base, item), categories, model_factory),
            item.get("threshold"),
        )
        for item in cfg.variants
    ]
    device = best_device() if cfg.device == "auto" else torch.device(cfg.device)
    frame = compare(
        cfg.coco_path,
        cfg.images_dir,
        cfg.manifest_path,
        cfg.split,
        variants,
        thresholds=tuple(cfg.thresholds),
        threshold=cfg.threshold,
        limit=cfg.limit,
        device=device,
    )
    write_report(frame, cfg.out_dir, describe_run(cfg))
    return frame


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="compare")
def main(cfg: DictConfig) -> None:
    if describe_run(cfg)["provisional"]:
        print(PROVISIONAL_BANNER, file=sys.stderr)
    print(to_markdown(run(cfg)))


def _check_thresholds(cfg: DictConfig) -> None:
    floor = cfg.postprocess.confidence_threshold
    grid = [cfg.threshold] if cfg.threshold is not None else list(cfg.thresholds)
    grid += [item.threshold for item in cfg.variants if item.get("threshold") is not None]
    below = [value for value in grid if value < floor]
    if below:
        raise ValueError(
            f"soglie sotto la soglia di confidenza del modello ({floor}): "
            f"{', '.join(str(value) for value in below)}"
        )


def _format(value: Any) -> str:
    if isinstance(value, float):
        return NOT_AVAILABLE if math.isnan(value) else f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
