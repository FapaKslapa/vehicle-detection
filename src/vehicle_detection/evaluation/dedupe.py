"""Class-agnostic duplicate removal on saved predictions, to measure its effect on counting.

NMS inside the detector and SAHI's tile merging both run per class, so one vehicle can
survive as both "car" and "sedan" (``cross_class_duplicate`` in ``evaluation.errors``) and
is then counted twice. This applies class-agnostic NMS to the predictions of an existing
``compare`` run, without running the model again, and writes a new run folder that
``just analyse`` and ``experiments.bootstrap`` read like any other ``compare`` output: the
original variants unchanged, plus one ``<variant>-agn<IoU>`` variant per IoU.

The counting threshold of each new variant is re-tuned on the split, as ``compare`` does,
because removing boxes moves the best threshold. On the test split tuning is refused:
pass ``--thresholds-from`` with the ``comparison.csv`` of the matching validation run.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from vehicle_detection.evaluation.analyse import select_subset
from vehicle_detection.evaluation.errors import iou, xyxy
from vehicle_detection.experiments.metrics import (
    best_threshold,
    counting_metrics,
    detection_metrics,
    match_metrics,
)
from vehicle_detection.reporting.tables import markdown_table

DEFAULT_IOUS = (0.5, 0.7)
THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))
TEST_SPLIT = "test"
COLUMNS = (
    "variant",
    "source",
    "dedupe_iou",
    "images",
    "detections",
    "ap",
    "ap50",
    "ap_small",
    "ap_medium",
    "ap_large",
    "precision",
    "recall",
    "f1",
    "threshold",
    "threshold_source",
    "mae",
    "rmse",
)


def suppress_across_classes(
    results: Sequence[Mapping[str, Any]], iou_threshold: float
) -> list[dict[str, Any]]:
    """Greedy NMS that ignores the class: per image, a box is dropped when a higher-scoring
    box of any class overlaps it with IoU >= ``iou_threshold``. Kept boxes stay in input order.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError(f"IoU threshold must be in (0, 1]: {iou_threshold}")
    by_image: dict[int, list[int]] = defaultdict(list)
    for position, result in enumerate(results):
        by_image[result["image_id"]].append(position)
    kept: list[int] = []
    for positions in by_image.values():
        boxes: list[tuple[float, float, float, float]] = []
        for position in sorted(positions, key=lambda p: -results[p]["score"]):
            box = xyxy(results[position]["bbox"])
            if all(iou(box, other) < iou_threshold for other in boxes):
                boxes.append(box)
                kept.append(position)
    return [dict(results[position]) for position in sorted(kept)]


def variant_name(source: str, iou_threshold: float) -> str:
    return f"{source}-agn{round(iou_threshold * 100)}"


def evaluate(
    gt: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    threshold: float | None,
    grid: Sequence[float] = THRESHOLDS,
) -> dict[str, Any]:
    detection = detection_metrics(gt, results)
    if threshold is None:
        chosen, counting = best_threshold(gt, results, grid)
        source = "tuned_on_split"
    else:
        chosen, counting = threshold, counting_metrics(gt, results, threshold)
        source = "fixed"
    matching = match_metrics(gt, results, chosen)
    return {
        "images": len(gt["images"]),
        "detections": len(results),
        "ap": detection.ap,
        "ap50": detection.ap50,
        "ap_small": detection.ap_small,
        "ap_medium": detection.ap_medium,
        "ap_large": detection.ap_large,
        "precision": matching.precision,
        "recall": matching.recall,
        "f1": matching.f1,
        "threshold": chosen,
        "threshold_source": source,
        "mae": counting.mae,
        "rmse": counting.rmse,
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir.with_name(f"{run_dir.name}_agnostic")
    if out_dir.resolve() == run_dir.resolve():
        raise ValueError("--out must differ from --run-dir, the source run is never modified")
    provenance = json.loads((run_dir / "provenance.json").read_text())
    config = provenance.get("config", {})
    split = args.split or config["split"]
    fixed = _thresholds_from(args.thresholds_from)
    if split == TEST_SPLIT and fixed is None:
        raise ValueError(
            "the threshold is not tuned on test: pass --thresholds-from "
            "<validation run>/comparison.csv"
        )
    gt = select_subset(
        args.coco or config["coco_path"],
        args.manifest or config["manifest_path"],
        split,
        config.get("limit"),
    )
    comparison = pd.read_csv(run_dir / "comparison.csv")
    source_thresholds = dict(zip(comparison["variant"], comparison["threshold"], strict=True))
    variants = list(args.variants) if args.variants else list(comparison["variant"])
    missing = [name for name in variants if name not in source_thresholds]
    if missing:
        raise ValueError(f"not in {run_dir / 'comparison.csv'}: {missing}")

    predictions_dir = out_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in variants:
        results = json.loads((run_dir / "predictions" / f"{name}.json").read_text())
        shutil.copyfile(run_dir / "predictions" / f"{name}.json", predictions_dir / f"{name}.json")
        rows.append(
            {
                "variant": name,
                "source": name,
                "dedupe_iou": math.nan,
                **evaluate(gt, results, float(source_thresholds[name])),
            }
        )
        for iou_threshold in args.iou:
            derived = variant_name(name, iou_threshold)
            kept = suppress_across_classes(results, iou_threshold)
            (predictions_dir / f"{derived}.json").write_text(json.dumps(kept))
            threshold = None
            if fixed is not None:
                if derived not in fixed:
                    raise ValueError(f"no threshold for {derived} in {args.thresholds_from}")
                threshold = fixed[derived]
            rows.append(
                {
                    "variant": derived,
                    "source": name,
                    "dedupe_iou": iou_threshold,
                    **evaluate(gt, kept, threshold),
                }
            )
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    frame.to_csv(out_dir / "comparison.csv", index=False)
    table = markdown_table(frame)
    (out_dir / "comparison.md").write_text(table + "\n")
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                **provenance,
                "config": {**config, "split": split},
                "derived_from": str(run_dir),
                "class_agnostic_nms_iou": list(args.iou),
                "thresholds_from": args.thresholds_from,
            },
            indent=2,
        )
    )
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="output folder of experiments.compare")
    parser.add_argument("--out", help="default: <run-dir>_agnostic next to the source run")
    parser.add_argument("--iou", type=float, nargs="+", default=list(DEFAULT_IOUS))
    parser.add_argument("--variants", nargs="+", help="default: every variant in the run")
    parser.add_argument("--thresholds-from", help="comparison.csv of a validation run")
    parser.add_argument("--coco", help="default: from provenance.json")
    parser.add_argument("--manifest", help="default: from provenance.json")
    parser.add_argument("--split", help="default: from provenance.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(markdown_table(run(build_parser().parse_args(argv))))
    return 0


def _thresholds_from(path: str | None) -> dict[str, float] | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    return {
        name: float(value) for name, value in zip(frame["variant"], frame["threshold"], strict=True)
    }


if __name__ == "__main__":
    raise SystemExit(main())
