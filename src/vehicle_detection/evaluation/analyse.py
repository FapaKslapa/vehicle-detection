"""Error analysis, counting and report figures for a ``compare`` run.

Reads ``comparison.csv``, ``predictions/<variant>.json`` and ``provenance.json`` from a run
directory written by ``experiments.compare`` and writes to ``<run-dir>/analysis``:

  summary.md                     the headline tables for every variant
  outcomes.csv                   prediction and ground-truth outcomes per variant
  localisation.csv               class-aware vs class-agnostic AP, precision, recall, F1
  counting.csv                   count MAE, RMSE, bias, relative MAE (total and per class)
  recall_breakdown.csv           recall by class, size, image edge, density, location, scene
  error_types.png, recall_by_*.png
  <variant>/confusion.csv, confusion.png
  <variant>/counts_per_image.csv, counts.png
  <variant>/gt_boxes.csv         every ground-truth box with its attributes and outcome
  <variant>/predictions.csv      every prediction above the threshold with its outcome
  <variant>/gallery/*.jpg        the worst images, annotated

Each variant is analysed at the threshold ``compare`` reported for it, so the numbers line
up with ``comparison.csv``. The split, image subset and paths come from ``provenance.json``
unless overridden.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from vehicle_detection.counting.counts import TOTAL, count_summary, count_table
from vehicle_detection.data.coco_split import load_coco, split_coco_by_manifest
from vehicle_detection.data.manifest import load_manifest
from vehicle_detection.evaluation.breakdown import (
    BREAKDOWN_ATTRIBUTES,
    DEFAULT_DENSITY_EDGES,
    DEFAULT_SIZE_EDGES,
    outcome_table,
    recall_breakdown,
    truth_attributes,
)
from vehicle_detection.evaluation.errors import (
    PredictionOutcome,
    TruthOutcome,
    agnostic_matches,
    category_names,
    class_agnostic,
    confusion_matrix,
    label_errors,
    localisation_summary,
)
from vehicle_detection.experiments.metrics import detection_metrics
from vehicle_detection.reporting.figures import (
    ERROR_PANELS,
    plot_confusion,
    plot_count_grid,
    plot_error_types,
    plot_recall_heatmap,
)
from vehicle_detection.reporting.gallery import draw_image, pick_images
from vehicle_detection.reporting.tables import markdown_table

PROVISIONAL_PREFIX = "provisional"
PROVISIONAL_BANNER = "PROVISIONAL SPLIT: these numbers are not valid for the report"
CONSISTENCY_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class RunSettings:
    run_dir: Path
    out_dir: Path
    coco_path: Path
    manifest_path: Path
    split: str
    limit: int | None
    images_dir: Path | None
    provisional: bool


@dataclass(frozen=True, slots=True)
class VariantAnalysis:
    name: str
    threshold: float
    outcomes: dict[str, Any]
    localisation: dict[str, Any]
    counting: pd.DataFrame
    breakdown: pd.DataFrame
    warnings: tuple[str, ...]


def select_subset(
    coco_path: str | Path, manifest_path: str | Path, split: str, limit: int | None = None
) -> dict[str, Any]:
    """Same images, in the same order, as ``inference.run.select_split_subset``.

    Re-implemented here so the analysis does not need torch; a test keeps the two equal.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be positive: {limit}")
    subsets = split_coco_by_manifest(load_coco(coco_path), load_manifest(manifest_path))
    if split not in subsets:
        raise ValueError(f"split {split!r} not in the manifest (has {', '.join(sorted(subsets))})")
    subset = subsets[split]
    images = sorted(subset["images"], key=lambda image: image["file_name"])[:limit]
    kept = {image["id"] for image in images}
    return {
        **subset,
        "images": images,
        "annotations": [a for a in subset["annotations"] if a["image_id"] in kept],
    }


def resolve_settings(args: argparse.Namespace) -> RunSettings:
    run_dir = Path(args.run_dir)
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    config = provenance.get("config", {})

    def pick(value: Any, key: str) -> Any:
        return value if value is not None else config.get(key)

    coco_path = pick(args.coco, "coco_path")
    manifest_path = pick(args.manifest, "manifest_path")
    split = pick(args.split, "split")
    missing = [
        flag
        for flag, value in (
            ("--coco", coco_path),
            ("--manifest", manifest_path),
            ("--split", split),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"no provenance.json in {run_dir} and no {', '.join(missing)} given on the command line"
        )
    images_dir = pick(args.images_dir, "images_dir")
    return RunSettings(
        run_dir=run_dir,
        out_dir=Path(args.out) if args.out else run_dir / "analysis",
        coco_path=Path(coco_path),
        manifest_path=Path(manifest_path),
        split=str(split),
        limit=pick(args.limit, "limit"),
        images_dir=Path(images_dir) if images_dir else None,
        provisional=bool(
            provenance.get("provisional", Path(manifest_path).name.startswith(PROVISIONAL_PREFIX))
        ),
    )


def load_run(
    run_dir: Path, variants: Sequence[str] | None, threshold: float | None
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float], pd.DataFrame | None]:
    comparison_path = run_dir / "comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.is_file() else None
    available = sorted(path.stem for path in (run_dir / "predictions").glob("*.json"))
    if comparison is not None:
        ordered = [name for name in comparison["variant"] if name in available]
        available = ordered + [name for name in available if name not in ordered]
    names = list(variants) if variants else available
    absent = [name for name in names if name not in available]
    if absent:
        raise FileNotFoundError(f"no predictions in {run_dir / 'predictions'} for: {absent}")
    if not names:
        raise FileNotFoundError(f"no predictions/*.json in {run_dir}")
    thresholds: dict[str, float] = {}
    for name in names:
        if threshold is not None:
            thresholds[name] = threshold
        elif comparison is not None and name in set(comparison["variant"]):
            value = comparison.loc[comparison["variant"] == name, "threshold"].iloc[0]
            thresholds[name] = float(value)
        else:
            raise ValueError(f"no threshold for {name}: not in comparison.csv, pass --threshold")
    predictions = {
        name: json.loads((run_dir / "predictions" / f"{name}.json").read_text()) for name in names
    }
    return predictions, thresholds, comparison


def analyse_variant(
    name: str,
    gt: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    threshold: float,
    manifest: pd.DataFrame | None,
    settings: RunSettings,
    comparison_row: Mapping[str, Any] | None,
    size_edges: Sequence[float],
    density_edges: Sequence[int],
    gallery: int,
) -> VariantAnalysis:
    _check_images(gt, results, name)
    out_dir = settings.out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    names = category_names(gt)
    labels = label_errors(gt, results, threshold)
    matrix = confusion_matrix(gt, results, threshold)
    pairs = agnostic_matches(gt, results, threshold)
    agnostic = localisation_summary(matrix)
    table = outcome_table(
        truth_attributes(gt, manifest, size_edges, density_edges),
        labels.truths,
        pairs,
        results,
        names,
    )
    breakdown = recall_breakdown(table)
    counts = count_table(gt, results, threshold)
    counting = count_summary(counts, gt["categories"])

    correct = labels.prediction_counts()[PredictionOutcome.CORRECT.value]
    precision = _ratio(correct, len(labels.predictions))
    recall = _ratio(correct, len(labels.truths))
    f1 = _ratio(2 * precision * recall, precision + recall)
    warnings = _consistency(comparison_row, precision, recall, counting)
    aware = _class_aware_ap(comparison_row, gt, results)
    agnostic_metrics = detection_metrics(*class_agnostic(gt, results))
    localisation = {
        "variant": name,
        "ap": aware[0],
        "ap_any_class": agnostic_metrics.ap,
        "ap50": aware[1],
        "ap50_any_class": agnostic_metrics.ap50,
        "precision": precision,
        "precision_any_class": agnostic.precision,
        "recall": recall,
        "recall_any_class": agnostic.recall,
        "f1": f1,
        "f1_any_class": agnostic.f1,
        "class_accuracy_when_found": agnostic.class_accuracy,
    }
    outcomes = {
        "variant": name,
        "images": len(gt["images"]),
        "threshold": threshold,
        "predictions": len(labels.predictions),
        "vehicles": len(labels.truths),
        **labels.prediction_counts(),
        **labels.truth_counts(),
    }

    matrix.to_csv(out_dir / "confusion.csv")
    table.to_csv(out_dir / "gt_boxes.csv", index=False)
    labels.predictions.assign(category=labels.predictions["category_id"].map(names)).to_csv(
        out_dir / "predictions.csv", index=False
    )
    counts.to_csv(out_dir / "counts_per_image.csv", index=False)
    plot_confusion(matrix, out_dir / "confusion.png", f"{name}: which class the found vehicles got")
    total = counting.set_index("target").loc[TOTAL]
    plot_count_grid(
        counts,
        out_dir / "counts.png",
        f"{name}: vehicles per image, predicted vs true",
        f"MAE {total['mae']:.2f}, RMSE {total['rmse']:.2f}, bias {total['bias']:+.2f} "
        f"(threshold {threshold:g})",
    )
    _write_gallery(name, gt, results, labels, counts, names, settings, out_dir, gallery)
    return VariantAnalysis(
        name=name,
        threshold=threshold,
        outcomes=outcomes,
        localisation=localisation,
        counting=counting.assign(variant=name, threshold=threshold),
        breakdown=breakdown.assign(variant=name),
        warnings=tuple(warnings),
    )


def write_summary(
    analyses: Sequence[VariantAnalysis], gt: Mapping[str, Any], settings: RunSettings
) -> str:
    out = settings.out_dir
    outcomes = pd.DataFrame([a.outcomes for a in analyses])
    localisation = pd.DataFrame([a.localisation for a in analyses])
    counting = pd.concat([a.counting for a in analyses], ignore_index=True)
    counting = counting[
        [
            "variant",
            "threshold",
            *[c for c in counting.columns if c not in ("variant", "threshold")],
        ]
    ]
    breakdown = pd.concat([a.breakdown for a in analyses], ignore_index=True)
    breakdown = breakdown[["variant", *[c for c in breakdown.columns if c != "variant"]]]
    outcomes.to_csv(out / "outcomes.csv", index=False)
    localisation.to_csv(out / "localisation.csv", index=False)
    counting.to_csv(out / "counting.csv", index=False)
    breakdown.to_csv(out / "recall_breakdown.csv", index=False)

    per_image = outcomes.set_index("variant")[[key for key, _ in ERROR_PANELS]].div(
        outcomes.set_index("variant")["images"], axis=0
    )
    plot_error_types(per_image, out / "error_types.png", "What the errors are")
    by_variant = {a.name: a.breakdown for a in analyses}
    for attribute in BREAKDOWN_ATTRIBUTES:
        plot_recall_heatmap(
            by_variant,
            attribute,
            out / f"recall_by_{attribute}.png",
            f"Recall by {attribute.replace('_', ' ')}",
        )

    totals = counting[counting["target"] == TOTAL].drop(columns=["target"])
    class_mae = counting[counting["target"] != TOTAL].pivot_table(
        index="variant", columns="target", values="mae", sort=False
    )
    class_mae = class_mae.reindex(index=[a.name for a in analyses])
    size = breakdown[breakdown["attribute"] == "size_bin"].pivot_table(
        index="variant", columns="value", values="recall_any_class", sort=False
    )
    size = size.reindex(index=[a.name for a in analyses])
    shares = outcomes.set_index("variant")
    prediction_keys = [o.value for o in PredictionOutcome]
    truth_keys = [o.value for o in TruthOutcome]
    prediction_shares = shares[prediction_keys].div(shares["predictions"], axis=0)
    truth_shares = shares[truth_keys].div(shares["vehicles"], axis=0)

    lines = [f"# Error analysis: {settings.run_dir}", ""]
    if settings.provisional:
        lines += [f"**{PROVISIONAL_BANNER}**", ""]
    lines += [
        f"Split `{settings.split}` of `{settings.manifest_path}`: {len(gt['images'])} images, "
        f"{len(gt['annotations'])} vehicles. Each variant is analysed at its threshold from "
        "`comparison.csv`.",
        "",
        "## Finding the vehicle vs naming its class",
        "",
        "`*_any_class` ignores the class when pairing boxes. A large gap between the two "
        "columns means the model finds vehicles but gives them the wrong class.",
        "",
        markdown_table(localisation),
        "",
        "## What the predictions are (share of predictions above the threshold)",
        "",
        markdown_table(prediction_shares, index=True),
        "",
        "## What happened to the vehicles (share of true vehicles)",
        "",
        markdown_table(truth_shares, index=True),
        "",
        "## Counting: total vehicles per image",
        "",
        markdown_table(totals),
        "",
        "## Counting: MAE per class",
        "",
        markdown_table(class_mae, index=True),
        "",
        "## Recall (any class) by vehicle size, sqrt(area) in px",
        "",
        markdown_table(size, index=True),
        "",
    ]
    notes = [warning for a in analyses for warning in a.warnings]
    if notes:
        lines += ["## Warnings", "", *[f"- {note}" for note in notes], ""]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text)
    return text


def run(args: argparse.Namespace) -> str:
    settings = resolve_settings(args)
    predictions, thresholds, comparison = load_run(settings.run_dir, args.variants, args.threshold)
    gt = select_subset(settings.coco_path, settings.manifest_path, settings.split, settings.limit)
    manifest = load_manifest(settings.manifest_path)
    settings.out_dir.mkdir(parents=True, exist_ok=True)
    analyses = []
    for name, results in predictions.items():
        row = None
        if comparison is not None and name in set(comparison["variant"]):
            row = comparison.loc[comparison["variant"] == name].iloc[0].to_dict()
        print(f"analysing {name} at threshold {thresholds[name]:g}", file=sys.stderr)
        analyses.append(
            analyse_variant(
                name,
                gt,
                results,
                thresholds[name],
                manifest,
                settings,
                row,
                args.size_edges,
                args.density_edges,
                args.gallery,
            )
        )
    return write_summary(analyses, gt, settings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="output folder of experiments.compare")
    parser.add_argument("--out", help="default: <run-dir>/analysis")
    parser.add_argument("--variants", nargs="+", help="default: every variant in the run")
    parser.add_argument("--threshold", type=float, help="override every variant's threshold")
    parser.add_argument("--coco", help="default: from provenance.json")
    parser.add_argument("--manifest", help="default: from provenance.json")
    parser.add_argument("--split", help="default: from provenance.json")
    parser.add_argument("--limit", type=int, help="default: from provenance.json")
    parser.add_argument("--images-dir", help="default: from provenance.json")
    parser.add_argument(
        "--size-edges", type=float, nargs="+", default=list(DEFAULT_SIZE_EDGES), metavar="PX"
    )
    parser.add_argument(
        "--density-edges", type=int, nargs="+", default=list(DEFAULT_DENSITY_EDGES), metavar="N"
    )
    parser.add_argument(
        "--gallery", type=int, default=24, help="annotated worst images per variant (0 = none)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(run(args))
    return 0


def _check_images(gt: Mapping[str, Any], results: Sequence[Mapping[str, Any]], name: str) -> None:
    known = {image["id"] for image in gt["images"]}
    outside = sorted({r["image_id"] for r in results} - known)
    if outside:
        raise ValueError(
            f"{name}: {len(outside)} predicted images are not in the selected split subset "
            f"(e.g. {outside[:5]}); check --split and --limit against the compare run"
        )


def _class_aware_ap(
    row: Mapping[str, Any] | None, gt: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> tuple[float, float]:
    if row is not None and "ap" in row and "ap50" in row:
        return float(row["ap"]), float(row["ap50"])
    metrics = detection_metrics(gt, results)
    return metrics.ap, metrics.ap50


def _consistency(
    row: Mapping[str, Any] | None, precision: float, recall: float, counting: pd.DataFrame
) -> list[str]:
    if row is None:
        return []
    total = counting.set_index("target").loc[TOTAL]
    checks = (
        ("precision", precision),
        ("recall", recall),
        ("mae", float(total["mae"])),
        ("rmse", float(total["rmse"])),
    )
    warnings = []
    for column, value in checks:
        reported = row.get(column)
        if reported is None or (isinstance(reported, float) and math.isnan(reported)):
            continue
        if abs(float(reported) - value) > CONSISTENCY_TOLERANCE:
            warnings.append(
                f"{row['variant']}: {column} is {value:.6f} here but {float(reported):.6f} in "
                "comparison.csv (different threshold, split or limit?)"
            )
    return warnings


def _write_gallery(
    name: str,
    gt: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    labels: Any,
    counts: pd.DataFrame,
    names: Mapping[int, str],
    settings: RunSettings,
    out_dir: Path,
    limit: int,
) -> None:
    if limit <= 0:
        return
    if settings.images_dir is None or not settings.images_dir.is_dir():
        print(
            f"{name}: no images folder ({settings.images_dir}), skipping the gallery",
            file=sys.stderr,
        )
        return
    gallery_dir = out_dir / "gallery"
    gallery_dir.mkdir(exist_ok=True)
    images = {image["id"]: image for image in gt["images"]}
    rows = []
    chosen = pick_images(counts, labels.predictions, labels.truths, limit)
    for rank, image_id in enumerate(chosen, start=1):
        image = images[image_id]
        truths = labels.truths[labels.truths["image_id"] == image_id]
        predictions = labels.predictions[labels.predictions["image_id"] == image_id]
        picture = draw_image(
            settings.images_dir / image["file_name"],
            [gt["annotations"][i] for i in truths["annotation_index"]],
            list(truths["outcome"]),
            [results[i] for i in predictions["result_index"]],
            list(predictions["outcome"]),
            names,
        )
        target = gallery_dir / f"{rank:02d}_{Path(image['file_name']).stem}.jpg"
        picture.save(target, quality=90)
        error = counts.loc[counts["image_id"] == image_id, "error_total"].iloc[0]
        rows.append({"rank": rank, "file_name": image["file_name"], "count_error": int(error)})
    pd.DataFrame(rows).to_csv(gallery_dir / "index.csv", index=False)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
