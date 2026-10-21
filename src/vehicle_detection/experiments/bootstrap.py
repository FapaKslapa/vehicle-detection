from __future__ import annotations

import contextlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from vehicle_detection.data.manifest import load_manifest
from vehicle_detection.experiments.compare import CONFIG_DIR, to_markdown
from vehicle_detection.experiments.metrics import MAX_DETECTIONS, match_counts
from vehicle_detection.inference.run import select_split_subset

METRICS = ("ap", "f1", "mae")
UNITS = ("image", "scene")
INTERVAL = (2.5, 97.5)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True, slots=True)
class CountStatistics:
    true_positives: np.ndarray
    false_positives: np.ndarray
    false_negatives: np.ndarray
    absolute_errors: np.ndarray


class AveragePrecision:
    def __init__(self, gt: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            truth = COCO()
            truth.dataset = {**gt}
            truth.createIndex()
            detections = truth.loadRes([dict(result) for result in results])
            evaluator = COCOeval(truth, detections, iouType="bbox")
            evaluator.params.maxDets = list(MAX_DETECTIONS)
            evaluator.evaluate()
        self._max_detections = MAX_DETECTIONS[-1]
        self._categories = self._collect(evaluator)

    def _collect(self, evaluator: COCOeval) -> list[dict[str, np.ndarray]]:
        image_count = len(evaluator.params.imgIds)
        area_count = len(evaluator.params.areaRng)
        categories = []
        for category_index in range(len(evaluator.params.catIds)):
            images, scores, matches, ignored = [], [], [], []
            ground_truth = np.zeros(image_count, dtype=np.int64)
            for image_index in range(image_count):
                entry = evaluator.evalImgs[category_index * area_count * image_count + image_index]
                if entry is None:
                    continue
                ground_truth[image_index] = int(np.count_nonzero(entry["gtIgnore"] == 0))
                kept = entry["dtScores"][: self._max_detections]
                images.append(np.full(len(kept), image_index, dtype=np.int64))
                scores.append(kept)
                matches.append(entry["dtMatches"][:, : self._max_detections])
                ignored.append(entry["dtIgnore"][:, : self._max_detections])
            thresholds = evaluator.params.iouThrs.size
            categories.append(
                {
                    "images": np.concatenate(images) if images else np.zeros(0, dtype=np.int64),
                    "scores": np.concatenate(scores) if scores else np.zeros(0),
                    "matches": np.concatenate(matches, axis=1)
                    if matches
                    else np.zeros((thresholds, 0)),
                    "ignored": np.concatenate(ignored, axis=1)
                    if ignored
                    else np.zeros((thresholds, 0), dtype=bool),
                    "ground_truth": ground_truth,
                }
            )
        return categories

    def __call__(self, weights: np.ndarray) -> float:
        values = []
        for category in self._categories:
            positives = int(np.dot(weights, category["ground_truth"]))
            if positives == 0:
                continue
            values.append(self._category_precision(category, weights, positives))
        return float(np.mean(values)) if values else math.nan

    @staticmethod
    def _category_precision(
        category: Mapping[str, np.ndarray], weights: np.ndarray, positives: int
    ) -> float:
        repeats = weights[category["images"]]
        index = np.repeat(np.arange(len(repeats)), repeats)
        if index.size == 0:
            return 0.0
        order = np.argsort(-category["scores"][index], kind="mergesort")
        index = index[order]
        matched = category["matches"][:, index]
        ignored = category["ignored"][:, index].astype(bool)
        true_positives = np.cumsum(np.logical_and(matched, ~ignored), axis=1, dtype=float)
        false_positives = np.cumsum(
            np.logical_and(~matched.astype(bool), ~ignored), axis=1, dtype=float
        )
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives + np.spacing(1))
        precision = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
        total = 0.0
        for row in range(precision.shape[0]):
            positions = np.searchsorted(recall[row], RECALL_THRESHOLDS, side="left")
            reached = positions < precision.shape[1]
            sampled = np.zeros(RECALL_THRESHOLDS.size)
            sampled[reached] = precision[row][positions[reached]]
            total += sampled.mean()
        return total / precision.shape[0]


def count_statistics(
    gt: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    threshold: float,
    image_ids: Sequence[int],
) -> CountStatistics:
    position = {image_id: index for index, image_id in enumerate(image_ids)}
    size = len(image_ids)
    true_positives = np.zeros(size)
    false_positives = np.zeros(size)
    false_negatives = np.zeros(size)
    for (image_id, _), (tp, fp, fn) in match_counts(gt, results, threshold).items():
        true_positives[position[image_id]] += tp
        false_positives[position[image_id]] += fp
        false_negatives[position[image_id]] += fn
    true_total = np.zeros(size)
    for annotation in gt["annotations"]:
        true_total[position[annotation["image_id"]]] += 1
    predicted_total = np.zeros(size)
    for result in results:
        if result["score"] >= threshold:
            predicted_total[position[result["image_id"]]] += 1
    return CountStatistics(
        true_positives,
        false_positives,
        false_negatives,
        np.abs(predicted_total - true_total),
    )


def metric_values(
    ap: AveragePrecision, counts: CountStatistics, weights: np.ndarray
) -> dict[str, float]:
    tp = float(np.dot(weights, counts.true_positives))
    fp = float(np.dot(weights, counts.false_positives))
    fn = float(np.dot(weights, counts.false_negatives))
    denominator = 2 * tp + fp + fn
    return {
        "ap": ap(weights),
        "f1": 2 * tp / denominator if denominator else 0.0,
        "mae": float(np.dot(weights, counts.absolute_errors) / weights.sum()),
    }


def resample_weights(
    size: int,
    resamples: int,
    seed: int,
    groups: Sequence[str] | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if groups is None:
        draws = rng.integers(0, size, size=(resamples, size))
        return np.stack([np.bincount(row, minlength=size) for row in draws])
    labels = sorted(set(groups))
    membership = np.array([[group == label for group in groups] for label in labels], dtype=int)
    draws = rng.integers(0, len(labels), size=(resamples, len(labels)))
    return np.stack([membership[row].sum(axis=0) for row in draws])


def bootstrap(
    gt: Mapping[str, Any],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    thresholds: Mapping[str, float],
    reference: str,
    resamples: int = 1000,
    seed: int = 0,
    groups: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if reference not in predictions:
        raise ValueError(f"variante di riferimento assente: {reference}")
    if resamples < 1:
        raise ValueError(f"resamples deve essere positivo: {resamples}")
    image_ids = [image["id"] for image in gt["images"]]
    if groups is not None and len(groups) != len(image_ids):
        raise ValueError("serve un gruppo per ogni immagine")
    weights = resample_weights(len(image_ids), resamples, seed, groups)
    full = np.ones(len(image_ids), dtype=np.int64)
    estimates: dict[str, dict[str, float]] = {}
    samples: dict[str, dict[str, list[float]]] = {}
    for name, results in predictions.items():
        ap = AveragePrecision(gt, results)
        counts = count_statistics(gt, results, thresholds[name], image_ids)
        estimates[name] = metric_values(ap, counts, full)
        collected: dict[str, list[float]] = {metric: [] for metric in METRICS}
        for row in weights:
            for metric, value in metric_values(ap, counts, row).items():
                collected[metric].append(value)
        samples[name] = collected
    rows = []
    differences = []
    for name in predictions:
        for metric in METRICS:
            values = np.array(samples[name][metric])
            low, high = np.nanpercentile(values, INTERVAL)
            rows.append(
                {
                    "variant": name,
                    "metric": metric,
                    "estimate": estimates[name][metric],
                    "low": low,
                    "high": high,
                }
            )
            if name == reference:
                continue
            delta = values - np.array(samples[reference][metric])
            low, high = np.nanpercentile(delta, INTERVAL)
            differences.append(
                {
                    "variant": name,
                    "reference": reference,
                    "metric": metric,
                    "difference": estimates[name][metric] - estimates[reference][metric],
                    "low": low,
                    "high": high,
                    "excludes_zero": bool(low > 0 or high < 0),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(differences)


def load_predictions(
    run_dir: str | Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    run_dir = Path(run_dir)
    comparison = pd.read_csv(run_dir / "comparison.csv")
    thresholds = dict(zip(comparison["variant"], comparison["threshold"], strict=True))
    predictions = {
        name: json.loads((run_dir / "predictions" / f"{name}.json").read_text())
        for name in thresholds
    }
    return predictions, thresholds


def scene_groups(gt: Mapping[str, Any], manifest_path: str | Path) -> list[str]:
    frame = load_manifest(manifest_path)
    scene_of = dict(zip(frame["image_id"], frame["session_id"], strict=True))
    return [scene_of[Path(image["file_name"]).stem] for image in gt["images"]]


def run(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.unit not in UNITS:
        raise ValueError(f"unità non supportata: {cfg.unit} (valide: {', '.join(UNITS)})")
    gt = select_split_subset(cfg.coco_path, cfg.manifest_path, cfg.split)
    predictions, thresholds = load_predictions(cfg.run_dir)
    groups = scene_groups(gt, cfg.manifest_path) if cfg.unit == "scene" else None
    intervals, differences = bootstrap(
        gt, predictions, thresholds, cfg.reference, cfg.resamples, cfg.seed, groups
    )
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(out_dir / "intervals.csv", index=False)
    differences.to_csv(out_dir / "differences.csv", index=False)
    (out_dir / "intervals.md").write_text(to_markdown(intervals) + "\n")
    (out_dir / "differences.md").write_text(to_markdown(differences) + "\n")
    return intervals, differences


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="bootstrap")
def main(cfg: DictConfig) -> None:
    intervals, differences = run(cfg)
    print(to_markdown(intervals))
    print()
    print(to_markdown(differences))


if __name__ == "__main__":
    main()
