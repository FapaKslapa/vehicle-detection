"""Per-image vehicle counts, in total and per class, and their errors.

``count_table`` is the counting output of the system on a labelled split (true and
predicted counts side by side); ``predicted_counts`` is the same output for unseen images
with no labels. ``count_summary`` reports MAE and RMSE as the assignment asks, plus the
signed bias (positive = over-counting) and the MAE relative to the mean true count, which
makes the error readable: TRANSSET images hold only 4.6 vehicles on average.

The total-count MAE and RMSE and the per-class MAE equal ``experiments.metrics``'s
``counting_metrics`` for the same threshold.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

TOTAL = "total"
SUMMARY_COLUMNS = (
    "target",
    "mean_true",
    "mean_predicted",
    "mae",
    "rmse",
    "bias",
    "relative_mae",
    "exact_share",
)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def category_slugs(categories: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    ordered = sorted(categories, key=lambda category: category["id"])
    slugs = {category["id"]: slug(category["name"]) for category in ordered}
    repeated = sorted(name for name, n in Counter(slugs.values()).items() if n > 1)
    if repeated or TOTAL in slugs.values():
        raise ValueError(f"category names clash once turned into columns: {repeated or TOTAL}")
    return slugs


def predicted_counts(
    images: Sequence[Mapping[str, Any]],
    results: Iterable[Mapping[str, Any]],
    categories: Sequence[Mapping[str, Any]],
    threshold: float,
) -> pd.DataFrame:
    """Total and per-class counts for each image, from detections alone."""
    slugs = category_slugs(categories)
    counts = _counts(result for result in results if result["score"] >= threshold)
    rows = []
    for image in images:
        image_counts = counts.get(image["id"], Counter())
        row: dict[str, Any] = {
            "image_id": image["id"],
            "file_name": image["file_name"],
            TOTAL: sum(image_counts[c] for c in slugs),
        }
        row.update({name: image_counts[c] for c, name in slugs.items()})
        rows.append(row)
    return pd.DataFrame(rows, columns=["image_id", "file_name", TOTAL, *slugs.values()])


def count_table(
    gt: Mapping[str, Any], results: Iterable[Mapping[str, Any]], threshold: float
) -> pd.DataFrame:
    """One row per image: true and predicted counts, total and per class."""
    slugs = category_slugs(gt["categories"])
    truth = _counts(gt["annotations"])
    predicted = _counts(result for result in results if result["score"] >= threshold)
    rows = []
    for image in gt["images"]:
        true_counts = truth.get(image["id"], Counter())
        predicted_counts_ = predicted.get(image["id"], Counter())
        true_total = sum(true_counts[c] for c in slugs)
        predicted_total = sum(predicted_counts_[c] for c in slugs)
        row: dict[str, Any] = {
            "image_id": image["id"],
            "file_name": image["file_name"],
            "true_total": true_total,
            "predicted_total": predicted_total,
            "error_total": predicted_total - true_total,
        }
        for category, name in slugs.items():
            row[f"true_{name}"] = true_counts[category]
            row[f"predicted_{name}"] = predicted_counts_[category]
            row[f"error_{name}"] = predicted_counts_[category] - true_counts[category]
        rows.append(row)
    columns = ["image_id", "file_name", "true_total", "predicted_total", "error_total"]
    for name in slugs.values():
        columns += [f"true_{name}", f"predicted_{name}", f"error_{name}"]
    return pd.DataFrame(rows, columns=columns)


def count_summary(table: pd.DataFrame, categories: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if table.empty:
        raise ValueError("no images to summarise")
    targets = [TOTAL, *category_slugs(categories).values()]
    rows = []
    for target in targets:
        true = table[f"true_{target}"].astype(float)
        predicted = table[f"predicted_{target}"].astype(float)
        error = predicted - true
        mean_true = float(true.mean())
        mae = float(error.abs().mean())
        rows.append(
            {
                "target": target,
                "mean_true": mean_true,
                "mean_predicted": float(predicted.mean()),
                "mae": mae,
                "rmse": math.sqrt(float((error**2).mean())),
                "bias": float(error.mean()),
                "relative_mae": mae / mean_true if mean_true else math.nan,
                "exact_share": float((error == 0).mean()),
            }
        )
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def _counts(items: Iterable[Mapping[str, Any]]) -> dict[int, Counter[int]]:
    counts: dict[int, Counter[int]] = {}
    for item in items:
        counts.setdefault(item["image_id"], Counter())[item["category_id"]] += 1
    return counts
