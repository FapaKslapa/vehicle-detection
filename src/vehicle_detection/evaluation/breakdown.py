"""Recall broken down by properties of the ground-truth vehicles.

COCO's small/medium/large cut-offs (32 and 96 px) put only 2% of TRANSSET boxes in
"small", so size is binned on this dataset's own quartiles of sqrt(area) instead:
60, 77 and 100 px on the full 1024x1024 images.

``size_and_edge`` crosses size with whether the box touches the image edge (the frames are
crops of 4K video, so those vehicles are partly outside the picture). It answers whether
the small vehicles the model misses are mostly the cut-off ones.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from vehicle_detection.evaluation.errors import TruthOutcome, category_names

DEFAULT_SIZE_EDGES: tuple[float, ...] = (60.0, 77.0, 100.0)
DEFAULT_DENSITY_EDGES: tuple[int, ...] = (3, 6, 10)
EDGE_MARGIN = 0.5
BREAKDOWN_ATTRIBUTES = (
    "category",
    "size_bin",
    "at_edge",
    "size_and_edge",
    "density_bin",
    "location",
    "scene",
)
UNKNOWN = "unknown"
INSIDE = "inside"
AT_EDGE = "edge"


def bin_labels(edges: Sequence[float], integer: bool = False) -> list[str]:
    if not edges or list(edges) != sorted(set(edges)):
        raise ValueError(f"bin edges must be strictly increasing: {list(edges)}")

    def show(value: float) -> str:
        return f"{value:g}"

    labels = [f"<{show(edges[0])}"]
    for low, high in zip(edges, edges[1:], strict=False):
        labels.append(f"{show(low)}-{show(high - 1)}" if integer else f"{show(low)}-{show(high)}")
    labels.append(f">={show(edges[-1])}")
    return labels


def bin_of(value: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    for edge, label in zip(edges, labels, strict=False):
        if value < edge:
            return label
    return labels[-1]


def size_and_edge_label(size_bin: str, at_edge: bool) -> str:
    return f"{size_bin} {AT_EDGE if at_edge else INSIDE}"


def truth_attributes(
    gt: Mapping[str, Any],
    manifest: pd.DataFrame | None = None,
    size_edges: Sequence[float] = DEFAULT_SIZE_EDGES,
    density_edges: Sequence[int] = DEFAULT_DENSITY_EDGES,
) -> pd.DataFrame:
    names = category_names(gt)
    images = {image["id"]: image for image in gt["images"]}
    per_image = Counter(annotation["image_id"] for annotation in gt["annotations"])
    size_labels = bin_labels(size_edges)
    density_labels = bin_labels(density_edges, integer=True)
    scene_of: dict[str, str] = {}
    location_of: dict[str, str] = {}
    if manifest is not None:
        scene_of = dict(zip(manifest["image_id"], manifest["session_id"], strict=True))
        location_of = dict(zip(manifest["image_id"], manifest["location"], strict=True))
    rows = []
    for position, annotation in enumerate(gt["annotations"]):
        image = images[annotation["image_id"]]
        stem = Path(image["file_name"]).stem
        x, y, width, height = annotation["bbox"]
        side = (width * height) ** 0.5
        vehicles = per_image[annotation["image_id"]]
        size_bin = bin_of(side, size_edges, size_labels)
        at_edge = _touches_edge(x, y, width, height, image["width"], image["height"])
        rows.append(
            {
                "annotation_index": position,
                "image_id": annotation["image_id"],
                "file_name": image["file_name"],
                "category": names[annotation["category_id"]],
                "side": side,
                "size_bin": size_bin,
                "at_edge": at_edge,
                "size_and_edge": size_and_edge_label(size_bin, at_edge),
                "vehicles_in_image": vehicles,
                "density_bin": bin_of(vehicles, density_edges, density_labels),
                "location": location_of.get(stem, UNKNOWN),
                "scene": scene_of.get(stem, UNKNOWN),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["category"] = pd.Categorical(frame["category"], categories=list(names.values()))
    frame["size_bin"] = pd.Categorical(frame["size_bin"], categories=size_labels, ordered=True)
    frame["size_and_edge"] = pd.Categorical(
        frame["size_and_edge"],
        categories=[size_and_edge_label(s, e) for s in size_labels for e in (False, True)],
        ordered=True,
    )
    frame["density_bin"] = pd.Categorical(
        frame["density_bin"], categories=density_labels, ordered=True
    )
    return frame


def outcome_table(
    attributes: pd.DataFrame,
    truths: pd.DataFrame,
    pairs: Mapping[int, int],
    results: Sequence[Mapping[str, Any]],
    names: Mapping[int, str],
) -> pd.DataFrame:
    """One row per ground-truth box with its attributes and what happened to it.

    ``found_correct_class`` comes from the class-aware matching, so per-class recall agrees
    with ``compare``. ``found_any_class`` and ``predicted_category`` come from the
    class-agnostic ``pairs`` (annotation index -> result index), so they agree with the
    confusion matrix.
    """
    outcomes = truths.set_index("annotation_index")[["outcome", "best_iou"]]
    table = attributes.join(outcomes, on="annotation_index")
    if table["outcome"].isna().any():
        raise ValueError("every ground-truth box needs an outcome from label_errors")
    predicted = {
        annotation: names[results[result]["category_id"]] for annotation, result in pairs.items()
    }
    table["predicted_category"] = table["annotation_index"].map(predicted)
    table["found_any_class"] = table["predicted_category"].notna()
    table["same_class_when_found"] = table["found_any_class"] & (
        table["predicted_category"] == table["category"].astype(str)
    )
    table["found_correct_class"] = table["outcome"] == TruthOutcome.DETECTED.value
    return table


def recall_breakdown(
    table: pd.DataFrame, attributes: Sequence[str] = BREAKDOWN_ATTRIBUTES
) -> pd.DataFrame:
    rows = []
    for attribute in attributes:
        grouped = table.groupby(attribute, observed=True, sort=True)
        for value, group in grouped:
            boxes = len(group)
            found = int(group["found_any_class"].sum())
            same = int(group["same_class_when_found"].sum())
            rows.append(
                {
                    "attribute": attribute,
                    "value": str(value),
                    "boxes": boxes,
                    "recall_any_class": found / boxes,
                    "recall_correct_class": float(group["found_correct_class"].mean()),
                    "class_accuracy_when_found": same / found if found else float("nan"),
                    "missed": int((group["outcome"] == TruthOutcome.MISSED.value).sum()),
                    "wrong_class": int((group["outcome"] == TruthOutcome.WRONG_CLASS.value).sum()),
                }
            )
    return pd.DataFrame(rows)


def _touches_edge(
    x: float, y: float, width: float, height: float, image_width: float, image_height: float
) -> bool:
    return (
        x <= EDGE_MARGIN
        or y <= EDGE_MARGIN
        or x + width >= image_width - EDGE_MARGIN
        or y + height >= image_height - EDGE_MARGIN
    )
