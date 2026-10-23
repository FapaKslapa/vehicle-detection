"""Static report figures (PNG), drawn with matplotlib's object API so no backend is needed.

Colour follows the data's job: one sequential blue ramp for every magnitude (shares,
recall, image counts), a neutral grey for exact zeros, and a single blue for bars. No
figure uses colour to tell variants apart, so any number of variants fits, and every
heatmap cell also prints its value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from vehicle_detection.evaluation.errors import NO_BOX

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ZERO = "#f0efec"
BAR = "#2a78d6"
WHITE = "#ffffff"
SEQUENTIAL = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
RAMP = LinearSegmentedColormap.from_list("sequential-blue", SEQUENTIAL)
LIGHT_TEXT_FROM = 0.55
HEADER_INCHES = 0.62
STYLE = {
    "font.family": "sans-serif",
    "font.size": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "text.color": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": AXIS,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
}
ERROR_PANELS = (
    ("duplicate", "Duplicate, same class"),
    ("cross_class_duplicate", "Duplicate, other class"),
    ("classification", "Wrong class"),
    ("localization", "Poor box, right class"),
    ("both", "Poor box, wrong class"),
    ("background", "Nothing there"),
    ("missed", "Missed vehicle"),
)
VALUE_NAMES = {"at_edge": {"False": "inside the image", "True": "cut by the edge"}}


def plot_confusion(matrix: pd.DataFrame, path: str | Path, title: str) -> Path:
    real = [label for label in matrix.index if label != NO_BOX]
    counts = matrix.loc[real, [*real, NO_BOX]].to_numpy(dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    text = [[_percent(value) if value > 0 else "" for value in row] for row in shares]
    false_alarms = int(matrix.loc[NO_BOX, real].sum())
    with matplotlib.rc_context(STYLE):
        figure = Figure(
            figsize=(1.6 + 0.8 * (len(real) + 1), HEADER_INCHES + 1.2 + 0.5 * len(real))
        )
        axes = figure.add_subplot()
        _cells(axes, shares, text, Normalize(0, 1))
        _tick_labels(
            axes,
            [f"{label}  (n={int(total)})" for label, total in zip(real, totals[:, 0], strict=True)],
            [*real, "missed"],
            rotation=35,
        )
        axes.axvline(len(real), color=SURFACE, linewidth=6)
        axes.set_xlabel("Class given to the vehicle (boxes paired at IoU >= 0.5, class ignored)")
        axes.set_ylabel("True class")
        _titles(
            figure,
            title,
            "Each row is one true class and adds up to 100%. "
            f"Boxes with no vehicle under them: {false_alarms}",
        )
        figure.tight_layout(rect=(0, 0, 1, _top(figure)))
        return _save(figure, path)


def plot_error_types(per_image: pd.DataFrame, path: str | Path, title: str) -> Path:
    """``per_image``: one row per variant, one column per error type, errors per image."""
    panels = [(key, label) for key, label in ERROR_PANELS if key in per_image.columns]
    variants = list(per_image.index)
    columns = 4
    rows = -(-len(panels) // columns)
    limit = float(per_image[[key for key, _ in panels]].to_numpy().max()) or 1.0
    with matplotlib.rc_context(STYLE):
        panel_height = 0.55 + 0.3 * len(variants)
        figure = Figure(figsize=(11, HEADER_INCHES + rows * panel_height))
        grid = figure.subplots(rows, columns, sharex=True, squeeze=False)
        for index, (axes, (key, label)) in enumerate(zip(grid.flat, panels, strict=False)):
            values = per_image[key].to_numpy(dtype=float)
            positions = np.arange(len(variants))
            axes.barh(positions, values, height=0.62, color=BAR)
            for position, value in zip(positions, values, strict=True):
                axes.text(
                    value + limit * 0.03, position, f"{value:.2f}", va="center", color=INK, size=8
                )
            axes.set_yticks(positions, variants if index % columns == 0 else [])
            axes.invert_yaxis()
            axes.set_xlim(0, limit * 1.35)
            axes.set_title(label, loc="left")
            axes.grid(axis="x", color=GRID, linewidth=0.8)
            axes.set_axisbelow(True)
            axes.spines["left"].set_visible(False)
            axes.tick_params(axis="y", length=0)
        for axes in list(grid.flat)[len(panels) :]:
            axes.set_visible(False)
        _titles(
            figure,
            title,
            "Errors per image, each variant at its counting threshold. Same scale in every panel.",
        )
        figure.tight_layout(rect=(0, 0, 1, _top(figure)), h_pad=1.5, w_pad=3.5)
        return _save(figure, path)


def plot_recall_heatmap(
    breakdowns: Mapping[str, pd.DataFrame], attribute: str, path: str | Path, title: str
) -> Path:
    variants = list(breakdowns)
    first = breakdowns[variants[0]]
    first = first[first["attribute"] == attribute]
    values = list(first["value"])
    boxes = dict(zip(first["value"], first["boxes"], strict=True))
    names = VALUE_NAMES.get(attribute, {})
    measures = (
        ("recall_any_class", "Found, any class"),
        ("recall_correct_class", "Found and given the right class"),
    )
    tables = {
        column: np.array(
            [
                breakdowns[variant]
                .loc[lambda frame: frame["attribute"] == attribute]
                .set_index("value")
                .reindex(values)[column]
                .to_numpy(dtype=float)
                for variant in variants
            ]
        )
        for column, _ in measures
    }
    finite = np.concatenate([table[np.isfinite(table)] for table in tables.values()])
    low = max(0.0, math.floor(finite.min(initial=1.0) * 10) / 10 - 0.1)
    norm = Normalize(vmin=low, vmax=max(float(finite.max(initial=1.0)), low + 0.1))
    column_labels = [f"{_wrap(names.get(value, value))}\nn={boxes[value]}" for value in values]
    with matplotlib.rc_context(STYLE):
        cell = 0.95 if max(len(label) for label in column_labels) > 14 else 0.8
        width = 1.4 + 2 * cell * len(values) + 0.6
        height = HEADER_INCHES + 1.1 + 0.36 * len(variants)
        figure = Figure(figsize=(max(width, 6.5), height))
        grid = figure.subplots(1, 2, squeeze=False)
        for index, (axes, (column, label)) in enumerate(zip(grid.flat, measures, strict=True)):
            table = tables[column]
            text = [[_percent(value) for value in row] for row in table]
            _cells(axes, np.nan_to_num(table), text, norm)
            _tick_labels(axes, variants if index == 0 else [], column_labels, rotation=0)
            axes.set_title(label, loc="left")
        _titles(
            figure,
            title,
            f"Share of true vehicles found (n = vehicles per group). Colour spans "
            f"{low:.0%}-{norm.vmax:.0%}; read the printed values.",
        )
        figure.tight_layout(rect=(0, 0, 1, _top(figure)), w_pad=2)
        return _save(figure, path)


def plot_count_grid(table: pd.DataFrame, path: str | Path, title: str, subtitle: str) -> Path:
    true = table["true_total"].to_numpy(dtype=int)
    predicted = table["predicted_total"].to_numpy(dtype=int)
    size = int(max(true.max(initial=0), predicted.max(initial=0))) + 1
    grid = np.zeros((size, size))
    np.add.at(grid, (predicted, true), 1)
    norm = Normalize(vmin=1, vmax=max(float(grid.max()), 2.0))
    with matplotlib.rc_context(STYLE):
        figure = Figure(figsize=(6.4, HEADER_INCHES + 5.2))
        axes = figure.add_subplot()
        for (row, column), value in np.ndenumerate(grid):
            if value > 0:
                axes.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor=RAMP(norm(value)),
                        edgecolor=SURFACE,
                        linewidth=1,
                    )
                )
        axes.plot([-0.5, size - 0.5], [-0.5, size - 0.5], color=MUTED, linewidth=1, linestyle="--")
        axes.set_xlim(-0.5, size - 0.5)
        axes.set_ylim(-0.5, size - 0.5)
        axes.set_aspect("equal")
        step = 5 if size > 12 else 1
        axes.set_xticks(range(0, size, step))
        axes.set_yticks(range(0, size, step))
        axes.set_xlabel("True vehicles in the image")
        axes.set_ylabel("Predicted vehicles in the image")
        bar = figure.colorbar(ScalarMappable(norm=norm, cmap=RAMP), ax=axes, shrink=0.75)
        bar.set_label("Images")
        bar.outline.set_visible(False)
        _titles(figure, title, f"{subtitle}. Dashed line = exact count; above it = over-counted.")
        figure.tight_layout(rect=(0, 0, 1, _top(figure)))
        return _save(figure, path)


def _cells(axes: Axes, values: np.ndarray, text: Sequence[Sequence[str]], norm: Normalize) -> None:
    rows, columns = values.shape
    for (row, column), value in np.ndenumerate(values):
        colour = RAMP(norm(value)) if value > 0 else ZERO
        axes.add_patch(
            Rectangle((column, row), 1, 1, facecolor=colour, edgecolor=SURFACE, linewidth=2)
        )
        label = text[row][column]
        if label:
            light = value > 0 and norm(value) > LIGHT_TEXT_FROM
            axes.text(
                column + 0.5,
                row + 0.5,
                label,
                ha="center",
                va="center",
                size=8,
                color=WHITE if light else INK,
            )
    axes.set_xlim(0, columns)
    axes.set_ylim(rows, 0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(length=0)


def _tick_labels(axes: Axes, rows: Sequence[str], columns: Sequence[str], rotation: float) -> None:
    axes.set_yticks(np.arange(len(rows)) + 0.5 if rows else [], rows)
    axes.set_xticks(
        np.arange(len(columns)) + 0.5,
        columns,
        rotation=rotation,
        ha="right" if rotation else "center",
        rotation_mode="anchor",
    )


def _titles(figure: Figure, title: str, subtitle: str) -> None:
    height = figure.get_figheight()
    figure.text(
        0.01, 1 - 0.08 / height, title, ha="left", va="top", size=11, weight="bold", color=INK
    )
    figure.text(0.01, 1 - 0.34 / height, subtitle, ha="left", va="top", size=9, color=INK_SECONDARY)


def _top(figure: Figure) -> float:
    return 1 - HEADER_INCHES / figure.get_figheight()


def _wrap(label: str) -> str:
    words = str(label).split(" ")
    if len(words) == 1 or len(label) <= 10:
        return str(label)
    middle = len(words) // 2
    return " ".join(words[:middle]) + "\n" + " ".join(words[middle:])


def _percent(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{value * 100:.0f}%"


def _save(figure: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    return path
