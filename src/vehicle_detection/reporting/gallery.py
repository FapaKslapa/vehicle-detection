"""Annotated images of the worst failures, for looking at what actually goes wrong.

Every box carries a short text tag as well as a colour, so the outcome can be read
without relying on colour (and in a black and white printout).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from vehicle_detection.evaluation.errors import Box, PredictionOutcome, TruthOutcome, xyxy

TRUTH_COLOUR = "#ffffff"
LEGEND_BACKGROUND = "#1a1a19"
LEGEND_INK = "#f0efec"
OUTCOME_STYLE: Mapping[str, tuple[str, str]] = {
    PredictionOutcome.CORRECT.value: ("OK", "#2a78d6"),
    PredictionOutcome.DUPLICATE.value: ("DUP", "#e87ba4"),
    PredictionOutcome.CROSS_CLASS_DUPLICATE.value: ("XDUP", "#e87ba4"),
    PredictionOutcome.CLASSIFICATION.value: ("CLS", "#eb6834"),
    PredictionOutcome.LOCALIZATION.value: ("LOC", "#eda100"),
    PredictionOutcome.BOTH.value: ("LOC+CLS", "#eda100"),
    PredictionOutcome.BACKGROUND.value: ("BG", "#e34948"),
    TruthOutcome.MISSED.value: ("MISS", "#6250d6"),
}
TAGS_BELOW = frozenset(
    {PredictionOutcome.DUPLICATE.value, PredictionOutcome.CROSS_CLASS_DUPLICATE.value}
)
DARK_TEXT_ON = frozenset({"#eda100"})
LEGEND = (
    (None, "true vehicle"),
    (PredictionOutcome.CORRECT.value, "OK  correct"),
    (PredictionOutcome.CLASSIFICATION.value, "CLS  wrong class"),
    (PredictionOutcome.DUPLICATE.value, "DUP / XDUP  extra box, same / other class"),
    (PredictionOutcome.LOCALIZATION.value, "LOC  poor box"),
    (PredictionOutcome.BACKGROUND.value, "BG  nothing there"),
    (TruthOutcome.MISSED.value, "MISS  missed vehicle"),
)
LEGEND_LINE = 24
LEGEND_PADDING = 8
SWATCH = 14


def pick_images(
    counts: pd.DataFrame, predictions: pd.DataFrame, truths: pd.DataFrame, limit: int
) -> list[int]:
    """Worst images first, alternating largest count error and most labelled mistakes."""
    if limit <= 0:
        return []
    mistakes = (
        predictions.loc[predictions["outcome"] != PredictionOutcome.CORRECT.value, "image_id"]
        .value_counts()
        .add(
            truths.loc[truths["outcome"] != TruthOutcome.DETECTED.value, "image_id"].value_counts(),
            fill_value=0,
        )
    )
    ranked = counts.assign(
        abs_error=counts["error_total"].abs(),
        mistakes=counts["image_id"].map(mistakes).fillna(0),
    )
    by_count = ranked.sort_values(["abs_error", "mistakes"], ascending=False, kind="stable")
    by_mistakes = ranked.sort_values(["mistakes", "abs_error"], ascending=False, kind="stable")
    chosen: list[int] = []
    for image_id in _interleave(list(by_count["image_id"]), list(by_mistakes["image_id"])):
        if int(image_id) not in chosen:
            chosen.append(int(image_id))
        if len(chosen) == limit:
            break
    return chosen


def draw_image(
    image_path: str | Path,
    annotations: Sequence[Mapping[str, Any]],
    truth_outcomes: Sequence[str],
    results: Sequence[Mapping[str, Any]],
    prediction_outcomes: Sequence[str],
    names: Mapping[int, str],
) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    font = _font(13)
    measure = ImageDraw.Draw(image)
    lines = _legend_lines(measure, image.width, font)
    legend_height = LEGEND_PADDING * 2 + LEGEND_LINE * len(lines)
    canvas = Image.new("RGB", (image.width, image.height + legend_height), LEGEND_BACKGROUND)
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for annotation, outcome in zip(annotations, truth_outcomes, strict=True):
        box = xyxy(annotation["bbox"])
        if outcome == TruthOutcome.MISSED.value:
            tag, colour = OUTCOME_STYLE[outcome]
            draw.rectangle(box, outline=colour, width=4)
            label = f"{tag} {names[annotation['category_id']]}"
            _tag(draw, box, label, colour, font, below=True, limits=image.size)
        else:
            draw.rectangle(box, outline=TRUTH_COLOUR, width=1)
    for result, outcome in zip(results, prediction_outcomes, strict=True):
        box = xyxy(result["bbox"])
        tag, colour = OUTCOME_STYLE[outcome]
        draw.rectangle(box, outline=colour, width=2)
        label = f"{tag} {names[result['category_id']]} {result['score']:.2f}"
        _tag(draw, box, label, colour, font, below=outcome in TAGS_BELOW, limits=image.size)
    _draw_legend(draw, lines, image.height + LEGEND_PADDING, font)
    return canvas


def _interleave(first: Sequence[int], second: Sequence[int]) -> list[int]:
    merged: list[int] = []
    for a, b in zip(first, second, strict=False):
        merged += [a, b]
    return merged


def _tag(
    draw: ImageDraw.ImageDraw,
    box: Box,
    text: str,
    colour: str,
    font: Any,
    below: bool,
    limits: tuple[int, int],
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width, height = right - left + 6, bottom - top + 4
    x = min(max(box[0], 0), limits[0] - width)
    y = box[3] + 1 if below else box[1] - height - 1
    if y < 0:
        y = box[1] + 1
    if y + height > limits[1]:
        y = box[3] - height - 1
    draw.rectangle((x, y, x + width, y + height), fill=colour)
    ink = "#0b0b0b" if colour in DARK_TEXT_ON else "#ffffff"
    draw.text((x + 3, y + 2 - top), text, fill=ink, font=font)


def _legend_lines(
    draw: ImageDraw.ImageDraw, width: int, font: Any
) -> list[list[tuple[str | None, str, int]]]:
    lines: list[list[tuple[str | None, str, int]]] = [[]]
    x = LEGEND_PADDING
    for outcome, text in LEGEND:
        item_width = SWATCH + 6 + int(draw.textlength(text, font=font)) + 18
        if lines[-1] and x + item_width > width:
            lines.append([])
            x = LEGEND_PADDING
        lines[-1].append((outcome, text, x))
        x += item_width
    return lines


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[Sequence[tuple[str | None, str, int]]],
    top: int,
    font: Any,
) -> None:
    for number, line in enumerate(lines):
        y = top + number * LEGEND_LINE + 4
        for outcome, text, x in line:
            swatch = (x, y, x + SWATCH, y + SWATCH)
            if outcome is None:
                draw.rectangle(swatch, outline=TRUTH_COLOUR, width=1)
            else:
                draw.rectangle(swatch, fill=OUTCOME_STYLE[outcome][1])
            draw.text((x + SWATCH + 6, y), text, fill=LEGEND_INK, font=font)


def _font(size: int) -> Any:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()
