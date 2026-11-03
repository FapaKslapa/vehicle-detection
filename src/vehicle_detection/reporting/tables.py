from __future__ import annotations

import math
from typing import Any

import pandas as pd

NOT_AVAILABLE = "n/a"


def markdown_table(frame: pd.DataFrame, index: bool = False, digits: int = 3) -> str:
    if index:
        frame = frame.reset_index()
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    lines = [header, separator]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_format(value, digits) for value in row) + " |")
    return "\n".join(lines)


def _format(value: Any, digits: int) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return NOT_AVAILABLE if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)
