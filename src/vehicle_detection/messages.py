from __future__ import annotations

from collections.abc import Sequence

PREVIEW_LIMIT = 5


def preview(items: Sequence[object], limit: int = PREVIEW_LIMIT) -> str:
    shown = ", ".join(str(item) for item in items[:limit])
    hidden = len(items) - limit
    return f"{shown} e altri {hidden}" if hidden > 0 else shown
