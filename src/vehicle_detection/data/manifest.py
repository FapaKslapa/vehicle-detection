from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("image_id", "session_id", "location", "split")
VALID_SPLITS = frozenset({"train", "val", "test"})


class ManifestError(ValueError):
    pass


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest non trovato: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ManifestError(f"colonne mancanti nel manifest: {', '.join(missing)}")
    frame = frame.loc[:, list(REQUIRED_COLUMNS)]
    blank = [column for column in REQUIRED_COLUMNS if (frame[column].str.strip() == "").any()]
    if blank:
        raise ManifestError(f"valori vuoti nel manifest, colonne: {', '.join(blank)}")
    unknown = sorted(set(frame["split"]) - VALID_SPLITS)
    if unknown:
        raise ManifestError(f"split non validi nel manifest: {', '.join(unknown)}")
    duplicated = sorted(set(frame.loc[frame["image_id"].duplicated(), "image_id"]))
    if duplicated:
        raise ManifestError(f"image_id duplicati nel manifest: {', '.join(duplicated)}")
    assert_disjoint_sessions(frame)
    return frame


def assert_disjoint_sessions(frame: pd.DataFrame) -> None:
    splits_per_session = frame.groupby("session_id")["split"].nunique()
    leaking = sorted(splits_per_session[splits_per_session > 1].index)
    if leaking:
        raise ManifestError(f"sessioni presenti in più split: {', '.join(leaking)}")
