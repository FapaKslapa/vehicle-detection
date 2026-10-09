from pathlib import Path

import pytest

from vehicle_detection.data.manifest import (
    ManifestError,
    assert_disjoint_sessions,
    load_manifest,
)

HEADER = "image_id,session_id,location,split"


def write(tmp_path: Path, *rows: str, header: str = HEADER) -> Path:
    path = tmp_path / "manifest.csv"
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def test_loads_valid_manifest(tmp_path):
    path = write(tmp_path, "a,s1,site1,train", "b,s2,site1,val", "c,s3,site2,test")
    frame = load_manifest(path)
    assert list(frame.columns) == ["image_id", "session_id", "location", "split"]
    assert frame["split"].tolist() == ["train", "val", "test"]


def test_missing_file(tmp_path):
    with pytest.raises(ManifestError, match="non trovato"):
        load_manifest(tmp_path / "absent.csv")


def test_missing_columns(tmp_path):
    path = write(tmp_path, "a,s1,train", header="image_id,session_id,split")
    with pytest.raises(ManifestError, match="location"):
        load_manifest(path)


def test_unknown_split(tmp_path):
    path = write(tmp_path, "a,s1,site1,validation")
    with pytest.raises(ManifestError, match="validation"):
        load_manifest(path)


def test_duplicate_image_ids(tmp_path):
    path = write(tmp_path, "a,s1,site1,train", "a,s1,site1,train")
    with pytest.raises(ManifestError, match="duplicati"):
        load_manifest(path)


def test_empty_values(tmp_path):
    path = write(tmp_path, "a,,site1,train")
    with pytest.raises(ManifestError, match="session_id"):
        load_manifest(path)


def test_session_in_two_splits(tmp_path):
    path = write(tmp_path, "a,s1,site1,train", "b,s1,site1,val", "c,s2,site1,test")
    with pytest.raises(ManifestError, match="s1"):
        load_manifest(path)


def test_assert_disjoint_sessions_accepts_clean_frame(tmp_path):
    frame = load_manifest(write(tmp_path, "a,s1,site1,train", "b,s2,site1,val"))
    assert_disjoint_sessions(frame)
