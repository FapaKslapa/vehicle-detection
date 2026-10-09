import json

import pandas as pd
import pytest

from vehicle_detection.data.coco_split import load_coco, split_coco_by_manifest
from vehicle_detection.data.manifest import ManifestError

COCO = {
    "images": [
        {"id": 1, "file_name": "image_0001.jpg", "width": 1024, "height": 1024},
        {"id": 2, "file_name": "image_0002.jpg", "width": 1024, "height": 1024},
        {"id": 3, "file_name": "image_0003.jpg", "width": 1024, "height": 1024},
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
        {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10]},
        {"id": 3, "image_id": 3, "category_id": 1, "bbox": [0, 0, 10, 10]},
        {"id": 4, "image_id": 3, "category_id": 1, "bbox": [20, 20, 10, 10]},
    ],
    "categories": [{"id": 1, "name": "car"}],
}


def manifest(*rows: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [name for name, _ in rows],
            "session_id": [f"s{index}" for index, _ in enumerate(rows)],
            "location": ["site"] * len(rows),
            "split": [split for _, split in rows],
        }
    )


def test_subsets_follow_the_manifest():
    frame = manifest(("image_0001", "train"), ("image_0002", "train"), ("image_0003", "val"))
    subsets = split_coco_by_manifest(COCO, frame)
    assert sorted(subsets) == ["train", "val"]
    assert [image["id"] for image in subsets["train"]["images"]] == [1, 2]
    assert [annotation["id"] for annotation in subsets["train"]["annotations"]] == [1, 2]
    assert [annotation["id"] for annotation in subsets["val"]["annotations"]] == [3, 4]
    assert subsets["val"]["categories"] == COCO["categories"]


def test_split_without_images_is_omitted():
    frame = manifest(("image_0001", "train"), ("image_0002", "train"), ("image_0003", "train"))
    assert sorted(split_coco_by_manifest(COCO, frame)) == ["train"]


def test_coco_image_missing_from_manifest():
    frame = manifest(("image_0001", "train"), ("image_0002", "train"))
    with pytest.raises(ManifestError, match="image_0003"):
        split_coco_by_manifest(COCO, frame)


def test_manifest_image_missing_from_coco():
    frame = manifest(
        ("image_0001", "train"),
        ("image_0002", "train"),
        ("image_0003", "val"),
        ("image_9999", "test"),
    )
    with pytest.raises(ManifestError, match="image_9999"):
        split_coco_by_manifest(COCO, frame)


def test_load_coco_roundtrip(tmp_path):
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(COCO))
    assert load_coco(path) == COCO


def test_load_coco_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="coco.json"):
        load_coco(tmp_path / "coco.json")


def test_load_coco_missing_keys(tmp_path):
    path = tmp_path / "coco.json"
    path.write_text(json.dumps({"images": []}))
    with pytest.raises(ValueError, match="annotations"):
        load_coco(path)
