import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vehicle_detection.data.manifest import load_manifest
from vehicle_detection.data.provisional import (
    ProvisionalConfig,
    build_provisional_manifest,
    detect_sessions,
)

SIZE = 96


def scene(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(20, 235, size=(SIZE, SIZE), dtype=np.uint8)


def frame(base: np.ndarray, car_x: int) -> np.ndarray:
    image = base.copy()
    image[40:56, car_x : car_x + 16] = 255
    return image


def write_scenes(root: Path, counts: dict[int, int]) -> tuple[Path, Path, list[str]]:
    images_dir = root / "images"
    images_dir.mkdir()
    names: list[str] = []
    index = 0
    for seed, count in counts.items():
        base = scene(seed)
        for k in range(count):
            index += 1
            name = f"file_{index:05d}.jpg"
            Image.fromarray(frame(base, 2 + 5 * k)).convert("RGB").save(
                images_dir / name, quality=100
            )
            names.append(name)
    coco = {
        "images": [
            {"id": i + 1, "file_name": n, "width": SIZE, "height": SIZE}
            for i, n in enumerate(names)
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "car"}],
    }
    coco_path = root / "coco.json"
    coco_path.write_text(json.dumps(coco))
    return coco_path, images_dir, names


@pytest.fixture
def config():
    return ProvisionalConfig(thumb_size=24, link_threshold=8.0, seed=0)


def test_similar_frames_share_a_session_whatever_their_order(tmp_path, config):
    _, images_dir, names = write_scenes(tmp_path, {1: 4, 2: 4})
    shuffled = [names[i] for i in (5, 0, 6, 2, 7, 1, 4, 3)]
    sessions = dict(zip(shuffled, detect_sessions(images_dir, shuffled, config), strict=True))
    assert len({sessions[n] for n in names[:4]}) == 1
    assert len({sessions[n] for n in names[4:]}) == 1


def test_a_different_scene_gets_its_own_session(tmp_path, config):
    _, images_dir, names = write_scenes(tmp_path, {1: 3, 2: 3})
    sessions = detect_sessions(images_dir, names, config)
    assert sessions[0] != sessions[3]


def test_manifest_has_contract_columns_and_loads(tmp_path, config):
    coco_path, images_dir, _ = write_scenes(tmp_path, {s: 4 for s in range(1, 8)})
    out = tmp_path / "provisional.csv"
    frame_ = build_provisional_manifest(coco_path, images_dir, out, config)
    assert list(frame_.columns) == ["image_id", "session_id", "location", "split"]
    assert len(load_manifest(out)) == 28
    assert set(frame_["location"]) <= {"loc0", "loc1"}


def test_no_session_spans_two_splits(tmp_path, config):
    coco_path, images_dir, _ = write_scenes(tmp_path, {s: 4 for s in range(1, 11)})
    frame_ = build_provisional_manifest(coco_path, images_dir, tmp_path / "m.csv", config)
    assert (frame_.groupby("session_id")["split"].nunique() == 1).all()


def test_split_proportions_are_close_to_target(tmp_path, config):
    coco_path, images_dir, _ = write_scenes(tmp_path, {s: 2 for s in range(1, 21)})
    frame_ = build_provisional_manifest(coco_path, images_dir, tmp_path / "m.csv", config)
    shares = frame_["split"].value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.7, abs=0.1)
    assert shares["val"] == pytest.approx(0.15, abs=0.1)
    assert shares["test"] == pytest.approx(0.15, abs=0.1)


def test_same_seed_gives_same_manifest(tmp_path, config):
    coco_path, images_dir, _ = write_scenes(tmp_path, {s: 2 for s in range(1, 13)})
    first = build_provisional_manifest(coco_path, images_dir, tmp_path / "a.csv", config)
    second = build_provisional_manifest(coco_path, images_dir, tmp_path / "b.csv", config)
    assert first.equals(second)


def test_missing_image_raises_italian_error(tmp_path, config):
    coco_path, images_dir, names = write_scenes(tmp_path, {1: 3})
    (images_dir / names[0]).unlink()
    with pytest.raises(FileNotFoundError, match="immagini non trovate"):
        build_provisional_manifest(coco_path, images_dir, tmp_path / "m.csv", config)


def test_invalid_ratios_are_rejected():
    with pytest.raises(ValueError):
        ProvisionalConfig(train=0.9, val=0.2)


def test_cli_writes_the_manifest(tmp_path):
    from vehicle_detection.data.provisional import main

    coco_path, images_dir, _ = write_scenes(tmp_path, {s: 3 for s in range(1, 8)})
    out = tmp_path / "out" / "provisional.csv"
    main(
        [
            "--coco",
            str(coco_path),
            "--images",
            str(images_dir),
            "--out",
            str(out),
            "--thumb-size",
            "24",
        ]
    )
    assert len(load_manifest(out)) == 21
