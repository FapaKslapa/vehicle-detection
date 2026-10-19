import json
import os

import numpy as np
import pytest
from PIL import Image

from vehicle_detection.data.manifest import load_manifest
from vehicle_detection.data.scene_embedding import embed_with_rotations
from vehicle_detection.data.scene_split import build_scene_split, main
from vehicle_detection.data.scenes import SceneConfig

SIZE = 64
ANGLES = (90, 180, 270)
CONFIG = SceneConfig(neighbors=5, min_group_size=3, max_group_share=0.4, seed=0)


class GrayGrid:
    def __init__(self):
        self.calls = 0
        self.batches = []

    def __call__(self, images):
        self.calls += 1
        self.batches.append(len(images))
        gray = images.astype(np.float32).mean(axis=-1)
        n, h, w = gray.shape
        pooled = gray.reshape(n, 8, h // 8, 8, w // 8).mean(axis=(2, 4))
        pooled = pooled - pooled.mean(axis=(1, 2), keepdims=True)
        return pooled.reshape(n, -1)


def write_scenes(root, scenes=6, frames=10, rotated=3):
    images_dir = root / "images"
    images_dir.mkdir()
    images, annotations, names = [], [], []
    for scene in range(scenes):
        base = np.random.default_rng(100 + scene).integers(20, 235, (SIZE, SIZE), dtype=np.uint8)
        for frame in range(frames + rotated):
            rng = np.random.default_rng(1000 * scene + frame)
            gray = np.clip(base.astype(int) + rng.integers(-3, 4, base.shape), 0, 255)
            gray = gray.astype(np.uint8)
            if frame >= frames:
                gray = np.rot90(gray, k=1 + frame % 3)
            name = f"file_{len(names) + 1:05d}"
            Image.fromarray(np.stack([gray] * 3, axis=-1)).save(images_dir / f"{name}.png")
            names.append(name)
            image_id = len(names)
            images.append(
                {"id": image_id, "file_name": f"{name}.png", "width": SIZE, "height": SIZE}
            )
            for klass in (1 + scene % 3, 1 + (scene + frame) % 3):
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": klass,
                        "bbox": [4, 4, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                    }
                )
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": "c"}],
    }
    coco_path = root / "coco.json"
    coco_path.write_text(json.dumps(coco))
    return coco_path, images_dir, names


@pytest.fixture
def dataset(tmp_path):
    return write_scenes(tmp_path)


def run(dataset, tmp_path, **kwargs):
    coco_path, images_dir, _ = dataset
    return build_scene_split(
        coco_path,
        images_dir,
        tmp_path / "out" / "v1.csv",
        GrayGrid(),
        CONFIG,
        angles=ANGLES,
        size=16,
        **kwargs,
    )


def test_embedding_shapes_and_order(dataset):
    _, images_dir, names = dataset
    base, rotated = embed_with_rotations(
        images_dir, [f"{n}.png" for n in names], GrayGrid(), ANGLES, size=16, batch_size=7
    )
    assert base.shape == (len(names), 64)
    assert rotated.shape == (len(names), 3, 64)


def test_embedding_batches_respect_the_batch_size(dataset):
    _, images_dir, names = dataset
    embedder = GrayGrid()
    embed_with_rotations(
        images_dir, [f"{n}.png" for n in names[:10]], embedder, ANGLES, size=16, batch_size=4
    )
    assert max(embedder.batches) <= 4
    assert sum(embedder.batches) == 10 * (1 + len(ANGLES))


def test_embedding_missing_image_raises_italian_error(dataset):
    _, images_dir, names = dataset
    with pytest.raises(FileNotFoundError, match="immagini non trovate"):
        embed_with_rotations(images_dir, ["nope.png"], GrayGrid(), ANGLES, size=16)


def test_every_scene_becomes_one_session_including_rotated_copies(dataset, tmp_path):
    report = run(dataset, tmp_path)
    manifest = load_manifest(tmp_path / "out" / "v1.csv")
    assert manifest.session_id.nunique() == 6
    assert report.groups == 6
    for scene in range(6):
        ids = [f"file_{scene * 13 + k + 1:05d}" for k in range(13)]
        rows = manifest[manifest.image_id.isin(ids)]
        assert rows.session_id.nunique() == 1


def test_manifest_has_contract_columns_and_no_session_crosses_splits(dataset, tmp_path):
    run(dataset, tmp_path)
    manifest = load_manifest(tmp_path / "out" / "v1.csv")
    assert list(manifest.columns) == ["image_id", "session_id", "location", "split"]
    assert len(manifest) == 78
    assert (manifest.groupby("session_id").split.nunique() == 1).all()
    assert set(manifest.split) == {"train", "val", "test"}


def test_leakage_report_is_zero_for_clean_scenes_and_beats_random(dataset, tmp_path):
    report = run(dataset, tmp_path)
    assert report.leakage.near_duplicate_pairs > 0
    assert report.leakage.crossing_share == 0.0
    assert report.random_baseline_share == pytest.approx(0.465)


def test_report_file_is_written_next_to_the_manifest(dataset, tmp_path):
    run(dataset, tmp_path)
    recorded = json.loads((tmp_path / "out" / "v1.report.json").read_text())
    assert recorded["groups"] == 6
    assert set(recorded["split_images"]) == {"train", "val", "test"}
    assert recorded["leakage"]["crossing_share"] == 0.0
    assert set(recorded["class_share_per_split"]["train"]) == {"1", "2", "3"}


def test_same_seed_gives_same_manifest(dataset, tmp_path):
    run(dataset, tmp_path)
    first = (tmp_path / "out" / "v1.csv").read_text()
    run(dataset, tmp_path)
    assert (tmp_path / "out" / "v1.csv").read_text() == first


def test_embeddings_are_cached(dataset, tmp_path):
    coco_path, images_dir, _ = dataset
    cache = tmp_path / "cache.npz"
    first, second = GrayGrid(), GrayGrid()
    for embedder in (first, second):
        build_scene_split(
            coco_path,
            images_dir,
            tmp_path / "out" / "v1.csv",
            embedder,
            CONFIG,
            angles=ANGLES,
            size=16,
            cache_path=cache,
        )
    assert first.calls > 0
    assert second.calls == 0


def test_stale_cache_with_other_angles_is_ignored(dataset, tmp_path):
    coco_path, images_dir, _ = dataset
    cache = tmp_path / "cache.npz"
    build_scene_split(
        coco_path,
        images_dir,
        tmp_path / "a.csv",
        GrayGrid(),
        CONFIG,
        angles=(90,),
        size=16,
        cache_path=cache,
    )
    embedder = GrayGrid()
    build_scene_split(
        coco_path,
        images_dir,
        tmp_path / "b.csv",
        embedder,
        CONFIG,
        angles=ANGLES,
        size=16,
        cache_path=cache,
    )
    assert embedder.calls > 0


def test_missing_images_raise_italian_error(dataset, tmp_path):
    coco_path, images_dir, names = dataset
    (images_dir / f"{names[0]}.png").unlink()
    with pytest.raises(FileNotFoundError, match="immagini non trovate"):
        build_scene_split(
            coco_path,
            images_dir,
            tmp_path / "v1.csv",
            GrayGrid(),
            CONFIG,
            angles=ANGLES,
            size=16,
        )


def test_cli_runs_with_an_injected_embedder(dataset, tmp_path, capsys):
    coco_path, images_dir, _ = dataset
    main(
        [
            "--coco",
            str(coco_path),
            "--images",
            str(images_dir),
            "--out",
            str(tmp_path / "cli" / "v1.csv"),
            "--size",
            "16",
            "--angles",
            "90",
            "180",
            "270",
            "--neighbors",
            "5",
            "--min-group-size",
            "3",
            "--max-group-share",
            "0.4",
        ],
        embedder=GrayGrid(),
    )
    assert (tmp_path / "cli" / "v1.csv").is_file()
    assert "gruppi" in capsys.readouterr().out


@pytest.mark.skipif(not os.environ.get("VD_TEST_DINO"), reason="imposta VD_TEST_DINO=1")
def test_dino_embedder_returns_one_vector_per_image():
    import torch

    from vehicle_detection.data.scene_embedding import DinoV2Embedder

    embedder = DinoV2Embedder(torch.device("cpu"))
    images = np.random.default_rng(0).integers(0, 255, (2, 224, 224, 3), dtype=np.uint8)
    vectors = embedder(images)
    assert vectors.shape == (2, 384)
    assert np.isfinite(vectors).all()


def test_report_includes_the_leakage_curve_at_several_thresholds(dataset, tmp_path):
    report = run(dataset, tmp_path)
    assert list(report.leakage_curve) == ["0.85", "0.9", "0.95", "0.97"]
    assert all(value == 0.0 for value in report.leakage_curve.values())
    recorded = json.loads((tmp_path / "out" / "v1.report.json").read_text())
    assert recorded["leakage_curve"]["0.9"] == 0.0
