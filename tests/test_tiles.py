import json
from pathlib import Path

import pytest

import vehicle_detection.data.tiles as tiles_module
from synthetic import ImageSpec, write_dataset
from vehicle_detection.data.manifest import ManifestError
from vehicle_detection.data.tiles import (
    TileOrigin,
    build_tile_dataset,
    parse_tile_name,
)
from vehicle_detection.data.tiling_config import TilingConfig

SPECS = (
    ImageSpec("image_0001", 1024, 1024, "train", "s1", ((100, 100, 40, 40), (490, 100, 40, 40))),
    ImageSpec("image_0002", 1024, 1024, "train", "s1", ((600, 600, 40, 40),)),
    ImageSpec("image_0003", 1024, 1024, "val", "s2", ((300, 300, 40, 40),)),
    ImageSpec("image_0004", 1024, 1024, "test", "s3", ((800, 200, 40, 40),)),
)


def load_split(out_dir: Path, split: str) -> dict:
    return json.loads((out_dir / split / "annotations_coco.json").read_text())


def build(tmp_path, specs=SPECS, config=None, overwrite=False):
    dataset = write_dataset(tmp_path / "data", specs)
    out_dir = tmp_path / "tiles"
    report = build_tile_dataset(
        dataset.coco_path,
        dataset.images_dir,
        dataset.manifest_path,
        out_dir,
        config or TilingConfig(),
        overwrite=overwrite,
    )
    return report, out_dir, dataset


def test_parse_tile_name_with_underscores_in_stem():
    origin = parse_tile_name("image_0001_0_410_0_922_512.png")
    assert origin == TileOrigin("image_0001", 410, 0, 922, 512)


def test_parse_tile_name_rejects_unknown_format():
    with pytest.raises(ValueError, match="holiday.jpg"):
        parse_tile_name("holiday.jpg")


def test_tiles_stay_in_the_split_of_their_source_image(tmp_path):
    report, out_dir, _ = build(tmp_path)
    expected = {
        "train": {"image_0001", "image_0002"},
        "val": {"image_0003"},
        "test": {"image_0004"},
    }
    for split, stems in expected.items():
        coco = load_split(out_dir, split)
        found = {parse_tile_name(image["file_name"]).stem for image in coco["images"]}
        assert found == stems
        for image in coco["images"]:
            assert (out_dir / split / image["file_name"]).is_file()
    assert report.splits["train"].source_images == 2
    assert report.splits["train"].tiles == 18
    assert report.splits["val"].tiles == 9
    assert report.splits["test"].tiles == 9


def test_box_below_min_area_ratio_is_dropped_from_the_tile(tmp_path):
    def boxes_in_origin_tile(name, config):
        _, out_dir, _ = build(tmp_path / name, config=config)
        coco = load_split(out_dir, "train")
        tile = next(
            image
            for image in coco["images"]
            if parse_tile_name(image["file_name"]) == TileOrigin("image_0001", 0, 0, 512, 512)
        )
        return [a["bbox"] for a in coco["annotations"] if a["image_id"] == tile["id"]]

    kept = boxes_in_origin_tile("loose", TilingConfig(min_area_ratio=0.5))
    dropped = boxes_in_origin_tile("strict", TilingConfig(min_area_ratio=0.6))
    assert [490.0, 100.0, 22.0, 40.0] in kept
    assert [490.0, 100.0, 22.0, 40.0] not in dropped
    assert [100.0, 100.0, 40.0, 40.0] in dropped


def test_tile_boxes_map_back_inside_source_boxes(tmp_path):
    _, out_dir, _ = build(tmp_path)
    sources = {spec.name: spec.boxes for spec in SPECS}
    for split in ("train", "val", "test"):
        coco = load_split(out_dir, split)
        files = {image["id"]: image["file_name"] for image in coco["images"]}
        for annotation in coco["annotations"]:
            origin = parse_tile_name(files[annotation["image_id"]])
            x, y, w, h = annotation["bbox"]
            left, top = x + origin.x_min, y + origin.y_min
            right, bottom = left + w, top + h
            assert any(
                sx - 1 <= left and sy - 1 <= top and right <= sx + sw + 1 and bottom <= sy + sh + 1
                for sx, sy, sw, sh in sources[origin.stem]
            )


def test_ignore_negative_samples_removes_empty_tiles(tmp_path):
    report, _, _ = build(tmp_path, config=TilingConfig(ignore_negative_samples=True))
    assert report.splits["train"].empty_tiles == 0
    assert report.splits["train"].tiles < 18


def test_empty_tiles_are_counted_when_kept(tmp_path):
    report, _, _ = build(tmp_path)
    assert report.splits["val"].empty_tiles == 8


def test_small_and_non_square_images(tmp_path):
    specs = (
        ImageSpec("image_0010", 300, 300, "train", "a", ((100, 100, 40, 40),)),
        ImageSpec("image_0011", 1024, 700, "val", "b", ((600, 600, 40, 40),)),
    )
    report, out_dir, _ = build(tmp_path, specs=specs)
    assert report.splits["train"].tiles == 1
    assert report.splits["val"].tiles == 6
    origins = {
        parse_tile_name(image["file_name"]) for image in load_split(out_dir, "val")["images"]
    }
    assert TileOrigin("image_0011", 0, 188, 512, 700) in origins


def test_jpg_extension(tmp_path):
    _, out_dir, _ = build(tmp_path, config=TilingConfig(image_extension=".jpg"))
    names = [image["file_name"] for image in load_split(out_dir, "val")["images"]]
    assert names and all(name.endswith(".jpg") for name in names)


def test_session_in_two_splits_writes_nothing(tmp_path):
    specs = (
        ImageSpec("image_0001", 1024, 1024, "train", "s1", ((100, 100, 40, 40),)),
        ImageSpec("image_0002", 1024, 1024, "val", "s1", ((100, 100, 40, 40),)),
    )
    dataset = write_dataset(tmp_path / "data", specs)
    out_dir = tmp_path / "tiles"
    with pytest.raises(ManifestError, match="s1"):
        build_tile_dataset(
            dataset.coco_path, dataset.images_dir, dataset.manifest_path, out_dir, TilingConfig()
        )
    assert not out_dir.exists()


def test_populated_output_requires_overwrite(tmp_path):
    _, out_dir, dataset = build(tmp_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        build_tile_dataset(
            dataset.coco_path, dataset.images_dir, dataset.manifest_path, out_dir, TilingConfig()
        )
    report = build_tile_dataset(
        dataset.coco_path,
        dataset.images_dir,
        dataset.manifest_path,
        out_dir,
        TilingConfig(),
        overwrite=True,
    )
    assert report.splits["train"].tiles == 18


def test_stale_split_blocks_before_any_write(tmp_path):
    dataset = write_dataset(tmp_path / "data", SPECS)
    out_dir = tmp_path / "tiles"
    (out_dir / "test").mkdir(parents=True)
    (out_dir / "test" / "stale.png").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        build_tile_dataset(
            dataset.coco_path, dataset.images_dir, dataset.manifest_path, out_dir, TilingConfig()
        )
    assert not (out_dir / "train").exists()


def test_missing_images_dir(tmp_path):
    dataset = write_dataset(tmp_path / "data", SPECS)
    with pytest.raises(FileNotFoundError, match="absent"):
        build_tile_dataset(
            dataset.coco_path,
            tmp_path / "absent",
            dataset.manifest_path,
            tmp_path / "tiles",
            TilingConfig(),
        )


def test_missing_source_images_are_listed_before_any_write(tmp_path):
    dataset = write_dataset(tmp_path / "data", SPECS)
    (dataset.images_dir / "image_0003.jpg").unlink()
    (dataset.images_dir / "image_0004.jpg").unlink()
    out_dir = tmp_path / "tiles"
    with pytest.raises(FileNotFoundError, match=r"image_0003\.jpg.*image_0004\.jpg"):
        build_tile_dataset(
            dataset.coco_path, dataset.images_dir, dataset.manifest_path, out_dir, TilingConfig()
        )
    assert not out_dir.exists()


def failing_on_second_call(monkeypatch):
    real = tiles_module.slice_coco
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(tiles_module, "slice_coco", flaky)
    return real


def test_failed_overwrite_keeps_the_previous_dataset(tmp_path, monkeypatch):
    _, out_dir, dataset = build(tmp_path)
    before = sorted(path.name for path in (out_dir / "train").iterdir())
    failing_on_second_call(monkeypatch)
    with pytest.raises(RuntimeError, match="boom"):
        build_tile_dataset(
            dataset.coco_path,
            dataset.images_dir,
            dataset.manifest_path,
            out_dir,
            TilingConfig(),
            overwrite=True,
        )
    assert sorted(path.name for path in (out_dir / "train").iterdir()) == before
    assert sorted(path.name for path in out_dir.iterdir()) == ["test", "train", "val"]


def test_failed_first_build_leaves_nothing_and_can_be_retried(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path / "data", SPECS)
    out_dir = tmp_path / "tiles"
    real = failing_on_second_call(monkeypatch)
    with pytest.raises(RuntimeError, match="boom"):
        build_tile_dataset(
            dataset.coco_path, dataset.images_dir, dataset.manifest_path, out_dir, TilingConfig()
        )
    assert list(out_dir.iterdir()) == []
    monkeypatch.setattr(tiles_module, "slice_coco", real)
    report = build_tile_dataset(
        dataset.coco_path, dataset.images_dir, dataset.manifest_path, out_dir, TilingConfig()
    )
    assert report.splits["train"].tiles == 18


@pytest.mark.parametrize("bbox", [[10, 10, 0, 5], [10, 10, 5, 0], [10, 10, -3, 5], []])
def test_degenerate_boxes_are_rejected_with_a_clear_error(tmp_path, bbox):
    dataset = write_dataset(tmp_path / "data", SPECS)
    coco = json.loads(dataset.coco_path.read_text())
    coco["annotations"][0]["bbox"] = bbox
    dataset.coco_path.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="box degenerat.*id 1"):
        build_tile_dataset(
            dataset.coco_path,
            dataset.images_dir,
            dataset.manifest_path,
            tmp_path / "tiles",
            TilingConfig(),
        )
    assert not (tmp_path / "tiles").exists()
