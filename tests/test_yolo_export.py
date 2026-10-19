import json
from pathlib import Path

import pytest
import yaml

from synthetic import ImageSpec, write_dataset
from vehicle_detection.data.tiles import build_tile_dataset
from vehicle_detection.data.tiling_config import TilingConfig
from vehicle_detection.data.yolo_export import export_yolo


def make_tiles(root: Path, split: str, images: dict[str, list], categories) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    coco_images, annotations = [], []
    for index, (name, boxes) in enumerate(images.items(), start=1):
        (split_dir / name).write_bytes(b"jpg")
        coco_images.append({"id": index, "file_name": name, "width": 200, "height": 100})
        for category_id, bbox in boxes:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": index,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
    coco = {"images": coco_images, "annotations": annotations, "categories": categories}
    (split_dir / "annotations_coco.json").write_text(json.dumps(coco))


CATEGORIES = [{"id": 3, "name": "van"}, {"id": 7, "name": "car"}, {"id": 5, "name": "suv"}]


def label_lines(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text().splitlines()]


def test_label_lines_are_normalised_centre_boxes(tmp_path):
    make_tiles(tmp_path / "tiles", "train", {"t.jpg": [(3, [20, 10, 40, 20])]}, CATEGORIES)
    export_yolo(tmp_path / "tiles", tmp_path / "yolo", splits=("train",))
    (line,) = label_lines(tmp_path / "yolo" / "labels" / "train" / "t.txt")
    cls, cx, cy, w, h = line
    assert cls == "0"
    assert [float(v) for v in (cx, cy, w, h)] == pytest.approx([0.2, 0.2, 0.2, 0.2])


def test_empty_tile_gets_empty_label_file(tmp_path):
    make_tiles(tmp_path / "tiles", "train", {"e.jpg": []}, CATEGORIES)
    export_yolo(tmp_path / "tiles", tmp_path / "yolo", splits=("train",))
    assert (tmp_path / "yolo" / "labels" / "train" / "e.txt").read_text() == ""


def test_non_contiguous_category_ids_map_to_contiguous_indices(tmp_path):
    boxes = [(7, [0, 0, 10, 10]), (3, [0, 0, 10, 10]), (5, [0, 0, 10, 10])]
    make_tiles(tmp_path / "tiles", "train", {"t.jpg": boxes}, CATEGORIES)
    export_yolo(tmp_path / "tiles", tmp_path / "yolo", splits=("train",))
    classes = [line[0] for line in label_lines(tmp_path / "yolo" / "labels" / "train" / "t.txt")]
    assert classes == ["2", "0", "1"]


def test_data_yaml_lists_names_in_index_order(tmp_path):
    make_tiles(tmp_path / "tiles", "train", {"t.jpg": []}, CATEGORIES)
    make_tiles(tmp_path / "tiles", "val", {"v.jpg": []}, CATEGORIES)
    data = yaml.safe_load(export_yolo(tmp_path / "tiles", tmp_path / "yolo").read_text())
    assert data["names"] == {0: "van", 1: "suv", 2: "car"}
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert Path(data["path"]) == (tmp_path / "yolo").resolve()


def test_images_are_linked_not_copied(tmp_path):
    make_tiles(tmp_path / "tiles", "train", {"t.jpg": []}, CATEGORIES)
    export_yolo(tmp_path / "tiles", tmp_path / "yolo", splits=("train",))
    link = tmp_path / "yolo" / "images" / "train" / "t.jpg"
    assert link.is_symlink()
    assert link.read_bytes() == b"jpg"


def test_boxes_are_clipped_to_the_tile(tmp_path):
    make_tiles(tmp_path / "tiles", "train", {"t.jpg": [(3, [180, 90, 40, 30])]}, CATEGORIES)
    export_yolo(tmp_path / "tiles", tmp_path / "yolo", splits=("train",))
    (line,) = label_lines(tmp_path / "yolo" / "labels" / "train" / "t.txt")
    cx, cy, w, h = (float(v) for v in line[1:])
    assert cx + w / 2 <= 1.0 + 1e-9
    assert cy + h / 2 <= 1.0 + 1e-9


def test_missing_split_raises_italian_error(tmp_path):
    make_tiles(tmp_path / "tiles", "train", {"t.jpg": []}, CATEGORIES)
    with pytest.raises(FileNotFoundError, match="split.*val"):
        export_yolo(tmp_path / "tiles", tmp_path / "yolo", splits=("train", "val"))


def test_export_reruns_cleanly(tmp_path):
    make_tiles(tmp_path / "tiles", "train", {"t.jpg": []}, CATEGORIES)
    export_yolo(tmp_path / "tiles", tmp_path / "yolo", splits=("train",))
    export_yolo(tmp_path / "tiles", tmp_path / "yolo", splits=("train",))
    assert (tmp_path / "yolo" / "labels" / "train" / "t.txt").is_file()


def test_exports_real_tile_builder_output(tmp_path):
    specs = (
        ImageSpec("a", 1024, 1024, "train", "s1", ((100, 100, 40, 40),)),
        ImageSpec("b", 1024, 1024, "val", "s2", ()),
    )
    dataset = write_dataset(tmp_path / "d", specs)
    build_tile_dataset(
        dataset.coco_path,
        dataset.images_dir,
        dataset.manifest_path,
        tmp_path / "tiles",
        TilingConfig(image_extension=".jpg"),
    )
    data_yaml = export_yolo(tmp_path / "tiles", tmp_path / "yolo")
    labels = sorted((tmp_path / "yolo" / "labels" / "train").glob("*.txt"))
    assert len(labels) == 9
    assert sum(1 for path in labels if path.read_text().strip()) == 1
    assert data_yaml.is_file()


def test_cli_exports_and_prints_the_data_yaml(tmp_path, capsys):
    from vehicle_detection.data.yolo_export import main

    make_tiles(tmp_path / "tiles", "train", {"t.jpg": []}, CATEGORIES)
    make_tiles(tmp_path / "tiles", "val", {"v.jpg": []}, CATEGORIES)
    main(["--tiles", str(tmp_path / "tiles"), "--out", str(tmp_path / "yolo")])
    assert str((tmp_path / "yolo" / "data.yaml").resolve()) in capsys.readouterr().out


def test_degenerate_boxes_are_not_written_as_zero_width_labels(tmp_path):
    make_tiles(
        tmp_path / "tiles",
        "train",
        {"t.jpg": [(7, [10, 10, 0, 20]), (3, [10, 10, 20, 20])]},
        CATEGORIES,
    )
    make_tiles(tmp_path / "tiles", "val", {"v.jpg": []}, CATEGORIES)
    export_yolo(tmp_path / "tiles", tmp_path / "yolo")
    assert len(label_lines(tmp_path / "yolo" / "labels" / "train" / "t.txt")) == 1


def test_export_refuses_tiles_newer_than_an_existing_export_marker(tmp_path):
    from vehicle_detection.data.yolo_export import check_export_is_current

    make_tiles(tmp_path / "tiles", "train", {"t.jpg": []}, CATEGORIES)
    make_tiles(tmp_path / "tiles", "val", {"v.jpg": []}, CATEGORIES)
    data_yaml = export_yolo(tmp_path / "tiles", tmp_path / "yolo")
    check_export_is_current(tmp_path / "tiles", data_yaml.parent)
    import os
    import time

    future = time.time() + 100
    os.utime(tmp_path / "tiles" / "train" / "annotations_coco.json", (future, future))
    with pytest.raises(RuntimeError, match="yolo-export"):
        check_export_is_current(tmp_path / "tiles", data_yaml.parent)
