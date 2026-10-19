import json

import pytest

from vehicle_detection.data.normalise import normalise_coco


def make(tmp_path, file_names, on_disk):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for name in on_disk:
        (images_dir / name).write_bytes(b"x")
    coco = {
        "images": [
            {"id": i + 1, "file_name": n, "width": 8, "height": 8} for i, n in enumerate(file_names)
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 5,
                "bbox": [1, 2, 3, 4],
                "area": 12,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 5, "name": "suv", "supercategory": "vehicle"}],
    }
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(coco))
    return path, images_dir


def test_extension_is_added(tmp_path):
    path, images_dir = make(
        tmp_path, ["file_00001", "file_00002"], ["file_00001.jpg", "file_00002.jpg"]
    )
    out = normalise_coco(path, images_dir, tmp_path / "out" / "coco.json").path
    names = [image["file_name"] for image in json.loads(out.read_text())["images"]]
    assert names == ["file_00001.jpg", "file_00002.jpg"]


def test_existing_extension_is_kept(tmp_path):
    path, images_dir = make(tmp_path, ["a.png"], ["a.png"])
    out = normalise_coco(path, images_dir, tmp_path / "coco_out.json").path
    assert json.loads(out.read_text())["images"][0]["file_name"] == "a.png"


def test_missing_image_raises_italian_error(tmp_path):
    path, images_dir = make(tmp_path, ["file_00001", "file_00002"], ["file_00001.jpg"])
    with pytest.raises(FileNotFoundError, match="immagini non trovate.*file_00002"):
        normalise_coco(path, images_dir, tmp_path / "coco_out.json")


def test_ambiguous_extensions_raise(tmp_path):
    path, images_dir = make(tmp_path, ["a"], ["a.jpg", "a.png"])
    with pytest.raises(ValueError, match="ambigu"):
        normalise_coco(path, images_dir, tmp_path / "coco_out.json")


def test_categories_and_annotations_untouched(tmp_path):
    path, images_dir = make(tmp_path, ["a"], ["a.jpg"])
    original = json.loads(path.read_text())
    out = json.loads(normalise_coco(path, images_dir, tmp_path / "coco_out.json").path.read_text())
    assert out["categories"] == original["categories"]
    assert out["annotations"] == original["annotations"]


def test_cli_writes_the_normalised_file(tmp_path, capsys):
    from vehicle_detection.data.normalise import main

    path, images_dir = make(tmp_path, ["file_00001"], ["file_00001.jpg"])
    out = tmp_path / "out" / "coco.json"
    main(["--coco", str(path), "--images", str(images_dir), "--out", str(out)])
    assert json.loads(out.read_text())["images"][0]["file_name"] == "file_00001.jpg"
    printed = capsys.readouterr().out
    assert "1 immagini" in printed
    assert "scartate" in printed


def annotated(tmp_path, boxes):
    path, images_dir = make(tmp_path, ["a"], ["a.jpg"])
    coco = json.loads(path.read_text())
    coco["images"][0].update(width=100, height=80)
    coco["annotations"] = [
        {
            "id": index,
            "image_id": 1,
            "category_id": 5,
            "bbox": list(box),
            "area": box[2] * box[3],
            "iscrowd": 0,
        }
        for index, box in enumerate(boxes, start=1)
    ]
    path.write_text(json.dumps(coco))
    return path, images_dir


def test_zero_size_boxes_are_dropped_and_counted(tmp_path):
    path, images_dir = annotated(tmp_path, [(10, 10, 20, 20), (30, 30, 0, 5), (40, 40, 5, 0)])
    report = normalise_coco(path, images_dir, tmp_path / "o.json")
    kept = json.loads(report.path.read_text())["annotations"]
    assert [a["id"] for a in kept] == [1]
    assert report.dropped == 2
    assert report.clipped == 0


def test_boxes_crossing_the_border_are_clipped_with_updated_area(tmp_path):
    path, images_dir = annotated(tmp_path, [(-1, 10, 21, 20), (90, 70, 20, 20)])
    report = normalise_coco(path, images_dir, tmp_path / "o.json")
    kept = json.loads(report.path.read_text())["annotations"]
    assert kept[0]["bbox"] == [0, 10, 20, 20]
    assert kept[0]["area"] == 400
    assert kept[1]["bbox"] == [90, 70, 10, 10]
    assert kept[1]["area"] == 100
    assert report.clipped == 2


def test_boxes_entirely_outside_the_image_are_dropped(tmp_path):
    path, images_dir = annotated(tmp_path, [(100, 10, 5, 5), (-20, 10, 10, 5)])
    report = normalise_coco(path, images_dir, tmp_path / "o.json")
    assert json.loads(report.path.read_text())["annotations"] == []
    assert report.dropped == 2


def test_valid_boxes_are_left_untouched(tmp_path):
    path, images_dir = annotated(tmp_path, [(10, 10, 20, 20)])
    original = json.loads(path.read_text())["annotations"]
    report = normalise_coco(path, images_dir, tmp_path / "o.json")
    assert json.loads(report.path.read_text())["annotations"] == original
    assert (report.dropped, report.clipped) == (0, 0)


def test_annotation_on_a_missing_image_is_a_clear_error(tmp_path):
    path, images_dir = make(tmp_path, ["a.png"], ["a.png"])
    coco = json.loads(path.read_text())
    coco["annotations"][0]["image_id"] = 99
    path.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="inesistenti.*99"):
        normalise_coco(path, images_dir, tmp_path / "out.json")
