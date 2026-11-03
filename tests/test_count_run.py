import json

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from PIL import Image

from synthetic import ImageSpec, make_detector, write_dataset
from vehicle_detection.counting.run import count_images, list_images, run
from vehicle_detection.inference.detections import Detection
from vehicle_detection.inference.run import CONFIG_DIR

SPECS = (
    ImageSpec("a", 1024, 1024, "train", "s1", ((100, 100, 40, 40),)),
    ImageSpec("b", 1024, 1024, "val", "s2", ((300, 300, 40, 40), (600, 600, 50, 50))),
    ImageSpec("c", 1024, 1024, "val", "s2", ()),
    ImageSpec("d", 1024, 1024, "test", "s3", ((200, 200, 30, 30),)),
)
EXPECTED = {"a.jpg": 1, "b.jpg": 2, "c.jpg": 0, "d.jpg": 1}
TWO_CLASSES = [{"id": 1, "name": "car"}, {"id": 2, "name": "sedan"}]


@pytest.fixture
def dataset(tmp_path):
    return write_dataset(tmp_path / "data", SPECS)


def compose_config(dataset, tmp_path, *overrides):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="count",
            overrides=[
                f"images_dir={dataset.images_dir}",
                f"categories_from={dataset.coco_path}",
                f"weights={tmp_path / 'w.pt'}",
                f"out_dir={tmp_path / 'out'}",
                "device=cpu",
                *overrides,
            ],
        )


def fake_factory(
    kind, weights, space, confidence_threshold, device, torchvision_name="x", input_size=None
):
    model = make_detector()
    model.confidence_threshold = confidence_threshold
    return model


def counts_by_file(outcome):
    return dict(zip(outcome.counts["file_name"], outcome.counts["total"], strict=True))


@pytest.mark.parametrize("mode", ["full", "sliced", "adaptive"])
def test_every_image_in_the_folder_is_counted_in_every_mode(dataset, tmp_path, mode):
    cfg = compose_config(dataset, tmp_path, f"mode={mode}", "tiling.slice_size=512")
    outcome = run(cfg, model_factory=fake_factory)
    assert counts_by_file(outcome) == EXPECTED
    assert outcome.totals()["total"] == 4
    assert outcome.totals()["car"] == 4


def test_the_outputs_are_written(dataset, tmp_path):
    run(compose_config(dataset, tmp_path), model_factory=fake_factory)
    out = tmp_path / "out"
    counts = pd.read_csv(out / "counts.csv")
    assert list(counts.columns) == ["image_id", "file_name", "total", "car"]
    assert len(counts) == 4
    detections = json.loads((out / "detections.json").read_text())
    assert len(detections["images"]) == 4
    assert len(detections["detections"]) == 4
    assert detections["categories"] == [{"id": 1, "name": "car"}]
    record = json.loads((out / "run.json").read_text())
    assert record["config"]["threshold"] == pytest.approx(0.35)
    assert record["totals"]["total"] == 4
    assert record["costs"]["images"] == 4
    assert record["weights_sha256"] is None


def test_images_get_their_real_size(dataset, tmp_path):
    outcome = run(compose_config(dataset, tmp_path), model_factory=fake_factory)
    assert {(i["width"], i["height"]) for i in outcome.images} == {(1024, 1024)}


def test_boxes_below_the_threshold_are_not_counted(dataset, tmp_path):
    outcome = run(compose_config(dataset, tmp_path, "threshold=0.95"), model_factory=fake_factory)
    assert outcome.totals()["total"] == 0
    assert len(outcome.counts) == 4


def test_a_threshold_below_the_detector_floor_is_refused_before_the_model_is_built(
    dataset, tmp_path
):
    built = []

    def factory(*args, **kwargs):
        built.append(True)
        return fake_factory(*args, **kwargs)

    cfg = compose_config(dataset, tmp_path, "threshold=0.3", "postprocess.confidence_threshold=0.5")
    with pytest.raises(ValueError, match="confidence floor"):
        run(cfg, model_factory=factory)
    assert not built


def test_an_empty_folder_is_refused_before_the_model_is_built(dataset, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    built = []

    def factory(*args, **kwargs):
        built.append(True)
        return fake_factory(*args, **kwargs)

    with pytest.raises(FileNotFoundError, match="no images"):
        run(compose_config(dataset, tmp_path, f"images_dir={empty}"), model_factory=factory)
    assert not built


def test_other_files_in_the_folder_are_ignored(tmp_path):
    for name in ("b.PNG", "a.jpg", "notes.txt", "c.jpeg"):
        (tmp_path / name).write_bytes(b"")
    assert [path.name for path in list_images(tmp_path)] == ["a.jpg", "b.PNG", "c.jpeg"]


class TwoBoxesPerImage:
    """Puts a "car" and a weaker "sedan" box on the same spot, like per-class NMS can."""

    def predict(self, image):
        return [Detection((10, 10, 50, 50), 0.9, 1), Detection((11, 10, 51, 50), 0.6, 2)]


def write_images(folder, count):
    folder.mkdir()
    for index in range(count):
        Image.new("RGB", (64, 64)).save(folder / f"{index}.png")
    return list_images(folder)


def test_a_vehicle_found_as_two_classes_is_counted_once(tmp_path):
    paths = write_images(tmp_path / "images", 2)
    outcome = count_images(paths, TwoBoxesPerImage(), TWO_CLASSES, 0.35, 0.5, warmup=0)
    assert outcome.totals() == {"total": 2, "car": 2, "sedan": 0}


def test_without_dedupe_the_vehicle_is_counted_twice(tmp_path):
    paths = write_images(tmp_path / "images", 2)
    outcome = count_images(paths, TwoBoxesPerImage(), TWO_CLASSES, 0.35, None, warmup=0)
    assert outcome.totals() == {"total": 4, "car": 2, "sedan": 2}


@pytest.mark.parametrize("dedupe_iou", [0.0, 1.5])
def test_a_dedupe_iou_outside_zero_to_one_is_refused(tmp_path, dedupe_iou):
    paths = write_images(tmp_path / "images", 1)
    with pytest.raises(ValueError, match="dedupe_iou"):
        count_images(paths, TwoBoxesPerImage(), TWO_CLASSES, 0.35, dedupe_iou, warmup=0)


def test_the_default_config_is_the_setting_chosen_on_val(dataset, tmp_path):
    cfg = compose_config(dataset, tmp_path)
    assert cfg.threshold == pytest.approx(0.35)
    assert cfg.dedupe_iou == pytest.approx(0.5)
    assert cfg.mode == "full"
    assert cfg.postprocess.confidence_threshold <= cfg.threshold


def test_dedupe_can_be_turned_off_from_the_command_line(dataset, tmp_path):
    cfg = compose_config(dataset, tmp_path, "dedupe_iou=null")
    assert run(cfg, model_factory=fake_factory).totals()["total"] == 4
