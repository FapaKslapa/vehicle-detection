import math

import pytest
from hydra import compose, initialize_config_dir

from synthetic import ImageSpec, make_detector, write_dataset
from vehicle_detection.experiments.compare import (
    CONFIG_DIR,
    Variant,
    compare,
    run,
    to_markdown,
    write_report,
)
from vehicle_detection.inference.detections import Detection
from vehicle_detection.inference.models import FullImagePredictor, RemappedPredictor

SPECS = (
    ImageSpec("a", 1024, 1024, "train", "s1", ((100, 100, 40, 40),)),
    ImageSpec("b", 1024, 1024, "val", "s2", ((100, 100, 20, 20), (500, 500, 100, 100))),
    ImageSpec("c", 1024, 1024, "val", "s2", ((300, 300, 100, 100),)),
    ImageSpec("d", 1024, 1024, "test", "s3", ((10, 10, 100, 100),)),
)


@pytest.fixture
def dataset(tmp_path):
    return write_dataset(tmp_path / "data", SPECS)


class MinSide:
    def __init__(self, predictor, min_side):
        self._predictor = predictor
        self._min_side = min_side

    def predict(self, image):
        return [
            d
            for d in self._predictor.predict(image)
            if min(d.box[2] - d.box[0], d.box[3] - d.box[1]) >= self._min_side
        ]


class Spurious:
    def __init__(self, predictor, score):
        self._predictor = predictor
        self._score = score

    def predict(self, image):
        return [*self._predictor.predict(image), Detection((900, 900, 950, 950), self._score, 1)]


def finder():
    return RemappedPredictor(FullImagePredictor(make_detector()), {0: 1})


def run_compare(dataset, variants, **kwargs):
    return compare(
        dataset.coco_path,
        dataset.images_dir,
        dataset.manifest_path,
        "val",
        variants,
        **kwargs,
    )


def test_one_row_per_variant_in_order(dataset):
    frame = run_compare(dataset, [Variant("full", finder()), Variant("sliced", finder())])
    assert list(frame["variant"]) == ["full", "sliced"]
    assert list(frame["images"]) == [2, 2]


def test_size_breakdown_separates_variants(dataset):
    blind = Variant("blind", MinSide(finder(), 30))
    sharp = Variant("sharp", finder())
    frame = run_compare(dataset, [blind, sharp]).set_index("variant")
    assert frame.loc["blind", "ap_small"] == pytest.approx(0.0)
    assert frame.loc["sharp", "ap_small"] == pytest.approx(1.0)
    assert frame.loc["blind", "ap_large"] == pytest.approx(1.0)
    assert frame.loc["sharp", "ap_large"] == pytest.approx(1.0)


def test_threshold_grid_picks_lowest_mae(dataset):
    noisy = Variant("noisy", Spurious(finder(), 0.3))
    frame = run_compare(dataset, [noisy], thresholds=(0.1, 0.5, 0.9))
    assert frame.loc[0, "threshold"] == 0.5
    assert frame.loc[0, "mae"] == 0.0


def test_fixed_threshold_is_used_without_tuning(dataset):
    noisy = Variant("noisy", Spurious(finder(), 0.3))
    frame = run_compare(dataset, [noisy], threshold=0.1)
    assert frame.loc[0, "threshold"] == 0.1
    assert frame.loc[0, "mae"] == pytest.approx(1.0)


def test_tuning_the_threshold_on_the_test_split_is_refused(dataset):
    with pytest.raises(ValueError, match="non va scelta sul test"):
        compare(
            dataset.coco_path,
            dataset.images_dir,
            dataset.manifest_path,
            "test",
            [Variant("full", finder())],
        )


def test_fixed_threshold_is_allowed_on_the_test_split(dataset):
    frame = compare(
        dataset.coco_path,
        dataset.images_dir,
        dataset.manifest_path,
        "test",
        [Variant("full", finder())],
        threshold=0.5,
    )
    assert frame.loc[0, "threshold_source"] == "fixed"


def test_threshold_source_says_how_the_threshold_was_chosen(dataset):
    tuned = run_compare(dataset, [Variant("full", finder())])
    assert tuned.loc[0, "threshold_source"] == "tuned_on_split"


def test_cost_columns_are_present(dataset):
    frame = run_compare(dataset, [Variant("full", finder())])
    assert frame.loc[0, "seconds_per_image"] >= 0
    assert frame.loc[0, "p95_seconds"] >= 0
    assert math.isnan(frame.loc[0, "peak_memory_mb"])


def test_empty_and_duplicate_variants_are_rejected(dataset):
    with pytest.raises(ValueError, match="almeno una variante"):
        run_compare(dataset, [])
    with pytest.raises(ValueError, match="duplicat"):
        run_compare(dataset, [Variant("x", finder()), Variant("x", finder())])


def test_markdown_has_header_and_one_line_per_variant(dataset):
    frame = run_compare(dataset, [Variant("full", finder()), Variant("sliced", finder())])
    lines = to_markdown(frame).splitlines()
    assert lines[0].startswith("| variant |")
    assert set(lines[1].replace("|", "").replace(" ", "")) <= {"-", ":"}
    assert len(lines) == 4
    assert "full" in lines[2]
    assert "sliced" in lines[3]


def test_markdown_shows_missing_values_as_na(dataset):
    frame = run_compare(dataset, [Variant("full", finder())])
    assert "n/a" in to_markdown(frame)


def test_write_report_creates_both_files(dataset, tmp_path):
    frame = run_compare(dataset, [Variant("full", finder())])
    write_report(frame, tmp_path / "report")
    assert (tmp_path / "report" / "comparison.csv").is_file()
    assert (tmp_path / "report" / "comparison.md").read_text().startswith("| variant |")


def fake_factory(kind, weights, space, confidence_threshold, device, torchvision_name="x"):
    return make_detector()


def test_run_builds_variants_from_config(dataset, tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="compare",
            overrides=[
                f"coco_path={dataset.coco_path}",
                f"images_dir={dataset.images_dir}",
                f"manifest_path={dataset.manifest_path}",
                f"out_dir={tmp_path / 'out'}",
                f"weights={tmp_path / 'w.pt'}",
                "device=cpu",
                "split=val",
                "tiling.slice_size=512",
            ],
        )
    frame = run(cfg, model_factory=fake_factory)
    assert list(frame["variant"]) == ["full", "sliced", "sliced-std", "adaptive"]
    assert (tmp_path / "out" / "comparison.md").is_file()


def compose_cfg(dataset, tmp_path, *overrides):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="compare",
            overrides=[
                f"coco_path={dataset.coco_path}",
                f"images_dir={dataset.images_dir}",
                f"manifest_path={dataset.manifest_path}",
                f"out_dir={tmp_path / 'out'}",
                f"weights={tmp_path / 'w.pt'}",
                "device=cpu",
                "split=val",
                *overrides,
            ],
        )


def test_default_confidence_floor_is_low_enough_for_ap(dataset, tmp_path):
    assert compose_cfg(dataset, tmp_path).postprocess.confidence_threshold <= 0.05


def test_thresholds_below_the_model_confidence_floor_are_rejected(dataset, tmp_path):
    cfg = compose_cfg(dataset, tmp_path, "postprocess.confidence_threshold=0.3")
    with pytest.raises(ValueError, match="sotto la soglia di confidenza"):
        run(cfg, model_factory=fake_factory)


def test_fixed_threshold_below_the_model_confidence_floor_is_rejected(dataset, tmp_path):
    cfg = compose_cfg(dataset, tmp_path, "postprocess.confidence_threshold=0.3", "threshold=0.1")
    with pytest.raises(ValueError, match="sotto la soglia di confidenza"):
        run(cfg, model_factory=fake_factory)


def test_run_writes_provenance_next_to_the_report(dataset, tmp_path):
    import json

    run(compose_cfg(dataset, tmp_path), model_factory=fake_factory)
    recorded = json.loads((tmp_path / "out" / "provenance.json").read_text())
    assert recorded["provisional"] is False
    assert "manifest_sha256" in recorded


def test_provisional_split_is_flagged_in_the_markdown_report(dataset, tmp_path):
    frame = run_compare(dataset, [Variant("full", finder())])
    write_report(frame, tmp_path / "report", {"provisional": True})
    first = (tmp_path / "report" / "comparison.md").read_text().splitlines()[0]
    assert "PROVISIONAL" in first
    assert (tmp_path / "report" / "provenance.json").is_file()


def test_merge_tuning_config_builds_every_variant(dataset, tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="compare_merge",
            overrides=[
                f"coco_path={dataset.coco_path}",
                f"images_dir={dataset.images_dir}",
                f"manifest_path={dataset.manifest_path}",
                f"out_dir={tmp_path / 'out'}",
                f"weights={tmp_path / 'w.pt'}",
                "device=cpu",
                "split=val",
                "tiling.slice_size=512",
                "limit=1",
            ],
        )
    frame = run(cfg, model_factory=fake_factory)
    names = list(frame["variant"])
    assert len(names) == len(set(names)) == 14
    assert "sliced-iou-30-c100" in names
