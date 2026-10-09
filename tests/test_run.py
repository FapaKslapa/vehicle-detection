import hashlib
import json

import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from synthetic import ImageSpec, make_detector, write_dataset
from vehicle_detection.data.manifest import ManifestError
from vehicle_detection.inference.models import FullImagePredictor, RemappedPredictor
from vehicle_detection.inference.run import (
    CONFIG_DIR,
    PredictionRun,
    build_predictor,
    describe_run,
    predict_split,
    run,
    write_run,
)

SPECS = (
    ImageSpec("a", 1024, 1024, "train", "s1", ((100, 100, 40, 40),)),
    ImageSpec("b", 1024, 1024, "val", "s2", ((300, 300, 40, 40), (600, 600, 50, 50))),
    ImageSpec("c", 1024, 1024, "val", "s2", ()),
    ImageSpec("d", 1024, 1024, "test", "s3", ((10, 10, 30, 30),)),
)


@pytest.fixture
def dataset(tmp_path):
    return write_dataset(tmp_path / "data", SPECS)


def predictor():
    return RemappedPredictor(FullImagePredictor(make_detector()), {0: 1})


def go(dataset, split="val", **kwargs):
    return predict_split(
        dataset.coco_path, dataset.images_dir, dataset.manifest_path, split, predictor(), **kwargs
    )


def test_results_use_coco_image_ids(dataset):
    result = go(dataset, "test")
    assert [(r["image_id"], r["category_id"]) for r in result.results] == [(4, 1)]


def test_only_requested_split_is_predicted(dataset):
    result = go(dataset, "val")
    assert result.image_count == 2
    assert {r["image_id"] for r in result.results} == {2}
    assert len(result.results) == 2


def test_empty_prediction_still_counts_the_image(dataset):
    result = go(dataset, "val")
    assert 3 not in {r["image_id"] for r in result.results}
    assert result.image_count == 2


def test_unknown_split_raises_italian_error(dataset):
    with pytest.raises(ValueError, match="split non presente nel manifest: holdout"):
        go(dataset, "holdout")


def test_missing_image_raises_italian_error(dataset):
    (dataset.images_dir / "b.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="immagini non trovate.*b.jpg"):
        go(dataset, "val")


def test_manifest_and_coco_must_agree(dataset):
    dataset.manifest_path.write_text("image_id,session_id,location,split\na,s1,site,train\n")
    with pytest.raises(ManifestError):
        go(dataset, "train")


def test_limit_caps_images_in_file_name_order(dataset):
    result = go(dataset, "val", limit=1)
    assert result.image_count == 1
    assert {r["image_id"] for r in result.results} == {2}


def test_non_positive_limit_is_rejected(dataset):
    with pytest.raises(ValueError, match="limit"):
        go(dataset, "val", limit=0)


def test_cost_report_excludes_warmup(dataset):
    calls = []

    class Counting:
        def predict(self, image):
            calls.append(1)
            return []

    result = predict_split(
        dataset.coco_path,
        dataset.images_dir,
        dataset.manifest_path,
        "val",
        Counting(),
        device=torch.device("cpu"),
        warmup=1,
    )
    assert len(calls) == 3
    assert result.costs.images == 2


def test_write_run_roundtrips_json(dataset, tmp_path):
    result = go(dataset, "val")
    write_run(result, tmp_path / "out")
    assert json.loads((tmp_path / "out" / "results.json").read_text()) == result.results
    cost = json.loads((tmp_path / "out" / "cost.json").read_text())
    assert cost["images"] == 2


def compose_config(dataset, tmp_path, *overrides):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="predict",
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


def fake_factory(kind, weights, space, confidence_threshold, device, torchvision_name="x"):
    model = make_detector()
    model.confidence_threshold = confidence_threshold
    return model


@pytest.mark.parametrize("mode", ["full", "sliced", "adaptive"])
def test_run_supports_every_mode(dataset, tmp_path, mode):
    cfg = compose_config(dataset, tmp_path, f"mode={mode}", "tiling.slice_size=512")
    outcome = run(cfg, model_factory=fake_factory)
    assert isinstance(outcome, PredictionRun)
    assert outcome.image_count == 2
    assert (tmp_path / "out" / "results.json").is_file()
    found = {r["image_id"] for r in outcome.results}
    assert found == {2}
    assert {r["category_id"] for r in outcome.results} == {1}


def test_unknown_mode_is_rejected(dataset, tmp_path):
    cfg = compose_config(dataset, tmp_path, "mode=quantum")
    with pytest.raises(ValueError, match="modalità"):
        run(cfg, model_factory=fake_factory)


def test_build_predictor_uses_coco_ids_for_each_kind(dataset, tmp_path):
    seen = {}

    def factory(kind, weights, space, confidence_threshold, device, torchvision_name="x"):
        seen["labels"] = dict(space.label_to_category)
        return make_detector()

    cfg: DictConfig = compose_config(dataset, tmp_path, "kind=torchvision")
    build_predictor(cfg, [{"id": 1, "name": "car"}], factory)
    assert seen["labels"] == {1: 1}


def test_select_split_subset_filters_images_and_annotations(dataset):
    from vehicle_detection.inference.run import select_split_subset

    subset = select_split_subset(dataset.coco_path, dataset.manifest_path, "val")
    assert [image["id"] for image in subset["images"]] == [2, 3]
    assert {annotation["image_id"] for annotation in subset["annotations"]} == {2}
    assert len(subset["annotations"]) == 2


def test_select_split_subset_applies_limit_in_file_name_order(dataset):
    from vehicle_detection.inference.run import select_split_subset

    subset = select_split_subset(dataset.coco_path, dataset.manifest_path, "val", limit=1)
    assert [image["id"] for image in subset["images"]] == [2]
    assert {annotation["image_id"] for annotation in subset["annotations"]} == {2}


def test_every_postprocess_config_is_valid():
    from vehicle_detection.inference.sliced import PostprocessConfig

    paths = sorted((CONFIG_DIR / "postprocess").glob("*.yaml"))
    assert {path.stem for path in paths} >= {"default", "eval"}
    for path in paths:
        PostprocessConfig.model_validate(yaml.safe_load(path.read_text()))


def test_predict_defaults_to_a_low_confidence_floor_for_ap(dataset, tmp_path):
    cfg = compose_config(dataset, tmp_path)
    assert cfg.postprocess.confidence_threshold <= 0.05


def test_write_run_records_provenance(dataset, tmp_path):
    write_run(go(dataset, "val"), tmp_path / "out", {"split": "val", "provisional": False})
    assert json.loads((tmp_path / "out" / "run.json").read_text()) == {
        "split": "val",
        "provisional": False,
    }


def test_describe_run_hashes_the_manifest_and_flags_provisional_splits(dataset, tmp_path):
    cfg = compose_config(dataset, tmp_path)
    description = describe_run(cfg)
    expected = hashlib.sha256(dataset.manifest_path.read_bytes()).hexdigest()
    assert description["manifest_sha256"] == expected
    assert description["provisional"] is False
    assert description["config"]["split"] == "val"
    provisional = dataset.manifest_path.with_name("provisional_v0.csv")
    provisional.write_bytes(dataset.manifest_path.read_bytes())
    flagged = describe_run(compose_config(dataset, tmp_path, f"manifest_path={provisional}"))
    assert flagged["provisional"] is True


def test_run_writes_the_provenance_file(dataset, tmp_path):
    run(compose_config(dataset, tmp_path, "mode=full"), model_factory=fake_factory)
    recorded = json.loads((tmp_path / "out" / "run.json").read_text())
    assert recorded["config"]["mode"] == "full"
    assert "manifest_sha256" in recorded
