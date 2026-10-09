from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from vehicle_detection.settings import ConfigError
from vehicle_detection.training.tile_train import (
    CONFIG_DIR,
    TileTrainConfig,
    device_argument,
    run,
    train,
)


class FakeModel:
    instances: list["FakeModel"] = []

    def __init__(self, weights):
        self.weights = weights
        self.train_kwargs = None
        self.trainer = None
        FakeModel.instances.append(self)

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        self.trainer = SimpleNamespace(save_dir=save_dir)
        return None


@pytest.fixture(autouse=True)
def environment(monkeypatch, tmp_path):
    FakeModel.instances.clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)


@pytest.fixture
def data_yaml(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("names: {0: car}\n")
    return path


def make_config(data_yaml, **overrides):
    return TileTrainConfig(weights="yolo26n.pt", data=data_yaml, **overrides)


def noop_tracking(settings, config):
    return None


def test_train_passes_config_to_ultralytics(data_yaml, tmp_path):
    config = make_config(data_yaml, epochs=3, imgsz=640, batch=4, fraction=0.5, name="x")
    best = train(config, trainer_factory=FakeModel, device="cpu", enable_tracking=noop_tracking)
    kwargs = FakeModel.instances[0].train_kwargs
    assert FakeModel.instances[0].weights == "yolo26n.pt"
    assert kwargs["data"] == str(data_yaml)
    assert (kwargs["epochs"], kwargs["imgsz"], kwargs["batch"], kwargs["fraction"]) == (
        3,
        640,
        4,
        0.5,
    )
    assert best == Path("runs").resolve() / "x" / "weights" / "best.pt"


def test_seed_and_deterministic_are_set(data_yaml):
    train(
        make_config(data_yaml, seed=7),
        trainer_factory=FakeModel,
        device="cpu",
        enable_tracking=noop_tracking,
    )
    kwargs = FakeModel.instances[0].train_kwargs
    assert kwargs["seed"] == 7
    assert kwargs["deterministic"] is True


def test_device_argument_maps_each_backend():
    assert device_argument("cuda") == "0"
    assert device_argument("mps") == "mps"
    assert device_argument("xpu") == "xpu:0"
    assert device_argument("cpu") == "cpu"


def test_device_comes_from_the_device_module(data_yaml, monkeypatch):
    import torch

    monkeypatch.setattr(
        "vehicle_detection.training.tile_train.best_device", lambda: torch.device("mps")
    )
    train(make_config(data_yaml), trainer_factory=FakeModel, enable_tracking=noop_tracking)
    assert FakeModel.instances[0].train_kwargs["device"] == "mps"


@pytest.mark.parametrize("field", ["epochs", "imgsz", "batch"])
def test_non_positive_integers_rejected(data_yaml, field):
    with pytest.raises(ValidationError):
        make_config(data_yaml, **{field: 0})


def test_fraction_must_be_in_unit_interval(data_yaml):
    with pytest.raises(ValidationError):
        make_config(data_yaml, fraction=1.5)


def test_missing_data_yaml_raises_italian_error(tmp_path):
    config = TileTrainConfig(weights="yolo26n.pt", data=tmp_path / "missing.yaml")
    with pytest.raises(FileNotFoundError, match="data.yaml non trovato"):
        train(config, trainer_factory=FakeModel, device="cpu", enable_tracking=noop_tracking)


def test_tracking_settings_are_validated_before_training(data_yaml, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    with pytest.raises(ConfigError, match="WANDB_API_KEY"):
        train(make_config(data_yaml), trainer_factory=FakeModel, device="cpu")
    assert FakeModel.instances == []


def test_tracking_hook_receives_validated_settings(data_yaml):
    received = []
    train(
        make_config(data_yaml),
        trainer_factory=FakeModel,
        device="cpu",
        enable_tracking=lambda settings, config: received.append((settings, config)),
    )
    assert received[0][0].wandb_mode == "offline"
    assert received[0][1].name == "E3b-yolo26n-512"


def test_hydra_config_composes_and_validates(data_yaml):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="tiles", overrides=[f"data={data_yaml}", "epochs=2"])
    best = run(cfg, trainer_factory=FakeModel, device="cpu", enable_tracking=noop_tracking)
    assert FakeModel.instances[0].train_kwargs["epochs"] == 2
    assert best.name == "best.pt"


def make_tracking(monkeypatch, mode):
    from vehicle_detection.settings import TrackingSettings

    monkeypatch.setenv("WANDB_MODE", mode)
    monkeypatch.setenv("WANDB_PROJECT", "demo")
    return TrackingSettings()


def test_default_tracking_hook_enables_wandb_in_ultralytics(monkeypatch, data_yaml):
    from vehicle_detection.training.tile_train import enable_wandb

    ultralytics_settings = {"wandb": False}
    enable_wandb(
        make_tracking(monkeypatch, "offline"),
        make_config(data_yaml),
        ultralytics_settings,
        wandb_init=lambda **kwargs: None,
    )
    assert ultralytics_settings["wandb"] is True


def test_default_tracking_hook_respects_disabled_mode(monkeypatch, data_yaml):
    from vehicle_detection.training.tile_train import enable_wandb

    calls = []
    ultralytics_settings = {"wandb": True}
    enable_wandb(
        make_tracking(monkeypatch, "disabled"),
        make_config(data_yaml),
        ultralytics_settings,
        wandb_init=lambda **kwargs: calls.append(kwargs),
    )
    assert ultralytics_settings["wandb"] is False
    assert calls == []


def test_default_tracking_hook_starts_the_run_with_the_configured_project(monkeypatch, data_yaml):
    from vehicle_detection.training.tile_train import enable_wandb

    calls = []
    enable_wandb(
        make_tracking(monkeypatch, "offline"),
        make_config(data_yaml, name="E3b-test"),
        {"wandb": False},
        wandb_init=lambda **kwargs: calls.append(kwargs),
    )
    assert calls[0]["project"] == "demo"
    assert calls[0]["name"] == "E3b-test"
    assert calls[0]["mode"] == "offline"
    assert calls[0]["config"]["epochs"] == 50
