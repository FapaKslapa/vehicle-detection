import pytest

from vehicle_detection.settings import (
    ConfigError,
    load_storage_settings,
    load_tracking_settings,
)

STORAGE_ENV = {
    "S3_ENDPOINT_URL": "http://localhost:3900/",
    "S3_BUCKET": "vehicle-detection",
    "AWS_ACCESS_KEY_ID": "GK0123456789abcdef",
    "AWS_SECRET_ACCESS_KEY": "0123456789abcdef0123456789abcdef",
}


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in [*STORAGE_ENV, "S3_REGION", "WANDB_API_KEY", "WANDB_MODE", "WANDB_PROJECT"]:
        monkeypatch.delenv(name, raising=False)


def set_env(monkeypatch, **overrides):
    for name, value in {**STORAGE_ENV, **overrides}.items():
        monkeypatch.setenv(name, value)


def test_storage_valid(monkeypatch):
    set_env(monkeypatch)
    settings = load_storage_settings()
    assert settings.endpoint == "http://localhost:3900"
    assert settings.s3_region == "garage"
    assert "GK0123" not in repr(settings)


def test_storage_reports_every_missing_variable():
    with pytest.raises(ConfigError) as error:
        load_storage_settings()
    message = str(error.value)
    for name in STORAGE_ENV:
        assert name in message


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("S3_ENDPOINT_URL", "garage:3900"),
        ("S3_BUCKET", "Bad_Bucket"),
        ("AWS_ACCESS_KEY_ID", "short"),
        ("AWS_SECRET_ACCESS_KEY", "0123456789abcdef0123456789abcdef\n"),
    ],
)
def test_storage_rejects_malformed(monkeypatch, name, value):
    set_env(monkeypatch, **{name: value})
    with pytest.raises(ConfigError, match=name):
        load_storage_settings()


def test_tracking_online_needs_key(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    with pytest.raises(ConfigError, match="WANDB_API_KEY"):
        load_tracking_settings()


def test_tracking_offline_needs_no_key(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert load_tracking_settings().wandb_mode == "offline"


def test_tracking_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "sometimes")
    with pytest.raises(ConfigError, match="WANDB_MODE"):
        load_tracking_settings()


def test_tracking_export_sets_the_wandb_environment(monkeypatch):
    import os

    for name in ("WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_API_KEY"):
        monkeypatch.setenv(name, "placeholder")
    monkeypatch.delenv("WANDB_API_KEY")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_PROJECT", "demo")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    settings = load_tracking_settings()
    for name in ("WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY"):
        monkeypatch.delenv(name)
    settings.export_to_environment()
    assert os.environ["WANDB_MODE"] == "offline"
    assert os.environ["WANDB_PROJECT"] == "demo"
    assert os.environ["WANDB_ENTITY"] == "team"
    assert "WANDB_API_KEY" not in os.environ
    for name in ("WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY"):
        monkeypatch.setenv(name, os.environ[name])


def test_tracking_export_includes_the_api_key_when_present(monkeypatch):
    import os

    monkeypatch.setenv("WANDB_API_KEY", "k" * 40)
    monkeypatch.setenv("WANDB_MODE", "online")
    settings = load_tracking_settings()
    monkeypatch.delenv("WANDB_API_KEY")
    settings.export_to_environment()
    assert os.environ["WANDB_API_KEY"] == "k" * 40
    monkeypatch.setenv("WANDB_API_KEY", os.environ["WANDB_API_KEY"])
