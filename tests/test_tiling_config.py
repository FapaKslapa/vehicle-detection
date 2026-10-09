import pytest
from pydantic import ValidationError

from vehicle_detection.data.tiling_config import TilingConfig


def test_defaults():
    config = TilingConfig()
    assert config.slice_size == 512
    assert config.overlap_ratio == 0.2
    assert config.min_area_ratio == 0.5
    assert config.ignore_negative_samples is False
    assert config.image_extension == ".png"


@pytest.mark.parametrize(
    "overrides",
    [
        {"slice_size": 32},
        {"slice_size": 5000},
        {"overlap_ratio": -0.1},
        {"overlap_ratio": 0.9},
        {"min_area_ratio": 0.0},
        {"min_area_ratio": 1.5},
        {"image_extension": ".bmp"},
        {"unknown_field": 1},
    ],
)
def test_rejects_invalid_values(overrides):
    with pytest.raises(ValidationError):
        TilingConfig(**overrides)


def test_is_immutable():
    config = TilingConfig()
    with pytest.raises(ValidationError):
        config.slice_size = 640
