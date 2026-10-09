from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pydantic import ValidationError

from synthetic import ImageSpec, write_dataset
from vehicle_detection.data.build_tiles import CONFIG_DIR, run
from vehicle_detection.data.tiling_config import TilingConfig

SPECS = (
    ImageSpec("image_0001", 1024, 1024, "train", "s1", ((100, 100, 40, 40),)),
    ImageSpec("image_0002", 1024, 1024, "val", "s2", ((300, 300, 40, 40),)),
)


def compose_config(*overrides: str) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="tiles", overrides=list(overrides))


def dataset_overrides(tmp_path: Path) -> list[str]:
    dataset = write_dataset(tmp_path / "data", SPECS)
    return [
        f"coco_path={dataset.coco_path}",
        f"images_dir={dataset.images_dir}",
        f"manifest_path={dataset.manifest_path}",
        f"out_dir={tmp_path / 'tiles'}",
    ]


@pytest.mark.parametrize(
    "path", sorted((CONFIG_DIR / "tiling").glob("*.yaml")), ids=lambda path: path.stem
)
def test_every_tiling_config_is_valid(path):
    TilingConfig.model_validate(yaml.safe_load(path.read_text()))


def test_config_dir_exists():
    assert (CONFIG_DIR / "tiles.yaml").is_file()


def test_run_builds_the_tiles(tmp_path):
    report = run(compose_config(*dataset_overrides(tmp_path)))
    assert report.splits["train"].tiles == 9
    assert (tmp_path / "tiles" / "val" / "annotations_coco.json").is_file()


def test_tiling_group_override_changes_the_slice_size(tmp_path):
    cfg = compose_config(*dataset_overrides(tmp_path), "tiling=e3_640")
    assert OmegaConf.to_container(cfg.tiling)["slice_size"] == 640
    assert run(cfg).splits["train"].tiles == 4


def test_invalid_tiling_values_fail_before_any_work(tmp_path):
    cfg = compose_config(*dataset_overrides(tmp_path), "tiling.overlap_ratio=0.9")
    with pytest.raises(ValidationError):
        run(cfg)
    assert not (tmp_path / "tiles").exists()


def test_missing_required_paths():
    with pytest.raises(MissingMandatoryValue):
        run(compose_config())
