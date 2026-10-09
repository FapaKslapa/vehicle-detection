from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from vehicle_detection.data.tiles import TileReport, build_tile_dataset
from vehicle_detection.data.tiling_config import TilingConfig

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def run(cfg: DictConfig) -> TileReport:
    tiling = TilingConfig.model_validate(OmegaConf.to_container(cfg.tiling, resolve=True))
    return build_tile_dataset(
        coco_path=cfg.coco_path,
        images_dir=cfg.images_dir,
        manifest_path=cfg.manifest_path,
        out_dir=cfg.out_dir,
        config=tiling,
        overwrite=cfg.overwrite,
    )


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="tiles")
def main(cfg: DictConfig) -> None:
    report = run(cfg)
    for split, split_report in report.splits.items():
        print(
            f"{split}: {split_report.source_images} immagini, {split_report.tiles} tile, "
            f"{split_report.annotations} box, {split_report.empty_tiles} tile vuoti"
        )


if __name__ == "__main__":
    main()
