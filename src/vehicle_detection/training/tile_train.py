from __future__ import annotations

from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vehicle_detection.device import best_device
from vehicle_detection.settings import TrackingSettings, load_tracking_settings

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "train"
DEVICE_ARGUMENTS = {"cuda": "0", "xpu": "xpu:0", "mps": "mps", "cpu": "cpu"}


class TileTrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    weights: str = Field(min_length=1)
    data: Path
    epochs: int = Field(default=50, ge=1)
    imgsz: int = Field(default=512, ge=32)
    batch: int = Field(default=16, ge=1)
    seed: int = 0
    name: str = Field(default="E3b-yolo26n-512", min_length=1)
    fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    project: Path = Path("runs")
    device: str | None = None
    resume: Path | None = None

    @field_validator("device", mode="before")
    @classmethod
    def _device_as_string(cls, value: Any) -> Any:
        return str(value) if isinstance(value, int) and not isinstance(value, bool) else value


def device_argument(device_type: str) -> str:
    return DEVICE_ARGUMENTS[device_type]


def enable_wandb(
    tracking: TrackingSettings,
    config: TileTrainConfig,
    ultralytics_settings: MutableMapping[str, Any] | None = None,
    wandb_init: Callable[..., Any] | None = None,
) -> None:
    if ultralytics_settings is None:
        from ultralytics import settings as ultralytics_settings
    enabled = tracking.wandb_mode != "disabled"
    ultralytics_settings["wandb"] = enabled
    if not enabled:
        return
    if wandb_init is None:
        import wandb

        wandb_init = wandb.init
    tracking.export_to_environment()
    wandb_init(
        project=tracking.wandb_project,
        entity=tracking.wandb_entity,
        name=config.name,
        config=config.model_dump(mode="json"),
        mode=tracking.wandb_mode,
    )


def train(
    config: TileTrainConfig,
    trainer_factory: Callable[[str], Any] | None = None,
    device: str | None = None,
    enable_tracking: Callable[[TrackingSettings, TileTrainConfig], None] = enable_wandb,
) -> Path:
    if not config.data.is_file():
        raise FileNotFoundError(f"data.yaml non trovato: {config.data}")
    if config.resume is not None and not config.resume.is_file():
        raise FileNotFoundError(f"checkpoint non trovato: {config.resume}")
    tracking = load_tracking_settings()
    if trainer_factory is None:
        from ultralytics import YOLO as trainer_factory
    enable_tracking(tracking, config)
    target = device or config.device or device_argument(best_device().type)
    if config.resume is not None:
        model = trainer_factory(str(config.resume))
        model.train(resume=True, device=target)
        return Path(model.trainer.save_dir) / "weights" / "best.pt"
    model = trainer_factory(config.weights)
    model.train(
        data=str(config.data),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        seed=config.seed,
        deterministic=True,
        fraction=config.fraction,
        device=target,
        project=str(config.project.resolve()),
        name=config.name,
        exist_ok=False,
    )
    return Path(model.trainer.save_dir) / "weights" / "best.pt"


def run(cfg: DictConfig, **kwargs: Any) -> Path:
    config = TileTrainConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    return train(config, **kwargs)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="tiles")
def main(cfg: DictConfig) -> None:
    print(run(cfg))


if __name__ == "__main__":
    main()
