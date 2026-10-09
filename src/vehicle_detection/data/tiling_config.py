from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TilingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    slice_size: int = Field(default=512, ge=64, le=4096)
    overlap_ratio: float = Field(default=0.2, ge=0.0, le=0.5)
    min_area_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    ignore_negative_samples: bool = False
    image_extension: Literal[".png", ".jpg"] = ".png"
