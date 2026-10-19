from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import Any

from sahi.slicing import get_slice_bboxes

from vehicle_detection.data.tiling_config import TilingConfig

SMALL_AREA = 32 * 32
MEDIUM_AREA = 96 * 96


class SizeBucket(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass(frozen=True, slots=True)
class SizeDistribution:
    counts: Mapping[str, int]
    shares: Mapping[str, float]
    median_side: float


@dataclass(frozen=True, slots=True)
class TilingCoverage:
    slice_size: int
    overlap_ratio: float
    tiles_per_image: float
    empty_tile_share: float
    whole: int
    cut: int
    lost: int

    @property
    def total(self) -> int:
        return self.whole + self.cut + self.lost


def size_bucket(area: float) -> SizeBucket:
    if area < SMALL_AREA:
        return SizeBucket.SMALL
    if area < MEDIUM_AREA:
        return SizeBucket.MEDIUM
    return SizeBucket.LARGE


def size_distribution(coco: Mapping[str, Any]) -> SizeDistribution:
    counts = {bucket.value: 0 for bucket in SizeBucket}
    sides: list[float] = []
    for annotation in coco["annotations"]:
        _, _, width, height = annotation["bbox"]
        counts[size_bucket(width * height).value] += 1
        sides.append((width * height) ** 0.5)
    total = sum(counts.values())
    shares = {name: (count / total if total else 0.0) for name, count in counts.items()}
    return SizeDistribution(
        counts=counts, shares=shares, median_side=float(median(sides)) if sides else 0.0
    )


def _overlap_area(box: Sequence[float], tile: Sequence[int]) -> float:
    x, y, width, height = box
    left, top = max(x, tile[0]), max(y, tile[1])
    right, bottom = min(x + width, tile[2]), min(y + height, tile[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def tiling_coverage(coco: Mapping[str, Any], config: TilingConfig) -> TilingCoverage:
    boxes_by_image: dict[int, list[Sequence[float]]] = {}
    for annotation in coco["annotations"]:
        boxes_by_image.setdefault(annotation["image_id"], []).append(annotation["bbox"])
    whole = cut = lost = tile_count = empty_tiles = 0
    for image in coco["images"]:
        tiles = get_slice_bboxes(
            image_height=image["height"],
            image_width=image["width"],
            slice_height=config.slice_size,
            slice_width=config.slice_size,
            auto_slice_resolution=False,
            overlap_height_ratio=config.overlap_ratio,
            overlap_width_ratio=config.overlap_ratio,
        )
        boxes = boxes_by_image.get(image["id"], [])
        tile_count += len(tiles)
        occupied = [False] * len(tiles)
        for box in boxes:
            area = box[2] * box[3]
            ratios = [_overlap_area(box, tile) / area if area else 0.0 for tile in tiles]
            for index, ratio in enumerate(ratios):
                if ratio >= config.min_area_ratio:
                    occupied[index] = True
            if any(ratio >= 1.0 - 1e-9 for ratio in ratios):
                whole += 1
            elif any(ratio >= config.min_area_ratio for ratio in ratios):
                cut += 1
            else:
                lost += 1
        empty_tiles += occupied.count(False)
    images = len(coco["images"])
    return TilingCoverage(
        slice_size=config.slice_size,
        overlap_ratio=config.overlap_ratio,
        tiles_per_image=tile_count / images if images else 0.0,
        empty_tile_share=empty_tiles / tile_count if tile_count else 0.0,
        whole=whole,
        cut=cut,
        lost=lost,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Statistiche di taglia e copertura dei tile")
    parser.add_argument("--coco", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--slice-size", type=int, nargs="+", default=[512, 640])
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--min-area-ratio", type=float, default=0.5)
    args = parser.parse_args(argv)
    coco = json.loads(Path(args.coco).read_text())
    report = {
        "sizes": asdict(size_distribution(coco)),
        "tiling": {
            str(size): asdict(
                tiling_coverage(
                    coco,
                    TilingConfig(
                        slice_size=size,
                        overlap_ratio=args.overlap,
                        min_area_ratio=args.min_area_ratio,
                    ),
                )
            )
            for size in args.slice_size
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"scritto {out}")


if __name__ == "__main__":
    main()
