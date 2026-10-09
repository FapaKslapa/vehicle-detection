from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vehicle_detection.data.coco_split import load_coco

EXTENSIONS = (".jpg", ".jpeg", ".png")
PREVIEW_LIMIT = 5


@dataclass(frozen=True, slots=True)
class NormaliseReport:
    path: Path
    dropped: int
    clipped: int


def normalise_coco(
    coco_path: str | Path, images_dir: str | Path, out_path: str | Path
) -> NormaliseReport:
    images_dir = Path(images_dir)
    coco = load_coco(coco_path)
    resolved = _resolve_file_names(coco, images_dir)
    images = [{**image, "file_name": resolved[image["id"]]} for image in coco["images"]]
    annotations, dropped, clipped = _clean_annotations(coco["annotations"], images)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({**coco, "images": images, "annotations": annotations}))
    return NormaliseReport(path=out_path, dropped=dropped, clipped=clipped)


def _resolve_file_names(coco: Mapping[str, Any], images_dir: Path) -> dict[int, str]:
    missing: list[str] = []
    resolved: dict[int, str] = {}
    for image in coco["images"]:
        name = image["file_name"]
        if (images_dir / name).is_file():
            resolved[image["id"]] = name
            continue
        existing = [f"{name}{ext}" for ext in EXTENSIONS if (images_dir / f"{name}{ext}").is_file()]
        if len(existing) > 1:
            raise ValueError(f"estensione ambigua per {name}: {', '.join(existing)}")
        if not existing:
            missing.append(name)
            continue
        resolved[image["id"]] = existing[0]
    if missing:
        shown = ", ".join(sorted(missing)[:PREVIEW_LIMIT])
        hidden = len(missing) - PREVIEW_LIMIT
        suffix = f" e altri {hidden}" if hidden > 0 else ""
        raise FileNotFoundError(f"immagini non trovate in {images_dir}: {shown}{suffix}")
    return resolved


def _clean_annotations(
    annotations: Sequence[Mapping[str, Any]], images: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], int, int]:
    size_of = {image["id"]: (image["width"], image["height"]) for image in images}
    kept: list[dict[str, Any]] = []
    dropped = clipped = 0
    for annotation in annotations:
        width, height = size_of[annotation["image_id"]]
        x, y, w, h = annotation["bbox"]
        x_min, y_min = max(0, x), max(0, y)
        x_max, y_max = min(width, x + w), min(height, y + h)
        if w <= 0 or h <= 0 or x_max <= x_min or y_max <= y_min:
            dropped += 1
            continue
        if (x_min, y_min, x_max, y_max) != (x, y, x + w, y + h):
            clipped += 1
            box = [x_min, y_min, x_max - x_min, y_max - y_min]
            kept.append({**annotation, "bbox": box, "area": box[2] * box[3]})
        else:
            kept.append(dict(annotation))
    return kept, dropped, clipped


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepara il COCO: estensioni e box validi")
    parser.add_argument("--coco", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = normalise_coco(args.coco, args.images, args.out)
    count = len(json.loads(report.path.read_text())["images"])
    print(
        f"scritto {report.path}: {count} immagini, "
        f"{report.dropped} annotazioni scartate, {report.clipped} ritagliate ai bordi"
    )


if __name__ == "__main__":
    main()
