from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

ANNOTATION_FILE = "annotations_coco.json"
DEFAULT_SPLITS = ("train", "val")


def export_yolo(
    tiles_dir: str | Path,
    out_dir: str | Path,
    splits: Sequence[str] = DEFAULT_SPLITS,
) -> Path:
    tiles_dir = Path(tiles_dir)
    out_dir = Path(out_dir).resolve()
    missing = [split for split in splits if not (tiles_dir / split / ANNOTATION_FILE).is_file()]
    if missing:
        raise FileNotFoundError(f"split di tile non trovati in {tiles_dir}: {', '.join(missing)}")
    cocos = {
        split: json.loads((tiles_dir / split / ANNOTATION_FILE).read_text()) for split in splits
    }
    categories = _merge_categories(cocos)
    index_of = {category_id: index for index, category_id in enumerate(sorted(categories))}
    for split, coco in cocos.items():
        _write_split(tiles_dir / split, out_dir, split, coco, index_of)
    names = {index_of[category_id]: categories[category_id] for category_id in categories}
    data = {"path": str(out_dir), "names": dict(sorted(names.items()))}
    data.update({split: f"images/{split}" for split in splits})
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return data_yaml


def _merge_categories(cocos: Mapping[str, Mapping[str, Any]]) -> dict[int, str]:
    merged: dict[int, str] = {}
    for split, coco in cocos.items():
        for category in coco["categories"]:
            known = merged.setdefault(category["id"], category["name"])
            if known != category["name"]:
                raise ValueError(
                    f"categoria {category['id']} con nomi diversi tra gli split: "
                    f"{known} e {category['name']} ({split})"
                )
    return merged


def _write_split(
    split_dir: Path,
    out_dir: Path,
    split: str,
    coco: Mapping[str, Any],
    index_of: Mapping[int, int],
) -> None:
    images_out = out_dir / "images" / split
    labels_out = out_dir / "labels" / split
    for target in (images_out, labels_out):
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
    by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    for image in coco["images"]:
        name = image["file_name"]
        (images_out / name).symlink_to((split_dir / name).resolve())
        lines = [_label_line(annotation, image, index_of) for annotation in by_image[image["id"]]]
        (labels_out / f"{Path(name).stem}.txt").write_text("".join(f"{line}\n" for line in lines))


def _label_line(
    annotation: Mapping[str, Any], image: Mapping[str, Any], index_of: Mapping[int, int]
) -> str:
    width, height = image["width"], image["height"]
    x, y, w, h = annotation["bbox"]
    x_min, y_min = max(0.0, x), max(0.0, y)
    x_max, y_max = min(float(width), x + w), min(float(height), y + h)
    centre_x = (x_min + x_max) / 2 / width
    centre_y = (y_min + y_max) / 2 / height
    box_w = (x_max - x_min) / width
    box_h = (y_max - y_min) / height
    index = index_of[annotation["category_id"]]
    return f"{index} {centre_x:.6f} {centre_y:.6f} {box_w:.6f} {box_h:.6f}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Esporta i tile in formato YOLO")
    parser.add_argument("--tiles", default="data/tiles")
    parser.add_argument("--out", default="data/yolo")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    args = parser.parse_args(argv)
    print(export_yolo(args.tiles, args.out, args.splits))


if __name__ == "__main__":
    main()
