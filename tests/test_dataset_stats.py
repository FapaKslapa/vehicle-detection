import json

import pytest

from vehicle_detection.data.tiling_config import TilingConfig
from vehicle_detection.experiments.dataset_stats import (
    SizeBucket,
    main,
    size_bucket,
    size_distribution,
    tiling_coverage,
)


def coco_with(boxes, size=1024):
    images = [{"id": 1, "file_name": "file_00001.jpg", "width": size, "height": size}]
    annotations = [
        {
            "id": index + 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": list(box),
            "area": box[2] * box[3],
            "iscrowd": 0,
        }
        for index, box in enumerate(boxes)
    ]
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "car"}],
    }


def test_size_bucket_uses_the_coco_area_thresholds():
    assert size_bucket(31 * 31) is SizeBucket.SMALL
    assert size_bucket(32 * 32) is SizeBucket.MEDIUM
    assert size_bucket(96 * 96 - 1) is SizeBucket.MEDIUM
    assert size_bucket(96 * 96) is SizeBucket.LARGE


def test_size_distribution_counts_each_bucket_and_gives_shares():
    coco = coco_with([(0, 0, 10, 10), (0, 0, 20, 20), (0, 0, 50, 50), (0, 0, 120, 120)])
    distribution = size_distribution(coco)
    assert distribution.counts == {"small": 2, "medium": 1, "large": 1}
    assert distribution.shares["small"] == pytest.approx(0.5)
    assert distribution.median_side == pytest.approx(35.0)


def test_size_distribution_of_an_empty_dataset_is_zero():
    distribution = size_distribution(coco_with([]))
    assert distribution.counts == {"small": 0, "medium": 0, "large": 0}
    assert distribution.median_side == 0.0


def test_tiling_coverage_counts_tiles_per_image():
    coverage = tiling_coverage(coco_with([]), TilingConfig(slice_size=512, overlap_ratio=0.0))
    assert coverage.tiles_per_image == 4
    assert coverage.empty_tile_share == pytest.approx(1.0)


def test_tiling_coverage_keeps_a_box_inside_one_tile_whole():
    coverage = tiling_coverage(
        coco_with([(10, 10, 40, 40)]), TilingConfig(slice_size=512, overlap_ratio=0.0)
    )
    assert coverage.whole == 1
    assert coverage.cut == 0
    assert coverage.lost == 0
    assert coverage.empty_tile_share == pytest.approx(0.75)


def test_tiling_coverage_marks_a_box_straddling_two_tiles_as_cut_not_whole():
    coverage = tiling_coverage(
        coco_with([(492, 10, 40, 40)]), TilingConfig(slice_size=512, overlap_ratio=0.0)
    )
    assert coverage.whole == 0
    assert coverage.cut == 1
    assert coverage.lost == 0


def test_tiling_coverage_loses_a_box_split_into_pieces_below_the_area_ratio():
    config = TilingConfig(slice_size=512, overlap_ratio=0.0, min_area_ratio=0.6)
    coverage = tiling_coverage(coco_with([(502, 502, 20, 20)]), config)
    assert coverage.lost == 1
    assert coverage.whole == 0


def test_overlap_rescues_a_box_that_would_otherwise_be_cut():
    box = [(492, 10, 40, 40)]
    without = tiling_coverage(coco_with(box), TilingConfig(slice_size=512, overlap_ratio=0.0))
    with_overlap = tiling_coverage(coco_with(box), TilingConfig(slice_size=512, overlap_ratio=0.2))
    assert with_overlap.whole >= without.whole
    assert with_overlap.whole == 1


def test_cli_writes_a_json_report(tmp_path, capsys):
    coco_path = tmp_path / "coco.json"
    coco_path.write_text(json.dumps(coco_with([(10, 10, 40, 40), (0, 0, 12, 12)])))
    out = tmp_path / "stats.json"
    main(["--coco", str(coco_path), "--out", str(out), "--slice-size", "512", "--overlap", "0.2"])
    report = json.loads(out.read_text())
    assert report["sizes"]["counts"]["small"] == 1
    assert report["tiling"]["512"]["whole"] == 2
    assert "scritto" in capsys.readouterr().out
