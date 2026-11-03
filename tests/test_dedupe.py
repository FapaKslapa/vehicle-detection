import json

import pandas as pd
import pytest

from vehicle_detection.evaluation.dedupe import main, suppress_across_classes, variant_name

CATEGORIES = [{"id": 1, "name": "car"}, {"id": 2, "name": "sedan"}]


def box(image_id, category_id, x, y, score, side=40):
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, side, side],
        "score": score,
    }


def test_a_weaker_box_of_another_class_on_the_same_vehicle_is_removed():
    kept = suppress_across_classes([box(1, 2, 102, 100, 0.6), box(1, 1, 100, 100, 0.9)], 0.5)
    assert kept == [box(1, 1, 100, 100, 0.9)]


def test_separate_vehicles_are_kept():
    results = [box(1, 1, 100, 100, 0.9), box(1, 2, 300, 300, 0.6)]
    assert suppress_across_classes(results, 0.5) == results


def test_images_do_not_suppress_each_other():
    results = [box(1, 1, 100, 100, 0.9), box(2, 2, 100, 100, 0.6)]
    assert suppress_across_classes(results, 0.5) == results


def test_the_iou_threshold_decides_what_counts_as_the_same_vehicle():
    # IoU of these two boxes is 0.6: removed at 0.5, kept at 0.7
    first, second = box(1, 1, 0, 0, 0.9, side=40), box(1, 2, 10, 0, 0.8, side=40)
    assert len(suppress_across_classes([first, second], 0.5)) == 1
    assert len(suppress_across_classes([first, second], 0.7)) == 2


def test_the_input_is_left_untouched_and_order_is_kept():
    results = [box(1, 1, 500, 500, 0.3), box(1, 2, 100, 100, 0.9), box(1, 1, 101, 100, 0.8)]
    copy = json.loads(json.dumps(results))
    kept = suppress_across_classes(results, 0.5)
    assert results == copy
    assert kept == [results[0], results[1]]


def test_the_iou_must_be_a_fraction():
    with pytest.raises(ValueError):
        suppress_across_classes([], 0)


def test_derived_variants_are_named_after_their_source_and_iou():
    assert variant_name("E1-full-1024", 0.5) == "E1-full-1024-agn50"


@pytest.fixture
def run_dir(tmp_path):
    images, annotations, rows = [], [], ["image_id,session_id,location,split"]
    for image_id in range(1, 5):
        images.append({"id": image_id, "file_name": f"{image_id}.jpg", "width": 512, "height": 512})
        rows.append(f"{image_id},scene_{image_id},loc0,val")
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 2,
                "bbox": [100, 100, 40, 40],
                "area": 1600,
                "iscrowd": 0,
            }
        )
    coco_path = tmp_path / "coco.json"
    coco_path.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": CATEGORIES})
    )
    manifest_path = tmp_path / "split.csv"
    manifest_path.write_text("\n".join(rows) + "\n")
    # every sedan is found, and also covered by a weaker "car" box: counted twice
    results = [b for i in range(1, 5) for b in (box(i, 2, 100, 100, 0.9), box(i, 1, 101, 100, 0.8))]
    run = tmp_path / "val_run"
    (run / "predictions").mkdir(parents=True)
    (run / "predictions" / "full.json").write_text(json.dumps(results))
    pd.DataFrame({"variant": ["full"], "threshold": [0.5]}).to_csv(
        run / "comparison.csv", index=False
    )
    (run / "provenance.json").write_text(
        json.dumps(
            {
                "provisional": False,
                "config": {
                    "coco_path": str(coco_path),
                    "manifest_path": str(manifest_path),
                    "split": "val",
                    "limit": None,
                },
            }
        )
    )
    return run


def test_removing_cross_class_duplicates_fixes_the_count(run_dir):
    main(["--run-dir", str(run_dir), "--iou", "0.5"])
    out = run_dir.with_name("val_run_agnostic")
    table = pd.read_csv(out / "comparison.csv").set_index("variant")
    assert list(table.index) == ["full", "full-agn50"]
    assert table.loc["full", "mae"] == pytest.approx(1.0)
    assert table.loc["full", "threshold_source"] == "fixed"
    assert table.loc["full-agn50", "mae"] == pytest.approx(0.0)
    assert table.loc["full-agn50", "threshold_source"] == "tuned_on_split"
    assert table.loc["full-agn50", "detections"] == 4
    assert (out / "predictions" / "full.json").is_file()
    assert (out / "predictions" / "full-agn50.json").is_file()
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["derived_from"] == str(run_dir)
    assert provenance["config"]["split"] == "val"


def test_the_source_run_is_never_overwritten(run_dir):
    with pytest.raises(ValueError, match="must differ"):
        main(["--run-dir", str(run_dir), "--out", str(run_dir)])


def test_thresholds_are_not_tuned_on_test(run_dir):
    with pytest.raises(ValueError, match="not tuned on test"):
        main(["--run-dir", str(run_dir), "--split", "test"])


def test_test_runs_take_thresholds_from_validation(run_dir, tmp_path):
    main(["--run-dir", str(run_dir), "--iou", "0.5"])
    validation = run_dir.with_name("val_run_agnostic") / "comparison.csv"
    manifest = tmp_path / "split.csv"
    manifest.write_text(manifest.read_text().replace(",val", ",test"))
    out = tmp_path / "test_run_agnostic"
    main(
        [
            "--run-dir",
            str(run_dir),
            "--split",
            "test",
            "--iou",
            "0.5",
            "--thresholds-from",
            str(validation),
            "--out",
            str(out),
        ]
    )
    table = pd.read_csv(out / "comparison.csv").set_index("variant")
    tuned = pd.read_csv(validation).set_index("variant").loc["full-agn50", "threshold"]
    assert table.loc["full-agn50", "threshold"] == pytest.approx(tuned)
    assert table.loc["full-agn50", "threshold_source"] == "fixed"
