import json

import pandas as pd
import pytest
from PIL import Image

from vehicle_detection.evaluation.analyse import main, select_subset

CATEGORIES = [{"id": 1, "name": "car"}, {"id": 2, "name": "sedan"}]
IMAGES = (("a", "val", "scene_1"), ("b", "val", "scene_1"), ("c", "val", "scene_2"))
SIZE = 256


@pytest.fixture
def run_dir(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    images, annotations, rows = [], [], ["image_id,session_id,location,split"]
    for image_id, (name, split, scene) in enumerate(IMAGES, start=1):
        Image.new("RGB", (SIZE, SIZE), "#404040").save(images_dir / f"{name}.jpg")
        images.append({"id": image_id, "file_name": f"{name}.jpg", "width": SIZE, "height": SIZE})
        rows.append(f"{name},{scene},loc0,{split}")
        for offset, category in ((20, 1), (120, 2)):
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": [offset, offset, 60, 60],
                    "area": 3600,
                    "iscrowd": 0,
                }
            )
    coco_path = tmp_path / "coco.json"
    coco_path.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": CATEGORIES})
    )
    manifest_path = tmp_path / "split.csv"
    manifest_path.write_text("\n".join(rows) + "\n")

    def found(image_id, category, offset, score=0.9):
        return {
            "image_id": image_id,
            "category_id": category,
            "bbox": [offset, offset, 60, 60],
            "score": score,
        }

    variants = {
        # finds every car and every sedan
        "perfect": [found(i, c, o) for i in (1, 2, 3) for o, c in ((20, 1), (120, 2))],
        # calls every sedan a car; in image 1 a weaker "sedan" box also covers the sedan
        "confused": [found(i, 1, o) for i in (1, 2, 3) for o in (20, 120)]
        + [found(1, 2, 122, 0.6)],
    }
    run = tmp_path / "run"
    (run / "predictions").mkdir(parents=True)
    for name, results in variants.items():
        (run / "predictions" / f"{name}.json").write_text(json.dumps(results))
    pd.DataFrame(
        {
            "variant": list(variants),
            "threshold": [0.5, 0.5],
            "ap": [1.0, 0.5],
            "ap50": [1.0, 0.5],
        }
    ).to_csv(run / "comparison.csv", index=False)
    (run / "provenance.json").write_text(
        json.dumps(
            {
                "provisional": False,
                "config": {
                    "coco_path": str(coco_path),
                    "manifest_path": str(manifest_path),
                    "images_dir": str(images_dir),
                    "split": "val",
                    "limit": None,
                },
            }
        )
    )
    return run


def test_the_analysis_writes_every_table_and_figure(run_dir):
    assert main(["--run-dir", str(run_dir), "--gallery", "2"]) == 0
    out = run_dir / "analysis"
    for name in (
        "summary.md",
        "outcomes.csv",
        "localisation.csv",
        "counting.csv",
        "recall_breakdown.csv",
        "error_types.png",
        "recall_by_size_bin.png",
        "recall_by_category.png",
        "recall_by_size_and_edge.png",
        "merged.csv",
        "merged_counting.csv",
    ):
        assert (out / name).is_file(), name
    for variant in ("perfect", "confused"):
        for name in (
            "confusion.csv",
            "confusion.png",
            "counts.png",
            "counts_per_image.csv",
            "gt_boxes.csv",
            "predictions.csv",
            "confusion_merged.csv",
            "confusion_merged.png",
        ):
            assert (out / variant / name).is_file(), f"{variant}/{name}"
        assert len(list((out / variant / "gallery").glob("*.jpg"))) == 2


def test_the_outcomes_tell_the_variants_apart(run_dir):
    main(["--run-dir", str(run_dir), "--gallery", "0"])
    outcomes = pd.read_csv(run_dir / "analysis" / "outcomes.csv").set_index("variant")
    assert outcomes.loc["perfect", "correct"] == 6
    # 3 cars, plus the image-1 sedan found by its weaker "sedan" box
    assert outcomes.loc["confused", "correct"] == 4
    # the "car" box on that sedan is now an extra box on a found vehicle
    assert outcomes.loc["confused", "cross_class_duplicate"] == 1
    assert outcomes.loc["confused", "classification"] == 2
    assert outcomes.loc["confused", "duplicate"] == 0
    assert outcomes.loc["confused", "wrong_class"] == 2
    localisation = pd.read_csv(run_dir / "analysis" / "localisation.csv").set_index("variant")
    assert localisation.loc["confused", "recall_any_class"] == pytest.approx(1.0)
    assert localisation.loc["confused", "class_accuracy_when_found"] == pytest.approx(0.5)


def test_counting_reports_the_overcount(run_dir):
    main(["--run-dir", str(run_dir), "--gallery", "0"])
    counting = pd.read_csv(run_dir / "analysis" / "counting.csv")
    total = counting[counting["target"] == "total"].set_index("variant")
    assert total.loc["perfect", "mae"] == pytest.approx(0.0)
    assert total.loc["confused", "bias"] == pytest.approx(1 / 3)


def test_a_subset_of_variants_can_be_analysed(run_dir):
    main(["--run-dir", str(run_dir), "--variants", "confused", "--gallery", "0"])
    outcomes = pd.read_csv(run_dir / "analysis" / "outcomes.csv")
    assert list(outcomes["variant"]) == ["confused"]


def test_predictions_outside_the_selected_images_are_refused(run_dir):
    with pytest.raises(ValueError, match="not in the selected split subset"):
        main(["--run-dir", str(run_dir), "--limit", "1", "--gallery", "0"])


def test_numbers_that_disagree_with_compare_are_flagged(run_dir):
    comparison = pd.read_csv(run_dir / "comparison.csv")
    comparison["mae"] = [0.0, 9.0]
    comparison.to_csv(run_dir / "comparison.csv", index=False)
    main(["--run-dir", str(run_dir), "--gallery", "0"])
    summary = (run_dir / "analysis" / "summary.md").read_text()
    assert "## Warnings" in summary
    assert "confused: mae" in summary


def test_the_subset_matches_the_one_used_for_predictions(run_dir):
    from vehicle_detection.inference.run import select_split_subset

    config = json.loads((run_dir / "provenance.json").read_text())["config"]
    for limit in (None, 1, 2):
        ours = select_subset(config["coco_path"], config["manifest_path"], "val", limit)
        theirs = select_split_subset(config["coco_path"], config["manifest_path"], "val", limit)
        assert ours == theirs


def test_merging_the_car_types_forgives_mix_ups_inside_the_group(run_dir):
    main(["--run-dir", str(run_dir), "--gallery", "0"])
    merged = pd.read_csv(run_dir / "analysis" / "merged.csv").set_index("variant")
    # car and sedan are both passenger-car types: calling a sedan a car is no longer an error
    assert merged.loc["confused", "class_accuracy_when_found"] == pytest.approx(0.5)
    assert merged.loc["confused", "class_accuracy_merged"] == pytest.approx(1.0)
    assert merged.loc["confused", "share_of_class_errors_within_group"] == pytest.approx(1.0)
    assert merged.loc["confused", "members"] == "car, sedan"
    summary = (run_dir / "analysis" / "summary.md").read_text()
    assert '## If car, sedan were one class ("passenger car")' in summary
    assert "## Recall (any class) by size, inside the image vs cut by the image edge" in summary


def test_the_merge_can_be_turned_off(run_dir):
    main(["--run-dir", str(run_dir), "--gallery", "0", "--merge-group", "none"])
    out = run_dir / "analysis"
    assert not (out / "merged.csv").exists()
    assert not (out / "perfect" / "confusion_merged.csv").exists()
    assert "were one class" not in (out / "summary.md").read_text()


def test_a_group_that_does_not_match_the_classes_is_skipped(run_dir):
    main(["--run-dir", str(run_dir), "--gallery", "0", "--merge-group", "big=truck,van"])
    assert not (run_dir / "analysis" / "merged.csv").exists()
