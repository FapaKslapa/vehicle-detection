import numpy as np
import pytest

from synthetic import ImageSpec, make_detector, render, write_dataset
from vehicle_detection.data.tiling_config import TilingConfig
from vehicle_detection.inference.detections import to_coco_results
from vehicle_detection.inference.sliced import PostprocessConfig, SlicedPredictor


def predictor(**postprocess) -> SlicedPredictor:
    return SlicedPredictor(make_detector(), TilingConfig(), PostprocessConfig(**postprocess))


def spec(width=1024, height=1024, *boxes) -> ImageSpec:
    return ImageSpec("image_0001", width, height, "train", "s1", tuple(boxes))


def assert_box(detection, expected, tolerance=3.0):
    assert detection.box == pytest.approx(expected, abs=tolerance)


@pytest.mark.parametrize("standard", [True, False])
def test_object_in_overlap_is_detected_once(standard):
    image = render(spec(1024, 1024, (520, 520, 40, 40)))
    detections = predictor(perform_standard_pred=standard).predict(image)
    assert len(detections) == 1
    assert_box(detections[0], (520, 520, 560, 560))


def test_object_straddling_a_tile_border_is_counted_once_with_the_union_box():
    image = render(spec(1024, 1024, (500, 100, 40, 40)))
    detections = predictor(perform_standard_pred=False).predict(image)
    assert len(detections) == 1
    assert_box(detections[0], (500, 100, 540, 140))


def test_distant_objects_are_both_reported():
    image = render(spec(1024, 1024, (100, 100, 40, 40), (800, 800, 40, 40)))
    assert len(predictor().predict(image)) == 2


def test_no_vehicles_gives_an_empty_list():
    assert predictor().predict(render(spec())) == []


def test_small_image_gives_a_single_detection():
    image = render(spec(300, 300, (100, 100, 40, 40)))
    detections = predictor().predict(image)
    assert len(detections) == 1
    assert_box(detections[0], (100, 100, 140, 140))


def test_non_square_image_uses_the_bottom_anchored_row():
    image = render(spec(1024, 700, (600, 600, 40, 40)))
    detections = predictor().predict(image)
    assert len(detections) == 1
    assert_box(detections[0], (600, 600, 640, 640))


def test_max_detections_caps_the_output():
    image = render(spec(1024, 1024, (100, 100, 40, 40), (800, 800, 40, 40)))
    assert len(predictor(max_detections=1).predict(image)) == 1


def test_reads_images_from_disk(tmp_path):
    dataset = write_dataset(tmp_path, [spec(1024, 1024, (520, 520, 40, 40))])
    detections = predictor().predict(dataset.images_dir / "image_0001.jpg")
    assert len(detections) == 1
    assert_box(detections[0], (520, 520, 560, 560))


def test_missing_image_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="absent.jpg"):
        predictor().predict(tmp_path / "absent.jpg")


def test_output_is_sorted_by_descending_score():
    image = render(spec(1024, 1024, (100, 100, 40, 40), (800, 800, 40, 40)))
    scores = [d.score for d in predictor().predict(image)]
    assert scores == sorted(scores, reverse=True)


def test_output_feeds_the_coco_results_format():
    image = render(spec(1024, 1024, (520, 520, 40, 40)))
    results = to_coco_results(7, predictor().predict(image))
    assert results[0]["image_id"] == 7
    assert results[0]["bbox"] == pytest.approx([520, 520, 40, 40], abs=3.0)


def test_postprocess_config_validation():
    with pytest.raises(ValueError):
        PostprocessConfig(match_threshold=0.0)
    with pytest.raises(ValueError):
        PostprocessConfig(max_detections=0)
    with pytest.raises(ValueError):
        PostprocessConfig(postprocess_type="BOGUS")


def test_array_input_is_not_modified():
    image = render(spec(1024, 1024, (520, 520, 40, 40)))
    before = image.copy()
    predictor().predict(image)
    assert np.array_equal(image, before)


@pytest.mark.parametrize("kind", ["GREEDYNMM", "NMM"])
@pytest.mark.parametrize(("full", "truncated"), [(0.9, 0.6), (0.6, 0.9), (0.8, 0.8)])
def test_border_object_is_counted_once_whatever_the_score_order(kind, full, truncated):
    image = render(spec(1024, 1024, (480, 100, 80, 40)))
    sliced = SlicedPredictor(
        make_detector(full, truncated),
        TilingConfig(),
        PostprocessConfig(postprocess_type=kind, perform_standard_pred=False),
    )
    detections = sliced.predict(image)
    assert len(detections) == 1
    assert_box(detections[0], (480, 100, 560, 140))


def test_containment_threshold_validation():
    with pytest.raises(ValueError):
        PostprocessConfig(containment_threshold=0.5)
    with pytest.raises(ValueError):
        PostprocessConfig(containment_threshold=1.1)


def test_nms_is_not_offered_because_it_cannot_merge_border_boxes():
    with pytest.raises(ValueError):
        PostprocessConfig(postprocess_type="NMS")


class SpyDetector(type(make_detector())):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shapes = []

    def perform_inference(self, image):
        self.shapes.append(image.shape)
        super().perform_inference(image)


@pytest.mark.parametrize("shape", [(600, 700), (600, 700, 4), (600, 700, 3)])
def test_models_always_receive_three_channel_tiles(shape):
    spy = SpyDetector(category_mapping={"0": "car"}, confidence_threshold=0.1)
    SlicedPredictor(spy, TilingConfig()).predict(np.zeros(shape, dtype=np.uint8))
    assert spy.shapes
    assert all(len(tile) == 3 and tile[2] == 3 for tile in spy.shapes)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((600, 700, 2), dtype=np.uint8),
        np.zeros(10, dtype=np.uint8),
        np.zeros((600, 700, 3), dtype=np.float64),
    ],
)
def test_unsupported_arrays_are_rejected_with_their_shape_and_dtype(image):
    with pytest.raises(ValueError, match="array immagine non supportato"):
        predictor().predict(image)


@pytest.mark.parametrize("kind", ["GREEDYNMM", "NMM"])
def test_low_model_confidence_does_not_switch_the_postprocess(kind, caplog):
    import logging

    detector = make_detector(0.9, 0.6)
    detector.confidence_threshold = 0.01
    image = render(spec(1024, 1024, (480, 100, 80, 40)))
    sliced = SlicedPredictor(
        detector,
        TilingConfig(),
        PostprocessConfig(postprocess_type=kind, perform_standard_pred=False),
    )
    sahi_logger = logging.getLogger("sahi")
    sahi_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING):
            detections = sliced.predict(image)
    finally:
        sahi_logger.removeHandler(caplog.handler)
    assert not [r for r in caplog.records if "Switching postprocess" in r.getMessage()]
    assert len(detections) == 1
