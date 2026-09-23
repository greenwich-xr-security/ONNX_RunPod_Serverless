import os
import sys
import types
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


if "mobile_sam" not in sys.modules:
    mobile_sam = types.ModuleType("mobile_sam")
    mobile_sam.SamPredictor = object
    mobile_sam.sam_model_registry = {}
    sys.modules["mobile_sam"] = mobile_sam

if "runpod" not in sys.modules:
    runpod = types.ModuleType("runpod")
    runpod.serverless = types.SimpleNamespace(start=lambda _config: None)
    sys.modules["runpod"] = runpod

os.environ["FORCE_RUNPOD_SERVERLESS"] = "1"
os.environ["RUNPOD_TEST_INPUT"] = __file__

import handler


def _make_hand(xmin: float, ymin: float, xmax: float, ymax: float):
    points = []
    for index in range(21):
        x = xmin if index % 2 == 0 else xmax
        y = ymin if (index // 2) % 2 == 0 else ymax
        points.append({"x": x, "y": y, "z": 0.0})
    return {"points": points, "handedness": "Right", "score": 0.99}


def test_hand_bbox_requires_21_finite_points():
    valid = _make_hand(10.2, 20.4, 30.7, 50.1)
    assert handler._hand_bbox(valid, 100, 100) == (10, 20, 31, 51)

    too_short = {**valid, "points": valid["points"][:-1]}
    assert handler._hand_bbox(too_short, 100, 100) is None

    non_finite = _make_hand(10, 20, 30, 50)
    non_finite["points"][0]["x"] = float("nan")
    assert handler._hand_bbox(non_finite, 100, 100) is None


def test_primary_hand_is_the_largest_valid_hand():
    small = _make_hand(10, 10, 20, 20)
    large = _make_hand(30, 15, 80, 75)

    selected, bbox = handler._select_primary_hand([small, large], 100, 100)

    assert selected is large
    assert bbox == (30, 15, 80, 75)


def test_square_crop_centres_bbox_and_adds_black_edge_padding():
    bbox = (0, 0, 20, 40)
    square = handler._make_square_bbox(bbox)
    assert square == (-13, -3, 32, 42)

    source = Image.new("RGB", (50, 50), (255, 0, 0))
    cropped = handler._crop_with_padding(source, square)

    assert cropped.size == (45, 45)
    assert cropped.getpixel((0, 20)) == (0, 0, 0)
    assert cropped.getpixel((13, 20)) == (255, 0, 0)


def test_square_crop_reserves_five_percent_border_per_side():
    bbox = (25, 10, 115, 70)
    square = handler._make_square_bbox(bbox, border_ratio=0.05)

    crop_side = square[2] - square[0]
    hand_side = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    total_border = crop_side - hand_side

    assert crop_side == 100
    assert total_border / 2 == crop_side * 0.05
    assert (square[0] + square[2]) / 2 == (bbox[0] + bbox[2]) / 2
    assert (square[1] + square[3]) / 2 == (bbox[1] + bbox[3]) / 2


def test_missing_hand_returns_sentinel_without_loading_sam_or_onnx():
    detection = types.SimpleNamespace(hand_landmarks=[], handedness=[])
    landmarker = Mock()
    landmarker.detect.return_value = detection
    image = Image.new("RGB", (80, 60), (255, 255, 255))

    with (
        patch.object(
            handler,
            "_resolve_selected_model",
            return_value=("model.onnx", "v2_m", "model.onnx"),
        ),
        patch.object(handler, "_decode_image", return_value=image),
        patch.object(handler, "_load_hand_landmarker", return_value=landmarker),
        patch.object(handler, "_load_sam_predictor") as load_sam,
        patch.object(handler, "_load_age_model") as load_age_model,
    ):
        result = handler._run_inference({})

    assert result == {
        "age": -1.0,
        "std": 0.0,
        "model_tag": "v2_m",
        "model_name": "model.onnx",
        "inference_id": None,
        "use_hand_landmarks": True,
        "use_hand_masking": True,
    }
    load_sam.assert_not_called()
    load_age_model.assert_not_called()


def test_inference_masks_and_logs_the_primary_hand_centred_crop():
    small = _make_hand(5, 5, 15, 15)
    large = _make_hand(60, 10, 90, 50)
    detection = object()
    landmarker = Mock()
    landmarker.detect.return_value = detection
    image = Image.new("RGB", (100, 60), (255, 255, 255))
    session = Mock()
    session.run.return_value = [
        np.array([27.5], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    ]
    captured = {}

    def fake_segment(rgb_image, hands):
        assert rgb_image.shape == (60, 100, 3)
        assert hands == [large]
        return np.zeros((60, 100), dtype=np.uint8)

    def fake_log(img, **_kwargs):
        captured["image"] = img.copy()
        return "inference-id"

    with (
        patch.object(
            handler,
            "_resolve_selected_model",
            return_value=("model.onnx", "v2_m", "model.onnx"),
        ),
        patch.object(handler, "_decode_image", return_value=image),
        patch.object(handler, "_load_hand_landmarker", return_value=landmarker),
        patch.object(handler, "_extract_hands", return_value=[small, large]),
        patch.object(handler, "_segment_hands", side_effect=fake_segment),
        patch.object(
            handler,
            "_load_age_model",
            return_value=(session, "input", 32),
        ),
        patch.object(handler, "_save_inference_log", side_effect=fake_log),
    ):
        result = handler._run_inference({})

    assert result["age"] == 27.5
    assert result["std"] == 1.0
    assert result["inference_id"] == "inference-id"
    assert captured["image"].size == (45, 45)
    session.run.assert_called_once()
    tensor = session.run.call_args.args[1]["input"]
    assert tensor.shape == (1, 3, 32, 32)


def test_landmark_disabled_path_keeps_frame_center_crop():
    image = Image.new("RGB", (100, 60), (255, 255, 255))
    session = Mock()
    session.run.return_value = [
        np.array([18.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    ]
    captured = {}

    def fake_log(img, **_kwargs):
        captured["image"] = img.copy()
        return None

    with (
        patch.object(
            handler,
            "_resolve_selected_model",
            return_value=("model.onnx", None, "model.onnx"),
        ),
        patch.object(handler, "_decode_image", return_value=image),
        patch.object(
            handler,
            "_load_age_model",
            return_value=(session, "input", 32),
        ),
        patch.object(handler, "_save_inference_log", side_effect=fake_log),
    ):
        handler._run_inference(
            {"use_hand_landmarks": False, "use_hand_masking": False}
        )

    assert captured["image"].size == (60, 60)
