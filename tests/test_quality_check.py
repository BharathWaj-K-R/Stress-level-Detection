import sys
from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from preprocessing.quality_check import (
    QualityThresholds,
    check_blur,
    check_handwriting_present,
    check_min_resolution,
    run_quality_checks,
)


def test_resolution_check_fails_for_tiny_image():
    tiny = np.zeros((64, 64), dtype=np.float32)
    ok, message = check_min_resolution(tiny, QualityThresholds(min_width=128, min_height=128))
    assert not ok
    assert "Minimum required size" in message


def test_handwriting_present_on_real_sample():
    image = Image.open(BASE_DIR / "test" / "low.png").convert("L")
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    ok, _, ink_ratio = check_handwriting_present(pixels)
    assert ok
    assert ink_ratio > 0.01


def test_blur_check_flags_flat_image():
    flat = np.ones((128, 128), dtype=np.float32)
    ok, message, variance = check_blur(flat)
    assert not ok
    assert message == "Image too blurry, please retake."
    assert variance < 0.0015


def test_run_quality_checks_passes_real_sample():
    image = Image.open(BASE_DIR / "test" / "medium.png").convert("L")
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    result = run_quality_checks(pixels)
    assert result["passed"] is True
    assert "ink_density_ratio" in result["metrics"]
