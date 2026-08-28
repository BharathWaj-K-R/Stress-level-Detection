import numpy as np
import pytest

from preprocessing.quality_check import (
    QualityThresholds,
    check_blur,
    check_handwriting_present,
    check_min_resolution,
    compute_laplacian_variance,
    run_quality_checks,
)

SIZE = 128
WHITE = np.ones((SIZE, SIZE), dtype=np.float32)
BLACK = np.zeros((SIZE, SIZE), dtype=np.float32)
RAND = np.random.default_rng(1).random((SIZE, SIZE)).astype(np.float32)


class TestCheckMinResolution:
    def test_exact_minimum_passes(self):
        ok, msg = check_min_resolution(WHITE)
        assert ok
        assert msg is None

    def test_too_small_fails(self):
        small = np.zeros((64, 64), dtype=np.float32)
        ok, msg = check_min_resolution(small)
        assert not ok
        assert msg is not None

    def test_larger_passes(self):
        big = np.zeros((256, 256), dtype=np.float32)
        ok, msg = check_min_resolution(big)
        assert ok

    def test_custom_thresholds(self):
        thresh = QualityThresholds(min_width=64, min_height=64)
        ok, _ = check_min_resolution(np.zeros((64, 64), dtype=np.float32), thresholds=thresh)
        assert ok

    def test_width_too_small(self):
        img = np.zeros((SIZE, 64), dtype=np.float32)
        ok, msg = check_min_resolution(img)
        assert not ok

    def test_height_too_small(self):
        img = np.zeros((64, SIZE), dtype=np.float32)
        ok, msg = check_min_resolution(img)
        assert not ok


class TestComputeLaplacianVariance:
    def test_uniform_image_low_variance(self):
        v = compute_laplacian_variance(WHITE)
        assert v < 1e-6

    def test_noisy_image_higher_variance(self):
        v = compute_laplacian_variance(RAND)
        assert v > 0.0

    def test_returns_float(self):
        assert isinstance(compute_laplacian_variance(RAND), float)


class TestCheckBlur:
    def test_uniform_fails_blur(self):
        ok, msg, val = check_blur(WHITE)
        assert not ok
        assert msg is not None
        assert isinstance(val, float)

    def test_noisy_passes(self):
        ok, msg, val = check_blur(RAND)
        assert ok
        assert msg is None

    def test_custom_threshold(self):
        thresh = QualityThresholds(min_laplacian_variance=0.0)
        ok, _, _ = check_blur(WHITE, thresholds=thresh)
        assert ok


class TestCheckHandwritingPresent:
    def test_white_image_no_ink_fails(self):
        ok, msg, ratio = check_handwriting_present(WHITE)
        assert not ok
        assert msg is not None
        assert isinstance(ratio, float)

    def test_black_image_has_ink_passes(self):
        ok, msg, ratio = check_handwriting_present(BLACK)
        assert ok

    def test_ratio_in_range(self):
        _, _, ratio = check_handwriting_present(RAND)
        assert 0.0 <= ratio <= 1.0


class TestRunQualityChecks:
    def test_good_image_passes(self):
        result = run_quality_checks(RAND)
        assert result["passed"] is True
        assert result["warnings"] == []
        assert "resolution" in result["metrics"]
        assert "laplacian_variance" in result["metrics"]
        assert "ink_density_ratio" in result["metrics"]

    def test_bad_image_fails(self):
        result = run_quality_checks(WHITE)
        assert result["passed"] is False
        assert len(result["warnings"]) > 0

    def test_metrics_keys_present(self):
        result = run_quality_checks(RAND)
        assert set(result["metrics"].keys()) == {"resolution", "laplacian_variance", "ink_density_ratio"}

    def test_resolution_metric_values(self):
        result = run_quality_checks(RAND)
        assert result["metrics"]["resolution"]["width"] == SIZE
        assert result["metrics"]["resolution"]["height"] == SIZE

    def test_small_image_fails(self):
        small = np.random.default_rng(2).random((32, 32)).astype(np.float32)
        result = run_quality_checks(small)
        assert result["passed"] is False

    def test_multiple_warnings_accumulated(self):
        small_white = np.ones((32, 32), dtype=np.float32)
        result = run_quality_checks(small_white)
        assert len(result["warnings"]) >= 2


class TestQualityThresholds:
    def test_defaults(self):
        t = QualityThresholds()
        assert t.min_width == 128
        assert t.min_height == 128
        assert t.min_laplacian_variance == pytest.approx(0.0015)
        assert t.min_ink_ratio == pytest.approx(0.01)

    def test_frozen(self):
        t = QualityThresholds()
        with pytest.raises(Exception):
            t.min_width = 256
