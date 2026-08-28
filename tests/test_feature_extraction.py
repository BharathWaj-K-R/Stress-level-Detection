import numpy as np
import pytest

from preprocessing.feature_extraction import (
    FeatureExtractionConfig,
    batch_extract_features,
    build_ink_mask,
    compute_handwriting_features,
    compute_ink_density_ratio,
    compute_slant_angle,
    compute_stroke_width_variance,
    ensure_grayscale_array,
    extract_feature_vector,
    extract_hog_features,
    extract_raw_pixel_features,
    get_feature_names,
    normalize_grayscale_array,
)

SIZE = 128
ZERO = np.zeros((SIZE, SIZE), dtype=np.float32)
ONES = np.ones((SIZE, SIZE), dtype=np.float32)
RAND = np.random.default_rng(0).random((SIZE, SIZE)).astype(np.float32)


class TestEnsureGrayscaleArray:
    def test_2d_passthrough(self):
        out = ensure_grayscale_array(ZERO)
        assert out.shape == (SIZE, SIZE)

    def test_1d_reshapes(self):
        flat = np.zeros(SIZE * SIZE, dtype=np.float32)
        out = ensure_grayscale_array(flat, image_size=SIZE)
        assert out.shape == (SIZE, SIZE)

    def test_3d_averages_channels(self):
        rgb = np.ones((SIZE, SIZE, 3), dtype=np.float32) * 0.5
        out = ensure_grayscale_array(rgb, image_size=SIZE)
        assert out.shape == (SIZE, SIZE)
        assert np.allclose(out, 0.5)

    def test_uint8_normalised(self):
        img = np.full((SIZE, SIZE), 255, dtype=np.float32)
        out = ensure_grayscale_array(img)
        assert out.max() <= 1.0

    def test_wrong_1d_size_raises(self):
        with pytest.raises(ValueError):
            ensure_grayscale_array(np.zeros(10), image_size=SIZE)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            ensure_grayscale_array(np.zeros((64, 64)), image_size=SIZE)

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            ensure_grayscale_array(np.zeros((1, SIZE, SIZE, 3)), image_size=SIZE)


class TestNormalizeGrayscaleArray:
    def test_2d_unchanged_range(self):
        out = normalize_grayscale_array(RAND)
        assert 0.0 <= out.min() and out.max() <= 1.0

    def test_uint8_normalised(self):
        img = np.full((SIZE, SIZE), 200, dtype=np.float32)
        out = normalize_grayscale_array(img)
        assert out.max() <= 1.0

    def test_1d_square_reshapes(self):
        flat = np.zeros(SIZE * SIZE, dtype=np.float32)
        out = normalize_grayscale_array(flat)
        assert out.shape == (SIZE, SIZE)

    def test_1d_non_square_raises(self):
        with pytest.raises(ValueError):
            normalize_grayscale_array(np.zeros(101))

    def test_3d_averages(self):
        rgb = np.ones((SIZE, SIZE, 3), dtype=np.float32) * 128.0
        out = normalize_grayscale_array(rgb)
        assert out.shape == (SIZE, SIZE)

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            normalize_grayscale_array(np.zeros((1, SIZE, SIZE, 3)))


class TestInkMask:
    def test_all_dark_all_ink(self):
        dark = np.zeros((SIZE, SIZE), dtype=np.float32)
        mask = build_ink_mask(dark)
        assert mask.all()

    def test_all_white_no_ink(self):
        white = np.ones((SIZE, SIZE), dtype=np.float32)
        mask = build_ink_mask(white)
        assert not mask.any()

    def test_custom_threshold(self):
        img = np.full((SIZE, SIZE), 0.5, dtype=np.float32)
        mask_low = build_ink_mask(img, threshold=0.3)
        mask_high = build_ink_mask(img, threshold=0.9)
        assert not mask_low.any()
        assert mask_high.all()


class TestInkDensityRatio:
    def test_zero_image_full_density(self):
        assert compute_ink_density_ratio(ZERO) == pytest.approx(1.0)

    def test_white_image_zero_density(self):
        assert compute_ink_density_ratio(ONES) == pytest.approx(0.0)

    def test_half_image(self):
        half = np.full((SIZE, SIZE), 0.5, dtype=np.float32)
        ratio = compute_ink_density_ratio(half)
        assert 0.0 <= ratio <= 1.0


class TestStrokeWidthVariance:
    def test_blank_returns_zero(self):
        assert compute_stroke_width_variance(ONES) == pytest.approx(0.0)

    def test_returns_float(self):
        result = compute_stroke_width_variance(RAND)
        assert isinstance(result, float)
        assert result >= 0.0


class TestSlantAngle:
    def test_blank_returns_zero(self):
        assert compute_slant_angle(ONES) == pytest.approx(0.0)

    def test_range(self):
        angle = compute_slant_angle(RAND)
        assert -90.0 <= angle <= 90.0


class TestHandwritingFeatures:
    def test_shape(self):
        feats = compute_handwriting_features(RAND)
        assert feats.shape == (3,)

    def test_dtype(self):
        feats = compute_handwriting_features(RAND)
        assert feats.dtype == np.float32


class TestExtractRawPixelFeatures:
    def test_length(self):
        feats = extract_raw_pixel_features(ZERO)
        assert feats.shape == (SIZE * SIZE,)

    def test_dtype(self):
        assert extract_raw_pixel_features(RAND).dtype == np.float32


class TestExtractHogFeatures:
    def test_positive_length(self):
        feats = extract_hog_features(RAND)
        assert feats.ndim == 1
        assert len(feats) > 0

    def test_dtype(self):
        assert extract_hog_features(RAND).dtype == np.float32

    def test_custom_config(self):
        cfg = FeatureExtractionConfig(mode="hog", image_size=SIZE, include_handwriting_features=False)
        feats = extract_hog_features(RAND, config=cfg)
        assert feats.ndim == 1


class TestExtractFeatureVector:
    def test_raw_no_handwriting(self):
        cfg = FeatureExtractionConfig(mode="raw_pixels", include_handwriting_features=False)
        feats = extract_feature_vector(RAND, config=cfg)
        assert feats.shape == (SIZE * SIZE,)

    def test_raw_with_handwriting(self):
        cfg = FeatureExtractionConfig(mode="raw_pixels", include_handwriting_features=True)
        feats = extract_feature_vector(RAND, config=cfg)
        assert feats.shape == (SIZE * SIZE + 3,)

    def test_hog_no_handwriting(self):
        cfg = FeatureExtractionConfig(mode="hog", include_handwriting_features=False)
        base = extract_hog_features(RAND)
        feats = extract_feature_vector(RAND, config=cfg)
        assert feats.shape == base.shape

    def test_hog_with_handwriting(self):
        cfg = FeatureExtractionConfig(mode="hog", include_handwriting_features=True)
        base = extract_hog_features(RAND)
        feats = extract_feature_vector(RAND, config=cfg)
        assert feats.shape == (len(base) + 3,)

    def test_invalid_mode_raises(self):
        cfg = FeatureExtractionConfig(mode="unknown")
        with pytest.raises(ValueError):
            extract_feature_vector(RAND, config=cfg)

    def test_output_dtype(self):
        feats = extract_feature_vector(RAND)
        assert feats.dtype == np.float32


class TestBatchExtractFeatures:
    def test_batch_shape(self):
        images = np.stack([RAND, ZERO, ONES])
        cfg = FeatureExtractionConfig(mode="raw_pixels", include_handwriting_features=False)
        out = batch_extract_features(images, config=cfg)
        assert out.shape == (3, SIZE * SIZE)

    def test_single_image_batch(self):
        images = RAND[np.newaxis]
        out = batch_extract_features(images)
        assert out.shape[0] == 1


class TestGetFeatureNames:
    def test_raw_no_handwriting_length(self):
        cfg = FeatureExtractionConfig(mode="raw_pixels", include_handwriting_features=False)
        names = get_feature_names(cfg)
        assert len(names) == SIZE * SIZE

    def test_raw_with_handwriting_length(self):
        cfg = FeatureExtractionConfig(mode="raw_pixels", include_handwriting_features=True)
        names = get_feature_names(cfg)
        assert len(names) == SIZE * SIZE + 3

    def test_hog_with_handwriting_ends_with_hw_names(self):
        cfg = FeatureExtractionConfig(mode="hog", include_handwriting_features=True)
        names = get_feature_names(cfg)
        assert names[-3:] == ["ink_density_ratio", "stroke_width_variance", "slant_angle_estimate"]

    def test_invalid_mode_raises(self):
        cfg = FeatureExtractionConfig(mode="bad_mode")
        with pytest.raises(ValueError):
            get_feature_names(cfg)


class TestFeatureExtractionConfig:
    def test_defaults(self):
        cfg = FeatureExtractionConfig()
        assert cfg.mode == "raw_pixels"
        assert cfg.image_size == SIZE
        assert cfg.include_handwriting_features is True

    def test_frozen(self):
        cfg = FeatureExtractionConfig()
        with pytest.raises(Exception):
            cfg.mode = "hog"
