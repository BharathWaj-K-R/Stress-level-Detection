from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, morphology


DEFAULT_IMAGE_SIZE = 128
INK_THRESHOLD = 0.82
HANDWRITING_FEATURE_NAMES = [
    "ink_density_ratio",
    "stroke_width_variance",
    "slant_angle_estimate",
]


@dataclass(frozen=True)
class FeatureExtractionConfig:
    mode: str = "raw_pixels"
    image_size: int = DEFAULT_IMAGE_SIZE
    include_handwriting_features: bool = True
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (8, 8)
    hog_cells_per_block: tuple[int, int] = (2, 2)


def ensure_grayscale_array(image: np.ndarray | Iterable[float], image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 1:
        expected = image_size * image_size
        if array.size != expected:
            raise ValueError(f"Expected flattened image of size {expected}, got {array.size}")
        array = array.reshape((image_size, image_size))
    elif array.ndim == 3:
        array = array.mean(axis=-1)
    elif array.ndim != 2:
        raise ValueError(f"Expected image with 1, 2, or 3 dimensions, got {array.ndim}")

    if array.shape != (image_size, image_size):
        raise ValueError(f"Expected image shape {(image_size, image_size)}, got {array.shape}")

    if array.max() > 1.0:
        array = array / 255.0

    return np.clip(array, 0.0, 1.0)


def normalize_grayscale_array(image: np.ndarray | Iterable[float]) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = array.mean(axis=-1)
    elif array.ndim not in (1, 2):
        raise ValueError(f"Expected image with 1, 2, or 3 dimensions, got {array.ndim}")

    if array.ndim == 1:
        side = int(np.sqrt(array.size))
        if side * side != array.size:
            raise ValueError("Flattened image does not represent a square image.")
        array = array.reshape((side, side))

    if array.max() > 1.0:
        array = array / 255.0

    return np.clip(array, 0.0, 1.0)


def build_ink_mask(image: np.ndarray, threshold: float = INK_THRESHOLD) -> np.ndarray:
    grayscale = normalize_grayscale_array(image)
    return grayscale < threshold


def compute_ink_density_ratio(image: np.ndarray, threshold: float = INK_THRESHOLD) -> float:
    ink_mask = build_ink_mask(image, threshold=threshold)
    return float(np.mean(ink_mask))


def compute_stroke_width_variance(image: np.ndarray, threshold: float = INK_THRESHOLD) -> float:
    grayscale = normalize_grayscale_array(image)
    ink_mask = build_ink_mask(grayscale, threshold=threshold)
    if not np.any(ink_mask):
        return 0.0

    edges = feature.canny(grayscale, sigma=1.0)
    if not np.any(edges):
        return 0.0

    skeleton = morphology.skeletonize(ink_mask)
    if not np.any(skeleton):
        return 0.0

    distance_to_edge = ndi.distance_transform_edt(~edges)
    stroke_widths = 2.0 * distance_to_edge[skeleton]
    if stroke_widths.size == 0:
        return 0.0

    return float(np.var(stroke_widths))


def compute_slant_angle(image: np.ndarray, threshold: float = INK_THRESHOLD) -> float:
    ink_mask = build_ink_mask(image, threshold=threshold)
    coordinates = np.column_stack(np.nonzero(ink_mask))
    if coordinates.shape[0] < 2:
        return 0.0

    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    angle_radians = np.arctan2(principal_axis[0], principal_axis[1])
    angle_degrees = float(np.degrees(angle_radians))

    if angle_degrees > 90:
        angle_degrees -= 180
    if angle_degrees < -90:
        angle_degrees += 180

    return angle_degrees


def compute_handwriting_features(image: np.ndarray) -> np.ndarray:
    grayscale = normalize_grayscale_array(image)
    return np.asarray(
        [
            compute_ink_density_ratio(grayscale),
            compute_stroke_width_variance(grayscale),
            compute_slant_angle(grayscale),
        ],
        dtype=np.float32,
    )


def extract_raw_pixel_features(image: np.ndarray) -> np.ndarray:
    grayscale = ensure_grayscale_array(image, image.shape[0])
    return grayscale.reshape(-1).astype(np.float32)


def extract_hog_features(image: np.ndarray, config: FeatureExtractionConfig | None = None) -> np.ndarray:
    config = config or FeatureExtractionConfig(mode="hog", image_size=image.shape[0])
    grayscale = ensure_grayscale_array(image, config.image_size)
    descriptor = feature.hog(
        grayscale,
        orientations=config.hog_orientations,
        pixels_per_cell=config.hog_pixels_per_cell,
        cells_per_block=config.hog_cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return descriptor.astype(np.float32)


def extract_feature_vector(image: np.ndarray, config: FeatureExtractionConfig | None = None) -> np.ndarray:
    config = config or FeatureExtractionConfig()
    grayscale = ensure_grayscale_array(image, config.image_size)

    if config.mode == "raw_pixels":
        base_features = extract_raw_pixel_features(grayscale)
    elif config.mode == "hog":
        base_features = extract_hog_features(grayscale, config=config)
    else:
        raise ValueError(f"Unsupported feature extraction mode: {config.mode}")

    if not config.include_handwriting_features:
        return base_features

    handwriting_features = compute_handwriting_features(grayscale)
    return np.concatenate([base_features, handwriting_features]).astype(np.float32)


def batch_extract_features(images: np.ndarray, config: FeatureExtractionConfig | None = None) -> np.ndarray:
    config = config or FeatureExtractionConfig()
    features = [extract_feature_vector(image, config=config) for image in images]
    return np.asarray(features, dtype=np.float32)


def get_feature_names(config: FeatureExtractionConfig | None = None) -> list[str]:
    config = config or FeatureExtractionConfig()
    zero_image = np.zeros((config.image_size, config.image_size), dtype=np.float32)

    if config.mode == "raw_pixels":
        names = [f"pixel_{index}" for index in range(config.image_size * config.image_size)]
    elif config.mode == "hog":
        hog_length = extract_hog_features(zero_image, config=config).shape[0]
        names = [f"hog_{index}" for index in range(hog_length)]
    else:
        raise ValueError(f"Unsupported feature extraction mode: {config.mode}")

    if config.include_handwriting_features:
        names.extend(HANDWRITING_FEATURE_NAMES)

    return names
