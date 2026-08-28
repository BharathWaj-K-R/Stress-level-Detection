from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from preprocessing.feature_extraction import compute_ink_density_ratio, normalize_grayscale_array


@dataclass(frozen=True)
class QualityThresholds:
    min_width: int = 128
    min_height: int = 128
    min_laplacian_variance: float = 0.0015
    min_ink_ratio: float = 0.01


def check_min_resolution(image: np.ndarray, thresholds: QualityThresholds | None = None) -> tuple[bool, str | None]:
    thresholds = thresholds or QualityThresholds()
    height, width = image.shape[:2]
    if width < thresholds.min_width or height < thresholds.min_height:
        return False, f"Image resolution is too low. Minimum required size is {thresholds.min_width}x{thresholds.min_height}."
    return True, None


def compute_laplacian_variance(image: np.ndarray) -> float:
    grayscale = normalize_grayscale_array(image)
    laplacian = ndi.laplace(grayscale)
    return float(np.var(laplacian))


def check_blur(image: np.ndarray, thresholds: QualityThresholds | None = None) -> tuple[bool, str | None, float]:
    thresholds = thresholds or QualityThresholds()
    variance = compute_laplacian_variance(image)
    if variance < thresholds.min_laplacian_variance:
        return False, "Image too blurry, please retake.", variance
    return True, None, variance


def check_handwriting_present(image: np.ndarray, thresholds: QualityThresholds | None = None) -> tuple[bool, str | None, float]:
    thresholds = thresholds or QualityThresholds()
    ink_ratio = compute_ink_density_ratio(image)
    if ink_ratio < thresholds.min_ink_ratio:
        return False, "No clear handwriting detected. Please upload a clearer handwriting sample.", ink_ratio
    return True, None, ink_ratio


def run_quality_checks(image: np.ndarray, thresholds: QualityThresholds | None = None) -> dict:
    thresholds = thresholds or QualityThresholds()
    results = {
        "passed": True,
        "warnings": [],
        "metrics": {},
    }

    resolution_ok, resolution_message = check_min_resolution(image, thresholds=thresholds)
    results["metrics"]["resolution"] = {"width": int(image.shape[1]), "height": int(image.shape[0])}
    if not resolution_ok and resolution_message:
        results["passed"] = False
        results["warnings"].append(resolution_message)

    blur_ok, blur_message, blur_value = check_blur(image, thresholds=thresholds)
    results["metrics"]["laplacian_variance"] = blur_value
    if not blur_ok and blur_message:
        results["passed"] = False
        results["warnings"].append(blur_message)

    handwriting_ok, handwriting_message, ink_ratio = check_handwriting_present(image, thresholds=thresholds)
    results["metrics"]["ink_density_ratio"] = ink_ratio
    if not handwriting_ok and handwriting_message:
        results["passed"] = False
        results["warnings"].append(handwriting_message)

    return results
