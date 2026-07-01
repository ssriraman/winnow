"""Technical metrics behave sensibly on synthetic images (no RAW/torch needed)."""

import numpy as np
from scipy.ndimage import gaussian_filter

from winnow.metrics import (
    calculate_exposure_metrics,
    calculate_sharpness,
    calculate_tenengrad,
    check_shake_fft,
)


def _noise_image(seed=0, size=256):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size), dtype=np.uint8)


def test_focus_metrics_rank_sharp_above_blurred():
    sharp = _noise_image()
    blurred = gaussian_filter(sharp, sigma=4)

    assert calculate_sharpness(sharp) > calculate_sharpness(blurred)
    assert calculate_tenengrad(sharp) > calculate_tenengrad(blurred)


def test_shake_fft_returns_finite_and_deterministic_score():
    # The FFT spread metric is calibrated for natural photos, not synthetic
    # patterns, so we assert its contract rather than a direction: a finite,
    # reproducible scalar for a given input.
    img = _noise_image()

    score = check_shake_fft(img)
    assert np.isfinite(score)
    assert check_shake_fft(img) == score


def test_exposure_metrics_detect_clipping():
    white = np.full((64, 64), 255, dtype=np.uint8)
    black = np.zeros((64, 64), dtype=np.uint8)
    mid = np.full((64, 64), 128, dtype=np.uint8)

    over_w, under_w = calculate_exposure_metrics(white)
    over_b, under_b = calculate_exposure_metrics(black)
    over_m, under_m = calculate_exposure_metrics(mid)

    assert over_w == 1.0 and under_w == 0.0
    assert over_b == 0.0 and under_b == 1.0
    assert over_m == 0.0 and under_m == 0.0
