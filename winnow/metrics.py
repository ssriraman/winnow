"""Technical image-quality metrics computed on grayscale arrays."""

import cv2
import numpy as np


def calculate_sharpness(image_gray):
    """Variance of Laplacian - simple but effective for focus."""
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()


def calculate_tenengrad(image_gray):
    """Sobel gradient energy - a more robust focus metric than Laplacian."""
    gx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.mean(magnitude**2)


def check_shake_fft(image_gray):
    """Detect motion blur via the spread of the FFT magnitude spectrum.

    Higher std-dev = more high-frequency content = sharper / less shake.
    """
    dft = cv2.dft(np.float32(image_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-6)
    return np.std(magnitude)


def calculate_exposure_metrics(image_gray):
    """Fraction of pixels clipped bright (>=250) and dark (<=5)."""
    over = np.sum(image_gray >= 250) / image_gray.size
    under = np.sum(image_gray <= 5) / image_gray.size
    return over, under
