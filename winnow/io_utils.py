"""Image decoding and file discovery.

The single source of truth for turning a path (RAW or standard image) into
pixels, so RAW-decode parameters stay consistent across both pipelines.
"""

from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import Image

from .config import IMAGE_GLOBS, RAW_EXTENSIONS


def is_raw(path) -> bool:
    return Path(path).suffix.lower() in RAW_EXTENSIONS


def load_rgb(path) -> np.ndarray:
    """Decode a RAW or standard image file to an RGB numpy array."""
    if is_raw(path):
        with rawpy.imread(str(path)) as raw:
            return raw.postprocess(use_camera_wb=True, no_auto_bright=True, bright=1.0)
    return np.array(Image.open(path).convert("RGB"))


def load_pil(path) -> Image.Image:
    """Decode any supported file to a PIL RGB image (for the NIMA transform)."""
    return Image.fromarray(load_rgb(path))


def load_gray(path) -> np.ndarray:
    """Decode any supported file to a grayscale array (for technical metrics)."""
    return cv2.cvtColor(load_rgb(path), cv2.COLOR_RGB2GRAY)


def find_images(directory, globs=IMAGE_GLOBS):
    """All RAW + JPEG images in ``directory`` (non-recursive)."""
    base = Path(directory)
    return [p for g in globs for p in base.glob(g)]


def find_raws(directory):
    """Canon RAW (``*.CR3``) files in ``directory`` (non-recursive)."""
    return list(Path(directory).glob("*.CR3"))
