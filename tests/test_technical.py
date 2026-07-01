"""End-to-end technical pass over JPEG/PNG (no RAW/torch needed)."""

import numpy as np
import pandas as pd
from PIL import Image

from winnow.config import TECHNICAL_COLUMNS, TechnicalCriteria
from winnow.technical import cull_directory


def _write_image(path, seed):
    arr = np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_cull_directory_logs_every_image_and_moves_keepers(tmp_path):
    _write_image(tmp_path / "a.jpg", 0)
    _write_image(tmp_path / "b.png", 1)

    # Permissive criteria so both are kept — this exercises decode + log + move
    # for mixed JPEG/PNG in one pass.
    cull_directory(
        str(tmp_path),
        criteria=TechnicalCriteria(min_focus=0, min_shake=0, max_over=1.0),
    )

    log = pd.read_csv(tmp_path / "analysis_log.csv")
    assert list(log.columns) == TECHNICAL_COLUMNS
    assert set(log["filename"]) == {"a.jpg", "b.png"}
    assert (tmp_path / "keepers" / "a.jpg").exists()
    assert (tmp_path / "keepers" / "b.png").exists()


def test_cull_directory_rejects_below_threshold(tmp_path):
    _write_image(tmp_path / "soft.jpg", 2)

    # Impossibly high focus requirement -> nothing qualifies, but it is still logged.
    cull_directory(
        str(tmp_path),
        criteria=TechnicalCriteria(min_focus=1e12, min_shake=0, max_over=1.0),
    )

    assert (tmp_path / "soft.jpg").exists()
    assert not (tmp_path / "keepers" / "soft.jpg").exists()
    assert (tmp_path / "analysis_log.csv").exists()
