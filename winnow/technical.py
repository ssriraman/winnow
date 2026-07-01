"""Technical culling pass: log metrics for every image, move the sharp keepers.

Handles RAW (rawpy) and standard images (JPEG/PNG via Pillow). Writes/appends to
``analysis_log.csv`` inside the target directory and moves files passing the
criteria into ``keepers/``.
"""

import csv
import shutil
from pathlib import Path

from tqdm import tqdm

from .config import LOG_FILENAME, TECHNICAL_COLUMNS, TechnicalCriteria
from .io_utils import find_images, load_gray
from .metrics import (
    calculate_exposure_metrics,
    calculate_tenengrad,
    check_shake_fft,
)


def get_image_metrics(path):
    """Return (focus, shake, over, under) for a single image."""
    gray = load_gray(path)
    focus = calculate_tenengrad(gray)
    shake = check_shake_fft(gray)
    over, under = calculate_exposure_metrics(gray)
    return focus, shake, over, under


def cull_directory(
    directory_path,
    output_keep="keepers",
    criteria: TechnicalCriteria = None,
    log_name: str = LOG_FILENAME,
    write_log: bool = True,
):
    """Move sharp keepers into ``output_keep``. When ``write_log`` is True, also
    append each file's metrics to ``directory_path/log_name`` (created on demand);
    the log is a side-output and is never required for the move to work."""
    criteria = criteria or TechnicalCriteria()
    path = Path(directory_path)
    keep_dir = path / output_keep
    keep_dir.mkdir(exist_ok=True)

    files = find_images(path)

    # Append mode creates the log if missing; header only written when empty.
    log_file = path / log_name
    f = open(log_file, mode="a", newline="") if write_log else None
    writer = None
    try:
        if f is not None:
            writer = csv.writer(f)
            if log_file.stat().st_size == 0:
                writer.writerow(TECHNICAL_COLUMNS)

        for file in tqdm(files, desc="Processing images"):
            try:
                focus, shake, over, under = get_image_metrics(file)
                if writer is not None:
                    writer.writerow(
                        [file.name, f"{focus:.2f}", f"{shake:.2f}", f"{over:.4f}", f"{under:.4f}"]
                    )
                if criteria.keep(focus, shake, over, under):
                    shutil.move(str(file), str(keep_dir / file.name))
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
    finally:
        if f is not None:
            f.close()
