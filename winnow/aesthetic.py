"""Aesthetic scoring pass, built on the shared :class:`~winnow.nima.NimaEstimator`.

Three entry points:

* :func:`batch_score_to_log` — score a directory into the log's ``aesthetic`` column.
* :func:`filter_by_percentile` — move the top N%% by score into ``aesthetic_keepers``.
* :func:`filter_by_threshold` — move images scoring ``>=`` a fixed cutoff.
"""

import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import DEFAULT_DEVICE, DEFAULT_TOP_PERCENT, LOG_FILENAME
from .io_utils import find_images
from .logstore import load_log


def batch_score_to_log(directory_path, log_path=LOG_FILENAME, device=DEFAULT_DEVICE):
    """Score every un-scored image (RAW/JPEG/PNG) in a dir and write it to the
    log's ``aesthetic`` column. Does not require a pre-existing log: a missing
    log is created, and files with no existing row are appended. Resumable: rows
    with ``aesthetic > 0`` are skipped."""
    from .nima import NimaEstimator  # lazy: pulls the optional pyiqa/torch stack

    estimator = NimaEstimator(device=device)

    # Missing log -> empty schema'd frame, so this runs standalone.
    df = load_log(log_path)

    processed_files = set(df[df["aesthetic"] > 0]["filename"])
    files = find_images(directory_path)

    print(f"Starting aesthetic batch processing for {len(files)} files...")

    scored_this_run = 0
    for file in tqdm(files, desc="Calculating Aesthetic Scores"):
        if file.name in processed_files:
            continue

        score = estimator.estimate(str(file))

        if (df["filename"] == file.name).any():
            df.loc[df["filename"] == file.name, "aesthetic"] = score
        else:
            # File not logged by the technical pass; record it ourselves.
            new_row = {col: None for col in df.columns}
            new_row["filename"] = file.name
            new_row["aesthetic"] = score
            df.loc[len(df)] = new_row

        # Save periodically to prevent data loss on long runs.
        scored_this_run += 1
        if scored_this_run % 10 == 0:
            df.to_csv(log_path, index=False)

    df.to_csv(log_path, index=False)
    print("Batch processing complete.")


def _prepare_dirs(source_dir):
    source = Path(source_dir)
    high_quality_dir = source / "aesthetic_keepers"
    others_dir = source / "others"
    high_quality_dir.mkdir(exist_ok=True)
    others_dir.mkdir(exist_ok=True)
    return source, high_quality_dir, others_dir


def filter_by_percentile(source_dir, top_n_percent=DEFAULT_TOP_PERCENT, device=DEFAULT_DEVICE):
    """Score a directory, then move the top N%% into ``aesthetic_keepers`` and
    the rest into ``others``."""
    source, high_quality_dir, others_dir = _prepare_dirs(source_dir)

    print("Loading NIMA model...")
    from .nima import NimaEstimator  # lazy: pulls the optional pyiqa/torch stack

    estimator = NimaEstimator(device=device)

    files = find_images(source)
    if not files:
        print(f"No image files found in {source}.")
        return

    # 1. Scoring pass.
    print(f"Critiquing {len(files)} files...")
    results = []
    for file in tqdm(files, desc="Scoring"):
        try:
            score = estimator.estimate(str(file))
            results.append({"path": file, "score": score})
            estimator.empty_cache()  # keep VRAM clean
        except Exception as e:
            print(f"Error scoring {file.name}: {e}")

    if not results:
        return

    best_image = max(results, key=lambda x: x["score"])
    print("\n--- Best Image Found ---")
    print(f"File: {best_image['path'].name}")
    print(f"Aesthetic Score: {best_image['score']:.2f}")

    # 2. Statistical thresholding.
    scores = [r["score"] for r in results]
    threshold = np.percentile(scores, 100 - top_n_percent)

    print("\n--- Analysis Summary ---")
    print(f"Calculated Threshold for Top {top_n_percent}%: {threshold:.2f}")
    print("Moving files to 'aesthetic_keepers' vs 'others'...\n")

    # 3. Moving pass.
    for item in tqdm(results, desc="Moving"):
        try:
            target = high_quality_dir if item["score"] >= threshold else others_dir
            shutil.move(str(item["path"]), str(target / item["path"].name))
        except Exception as e:
            print(f"Error moving {item['path'].name}: {e}")

    print("\nProcessing complete.")


def filter_by_threshold(source_dir, threshold=6.5, device=DEFAULT_DEVICE):
    """Score-and-move in a single pass using a fixed score cutoff."""
    source, high_quality_dir, others_dir = _prepare_dirs(source_dir)

    print("Loading NIMA model...")
    from .nima import NimaEstimator  # lazy: pulls the optional pyiqa/torch stack

    estimator = NimaEstimator(device=device)

    files = find_images(source)
    if not files:
        print("No image files found.")
        return

    pbar = tqdm(files, desc="Critiquing")
    for file in pbar:
        try:
            score = estimator.estimate(str(file))
            pbar.set_postfix({"score": f"{score:.2f}"})
            target = high_quality_dir if score >= threshold else others_dir
            shutil.move(str(file), str(target / file.name))
            estimator.empty_cache()
        except Exception as e:
            pbar.write(f"Error processing {file.name}: {e}")
