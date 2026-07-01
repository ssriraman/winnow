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

from .config import (
    DEFAULT_DEVICE,
    DEFAULT_HAMMING_THRESHOLD,
    DEFAULT_TOP_PERCENT,
    DUPLICATES_DIRNAME,
    LOG_FILENAME,
)
from .dedupe import cluster_by_hash, dhash, format_hash
from .io_utils import find_images, load_pil
from .logstore import load_log


def batch_score_to_log(directory_path, log_path=LOG_FILENAME, device=DEFAULT_DEVICE):
    """Score every un-scored image (RAW/JPEG/PNG) in a dir and write it to the
    log's ``aesthetic`` column, alongside a perceptual ``hash`` (for later
    ``aesthetic-filter --dedupe``). Does not require a pre-existing log: a
    missing log is created, and files with no existing row are appended.
    Resumable: rows with ``aesthetic > 0`` are skipped."""
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

        # One decode feeds both the perceptual hash and the NIMA score.
        img = load_pil(str(file))
        file_hash = format_hash(dhash(img))
        score = estimator.estimate_image(img)

        if (df["filename"] == file.name).any():
            df.loc[df["filename"] == file.name, ["aesthetic", "hash"]] = [score, file_hash]
        else:
            # File not logged by the technical pass; record it ourselves.
            new_row = {col: None for col in df.columns}
            new_row["filename"] = file.name
            new_row["aesthetic"] = score
            new_row["hash"] = file_hash
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


def _score_directory(estimator, files, dedupe):
    """Decode + score every file once, optionally computing a perceptual hash in
    the same pass. Returns a list of ``{"path", "score"[, "hash"]}`` dicts."""
    results = []
    for file in tqdm(files, desc="Scoring"):
        try:
            img = load_pil(str(file))
            result = {"path": file, "score": None}
            if dedupe:
                # Hash the full-res decode before estimate_image() down-scales it.
                result["hash"] = dhash(img)
            result["score"] = estimator.estimate_image(img)
            results.append(result)
            estimator.empty_cache()  # keep VRAM clean
        except Exception as e:
            print(f"Error scoring {file.name}: {e}")
    return results


def _resolve_duplicates(results, duplicates_dir, hash_threshold):
    """Cluster scored images by perceptual hash and keep only the top-scoring
    member of each near-duplicate cluster; move the losers into
    ``duplicates_dir``. Returns the surviving results (one per cluster)."""
    clusters = cluster_by_hash(results, threshold=hash_threshold)
    survivors, moved = [], 0
    for cluster in clusters:
        best = max(cluster, key=lambda r: r["score"])
        survivors.append(best)
        for item in cluster:
            if item is best:
                continue
            duplicates_dir.mkdir(exist_ok=True)
            try:
                shutil.move(str(item["path"]), str(duplicates_dir / item["path"].name))
                moved += 1
            except Exception as e:
                print(f"Error moving duplicate {item['path'].name}: {e}")
    if moved:
        print(
            f"\nDedupe: kept {len(survivors)} unique image(s), moved {moved} "
            f"near-duplicate(s) to '{duplicates_dir.name}'."
        )
    # Preserve original discovery order for stable downstream reporting.
    return sorted(survivors, key=lambda r: r["path"].name)


def filter_by_percentile(
    source_dir,
    top_n_percent=DEFAULT_TOP_PERCENT,
    device=DEFAULT_DEVICE,
    dedupe=False,
    hash_threshold=DEFAULT_HAMMING_THRESHOLD,
):
    """Score a directory, then move the top N%% into ``aesthetic_keepers`` and
    the rest into ``others``.

    With ``dedupe=True``, near-duplicate frames (Hamming distance
    ``<= hash_threshold`` on a perceptual hash) are collapsed first: only the
    highest-scoring frame of each cluster survives into the percentile decision;
    the rest are moved to ``duplicates`` and excluded from the percentile math."""
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
    results = _score_directory(estimator, files, dedupe)

    if not results:
        return

    # 1b. Optional dedupe: keep the best of each near-duplicate cluster.
    if dedupe:
        results = _resolve_duplicates(results, source / DUPLICATES_DIRNAME, hash_threshold)

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


def filter_by_threshold(
    source_dir,
    threshold=6.5,
    device=DEFAULT_DEVICE,
    dedupe=False,
    hash_threshold=DEFAULT_HAMMING_THRESHOLD,
):
    """Score a directory and move images scoring ``>=`` ``threshold`` into
    ``aesthetic_keepers`` (the rest into ``others``).

    With ``dedupe=True``, near-duplicate frames are collapsed to their
    highest-scoring member first (losers moved to ``duplicates``), so only the
    best frame of each cluster is tested against the cutoff."""
    source, high_quality_dir, others_dir = _prepare_dirs(source_dir)

    print("Loading NIMA model...")
    from .nima import NimaEstimator  # lazy: pulls the optional pyiqa/torch stack

    estimator = NimaEstimator(device=device)

    files = find_images(source)
    if not files:
        print("No image files found.")
        return

    # 1. Scoring pass (single decode per file; hash alongside when deduping).
    results = _score_directory(estimator, files, dedupe)
    if not results:
        return

    # 1b. Optional dedupe before applying the cutoff.
    if dedupe:
        results = _resolve_duplicates(results, source / DUPLICATES_DIRNAME, hash_threshold)

    # 2. Moving pass.
    for item in tqdm(results, desc="Moving"):
        try:
            target = high_quality_dir if item["score"] >= threshold else others_dir
            shutil.move(str(item["path"]), str(target / item["path"].name))
        except Exception as e:
            print(f"Error moving {item['path'].name}: {e}")
