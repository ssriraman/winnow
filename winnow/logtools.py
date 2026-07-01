"""Operations on the shared ``analysis_log.csv`` that don't re-decode images."""

import shutil
from pathlib import Path

from tqdm import tqdm

from .config import LOG_FILENAME, LogCullCriteria
from .logstore import load_log, log_exists


def prepare_log(log_path=LOG_FILENAME):
    """Add (or reset) the ``aesthetic`` column, default 0.0.

    Creates an empty schema'd log if none exists.
    """
    df = load_log(log_path)
    df["aesthetic"] = 0.0
    df.to_csv(log_path, index=False)
    print(f"{log_path} written with 'aesthetic' column ({len(df)} rows).")


def cull_from_log(
    log_path,
    source_dir,
    keep_dir_name="keepers",
    criteria: LogCullCriteria = None,
):
    """Move keepers selected from logged metrics, without re-decoding RAWs.

    Use this to tune thresholds after a scoring dry run. This command reads
    metrics from the log, so it no-ops with a message when no log is present.
    """
    if not log_exists(log_path):
        print(
            f"No log found at '{log_path}'; nothing to cull.\n"
            f"Run 'cull technical {source_dir}' first to generate metrics."
        )
        return

    criteria = criteria or LogCullCriteria()

    df = load_log(log_path)
    source = Path(source_dir)
    dest = source / keep_dir_name
    dest.mkdir(exist_ok=True)

    keepers = df[criteria.mask(df)]
    print(f"Moving {len(keepers)} files to {keep_dir_name}...")

    for filename in tqdm(keepers["filename"]):
        src_file = source / filename
        if src_file.exists():
            shutil.move(str(src_file), str(dest / filename))
