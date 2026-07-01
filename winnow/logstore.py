"""Read helpers for the optional ``analysis_log.csv``.

The log is never required. If it is missing, :func:`load_log` returns an empty
frame with the correct schema, so every command can run standalone and create
the log on demand rather than crashing on a missing file.
"""

from pathlib import Path

import pandas as pd

from .config import LOG_COLUMNS


def log_exists(log_path) -> bool:
    return Path(log_path).exists()


def load_log(log_path) -> pd.DataFrame:
    """Return the log as a DataFrame, or an empty schema'd frame if it is missing.

    Always guarantees an ``aesthetic`` column so callers never have to check.
    """
    path = Path(log_path)
    if path.exists():
        df = pd.read_csv(path)
        if "aesthetic" not in df.columns:
            df["aesthetic"] = 0.0
        return df
    return pd.DataFrame(columns=LOG_COLUMNS)
