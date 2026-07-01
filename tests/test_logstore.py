"""The optional log is never required and always exposes an 'aesthetic' column."""

import pandas as pd

from winnow.config import LOG_COLUMNS
from winnow.logstore import load_log, log_exists


def test_missing_log_returns_empty_schemad_frame(tmp_path):
    missing = tmp_path / "nope.csv"

    assert log_exists(missing) is False
    df = load_log(missing)
    assert list(df.columns) == LOG_COLUMNS
    assert len(df) == 0


def test_legacy_log_without_aesthetic_or_hash_columns_gets_them(tmp_path):
    path = tmp_path / "analysis_log.csv"
    pd.DataFrame(
        {"filename": ["a.CR3"], "focus": [600.0], "shake": [25.0], "over": [0.0], "under": [0.0]}
    ).to_csv(path, index=False)

    df = load_log(path)
    assert "aesthetic" in df.columns
    assert "hash" in df.columns
    assert len(df) == 1
