"""Log-only operations: prepare the log, and cull keepers from logged metrics."""

import pandas as pd

from winnow.logstore import load_log
from winnow.logtools import cull_from_log, prepare_log


def test_prepare_log_creates_schemad_log_with_aesthetic(tmp_path):
    log = tmp_path / "analysis_log.csv"

    prepare_log(str(log))

    assert log.exists()
    assert "aesthetic" in load_log(str(log)).columns


def _write_log(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_cull_from_log_moves_only_matching_files(tmp_path):
    (tmp_path / "keep.CR3").write_bytes(b"")
    (tmp_path / "drop.CR3").write_bytes(b"")
    log = tmp_path / "analysis_log.csv"
    # keep.CR3 satisfies the sharp clause (focus>350 & shake>19); drop.CR3 fails both.
    _write_log(
        log,
        {
            "filename": ["keep.CR3", "drop.CR3"],
            "focus": [400.0, 100.0],
            "shake": [25.0, 5.0],
            "over": [0.9, 0.9],
            "under": [0.9, 0.9],
            "aesthetic": [0.0, 0.0],
        },
    )

    cull_from_log(str(log), str(tmp_path), keep_dir_name="keepers")

    assert (tmp_path / "keepers" / "keep.CR3").exists()
    assert (tmp_path / "drop.CR3").exists()  # left in place
    assert not (tmp_path / "keepers" / "drop.CR3").exists()


def test_cull_from_log_noops_without_log(tmp_path, capsys):
    cull_from_log(str(tmp_path / "missing.csv"), str(tmp_path))

    assert "nothing to cull" in capsys.readouterr().out.lower()
    assert not (tmp_path / "keepers").exists()
