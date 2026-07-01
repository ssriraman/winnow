# winnow

> Shot 2,000 frames at an event? Point this at the folder and it throws out the
> soft, shaky, and blown-out ones — then optionally ranks what's left by how good
> it *looks* — so you start editing from the keepers instead of the dump.

A command-line pipeline that culls large batches of camera RAW files (primarily
Canon `.CR3`/`.CR2`, plus `.ARW`/`.DNG`/`.NEF`) — and standard JPEG/PNG images — on two
independent axes:

- **Technical quality** — focus (Tenengrad), motion blur (FFT), exposure clipping.
- **Aesthetic quality** — a pretrained [NIMA](https://github.com/chaofengc/IQA-PyTorch)
  neural score trained on the AVA dataset (~1–10, higher is better).

Both passes can coordinate through a shared `analysis_log.csv` and physically
move keepers into subdirectories. The aesthetic pass uses a GPU (CUDA) when one
is available and falls back to CPU otherwise.

The log is **optional**: it is created on demand and never required. Every
command runs standalone — `aesthetic-score` builds the log itself (recording
files the technical pass never logged), `technical --no-log` skips it entirely,
and `cull-from-log` simply reports that there's nothing to do if no log exists.

## Setup

The technical pass has no heavyweight dependencies:

```bash
uv sync
```

The **aesthetic pass is optional** and lives behind the `aesthetic` extra,
because it pulls in [`pyiqa`](https://github.com/chaofengc/IQA-PyTorch) + the
PyTorch stack:

```bash
uv sync --extra aesthetic          # or: pip install 'winnow[aesthetic]'
```

The pretrained NIMA weights are **downloaded and cached automatically by pyiqa
on first use** (into `~/.cache/torch/hub/pyiqa`), so the first `aesthetic-score`
run needs network access; subsequent runs are fully offline.

> **License note:** pyiqa and its AVA-trained NIMA weights are under a
> **noncommercial** license (PolyForm Noncommercial 1.0.0 + NTU S-Lab). The
> core technical-culling pipeline is unaffected — only the optional aesthetic
> pass inherits that restriction.

## Usage

After `uv sync`, the pipeline is available as the `winnow` command (or via
`python main.py` / `python -m winnow.cli`):

```bash
uv run winnow --help
```

| Command | What it does |
| --- | --- |
| `technical DIR` | Compute focus/shake/exposure for every supported image (RAW + JPEG/PNG), append to `DIR/analysis_log.csv` (skip with `--no-log`), and move sharp keepers into `DIR/keepers`. |
| `aesthetic-score DIR` | Write NIMA scores into the log's `aesthetic` column, creating/extending the log as needed (resumable — skips already-scored files). |
| `aesthetic-filter DIR` | Move the top `--top-percent N` (default 10) images into `DIR/aesthetic_keepers`, the rest into `DIR/others`. Use `--threshold T` for a fixed cutoff instead. |
| `cull-from-log DIR` | Re-run the keep/reject decision from logged metrics **without re-decoding RAWs** — use this to tune thresholds. |
| `prepare-log` | Add the `aesthetic` column to a pre-existing log. |

Examples:

```bash
uv run winnow technical ./data
uv run winnow technical ./data --min-focus 400 --min-shake 18
uv run winnow aesthetic-score ./data
uv run winnow aesthetic-filter ./data --top-percent 5
uv run winnow aesthetic-filter ./data --threshold 6.5
uv run winnow cull-from-log ./data --focus-gt 350 --shake-gt 19
```

All thresholds have sensible defaults (see `winnow/config.py`); every one
is overridable via flags.

### Sample run

```text
$ uv run winnow technical ./shoot
Analyzing 1,842 *.CR3 files...
Scoring: 100%|████████████████████████| 1842/1842 [04:11<00:00,  7.3it/s]
Kept 1,196 sharp frames -> shoot/keepers  (646 rejected: soft/shaky/clipped)
Wrote shoot/analysis_log.csv

$ uv run winnow aesthetic-filter ./shoot/keepers --top-percent 10
Loading NIMA model...
Critiquing 1196 files...
Scoring: 100%|████████████████████████| 1196/1196 [02:38<00:00,  7.5it/s]

--- Best Image Found ---
File: 3B9A4471.CR3
Aesthetic Score: 6.83

--- Analysis Summary ---
Calculated Threshold for Top 10%: 5.71
Moving 120 files to 'aesthetic_keepers', 1076 to 'others'.
```

## Layout

```
winnow/
  config.py     # thresholds, file patterns, log schema (single source of truth)
  io_utils.py   # RAW/image decode + file discovery
  metrics.py    # technical metrics (Tenengrad, FFT shake, exposure)
  nima.py       # NimaEstimator — pretrained NIMA via pyiqa (optional extra)
  technical.py  # technical culling pass
  aesthetic.py  # aesthetic scoring + percentile/threshold filters
  logtools.py   # log helpers (prepare, re-cull from log)
  cli.py        # `winnow` argparse entry point
main.py         # thin CLI shim
```
