"""Central configuration: thresholds, file patterns, and the shared log schema.

Every tunable number that used to live in a script's ``__main__`` block or be
hardcoded inline now has a home here. CLI flags override these defaults.
"""

from dataclasses import dataclass

# --- File discovery --------------------------------------------------------
# Extension matching is case-insensitive (see io_utils). RAW files are decoded
# via rawpy; standard images (JPEG/PNG) via Pillow.
RAW_EXTENSIONS = (".cr3", ".cr2", ".arw", ".dng", ".nef")
STANDARD_EXTENSIONS = (".jpg", ".jpeg", ".png")
# Everything the pipeline can decode and score.
IMAGE_EXTENSIONS = RAW_EXTENSIONS + STANDARD_EXTENSIONS

# --- Shared analysis log ---------------------------------------------------
LOG_FILENAME = "analysis_log.csv"
# Technical pass writes the first five columns; the aesthetic pass fills 'aesthetic'.
TECHNICAL_COLUMNS = ["filename", "focus", "shake", "over", "under"]
LOG_COLUMNS = TECHNICAL_COLUMNS + ["aesthetic"]

# --- NIMA aesthetic model --------------------------------------------------
# pyiqa metric name. "nima" is a NIMA model trained on the AVA aesthetic
# dataset (higher score = more aesthetic, ~1-10). Weights are downloaded and
# cached by pyiqa on first use. See the optional 'aesthetic' extra.
NIMA_METRIC = "nima"
# Cap the long edge before scoring to bound VRAM/transfer; pyiqa does its own
# final resize/crop internally, so this only removes wasteful upload of full RAWs.
NIMA_MAX_EDGE = 1024
DEFAULT_DEVICE = "cuda"
DEFAULT_TOP_PERCENT = 10.0
DEFAULT_AESTHETIC_THRESHOLD = 6.5


@dataclass
class TechnicalCriteria:
    """Keep-decision for the live technical pass (AND of all conditions).

    Mirrors the original ``processor/pipeline.py`` logic:
    ``focus > 500 and shake >= 20 and over <= 0.05``.
    """

    min_focus: float = 500.0
    min_shake: float = 20.0
    max_over: float = 0.05

    def keep(self, focus: float, shake: float, over: float, under: float) -> bool:
        return focus > self.min_focus and shake >= self.min_shake and over <= self.max_over


@dataclass
class LogCullCriteria:
    """Re-cull decision applied to logged metrics (OR of two clauses).

    Mirrors the original ``cull_from_log.py`` expression:
    ``(focus > 350 & shake > 19) | (over < 0.05 & under < 0.30)``. Intentionally
    distinct from :class:`TechnicalCriteria` — this is the threshold-tuning pass
    run after a dry run, without re-decoding RAWs.
    """

    focus_gt: float = 350.0
    shake_gt: float = 19.0
    over_lt: float = 0.05
    under_lt: float = 0.30

    def mask(self, df):
        """Return a boolean Series selecting keeper rows from a metrics DataFrame."""
        return (
            ((df["focus"] > self.focus_gt) & (df["shake"] > self.shake_gt))
            | ((df["over"] < self.over_lt) & (df["under"] < self.under_lt))
        )
