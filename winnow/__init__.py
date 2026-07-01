"""Two-axis photo culling pipeline (RAW + JPEG/PNG).

Two independent passes that share a common core:

* Technical culling  -> winnow.technical  (focus / motion-blur / exposure)
* Aesthetic scoring  -> winnow.aesthetic  (pretrained NIMA score, optional extra)

Both coordinate through ``analysis_log.csv``. See ``winnow.cli`` for the
``winnow`` command-line entry point.
"""

__all__ = [
    "config",
    "io_utils",
    "metrics",
    "nima",
    "technical",
    "aesthetic",
    "logtools",
    "logstore",
]
