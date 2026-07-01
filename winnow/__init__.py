"""RAW photo culling pipeline.

Two independent passes that share a common core:

* Technical culling  -> winnow.technical  (focus / motion-blur / exposure)
* Aesthetic scoring  -> winnow.aesthetic  (NIMA-style neural score)

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
