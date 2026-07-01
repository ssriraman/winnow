# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-01

### Added

- **Perceptual-hash deduplication for the aesthetic pass.** Bursts and brackets
  of the same moment can now be collapsed to a single best frame.
  - `aesthetic-filter --dedupe` groups visually near-identical frames (bursts,
    brackets, minor re-crops) using a perceptual difference hash (dHash), keeps
    only the highest aesthetic score in each group, and moves the rest to
    `DIR/duplicates`. Deduped frames are excluded from the top-percent /
    threshold decision, so a burst counts once instead of skewing the results.
  - `--hash-threshold N` tunes grouping strength (max Hamming distance, default
    `5` of 64 bits; larger is more aggressive). Works in both `--top-percent`
    and `--threshold` modes.
  - `aesthetic-score` now records each image's perceptual hash into a new `hash`
    column in `analysis_log.csv` (a `0x`-prefixed hex string), captured during
    the same decode as the score.

### Changed

- `analysis_log.csv` gains a `hash` column. Existing logs are upgraded
  transparently — the column is added on load, so older logs keep working.
- Internal: `NimaEstimator` exposes `estimate_image(img)` so callers decode each
  image once and reuse the pixels for both hashing and scoring (no double RAW
  decode).

### Notes

- The dedupe hashing uses only core dependencies (Pillow + numpy); no extra
  install is required for hashing itself. Scoring still needs the optional
  `aesthetic` extra (pyiqa + torch).
- Fully backward compatible: no breaking changes to existing commands or flags.

## [0.1.0]

- Initial release: two-axis (technical + aesthetic) RAW photo culling pipeline.
