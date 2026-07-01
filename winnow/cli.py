"""``winnow`` command-line entry point.

Subcommands map to the two independent passes plus the log helpers::

    winnow technical       DIR              # score + move sharp keepers, log metrics
    winnow aesthetic-score DIR              # write NIMA scores into the log
    winnow aesthetic-filter DIR            # move top-N%% (or >= threshold) into aesthetic_keepers
    winnow cull-from-log   DIR              # re-cull from logged metrics (no re-decode)
    winnow prepare-log                     # add the 'aesthetic' column to a log

The aesthetic pass's heavy deps (pyiqa/torch, the optional 'aesthetic' extra)
are imported lazily per-handler, so the log/technical commands start instantly
and work even when that extra isn't installed.
"""

import argparse

from .config import (
    DEFAULT_AESTHETIC_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_TOP_PERCENT,
    LOG_FILENAME,
    LogCullCriteria,
    TechnicalCriteria,
)


def _cmd_technical(args):
    from .technical import cull_directory

    criteria = TechnicalCriteria(
        min_focus=args.min_focus, min_shake=args.min_shake, max_over=args.max_over
    )
    cull_directory(
        args.directory,
        output_keep=args.keep_dir,
        criteria=criteria,
        write_log=not args.no_log,
    )


def _cmd_aesthetic_score(args):
    from .aesthetic import batch_score_to_log

    batch_score_to_log(args.directory, log_path=args.log, device=args.device)


def _cmd_aesthetic_filter(args):
    from .aesthetic import filter_by_percentile, filter_by_threshold

    if args.threshold is not None:
        filter_by_threshold(args.directory, threshold=args.threshold, device=args.device)
    else:
        filter_by_percentile(
            args.directory, top_n_percent=args.top_percent, device=args.device
        )


def _cmd_cull_from_log(args):
    from .logtools import cull_from_log

    criteria = LogCullCriteria(
        focus_gt=args.focus_gt,
        shake_gt=args.shake_gt,
        over_lt=args.over_lt,
        under_lt=args.under_lt,
    )
    cull_from_log(args.log, args.directory, keep_dir_name=args.keep_dir, criteria=criteria)


def _cmd_prepare_log(args):
    from .logtools import prepare_log

    prepare_log(args.log)


def build_parser():
    parser = argparse.ArgumentParser(prog="winnow", description="RAW photo culling pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- technical ---------------------------------------------------------
    p = sub.add_parser("technical", help="Score technical quality, log metrics, move keepers")
    p.add_argument("directory")
    p.add_argument("--keep-dir", default="keepers")
    p.add_argument("--min-focus", type=float, default=TechnicalCriteria.min_focus)
    p.add_argument("--min-shake", type=float, default=TechnicalCriteria.min_shake)
    p.add_argument("--max-over", type=float, default=TechnicalCriteria.max_over)
    p.add_argument(
        "--no-log", action="store_true", help="Skip writing analysis_log.csv (just move keepers)"
    )
    p.set_defaults(func=_cmd_technical)

    # --- aesthetic-score ---------------------------------------------------
    p = sub.add_parser("aesthetic-score", help="Write NIMA aesthetic scores into the log")
    p.add_argument("directory")
    p.add_argument("--log", default=LOG_FILENAME)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.set_defaults(func=_cmd_aesthetic_score)

    # --- aesthetic-filter --------------------------------------------------
    p = sub.add_parser(
        "aesthetic-filter",
        help="Move keepers by aesthetic score (top-percent by default, or --threshold)",
    )
    p.add_argument("directory")
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--top-percent", type=float, default=DEFAULT_TOP_PERCENT,
        help=f"Keep the top N%% by score (default: {DEFAULT_TOP_PERCENT})",
    )
    group.add_argument(
        "--threshold", type=float, default=None,
        help=f"Keep scores >= T instead of a percentile (e.g. {DEFAULT_AESTHETIC_THRESHOLD})",
    )
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.set_defaults(func=_cmd_aesthetic_filter)

    # --- cull-from-log -----------------------------------------------------
    p = sub.add_parser("cull-from-log", help="Move keepers from logged metrics (no re-decode)")
    p.add_argument("directory")
    p.add_argument("--log", default=LOG_FILENAME)
    p.add_argument("--keep-dir", default="keepers")
    p.add_argument("--focus-gt", type=float, default=LogCullCriteria.focus_gt)
    p.add_argument("--shake-gt", type=float, default=LogCullCriteria.shake_gt)
    p.add_argument("--over-lt", type=float, default=LogCullCriteria.over_lt)
    p.add_argument("--under-lt", type=float, default=LogCullCriteria.under_lt)
    p.set_defaults(func=_cmd_cull_from_log)

    # --- prepare-log -------------------------------------------------------
    p = sub.add_parser("prepare-log", help="Add the 'aesthetic' column to an existing log")
    p.add_argument("--log", default=LOG_FILENAME)
    p.set_defaults(func=_cmd_prepare_log)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
