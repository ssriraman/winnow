"""The argparse parser wires every subcommand to a handler."""

import pytest

from winnow.cli import build_parser


@pytest.mark.parametrize(
    "argv",
    [
        ["technical", "somedir"],
        ["aesthetic-score", "somedir"],
        ["aesthetic-filter", "somedir", "--threshold", "6.5"],
        ["cull-from-log", "somedir"],
        ["prepare-log"],
    ],
)
def test_every_subcommand_parses_and_binds_a_handler(argv):
    args = build_parser().parse_args(argv)
    assert callable(args.func)


def test_missing_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_aesthetic_filter_rejects_percent_and_threshold_together():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["aesthetic-filter", "d", "--top-percent", "5", "--threshold", "6.5"]
        )
