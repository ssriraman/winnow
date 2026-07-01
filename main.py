"""CLI entry point. Run with:  uv run python main.py <command> [options]

Example:  uv run python main.py technical ./data
See:      uv run python main.py --help
(Equivalent to `uv run python -m winnow.cli ...`.)
"""

from winnow.cli import main

if __name__ == "__main__":
    main()
