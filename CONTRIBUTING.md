# Contributing

Thanks for your interest in improving `winnow`!

## Development setup

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev              # core + test/lint tooling
uv sync --group dev --extra aesthetic   # also install the NIMA aesthetic pass
```

## Before opening a PR

```bash
uv run ruff check winnow tests   # lint
uv run pytest -q                      # tests
```

CI runs the same lint + tests on Python 3.10–3.12. The aesthetic pass is *not*
exercised in CI (it needs the torch stack and downloaded weights), so if you
change `winnow/nima.py` or `winnow/aesthetic.py`, please run it
locally against a few sample files and describe what you saw in the PR.

## Guidelines

- Keep the technical pass free of heavyweight/ML dependencies — those belong
  behind the optional `aesthetic` extra and must be imported lazily.
- New tunable numbers go in `winnow/config.py`, not inline.
- Never commit image data, RAW files, `analysis_log.csv`, or model weights
  (they're in `.gitignore` for a reason).

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache-2.0](LICENSE) license.
