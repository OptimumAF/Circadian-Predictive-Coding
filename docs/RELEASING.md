# Releasing

## Pre-release Checklist

1. Run quality checks:
   - `ruff check .`
   - `mypy src tests scripts`
   - `pytest -q`
2. Update `CHANGELOG.md` with release notes.
3. Verify README commands against current CLI options.
4. Ensure benchmark scripts still run on at least one smoke configuration.

## Suggested Tag Format

- `vMAJOR.MINOR.PATCH`
- Example: `v0.2.0`

## Release Notes

Release notes should emphasize:

- circadian algorithm changes
- benchmark protocol changes
- reproducibility or tooling changes
