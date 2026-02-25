# Contributing

## Code Style

- Use explicit names and small focused functions.
- Keep `core` free from IO and framework concerns.
- Explain non-obvious design choices with short "why" comments.
- Prefer typed dataclasses for public interfaces.

## Branching / PR Expectations

- One feature or fix per branch.
- Keep changes small and testable.
- Include tests for behavior changes.
- Update docs when user-visible behavior changes.

## Safe Feature Add Process

1. Restate requirement and identify affected layer.
2. Add/adjust tests first when practical.
3. Implement smallest boundary-respecting change.
4. Run checks locally:
   - `pytest -q`
   - `ruff check .`
   - `ruff format .`
   - `mypy src tests`
5. Update `README.md`, `ARCHITECTURE.md`, and ADRs when decisions change.

