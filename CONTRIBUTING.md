# Contributing

## Project Direction

This project is centered on Circadian Predictive Coding.
Backprop and predictive coding baselines are maintained as comparison anchors.

Contributions should improve one or more of:

- circadian algorithm quality
- benchmark rigor and reproducibility
- engineering reliability and clarity

## Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional benchmark dependencies:

```powershell
pip install -r requirements-resnet.txt
```

## Branch And PR Workflow

1. Create a focused branch from `main`/`master`.
2. Keep scope narrow and architecture-consistent.
3. Add or update tests for behavior changes.
4. Update docs for user-facing changes.
5. Open a PR using the repository template.

## Required Checks Before PR

```powershell
ruff check .
mypy src tests scripts
pytest -q
```

## Coding Standards

- Keep `core` pure and free from dataset/CLI concerns.
- Prefer explicit dataclasses for configuration surfaces.
- Fail early with actionable error messages.
- Avoid hidden coupling between model families.
- Keep benchmark comparisons fair:
  - same dataset split
  - same evaluation protocol
  - clear disclosure of differing hyperparameters

## Documentation Standards

When behavior changes:

- update `README.md` for usage changes
- update `ARCHITECTURE.md` for boundary or flow changes
- add/update ADRs for major decisions in `docs/adr/`
- add an entry in `CHANGELOG.md`

## Commit Guidance

- Use concise, descriptive commit messages.
- Separate refactors from behavior changes where practical.
- Do not include generated benchmark artifacts unless intentionally publishing results.

## Issue Triage Priorities

1. Reproducible correctness bugs
2. Benchmark regressions
3. Circadian adaptation stability/performance issues
4. Documentation and DX improvements
