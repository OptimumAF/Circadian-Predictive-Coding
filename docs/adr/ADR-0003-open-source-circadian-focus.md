# ADR-0003: Establish Open-Source Baseline With Circadian-First Positioning

## Context

The repository has matured beyond internal experimentation and now needs a professional open-source baseline:

- clear governance and contribution process
- reproducible CI checks
- security and community reporting channels
- explicit positioning of Circadian Predictive Coding as the main project focus

## Decision

Adopt a full open-source community and engineering baseline by adding:

- `LICENSE` (MIT)
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- `SUPPORT.md`
- `GOVERNANCE.md`
- GitHub issue/PR templates
- CI workflow and dependency update automation
- standardized tool config in `pyproject.toml`
- improved top-level docs centered on Circadian Predictive Coding

## Alternatives Considered

1. Keep documentation-only changes without process files.
   - Rejected: does not provide contributor safety, triage standards, or CI reliability.
2. Publish only algorithm code and defer governance.
   - Rejected: increases contributor friction and maintenance risk.
3. Maintain equal project emphasis on all three models.
   - Rejected: dilutes roadmap clarity. Baselines remain, but circadian is primary.

## Consequences

- The repository becomes collaboration-ready for external contributors.
- Contribution and review expectations are explicit and enforceable.
- Circadian development has clear product identity while retaining rigorous baseline comparisons.
