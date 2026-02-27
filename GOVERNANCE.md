# Governance

## Project Focus

The primary focus of this repository is the Circadian Predictive Coding algorithm and its evaluation
against standard backpropagation and traditional predictive coding baselines.

## Maintainers

Maintainers are responsible for:

- Reviewing and merging pull requests
- Curating releases and changelog entries
- Preserving architecture boundaries
- Keeping benchmark protocols reproducible

## Decision Process

- Small changes: maintainer review on pull request
- Major changes: document with an ADR under `docs/adr/`
- Breaking changes: require explicit migration notes in the PR and changelog

## Contribution Bar

To merge, changes should:

- Pass tests and lint checks
- Include docs updates for user-facing behavior changes
- Preserve fair, apples-to-apples comparisons across all three model families
- Avoid introducing hidden coupling across layers
