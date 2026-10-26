# ADR-0564 — Ship exact saved ticket evidence through package resources

## Context

The saved-task inspector works, but its examples reference unbundled local
artifact paths. A fresh wheel user cannot inspect those task outcomes. The CLI
already has an installed-resource pattern for the continuous comparison.

## Decision

Add one mutually exclusive `--example-ticket` option with four explicit families.
Reuse checkout-first lookup and `importlib.resources.as_file` lifetime. Map exact
report copies through existing setuptools resources/source manifest, with original
provenance and accurate project-generated numeric-state license notices. Keep
explicit source and continuous-example choices unchanged. Verify selected bytes,
zip-resource cleanup and exclusive publication, then build/install/export offline
in an isolated prefix without research-source imports. No new dependency/model.

Why this: reuse makes existing useful/negative evidence portable, without another
loader, experiment, dataset selection or orchestration framework.

## Alternatives

- Require the whole research archive: unnecessary and prevents installed use.
- Strip snapshots/negative endpoints: alters the original source and interpretation.
- Retrain examples: adds unneeded work and new numerical assumptions.
- Introduce a generic resource registry: excessive for four fixed reports and two
  existing source modes; explicit lookup keeps the public interface bounded.

## Consequences

The wheel/source archive grow by approximately1.83MB of original uncompressed
JSON. This contains project-trained numeric state, not third-party released
weights or SDK assets. Viewing does not execute those models or establish new
science. Fresh native execution, stronger comparison/release/GPU/clean-clone
requirements remain open; scoped self-review is not independent review.
