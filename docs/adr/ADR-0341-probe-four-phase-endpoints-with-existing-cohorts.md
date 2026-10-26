# ADR-0341: Probe four-phase endpoints with existing cohorts

## Context

Declared-context replay yielded one final B-cue success, tied mean return, lost
C skill and no D success. Existing E16 probe helpers support arbitrary bounded
endpoints, but the four-phase caller rejects fresh-task probe composition.
Saved reports do not retain owned endpoint weights.

## Decision

Compose existing disposable E16 cohorts at wake0/256 through the existing runner.
Use separate original 16-update budgets and the same final-data isolation. A new
single-control diagnosis may regenerate the benchmark once and must preserve its
accepted deterministic report projection exactly.

Validate initial completion before admitting benchmark work. Share the existing
per-cohort predicate without changing old two-cohort behavior. Retain one detached
initial record and check it around each existing one-wake stream advance; old
routes preserve their original advance call. Refuse changed evidence and retain
permanent charges. No slot schema or endpoint validator expansion is needed.

## Alternatives

Reimplementing the probe or a new runner duplicates accepted boundaries. Probing
saved scalar reports cannot recover a learner. Rerunning all matched controls is
unnecessary for the selected single-control learnability diagnosis.

## Consequences

Probe computation, input storage, teacher exposure, copies and wall/RSS costs are
explicit. CE gain alone is not isolated plasticity, deterministic E is not broad
generalization, and no comparative circadian advantage follows from correctness.
