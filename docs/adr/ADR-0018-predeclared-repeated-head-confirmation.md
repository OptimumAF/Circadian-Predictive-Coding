# ADR-0018: Freeze repeated matched-head confirmation before final test

## Context

The equal-trial matched-head tuner previously selected on outer validation
and immediately confirmed the selected heads on final test. A separate
repeated comparison needs a record of the selected configurations, seed set,
metrics, and budget scopes before any confirmation labels are read. A single
tiny result or a post-test seed choice would not support a fairness claim.

## Decision

Add a validation-only mode to the existing tuner. It uses the same bounded
candidate ledger and deterministic outer-validation selection, then returns
without accessing the test loader. A confirmation manifest freezes that
result's digest, one selected candidate per head, three or four disjoint
confirmation seeds, accuracy and cross-entropy metrics, and fixed-data,
per-head wall-time, and process-isolated capacity/memory scopes. The manifest
has a canonical SHA-256 digest to detect accidental changes. It is written
to disk before confirmation begins; it is not an authenticity signature.

The confirmation runner trains all fixed-data heads and seeds before opening
their final-test loaders. It reuses the selected settings in a separate
wall-time route and an untimed process-isolated memory route. It verifies
paired split, feature, backbone, and initial-head hashes per seed, equal
fixed-width starting/final parameters, and no circadian structural change.
It retains every declared raw run and reports mean and population standard
deviation without selecting a winning seed or head. Wall-time training begins
after feature and head setup, while memory children do not read final test.

The repeatable CPU smoke uses one selection seed, two candidates per head,
three disjoint confirmation seeds, eight synthetic examples per role, one
fixed-data epoch, and a 0.05-second wall-time deadline per head. Its three
JSON artifacts preserve selection, manifest, and all raw result rows.

## Alternatives considered

- Confirming directly from the tuning result would open final test before a
  durable, reviewable seed and budget declaration.
- Using final-test accuracy to choose a seed, configuration, metric, or
  budget would contaminate the confirmation.
- Combining epoch, wall-time, and memory observations into one score would
  hide unequal relaxation, guard, and sleep work and different memory scopes.

## Consequences and open work

The tiny random-feature CPU evidence is a correctness and descriptive
reproducibility gate, not an accuracy ranking. In its fixed-data scope all
three heads had 0.25 mean test accuracy. Under the 0.05-second wall-time
scope, circadian mean accuracy was 0.167 versus 0.5 backprop and 0.458 PC;
these negative results remain in the artifact. Circadian did fewer wake
updates and more guard/sleep work within the deadline. Separate-process RSS
contains the runtime, backbone, and feature cache and is only an observed
peak. Larger-data and real CUDA evidence, broader stream replay checks, and
full P1.8/P1.7 conclusions remain open.
