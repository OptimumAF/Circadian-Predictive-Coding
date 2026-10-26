# ADR0433: compose the BP readout at the owned joint checkpoint

## Context

The accepted BP diagnostic API uses actual SGD/momentum, but its saved-TRAIN
example has only one warmup call. It cannot observe the historical joint128
state. The existing MiniGrid pilot already owns checkpoint128 before F collection.

## Decision

Add `--bp-update-diagnostics` to the original BP joint128/F16/old-A/A16 route.
Run the fixed B-blue/A-blue/A-red TRAIN cycle on a disposable copy of that
checkpoint, with a separate three-update/eighteen-replay-row budget. Bind the
original allowance before sampler enrollment, then retain its full binding,
charges and completed clock. Register the scalar result in the parent's existing
evidence owner/bytes guards, and check it before later probes. Preserve paid
prefixes on failure and include native/replay work in overall and per-method
costs. Defaults, benchmark doses and reserved F/A probes remain unchanged.

Why this: the existing stream owns the required optimizer state and original
sources. Composition answers which state was observed without another runner,
checkpoint format or evidence framework. F is collected only after the TRAIN
observer has completed, so it cannot choose or train the diagnostic.

## Alternatives and consequences

The one-warmup CLI stays useful as an example but is insufficient for joint128.
Changing momentum or extending training would change the question. Observer
native work costs three calls/27 rows, including18 replay exposures, plus15
diagnostic CE forwards/135 rows and9 backwards/81 rows. Validation, copying,
projection, process memory and elapsed time are additional costs.

The synthetic public route passed independently reviewed engineering checks.
It supplies no real-source interference explanation, acquired retention,
recovery or circadian advantage claim. A small reviewed real-source protocol
is required before new scientific use; all negative results remain preserved.
