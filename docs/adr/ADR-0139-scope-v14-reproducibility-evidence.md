# ADR-0139: Scope exact v14 reproducibility evidence to the recorded environment

## Context

P5.7 asks for a repeated small run, model-order independence, and honest
CPU/GPU and cross-version tolerances. The P5.1 fixed v14 manifest seals
model order; changing it would create a new protocol. The corrected
configurable continual runner can legitimately permute method order.
The v14 bundle contains deterministic NumPy training and outcome bytes,
while run ID and measured runtime facts have different meanings.

## Decision

Repeat the same fixed-v14 preset in two separate local processes and
require exact deterministic payload bytes after both bundles verify.
Compare resolved configuration, source, seed, role, algorithm, precision,
dependency, and hardware facts; permit run identity to differ explicitly.
Test model-order independence in the corrected configurable continual
route, comparing trained states and outcomes while excluding measured
durations and normalizing order-bearing audit events. State clearly that
this is not a reversed fixed-v14 run.

For different CPUs, library builds or versions, and CPU/GPU execution,
there is no empirically validated score tolerance. Require protocol and
data-role identity first, publish all per-seed numeric deltas, and keep
equivalence unverified until a separate study predeclares a tolerance.
The fixed v14 track has no Torch/CUDA implementation, so a GPU tolerance
for it is not applicable. Record this policy and the observed hashes in
`docs/reproducibility-scope.md`.

## Alternatives

- Reorder the sealed v14 manifest: rejected because that changes the
  declared study and would bypass the existing validation gate.
- Treat matching aggregate scores as reproducibility: rejected because
  seed-level trajectories, source roles, and protocol identity could differ.
- Declare an unmeasured universal CPU/GPU numeric threshold: rejected
  because it could hide ranking changes and lacks paired evidence.
- Compare timing or memory byte for byte: rejected because they are
  measurements of execution conditions, not deterministic model output.

## Consequences

Exact bytes are verified for one recorded Windows/NumPy CPU environment,
and model-order invariance is tested in a valid configurable protocol.
The documentation does not claim bitwise or numeric equivalence on
untested devices or versions. A later cross-platform study must define
its own track, matched sources, and tolerance before evaluating scores.
