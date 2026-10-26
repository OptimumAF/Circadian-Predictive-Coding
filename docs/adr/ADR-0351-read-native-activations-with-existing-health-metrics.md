# ADR-0351: Feed native activations into the existing health reducer

## Context

The accepted public-history diagnostic removes an observed conflicting input pair,
but still fails both-cue task acquisition. The pure tanh health reducer already
exists. Legacy/coupled unit-history adapters do not read native Torch activations.
Changing dose, seeds or algorithms before observing the missing states would
obscure this dependency.

## Decision

A small adapter accepts exact validated feature/recurrent snapshots and native
feature arrays, owns copies, and returns detached float64 hidden rows computed
with actual float32 native operations. Feedforward health observes tanh priors,
including PC/CPC, without labels or inference relaxation. Recurrent health starts
from zero for each complete episode and reuses the existing forward one row at
a time. Its discarded output projections are explicit diagnostic computation.

Add an optional checked branch to that existing recurrent forward to reject
nonfinite pre-tanh values. Default arithmetic/order remains unchanged. Why:
tanh can map affine overflow to finite saturation, which must not become valid
health evidence. Do not duplicate recurrent math or intercept global functions.

An opt-in flag on the existing history CLI adds readouts only to known TRAIN
checkpoint cells. Reuse diagnose_activations unchanged, retain raw hidden matrices
and fixed definitions, and account diagnostic calls, rows, arrays, SVDs, recurrent
work and elapsed time separately. Failed health must preserve already paid raw
scores and successfully returned hidden rows. Original no-flag recipes remain.

## Alternatives

NumPy recomputation would change native precision/operation order. Reimplementing
the reducer would duplicate accepted metrics. Label-relaxed PC activations would
change information access. A separate runner or state archive is unnecessary.

## Consequences

The diagnostic adds work and storage but no SGD or training signal. Existing
state/RNG/roles/budgets remain binding; no final or probe input enters health.
Metrics describe activity and rank, not causal plasticity, importance, neuron age,
live skill or advantage. A frozen integration case regenerates prior states
because old artifacts do not serialize them; it is not independent replication.
