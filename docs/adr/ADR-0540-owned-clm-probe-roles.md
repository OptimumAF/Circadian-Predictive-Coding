# ADR-0540: Keep CLM probe roles and optimizer ownership separate

## Context

The CLM temporal recipe demonstrates limited adaptation and failed return-rule
transfer, not a loss-of-plasticity diagnosis. Existing generic probes admit a
different float64 AttributionBatch/ForkableLearner interface; CLM uses paired
CPUfloat32 embeddings and native head ownership. Converting reserved labels to
TRAIN or widening benchmark TRAIN admission would break evaluation isolation.

## Decision

Declare exact probe_train paired features and probe_train/probe_test/old-task
outer_selection readout features in core.clm_probe. Compose a disposable public
CLMHeadProbe around a private native learner that admits only these probe types.
Extract the original validation bodies into protected record-kind helpers; normal
public CLMHeadLearner still binds its original TRAIN/readout kinds. Share the
native paired objective, atomic staging, optimizer and permanent failure charges.

Copy warm heads/scale/modes and sourceRNG while preserving source native state,
logical progress and all physical counters. Probe owns fresh AdamW/OneCycle and
cleared gradients, with explicit finite updates/readouts and zero restores.
No public snapshot/fork/restore/export/install path. Source may already be spent:
this measures warm-weight learning under a separately paid optimizer protocol,
not continuation, a source budget renewal or a deployed update.

## Alternatives

- Coerce probe features into normal TRAIN: rejected, would lose role separation.
- Widen the generic AttributionBatch contract: rejected, would mix incompatible
  paired feature and target semantics without solving native ownership.
- Duplicate CLM training math: rejected, would create a drift-prone second engine.
- Continue spent source optimizer: answers a different question and cannot renew
  the original paid schedule. Keep future control protocols explicit.

## Consequences

The supported probe horizon is1..16 except10: unchanged native OneCyclepct_start.1
has a zero-length first phase at10 and fails its initial step. Reject before
copies/construction; no native schedule workaround. Readouts contain no labels;
probe_test/outer_selection targets stay in the caller's post-training scoring.
No final access, model download, new dependency, encoder or serialization system.
Private Python objects are trusted, not a malicious-code containment boundary.

This is a necessary R2.1 interface, not a new-task or generalization result.
Compose through the existing temporal runner and freeze/review a small reserved
new-task/old-task protocol before native use. Preserve all earlier negative
findings and fullR2.1/R5.1/R5.3/G5/release requirements.
[Software protocol](../../artifacts/runs/clm-owned-probe-20261022/PROTOCOL.md).

## Scoped outcome

97 tiny checks per platform and scoped statics pass after a test-only union
annotation repair; source/software/results independently accepted. The spent
source remains unchanged after separate probe updates/readouts and lost-return
failures. Full temporal/native probe integration and capacity evidence remain
open. [Executed evidence](../../artifacts/runs/clm-owned-probe-20261022/RESULTS.md).
