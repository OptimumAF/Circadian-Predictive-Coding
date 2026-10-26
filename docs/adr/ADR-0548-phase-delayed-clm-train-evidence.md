# ADR-0548 — Phase-delayed TRAIN memory beside CLM

## Context

Owned CLM updates, typed serving and isolated fresh-task diagnostics already
exist. Native probes show severe forgetting, including rehearsed TRAIN points;
state-only controls give mixed results. Conventional causal evidence memory
already works in the reference/native workflow paths. Repeating a checkpoint,
parameter-scope variant or accepted typed native case does not address this gap.

## Decision

Add an opt-in observer to the existing catalog stream's checkpoint hook. Reuse
DecisionEvidenceStore, apply_decision_feedback and decision_from_evidence.
Require original raw reuse consent and the actual complete canonical none-recipe
catalog receipt/label prefix before admitting TRAIN mirrors. Record at phase
checkpoints; preserve earlier observation and label clocks without backdating.
Keep head updates and encoder calls unchanged, query current-policy evidence
separately, and score only after TRAIN. Bound events, text, context and requests.

Why this: the task already has fixed raw templates and a protected synchronous
checkpoint boundary. Composition supplies usable memory without another runner,
optimizer change or access to private probes or final labels.

## Alternatives

Per-label general event hooks would broaden the app interface without a present
need. Reconstructing a public actor from saved predictions cannot recover trained
weights. Another rate/scope/dose variant would repeat a partially explored learner
hypothesis without establishing conventional memory behavior first.

## Consequences

Memory is phase-delayed, requires an explicit policy-version signal and lexical
overlap, and may be unavailable for previously unobserved current-version tasks.
This is fixed-corpus trusted mapping, not general external-feature attestation.
Raw storage/retrieval costs are additional to CLM compute; head-only comparisons
are not equal-memory or equal-information comparisons. One-hot retrieval is not
calibrated head confidence. Native usefulness and broad scientific/release claims
need separately frozen evidence; no such allowance follows from software checks.
