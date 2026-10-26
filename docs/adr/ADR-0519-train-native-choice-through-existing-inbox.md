# ADR-0519 — Train native Choice through the existing inbox

## Context

The accepted native six-case reference supplies inference, not adaptation. The
existing NativeLearner and ExperienceInbox accept opaque inputs, so typed text
requests do not need an array projection or another orchestration framework.
The pinned Open-Jev training CLI requires a GPU/Git/data setup beyond this local
one-update slice; its hard Choice objective and optimizer are directly reusable.

## Decision

Implement an optional CPU float32 adapter with the original trainable selector,
NLL plus .1 squared Brier, separate native AdamW learning rates and clipping.
Deliver TRAIN sources and delayed labels through ExperienceInbox. Use four
frozen development readouts, keep inference temperature inherited and explicitly
unrecalibrated, and save only a native inference package to a fresh directory.

Own trainable weights, AdamW, gradients and private CPU RNG in model-bound local
snapshots. Frozen base parameters use identity/Tensor-version guards; they are
not copied. Attempt/work counters survive restore. Uncertain mutation poisons
the learner and raises TrainingUpdateFailed; no retry or budget renewal.

## Alternatives

An array learner would lose native prompts and scalar-head semantics. Running
the complete upstream CLI would add unrelated datasets/calibration and GPU
requirements. Forking the whole base would add substantial memory without a
need in an ordinary inbox update. None is required for this slice.

## Consequences

No actor fork, hostile-Python security boundary, portable optimizer resume,
calibration quality, BF16 parity or acquisition/retention/return claim follows.
Native update correctness is a prerequisite to a separately frozen adaptation
study; all full R5.4 comparison criteria remain open. Probe/final labels are
unavailable to this TRAIN delivery path; readout labels stay outside callbacks.
