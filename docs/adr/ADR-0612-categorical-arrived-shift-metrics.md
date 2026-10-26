# ADR-0612: Extend arrived shift diagnostics with explicit class semantics

## Context

The existing arrived-only observer accepts binary B×1 probabilities and soft
targets. Working ticket learners use six action classes. The stream already
supports categorical TRAIN batches and has a separate unlabeled feature callback;
the existing detector/calibrator/incidents accept bounded scalar ShiftMetrics.

## Decision

Add a separate pure categorical summary, reusing current categorical validators,
bounded feature shapes and the original ShiftMetrics family. Use normalized
entropy and mean1-p_true residual. Add explicit observer num_classes default2;
K>=3 binds exact batch class metadata, chooses the new summary, and identifies
metric definitions and diagnostic input-row costs in its report. Binary output,
soft-target meaning, policies, thresholds and numerical formula remain.

Why this: scalar detector logic needs no new framework or learner, while an
implicit binary-to-categorical conversion could alter preserved experiments.
Shapes are bounded before shared array validators allocate checks. No incoming
probabilities are renormalized to make a malformed predictor appear valid.

## Alternatives

Argmax-only mistakes discard uncertainty. Feeding a single selected probability
to binary entropy misstates six-class uncertainty and target semantics. A new
detector/scheduler duplicates existing machinery. Moving label-based diagnostics
to a prearrival callback would violate causal data access.

## Consequences

Diagnostics remain retrospective: after update k, score arrived TRAIN with copied
state k-1. Trusted stream authority admits labels; arbitrary direct calls cannot
prove arrival. Delegates receive copied features/states, and failures retain paid
attempts/normal returns before refusal. Fixed calibration/incidents work without
rebasing or threshold tuning. Actual model forwards/copies still require caller
budgets; fixture zero-model accounting is not evidence of native overhead.

174 focused cases/platform and scoped statics pass. Existing delayed K6 stream
behavior/receipts/budgets are exact under a scripted delegate; this is software
and isolation evidence, not learned adaptation, deployable false-alarm control
or a scientific advantage. Actual ticket integration/observation remains next;
R2.7 and wider parents stay open. Original negative studies are preserved.
