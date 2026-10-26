# ADR-0549 — Evidence-first serving with a frozen fallback

## Context

Native TRAIN memory can recall declared facts without changing the CLM head.
It also returns unavailable for A parcel and B password. Existing retrieval mode
refuses missing evidence; existing context mode always invokes the predictor.
A useful caller needs a bounded explicit way to combine memory availability
with an existing frozen typed predictor.

## Decision

Add mode="fallback" to EvidenceDecisionPredictor. Retrieve under the original
cutoff/version/allowed IDs; answer from eligible memory. Only an empty selection
invokes the supplied callback, once, with a detached unchanged original request.
Reuse existing typed return validation, write/request guards, error retention
and finite call limits. Record choice_source without changing old mode reports.

Why this: deterministic availability requires no confidence threshold, training,
metric selection, new runner or model-specific port. Existing CLM/other frozen
predictors already implement the callable boundary and retain their own budgets.

## Alternatives

Always-context adds inference even when memory suffices and changes the input
recipe. Choosing between model/memory scores would introduce a calibration and
tuning requirement. Silent fallback in retrieval mode would change existing
callers' refusal and cost semantics. A separate orchestration class duplicates
the current protected callback path.

## Consequences

Memory can still be wrong; a valid adverse fallback stays adverse. No native
hybrid performance or generalization follows from scalar composition checks.
The explicit mode shares one wrapper call allowance across both sources;
callback budgets and physical work remain owned by the frozen predictor.
Final/probe labels, source training, promotion and external effects remain absent.
