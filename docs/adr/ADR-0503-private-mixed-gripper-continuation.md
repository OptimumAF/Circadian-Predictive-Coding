# ADR-0503 — Explicit private mixed-gripper continuation

## Context

The accepted fixed PLC study retained known Pick successes but failed Push before
and after8head-GN updates. Cached first inputs exactly match a training input;
XYZ commands improve but gripper remainsopen(+1) againstclosed(−1) supervision.
This is a fit failure before drift, not proof of exact gradient saturation.
The native BP core already implements XYZ-half-MSE plus gripper-logit2z-BCE;
CTC previously exposed only common-MSE/head-GN. Final/probe information must stay
unavailable, and serving/source objective and paid state must stay unchanged.

## Decision

Add opt-in update_rule='bp_mixed' to the existing continuation API; keep
head_gauss_newton as default with unchanged reports/semantics. Admit same-spec
TRAIN targets only, exact normalized gripper axis3 endpoints. Copy the source
without RNG or refitting, preserving momentum/counters/normalization. Explicitly
change ONLY the private fork's objective; accept already-mixed sources without
another objective change. An explicitly supplied StableActor preserves the
source's original objective/state. Use the existing runtime/inbox/budget/clock;
five bounded named owners include one immutable private training seed.

Record returned SGD diagnostics without another model evaluation. Successful
work counters are explicitly source-derived, separate from native failure work.
Hold scalar receipt seals and one latest native state, not a model history. Reuse
the existing native birth/commit guard for SGD so public snapshot callbacks cannot
silently substitute its admitted source. Gradient/proposal arithmetic unchanged.

## Alternatives

A larger head-only dose or new teacher trajectory would not first address the
observed same-input gripper mismatch. A sign decoder would still open for+1.
Changing the source/actor objective would obscure the original serving control.
A separate runtime/loader would duplicate already supported composition.

## Consequences

Default historical common-MSE/head-GN behavior and negative evidence remain.
Mixed loss numbers cannot be compared directly to original MSE. All-layer SGD
has different compute and retention risks; software correctness predicts neither
physical success nor CPC benefit. A later physical experiment needs a separately
frozen data/dose/seed/metric protocol, accounting for source reconstruction, all
updates/reference exposure/diagnostics/native interactions. Other full-goal and
release requirements remain open.
