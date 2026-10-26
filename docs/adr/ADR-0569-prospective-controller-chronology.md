# ADR-0569 — Prospective controller utility and bounded stage admission

## Context

R7.2 supplies causal reference actions. R7.3 requires fixed normalized utility
and separate controller development, validation and sealed evaluation. Existing
native task, role, clock and stream ports already execute the relevant behavior;
older binary-specific role splitters do not describe these categorical tasks.

## Decision

Place pure scalar objective/metadata records in core and bounded three-stage
admission in app. Declare unit weights and explicit wall/RSS normalizers before
execution. Reject unknown costs. Bind one freeze identity to original fixed
policy/objective/streams/clock; expose stage feedback only through its appropriate
interface. Preserve paid failed attempts and stop further source callbacks.

Compose opt-in --outer-study in the existing serving script. Reuse its native
owners and admission/guard paths with actual new-context and old-return arrays.
Use a separate outer trial clock. Open the sealed episode source only after
freeze, then evaluate original and candidate with detached full-state readonly
readers. Canonicalize fresh observation dictionaries with JSON before the
unchanged strict snapshot decoder. Include source construction inside cost
measurement. Keep full-stream completion unchanged.

## Alternatives

A new orchestration/source-hash framework duplicates existing boundaries.
Repeated scoring against old endpoints is retrospective tuning. Broadly relaxing
the snapshot decoder would weaken an unrelated correctness boundary. Treating
missing work as zero would reward failed trials. Treating distinct IDs as
statistical independence would overstate this engineering evidence.

## Consequences

One fixed reference can execute and report all stages with bounded state and
causal callback admission. It is not trained-policy version binding, statistical
meta-generalization or an efficiency/CPC claim. Delayed/partial information,
counterfactual trial costs and policy fitting remain R7.4 work. Shared grammar
and pretraining require broader independent task streams for scientific claims.
Single-agent source/procedure/result review is self-review, not independent
acceptance. Historical negative evidence remains unchanged.
