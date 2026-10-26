# ADR-0413: Public weighted PC checkpoint readouts

## Context

Weighted joint PC/neutral fit16/54 TRAIN decisions in the fixed study. The exact
weighted numerical diagnostic is accepted, while public checkpoint readouts had
only the unweighted schema. A diagnostic must measure the actual proposal and
pay its costs without changing training, probes or original evidence guards.

## Decision

Allow existing weighted_catalogue and pc_update_diagnostics together only for
the original explicit PC/neutral joint recipe. At0/128, after six paid raw scores,
derive fixed aggregate weights through existing catalogue preparation from only
the original six TRAIN sources and call the accepted detached numerical API.

Extend existing work aggregation and record-container identity/value guards for
weight derivation, numerical proposal, fixed-state energy and weighted payloads.
Retain derivation failures and partially returned diagnostic work. Preserve the
original pilot budget, benchmark/actor/source/receipt/checkpoint guards and two
ordinary isolated probe templates. Defaults remain unchanged; BP still refuses
PC diagnostics. No new runner, core objective or training update is introduced.

## Alternatives

- An unweighted diagnostic would describe the wrong finite proposal.
- A new orchestration framework would duplicate existing checkpoint guards.
- Ignoring new fields during paid-work or owner checks would weaken evidence.

## Consequences

284 current cases qualify through retained composite evidence, scoped statics,
actual CLI description and independent review. Two new weighted full native
routes and original unweighted full routes execute. The initial stale CLI
negative fixture failure remains saved; its repaired BP-negative family passes
and new PC/neutral-positive cases cover the extension. This is engineering
acceptance, not scientific explanation of underfit or comparative benefit.
