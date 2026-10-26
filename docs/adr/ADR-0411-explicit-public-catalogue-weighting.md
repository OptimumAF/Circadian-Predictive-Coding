# ADR-0411: Explicit public catalogue weighting

## Context

The typed catalogue update and existing weighted stream support atomic BP/PC/
neutral continuation, but the public joint runner still used ordinary learners.
Online per-episode weighting has different source semantics and exposure; changing
its flags to mean catalogue weighting would silently change existing experiments.

## Decision

Add weighted_catalogue/--weighted-catalogue to the existing joint API/CLI and a
small route module. Require the fixed explicit BP/PC/neutral joint recipe with
distinct F probes, old-A readouts and A-return. Validate six original complete
TRAIN sources and every one of128 independent54-row transports before constructing
the bound from_catalogue learner. Weight aggregate TRAIN class mass only.
Retain original seed, architecture, policies, doses, metrics and support checks;
name the different objective and protocol explicitly.

Create separate ordinary unweighted source templates, prove complete initial
state identity with the zero-update benchmark source, and restore checkpoints
only on disposable probe forks. Reuse existing paid budgets, source/actor/state/
scorer guards and original ordinary probe diagnostics. Never train on final data.
Original online flags remain distinct and reject the joint mode as before.
Existing PC realized-update diagnostics refuse the new flag because they describe
an unweighted update law. Readonly margins/live-port admission is retained;
those optional combinations are outside this increment's native coverage.

## Alternatives

- Reinterpreting online flags would mix objectives and historical baselines.
- Training probes on the weighted schedule would break source authority/isolation.
- A new runner or duplicated update law would add avoidable maintenance cost.

## Consequences

All three full native routes and an ordinary joint BP regression pass, alongside
243 scripted cases, scoped statics, actual CLI description and independent review.
This is implementation evidence from synthetic sources, not learning benefit.
Factory validation creates a transient second128-wake array plan (~16.4MB),
adding time/RSS without extra native updates. Source/candidate/actor/fork copies
and ordinary templates are also paid costs. A small frozen real-data protocol
with qualified historical joint controls remains next; old partial results and
the missing probe-probability evidence criterion remain unchanged/open.
