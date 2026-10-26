# ADR-0598 — Bind existing controller learning to the contextual source

## Context

The contextual conventional learner demonstrates acquisition/retention, and fixed
references have scoped task evidence. The older fitted controller uses a different
native BP source and cannot be transplanted. A new scheduler is unnecessary.

## Decision

Add opt-in exact contextual policies to the existing live helper, with complete
source/version binding for fitted policies before model construction and a
mutation pin throughout execution. A finite helper runs four ordinary TRAIN-only
counterfactual trials through the existing schedule, freezes the original small
chooser, then applies it once through the original worker/promotion ports.

Why this: reuse the existing controller's causal and version contracts while
measuring classifier fit separately from deployed task usefulness. Do not fit
from accepted fixed-control task outcomes, probes or final evaluation.

## Alternatives

Transplanting the BP fit would violate source identity. Selecting a policy from
the application would leak its outcome into fit. Adding another runtime would
duplicate the existing stream/admission behavior. None is required here.

## Consequences

The original fixed-reference defaults stay intact. Full discarded-trial work and
monitor/ownership/copy costs are paid. Four clean/mistakes estimates cannot express
pending repair or hidden shadow state; task-level benefit and unseen-stream
generalization remain unproven. Publication can stale the fitted policy and is
not an authorization to renew or rebind it. No CPC/efficiency/release claim follows
from this development composition. See the frozen local protocol and results.
