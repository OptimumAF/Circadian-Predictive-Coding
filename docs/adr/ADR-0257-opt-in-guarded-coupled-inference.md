# ADR-0257: opt-in guarded coupled inference

## Context

R1.5 requires bounded latent computation, divergence detection and safe fallback.
The coherent coupled PC/CPC learners currently share fixed-step inference and
atomic parameter/chemistry publication. Existing results and defaults must retain
their meaning before any efficiency comparison.

## Decision

Add a pure inference kernel with a frozen per-call policy. The existing step
argument becomes its proposal cap when opted in. Accept only finite nonincreasing
energy candidates; after a declared minimum, check latent gradient infinity norm
and then relative energy drop. Reject unsafe candidates and return the last
accepted pair, without retry. Invalid initial problems raise. Report all attempted/
accepted steps, energy evaluations, norms and comparisons. Preserve the default
fixed loop exactly. PC/CPC use the same result before existing atomic publication.
Adapter fixed policies are copied at construction and fork, outside native state.

Why this: stopping and fallback must be explicit and bounded. Reusing complete
accepted evaluations avoids an extra gradient kernel solely for stopping while
keeping its reduction/comparison overhead visible.

## Alternatives

Changing the fixed default would invalidate legacy comparison semantics. Hidden
backtracking would spend unreported computation and select a new rate. Persisting
a per-call policy inside native snapshots would mix model continuation with
adapter learning settings. Aborting every unsafe proposal would omit the requested
valid-state fallback; returning unchecked candidates would lose its safety property.

## Consequences

Fallback can occur before the minimum and may still drive a valid parameter step.
Strict finite energy rejection may respond to rounding. Native snapshot formats
stay compatible; adapter diagnostics remain the existing native energy port.
No inference-at-prediction, data-role authorization, convergence, parameter descent
or speed guarantee follows. Matched work/time experiments including all stop-check
overhead remain necessary for parent R1.5 acceptance.

`energy_comparisons` specifically counts proposed/current monotonicity checks.
Threshold comparisons and relative-drop arithmetic are not individually tallied;
the counters are not a complete operation count. Any timing comparison must
measure the whole stopping procedure rather than infer savings from counters.
