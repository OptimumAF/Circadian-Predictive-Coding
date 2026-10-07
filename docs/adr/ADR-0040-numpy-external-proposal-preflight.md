# ADR-0040: Validate NumPy external topology requests before mutation

## Context

`apply_neuron_proposals()` and policy-driven `sleep_event()` accepted
`NeuronChangeProposal` values, but the adapter silently ignored invalid
remove indices, aggregated duplicate removals, and could fill requested
split counts with neurons on cooldown. The direct path could exceed the
configured split/prune count and width limits; a split could select the
same original neuron an explicit request then pruned. In these cases the
reported proposal and actual topology could disagree.

## Decision

Validate the complete external/policy request before any tensor or local
RNG mutation. Only the adaptive `hidden` layer is supported. Require a
nonnegative integer add count and unique in-range integer remove indices.
Enforce configured split/prune budgets, the existing per-action change
fraction rule, intermediate maximum width, minimum width including
pending gradual prunes, and prune age/cooldown/mark eligibility. Policy
requests that reach structural selection during sleep must fit the
resolved sleep budgets; they are no longer silently truncated. A sleep
event with both structural budgets at zero still skips policy invocation,
as before. A direct external request uses the configured count and
fraction limits without an adaptive sleep-budget scale.

For requested additions, rank neurons by the existing split score,
preferring those above the current threshold, then eligible fallback
neurons. Policy control may request growth below the automatic threshold,
but cannot bypass split cooldown or a pending prune. An explicit prune
request owns its original neuron and excludes it from split sources; the
next eligible source is chosen, or the request is rejected. Split indices
refer to the pre-event topology, and the existing split-then-prune tensor
order remains. This preserves valid policy behavior while making overlap
deterministic.

Why reject instead of clamping: a policy's requested work is an
experimental input. Silent clipping changes the intervention and can
make telemetry claim removals that did not occur.

## Alternatives and consequences

Clamping or ignoring invalid requests would retain historical permissive
behavior but conceal mistakes. The new rejection is an explicit behavior
change for malformed requests and policy requests over a resolved sleep
budget. Automatic NumPy and Torch selectors are still separate and remain
P3.4b; stable neuron IDs and parent lineage remain P3.4c. P3.7 will add
atomic recovery for errors that arise after a valid proposal commits.

## Evidence

`tests/test_numpy_proposal_preflight.py` first failed because an invalid
remove index was silently ignored. It now checks range, duplicates,
negative count, count/fraction and width limits, age/cooldown, unsupported
layers, policy budget rejection, no state/RNG mutation on rejection, and
prune-over-split priority. The development log records the full gate.
