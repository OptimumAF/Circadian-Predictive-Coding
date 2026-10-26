# ADR-0333: Match warm-start state before reward comparison

## Context

The accepted dual-objective route preserves BC momentum across reward updates.
Its zero-reward native checks move weights through that momentum. A BC-only
control is needed before comparing endpoints; parameter movement alone cannot
establish reward learning.

## Decision

Add an explicit paired mode to the existing recipe: same fresh architecture,
seed and eight demonstrations in each arm; full native checkpoint equality at8
before any sampled reward work. Freeze the BC-only arm at native8 and continue
the other to16 with its original budget. Hold at most two owned completion
closures so both train before final evaluation. Preserve actual native counts,
original charges and all partial failures; no new orchestration framework.

Require every final context/cue exactly once, retaining both values and all signed
metric differences. Keep different computation, on-policy exposure and carried
momentum explicit. Freeze/review one small protocol before its native comparison.

## Alternatives

Loading old parameter-only results would lose full optimizer/gradient matching.
Training a stronger baseline or changing seeds/metrics from final outcomes would
invalidate the comparison. Relabeling control8 as16 would misstate actual work.

## Consequences

The observed comparison can remain negative or inconclusive. The sole case has
lower reward-arm CE, worse accuracy and no navigation success; it establishes no
useful adaptation or CPC benefit. Reserved per-decision diagnostics are a separate
next increment, preserving the original metrics and evaluation isolation.
