# ADR-0372: Explicit replay removal in the existing rehearsal recipe

## Context

R2.8 requires component removals. The rehearsal pilot compares rotating and
recent-only retention; both allocate replay storage, sessions and updates. The
shared attribution stream already supports an absent rehearsal helper. Compatible
accepted native no-replay benchmarks exist and should be reused within scope.

## Decision

Add no_replay=False to existing single-case composition and --no-replay to its
CLI. True passes rehearsal=None and allocates only the existing wake sessions;
no replay policy, retention store, helper or replay budget is constructed.
Original geometry, initialization, data, wake clocks, metric directions, delayed
arrival, final-only scoring and source isolation remain fixed.

Why: removing the component must remove its allocations and native exposures.
The result declares replay_disabled=True; no_replay_verified requires complete
validated execution. Shared replay-plan equality is None when no plans exist.
Partial paid wake work survives failure. The fixed recent/tracked presets stay
separate, and conflicting combinations are rejected before work.

Optional existing isolated sixteen-step probes remain supported in independent
forks/sessions, including their original deadlines and charges. Their results do
not reach the benchmark learner or final-data selection. New plans account for
72wakeupdates/576rows, or328totalupdates/2624rows with the probes.

## Alternatives

A helper with a zero quota still allocates bookkeeping and risks changing event
semantics. Another stream runner would duplicate existing orchestration. Reusing
accepted mathematics evidence plus current fake-native integration tests covers
this wrapper increment without repeating completed benchmark training.

## Consequences

This supplies one component-removal control. It establishes no minimal successful
bundle, equivalence, equal compute, replay effect or circadian advantage. Prior
native evidence retains its original source and measured resource scope; it is
not a native execution of the new wrapper. Broader R2.8 remains unfinished.
