# ADR-0422: Report TRAIN input bounds with original fit gaps

## Context

R2.1's acquisition premise remains unmet. Accepted catalogue support already
records54 distinct raw/canonical TRAIN inputs and no conflicting labels; another
alias inventory or baseline rerun would not add evidence. The existing helper
only counts label sets, so it cannot express frequency-aware error floors on
catalogues that do contain conflicts.

## Decision

Add a pure exact-input categorical frequency reducer and compose it through the
existing TRAIN source validator and MiniGrid support adapter. Add default-off
joint-pilot reporting after benchmark training, publishing through existing
evidence seals, and a bounds-only branch in the existing saved-source linear
control CLI. Relate normalized-simplex bounds to unchanged, unweighted TRAIN
scores with signed gaps. No new runner, loader, provenance system or controller
decision is required.

## Alternatives

Reusing only label sets loses multiplicities. Treating lookup attainability as
finite-network capacity would overclaim. Clamping gaps or changing probability
normalization would hide accepted numerical roundoff and change metrics.
Re-running native benchmarks is unnecessary for this saved-source question.

## Consequences

The core has no role, model or IO authority; the adapter admits the exact six
original TRAIN episodes and accounts for reductions, projections and arrays.
Default reports remain unchanged. A late failure preserves paid evidence and
does not publish successful derived outcomes. Current ceiling1/floor0 reuse an
old implication; a23/54 accuracy gap records remaining underfit without claiming
H8 capacity, convergence, episode success, adaptation or circadian benefit.
