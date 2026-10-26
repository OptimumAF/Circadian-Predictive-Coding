# ADR-0391: Read bound prediction gradients in the existing TRAIN diagnostic

## Context

Matched neural MiniGrid acquisition remains unsuccessful in the preserved cases.
Per-decision/activation diagnostics and isolated fresh-task probes exist, but the
existing gradient reader accepts only recurrent snapshots. Bound feature heads
explicitly refused gradient diagnostics. More capture infrastructure does not
address this learner gap.

## Decision

Add a separate opt-in prior-gradient reader in adapters and compose it through
the existing TRAIN fit readout, stream and CLI. Use unchanged BP/PC/neutral/active
snapshots, the same public projection and prediction function, and mean CE on
original TRAIN labels. Differentiate detached clones; use the existing pure core
geometry reducer. Read complete original six TRAIN sources after fixed learning,
before separate probes. Preserve complete state/RNG/gradient metadata and paid
failed prefixes. Keep old recurrent diagnostic/default protocols unchanged.

Why this: a common prediction objective makes a precisely defined readout possible
without inventing a native PC update direction or another orchestration framework.
The prior is `tanh(x W1+b1)`; this derivative does not differentiate latent inference
or chemistry. Cosine is descriptive and does not establish causal interference.

## Alternatives

- Reuse the recurrent flag: rejected because it would silently change objective
  meaning and admission for an existing protocol.
- Differentiate the native PC latent objective: requires a distinct mathematical
  definition and comparison protocol; remains separate work.
- Add another learner runner or rerun failed gameplay: neither is a prerequisite.

## Consequences

Four parameter groups plus joint geometry and two losses are scalar JSON.
Projection, forward/backward rows, logical bytes and reader time are additional
paid diagnostics, not training/probe updates or full process-memory measurement.
Detached source validation/copies/guards and autograd still cost time/memory.
The original whole-pilot budget includes the procedure; guard/copy time is outside
the nested reader-seconds field; internal validation, copies and reducers are
included in reader seconds. No temporary optimizer is constructed by the
feature-state copier. Explicit bound-neural history/full-six-source admission
excludes A-only/count/recurrent combinations. No new dependency/config/env var.

Scoped123 tests/statics/independent review qualify implementation only. A new
scientific use must freeze its protocol, preserve old negatives and measure all
costs, including actual RSS. Full goal and neural adaptation claims remain open.
