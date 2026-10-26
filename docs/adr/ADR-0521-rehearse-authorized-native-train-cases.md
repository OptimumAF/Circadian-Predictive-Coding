# ADR-0521 — Rehearse two authorized native TRAIN cases

## Context

The accepted continuous-AdamW stream retains A but fails B. A declared reset
learns B while losing one A decision and exceeding the anchor drift guard.
Neither recipe satisfies acquisition plus retention. Old-run TRAIN permissions
did not authorize replay and cannot be changed retrospectively.

## Decision

Extend the existing phase delivery and control CLI with explicit rehearsal,
default off. New opted-in initial A exposures prospectively grant replay
permission. Before B, validate exactly two old A TRAIN sources, six successful
authorized deliveries and complete receipts against the inbox's applied updates.
Each replay identifies the first paid source exposure. B0/replayA0/B1/replayA1
repeats three times, using one original inbox/budget/clock for18updates.

Why this: a small old-task rehearsal rule directly addresses observed forgetting
through the existing runtime. One exposure helper owns arrival, delayed label,
paid receipt and callback ordering; splitting this transaction further would
obscure its causal order. No separate replay framework or model state chain.

The control CLI exposes `--rehearsal --optimizer-reset {0,1}`. Continuous mode
requires rehearsal to avoid falsely labeling a no-reset run as a reset control.
Both rehearsal arms use generic endpoint metrics and the same fixed exposures.
Legacy defaults/report shape remain unchanged. Adapter math/state stays intact.

## Alternatives

- More B-only updates do not address retention and alter the frozen old study.
- Another return/reset audit measures another endpoint without improving learning.
- A new buffer/runtime duplicates existing experience/inbox responsibilities.

## Consequences

Rehearsal retains two distinct old examples and adds six paid TRAIN exposures;
it may still fail acquisition, retention or anchor guards. Development/final/
probe rows are never replay sources. Equal18exposures between rehearsal arms
does not imply identical total computation; historical no-replay12controls have
less optimization. No recovery, generalization or circadian benefit follows
from implementation correctness or this small development comparison.
