# ADR-0327: Reuse the key/door runner for the recurrent control

## Context

The recurrent BC component now has verified BPTT/SGD, complete continuation and
isolated live activation. The actual key/door recipe still has only feedforward
methods; its accepted320 illegal/zero-success results must remain preserved.

## Decision

Extend the existing recipe through a common private execution function. Keep
the original public route/protocol/eight arms intact; add an explicit recurrent
CLI route with only two replay modes and reuse accepted baseline evidence.
Use the same eight complete training episodes, fixed seed/rate, final contexts,
checkpoint points, replay packets, permanent exposure charges and metrics.

Classify each complete episode from zero state and start a fresh frozen owned
session for every live slot. Keep full learned-state guards, partial failures,
unreached slots and actual greedy illegal outcomes. Report recurrent architecture,
parameter/gradient/momentum/activation bytes and unrolled work explicitly; do not
label cross-architecture initialization or PC-neutral parity as applicable.

Freeze/review the descriptor before a sole small local native run. No final data
selects updates, hyperparameters, probes, checkpoints or subsequent favorable runs.

## Alternatives

- Repeating all accepted baseline training adds computation without a changed
  baseline source/assumption requiring it.
- Another orchestration framework duplicates existing role/replay/runtime guards.
- Carrying hidden state between final episodes contaminates their independence.

## Consequences

The new control exercises the real recurring game through ordinary repository
interfaces. Exposure and context comparisons are available, but architecture,
initialization, objective and computation differ. Tiny fixed exposure may fail;
zero success is a valid retained result. Recurrent BC is not recurrent RL and
does not by itself establish acquisition, retention, recovery or generalization.
