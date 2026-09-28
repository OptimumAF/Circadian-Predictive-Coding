# ADR-0029: Guard circadian and Torch head update commits

## Context

The NumPy circadian trainer can decay a marked neuron, update chemical and
reward state, and write output parameters before a later hidden candidate
overflows. A gradual prune can even remove a neuron before that failure.
The Torch PC heads likewise wrote parameters before checking candidate
finiteness and returned an infinite squared-error diagnostic after mutation.
The Torch circadian head also decayed cooldowns and advanced adaptive state
before late numeric failures.

## Decision

The NumPy circadian step validates stored numerical state, computes its
pre-update diagnostic, and stages candidate parameters, traffic, age, and
adaptive values before committing. Provisional chemical, importance, and
reward values are restored if their calculation or candidate check fails.
When a gradual prune is active, the step snapshots the arrays that prune
decay can change, including topology and per-neuron metadata, and restores
them after a nonfinite rejection. Cooldowns decay only after candidate
validation. Replay storage and accepted sleep events retain their existing
separate behavior; P3 owns a general full-state sleep transaction.

Both Torch PC heads now check all candidate parameters and their existing
diagnostic before assignment. The circadian head restores provisional
chemical, importance, and reward state after a failed candidate check;
cooldowns decay after that check. The finite reductions feed the diagnostic
`.item()` read that was already present, so the valid path retains its one
post-update host synchronization. It adds reduction kernels and tensor
operations, which belong in the measured training time under P1.8. Entry
validation and relaxed-state checks still each synchronize separately, and
reward modulation may add its own host read. This statement is about call
placement, not a measured CUDA throughput claim.

## Evidence and limits

Deterministic finite input/rate fixtures produce nonfinite hidden-weight
candidates after finite forward relaxation. Rejection preserves parameters,
traffic, adaptive values, cooldowns, and pending-prune topology at TTL 2
and TTL 1. Separate fixtures reject an overflowing squared-error
diagnostic and a nonfinite stored parameter. Valid gradual pruning still
decrements TTL and trains. CPU tests and the full quality gate are recorded
in `docs/development-log.md`.

These checks reject numerical failures at a wake-step commit boundary.
They do not assert stability for arbitrary rates, guard every evaluation
forward pass, or complete the broader checkpoint/sleep rollback gate.
Actual CUDA timing remains open under P1.8.
