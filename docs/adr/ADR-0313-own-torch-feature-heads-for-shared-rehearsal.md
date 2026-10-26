# ADR-0313: Own native Torch feature heads for shared rehearsal

## Context

Existing shared rehearsal isolates final evaluation, forks candidates and charges
replay. Native Torch PC/CPC and matched BP exist, but PC/BP lack a complete owned
port and BP optimizer state is external. Native Torch sleep replay is unsupported.

## Decision

Add an opt-in CPU fixed-feature adapter around existing native heads. Bind a small
policy and own all parameters, CPC adaptive/RNG state, BP SGD/gradients. Stage
complete native training/restore before publishing one payload. Reuse the generic
attribution runner and shared rehearsal. Preserve native objectives, clocks,
sleep behavior and defaults; identify external wake/replay costs separately.

A fixed eight-arm CLI demonstrates switched replay and within-mode matched caps,
proposals and exposure. Its explicit Torch engineering RSS allowance leaves
scientific pilot ceilings unchanged. Score signs are not correctness criteria.

## Alternatives

- Parameter-only wrappers lose momentum, chemistry, traffic, ages or RNG.
- NumPy learners under a Torch name change the native objective.
- Native sleep replay duplicates tested admission and introduces unrelated
  structural/sleep policy changes.

## Consequences

Owned feature heads use existing delayed labels and candidate-only replay.
Staging adds copying/construction work. Reports identify native calls, exposure,
tensor bytes and process RSS without claiming efficiency. BP CE differs from PC
energy and equal exposure is not equal compute. Component continuation is tested;
runtime authority and whole-stream resume are not provided. R2.4 stays open.
