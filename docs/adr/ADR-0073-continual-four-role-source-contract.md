# ADR-0073: Declare four continual roles before decision wiring

## Context

The opt-in v5 run uses one validation split and intentionally reproduces
v4 seed reports. Splitting that training source again inside v5 would
change its verified scores and format-5 development identity. Inner
guard and outer selection cannot share one role if their decisions are
to be evaluated independently.

## Decision

Add a separate deterministic four-role source contract in
`src/infra/continual_roles.py` for a later versioned continual runner.
Within each phase and class, a seeded permutation reserves disjoint
inner-guard and outer-selection rows; the remaining rows train. Stable
IDs name original development row positions, phase, and seed. Content
hashes bind IDs, role, and values. Final-test IDs and expected count are
declared without reading either final source field. An explicit release
function validates and hashes final input and labels after the app's
global freeze gate.

The returned policy declares development input/label availability at
that phase's arrival and final input/label availability at global
freeze. These are intended events, not an observation that a runner
enforced them. The app must separately record actual accesses and
per-method task information when it wires the new protocol.

## Alternatives

- Reuse v5's validation rows for both decisions. That makes inner guard
  outcomes part of outer selection evidence.
- Change v5's split in place. That invalidates its v4-equivalent report
  and saved checkpoint identity.
- Hash final-test values while splitting. That would open held-out
  labels before all seeds and settings freeze.

## Consequences

The splitter is deterministic, class-covered, and disjoint across roles
and phases on the fixed two-phase fixture. Raising source sentinels
observe no final field read before explicit release. Altering only
final labels changes only the released final hash. This module does
not train models, make guard decisions, choose settings, record actual
release events, or serialize a checkpoint. The synthetic source can
still allocate final arrays internally; the app must enforce their
release timing. P1.3c3b2c2–c3 and their parent remain open.
