# ADR-0143: Guard sleep factors in a train-only gate before scoring

## Context

The development replay factor held width fixed and left structure and
nonstructural sleep effects untested. The historical v12 structural
factor exercises one split and one prune with fixed final width, but its
final outcomes cannot select a new P6.3 setting. The arrived runner has
an inner-guard rollback path that records proposed and applied work.

## Decision

Freeze a separate, unscored nine-arm A→B preflight on seeds 67/71/73.
Use ordinary PC, backprop, exact neutral circadian sham, structure-only,
homeostasis-only, gating sham, gating with chemical reset, and planned
width-12 PC/backprop controls. All arms share source exposure and model
initialization within width, use 12+12 wake updates, and have zero
replay. Five circadian arms attempt one A-boundary component sleep through
the existing inner-guard helper at tolerance zero. Preserve a rejected
proposal in the audit while restoring its model state. Verify train-only
roles, work, parameter parity, guard evaluations, stable structural IDs,
transient and final width, active homeostasis and chemical reset, and
observed worker RSS before opening outer-selection scores.

The arrived-role validator requires a replay-enabled source config; that
config carries only the source geometry. Every actual model uses a
separately constructed no-replay arm config. The public adapter binds
source/manifest identities, saves an exclusive request and result/audit
or failure, and enforces a 700-update prelaunch cap, 120-second child
limit and observed 256-MiB worker RSS ceiling.

## Alternatives

- Reuse v12 final results as new selection evidence: rejected because
  their final roles were already opened under another protocol.
- Disable the guard to force structural activity: rejected because it
  would hide rollback cost and favor an unchecked topology change.
- Compare chemical reset under neutral plasticity: rejected because
  a neutral plasticity factor would make the reset inert by design.
- Match width 12 to a later observed final width: rejected as a
  retrospective capacity oracle. It is planned before this gate.

## Consequences

The preflight can establish that the proposed factors are executable,
bounded, role-isolated, and auditable. It cannot establish an accuracy or
retention benefit. A guard-rejected factor remains a valid negative
feasibility result, and scored development, confirmation, full-minus-one,
and process cost comparisons remain separate tasks.
