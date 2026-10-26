# ADR-0588: Expose public policy context without changing task truth

## Context

The native warm/fresh probe does not acquire revised invoice behavior. Its D7
encoding gives v1route and v2inspect the same ready-invoice input while both actions
are allowed. Simultaneous retention of both rules is unidentifiable from those
features. This does not prove why the fixed optimization probe failed to switch.

## Decision

Add exact-bool opt-in context to the existing request, teacher, episode and local
example. Append only policy_v1/policy_v2 from validated public state; select the
existing seven features plus these tokens as D9. Distinct TRAIN group/report
markers prevent silently treating changed observations as old evidence. Preserve
all defaults, typed capability masks, revisions, effects and task-success rules.
Existing example model dimensions and nominal costs follow the selected schema.

Why this: a learner needs observable context to distinguish conflicting rules.
Supplying task targets or automatically masking out wrong actions would bypass
that learning question; neither occurs. No new model, runner, learner equation,
authority or dependency is required for the observation path.

## Alternatives and consequences

- Tune probe dose/seeds or add circadian mechanisms: does not remove aliasing.
- Require the model to satisfy both labels on D7: not a valid joint-fit gate.
- Change defaults or replace old checkpoints: would invalidate frozen evidence.

80focused checks/platform/statics and one zero-model TRAIN-table case verify
observation alignment/effect behavior, including both correct revocation refusals.
This proves plumbing, not native learning or generalization. The next conventional
context-aware fit needs a separately frozen protocol and new D9 initialization/
capacity accounting; old D7 learned sources cannot be declared equivalent.
Source/procedure/results are self-reviewed, not independent acceptance. See
issue72 and artifacts/runs/ticket-policy-context-20261023/RESULTS.md.
