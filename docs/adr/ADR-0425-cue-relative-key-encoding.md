# ADR-0425: Preserve the public cue in a separate relative-key state family

## Context

The original constructive H8 BP fit remains52/54, with both A cue-turn rows
wrong. Their measured local gradient opposition does not establish its cause.
The environment already exposes an initial red/blue Ball cue and red/blue keys.
A cue-relative key representation is a testable hypothesis requiring no private
state, labels, action, context, teacher or future observation as feature inputs.

## Decision

Validate the complete existing raw296 public history, retain the original cue
and non-key information, and map key colors to match/nonmatch. Keep the original
wall/direction normalization. Require a unique red/blue initial cue and red/blue
keys. Retaining the original cue makes the permutation invertible relative to
the old canonical representation. Document the apply-once raw-input contract:
red-cue projection is not idempotent.

Provide a small pure core codec and a distinct exact policy/snapshot/learner
family composing the unchanged native learner. All four methods share the same
projection. Original types, defaults, policies and historical results remain.
Reuse the existing stream; do not create another runner or optimizer.

## Alternatives

Discarding original cue color would remove public information and could alias
inputs. Extending the old invariant snapshot family would admit a new encoding
to existing readers that hardcode the old projection, silently calculating
inconsistent diagnostics. A separate family makes that boundary explicit.
Changing native math at the same time would confound the representation test.

## Consequences

Pure semantics, all-method native parity and a two-update ordinary delayed
training-only stream are covered by110passing scoped cases and scoped statics.
The tag cannot authenticate arbitrary manually constructed weights or detect
already projected coordinates of the same shape. Stateful history therefore
caches pre-relative initial coordinates, and public docs specify alternative
single projection boundaries. Role admission remains outside the pure codec.

Pilot/CLI registration, projection-aware diagnostic readers and any scientific
learning/retention/recovery claim remain separate unfinished work. The source
preflight and tests establish representation and integration behavior only.
See [usage](../cue-relative-encoding.md) and
[evidence](../../artifacts/runs/cue-relative-encoding-20261018/RESULTS.md).
