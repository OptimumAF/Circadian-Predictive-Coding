# ADR-0375: Change controls independently of spatial observations

## Context

R4.3 calls for a second controlled game stream with changed movement dynamics or
action mapping. Existing key/door rotation changes both physical orientation and
spatial observation directions; it does not isolate a control intervention. The
ordinary reward collector/actor runtime already provides private sampling,
delayed updates, frozen behavior state and separate final episodes.

## Decision

Add a four-index movement_action_map to KeyDoorLayout, identity by default.
Entry[action] names a spatial direction before the existing inverse rotation.
Legal support and movement use the mapping; observation neighbor channels keep
their spatial direction order. Interaction4 is fixed; teachers invert the map.
Validate exact immutable permutations at construction and owned-copy boundaries.

The explicit key_door_layout_record serializer preserves the five original
identity-layout fields; mapped records also declare the permutation and spatial
observation basis. Existing descriptor/evaluator callsites use that serializer.
Mapped reward collection records its layout; identity collection shape stays
unchanged. Native reward packet/loss/state schemas are unchanged.

Add one explicit legal-action-remap scenario to the existing CLI/reward runner.
A/B/C geometry, appearance and keys agree; only B controls change. C uses the
identity law, so it is not a novel third task. Per-arm policies collect their own
trajectories; final packets wait for all original eight commits. The scenario
reuses existing checkpoints/evaluation, not another orchestration framework.

## Alternatives

Reordering observation channels with controls could cancel the intervention.
Silently adding identity metadata to old serialized protocols would break old
schemas. A new environment/runner or altered reward objective is unnecessary.

## Consequences

Default behaviors and layouts remain; the optional dataclass field is an additive
Python API change. Consumers directly using dataclasses.asdict see that new
field; repository legacy wire callsites deliberately use the explicit serializer.
Legal support remains additional information and can force corridor controls, so
success alone would not prove learned remapping. The engineering qualifier is
not a matched repeated-seed experiment. Full R4.3/equal-interaction comparison,
useful learning and release claims stay open. The existing inspector does not
support this new mapped protocol; retained raw JSON is authoritative.
