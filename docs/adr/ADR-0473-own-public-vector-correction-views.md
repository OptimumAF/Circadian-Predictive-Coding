# ADR-0473: Own public-vector correction views at update admission

Date: 2026-10-19

## Context

The frozen public21 rollout can predict from a validated raw789 observation, but
the correction collector retains raw789 trajectories and the private updater
requires the learner's exact specification. The composition previously refused
projection plus corrections. Native p32 results remain negative despite lower
offline mean error; useful corrective adaptation is a concrete missing outcome.

## Decision

Add a pure core raw/model correction-view contract. Revalidate and own the raw
batch, project complete current/next TRAIN observations, and derive paired model
labels from those same transitions without changing targets, actions or clocks.
The updater accepts the original raw batch and an explicit frozen projection;
after arrival admission it derives the model view itself before forking. Continue
the existing fixed eight-update private runtime and all optimizer/receipt guards.

Collection predicts from the projection of the exact captured raw tuple used by
the existing current-port query check. Keep raw execution/query evidence and
recheck it and all owners after the last episode getter before releasing labels.
Forward the explicit option through the existing corrective pipeline. Preserve
default raw789 calls/report shapes and unrelated incompatible-option refusals.

## Alternatives

- Trust a caller-projected batch: loses evidence tying targets to original raw
  observations and permits unvalidated discarded pixels.
- Convert the executed episode to21D: erases the collector's raw evidence.
- Add another runner or fitter: duplicates working boundaries without solving
  ownership or pairing and would change exposure/statistics unnecessarily.

## Consequences

The new core contract remains independent of adapters and infrastructure. Reports
separate one-time projection/pairing payloads from raw collection and21D update
work. Copies and process overhead are not claimed as measured memory. Frozen
statistics, full source optimizer state, permanent charges and preceding-success
validation remain required. Synthetic correctness and independent review do not
establish native adaptation, circadian benefit or generalization. Any new native
study needs a small frozen prospective protocol and retains all earlier failures.
