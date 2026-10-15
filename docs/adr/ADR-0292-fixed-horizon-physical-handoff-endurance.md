# ADR-0292: Fixed-horizon handoff endurance with physical resource sampling

Date:2026-10-15. Status: accepted design; validation tracked in R3.7v.

## Context

Three-cycle continuation and deterministic resource refusal controls pass.
The original failure/long-run requirement still needs a longer finite exercise
with real monotonic time and OS RSS observations across repeated ownership changes.

## Decision

Parameterize the existing private repeated loop for its original three-cycle
mode and a fixed16-cycle endurance mode. Preserve existing public modes and
schemas. Endurance uses one original controller/resources and compact records,
while full state/cursor checks occur before compaction. Retain native envelopes
for every cycle and compare uninterrupted references and PC-neutral controls.

Predeclare update/preparation/cycle/time/RSS/output bounds. Use the existing
monotonic budget and manually sampled OS RSS reader before model construction;
fail explicitly on unavailable or exceeded resources. Keep original start/peak/
sample history across owners and end with spent-limit refusals.

## Alternatives

Copying another large demo would duplicate the transaction logic. Per-cycle
controllers or resource objects reset quotas and cannot demonstrate continuity.
Unbounded runs or large parameter sweeps spend resources without a fixed
correctness horizon. Retaining every full cursor repeatedly would inflate output
without improving the in-memory equality checks.

## Consequences

The opt-in mode extends bounded repetition evidence with actual resource sources.
Observed sample peaks are not hard OS limits or model-memory attribution, and a
finite successful run does not establish indefinite stability or full recovery.
No native algorithm, baseline, scientific protocol, dependency or app/core API
changes are introduced.
