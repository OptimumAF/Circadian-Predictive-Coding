# ADR-0476: Observe signed physical joint limits without changing control

## Context

R4.5 requires physical constraint observations. The existing CPU PickCube port
validates normalized actions but does not measure actual joint-limit exceedance.
Prior learned-actor successes/failures and lighting loss tradeoffs cannot answer
this missing question. The pinned SDK exposes physical limits and qpos already
present in both public observation encodings.

## Decision

Use a pure bounded per-joint monitor and opt-in composition in the existing port.
Own one native reset-time limit snapshot with exact Panda names and7rad/2m units.
Sample the existing qpos9 prefix, keeping signed extrema and per-joint/sample
counts with zero epsilon. Separate discovery/read work from accepted samples;
preserve partial work and permanently refuse changed binding or invalid output.
Defaults/features/actions remain unchanged. Use current teacher code for a
separately budgeted native observation, not another runner or comparative sweep.

## Alternatives

- Inferring physical bounds from normalized action limits confuses target commands
  with observed joint positions.
- A chosen epsilon could hide real small deviations; preserve raw magnitudes first.
- Keeping every qpos sample duplicates the bounded trajectory storage and is not
  needed for extrema/counts. Consumers can independently read the existing trace.
- Reading qpos again adds native work and may measure a different observation time.

## Consequences

Readouts are bounded and scientifically interpretable, without a mixed-unit
aggregate. Limits are a declared snapshot; later changes/inter-control substeps
and collision/velocity safety are not observed. Numerical correctness and scoped
native execution do not establish repository readiness or circadian advantage.
Higher-level result propagation remains a separate explicit API extension.

## Follow-up: frozen rollout composition, 2026-10-19

That API extension is now implemented as an explicit opt-in on the existing frozen
rollout and smoke CLI. Core validates owned bounded summaries and recomputes raw
qpos samples; infra validates paid counters independently of the summary getter.
Final owner/context checks and withholding of failed corrections follow reporting.
Default fields/calls remain unchanged. No new native run, task energy, orchestration
framework or safety claim accompanies this composition.

[Executed checks and limitations](../../artifacts/runs/joint-rollout-integration-20261019/RESULTS.md).
