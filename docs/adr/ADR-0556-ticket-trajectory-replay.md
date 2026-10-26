# ADR-0556 — Reuse existing bounded trajectory rehearsal for full ticket tasks

## Context

The accepted chronological conventional case learns/relearns each latest task
but forgets the other. Existing StreamRehearsal/episode codec/retention policies
already support native categorical trajectory replay; ticket batches lacked
complete teacher episode metadata. Rebuilding replay or extending the failed
native observer tooling would add less useful functionality.

## Decision

Compile only original synthetic TRAIN teacher transitions, validate their exact
features/targets and use existing whole-episode retention/consolidation. Add
explicit recent/coverage options to the same recipe; preserve default no-replay.
Fixed two-episode cap, six one-episode events at midpoint/end of the phases and
separate permanent replay authority/costs. No new algorithm/controller/framework.

Add a detached quiescent stream consolidation-receipt getter. Native counters
must equal wakes plus observed published replay, never merely ignore differences.
Keep source/native rows, event costs, old receipt prefix and budgets separate.
Failed native replay retains all attempted work and refuses further progress.

Why this: complete-task retention can now be tested through existing working
components, with matched replay quotas and a strong recent-only reference.

## Consequences and alternatives

Coverage uses stronger declared TRAIN context information and changes order and
occupancy; it is a policy-bundle comparison. The saved no-replay arm has less work.
Neither equal caps nor equal update counts establish equal information or FLOPs.
Synthetic packet reuse does not confer general managed-data consent; final/probe
data is excluded. Defensive value copies are not hostile-process attestation.
All negative results stay valid. No default policy change, threshold/dose tuning,
probe/controller addition or fresh native-model comparison is justified by one
small case. Full acquisition/retention/generalization/release requirements remain.
