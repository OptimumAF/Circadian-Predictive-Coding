# ADR-0414: Compose existing fixed64 recipes with weighted catalogues

## Context

The fixed weighted-PC S2 reconstruction reproduces the negative16/54 TRAIN fit.
Both original checkpoint readouts show local descent, without identifying why
acquisition fails. Named pc_relax64/neutral_cpc_relax64 controls and exact weighted
native S64 support already exist. The public weighted catalogue route admitted
only the S2 names and constructed S2 policies, blocking this existing composition.

## Decision

Admit only the two existing fixed64 names in the weighted joint route and CLI.
Choose their complete policy through minigrid_relaxation_control.policy; keep
base native pc/neutral_cpc identity and distinct public recipe names. Derive
benchmark nominal work from that policy. Existing stream, solver, source/actor/
receipt/scorer/budget guards and original128x54/F16/A16 doses are reused.

Ordinary S64 probe templates must equal the weighted owner's complete zero-update
snapshot, followed by original source snapshot and identity checks. Why this:
the online episode-weighted helper intentionally admits S2 only, so this narrowly
scoped branch reuses its full-state proof without broadening online semantics.
Original S2 catalogue probes continue delegating through the original helper.
Fixed64 PC diagnostics and action margins remain unsupported and fail early.

## Alternatives

- Add a configurable relaxation horizon: unnecessary new parameter/search space.
- Broaden the online episode route: changes unrelated supported semantics.
- Create another runner: duplicates the accepted stream/probe orchestration.

## Consequences

The separately named route pays442368 benchmark nominal latent-row iterations
and479232 including ordinary F/A probes, compared with14976 for original S2.
Wall time, RSS, validation/copies and serialization remain additional costs.
No equal-compute, convergence, acquisition, causal diagnosis or advantage follows
from implementation checks. Original defaults, negative experiments and unfinished
requirements remain. A separately frozen/reviewed small protocol is required
before any new real-data comparison; no experiment is part of this ADR.

Acceptance evidence: artifacts/runs/weighted-catalogue-relax64-20261018/RESULTS.md
(280focused/static/CLI/native/source+actualreview gates accepted).
