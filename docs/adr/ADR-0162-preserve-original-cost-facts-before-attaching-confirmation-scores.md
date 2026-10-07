# ADR-0162: Preserve original cost facts before attaching confirmation scores

Status: accepted; P6.11b1 correctness, complete actual repetition and independent readback pass.
Date: 2026-10-01.

## Context

P6.7c has two identical complete scored results. P6.11b requires the original
cost distinctions alongside every seed and fixed contrast. The existing
72,050-byte reference report retains grouped work and whole-run resources.
Per-method work, checkpoint capacity and inactive/rejected event context are
in each original 134,554,378-byte training result.

## Decision

Split b into b1, a complete verified cost join, and b2, the exhaustive scored
seed/interval/cost publication. Preserve the original b acceptance unchanged.
Reuse the unchanged complete training readers sequentially, projecting small
cost facts before each decoded checkpoint graph is discarded. Require the
entire original result fingerprint, complete work validation, equality to the
independent audit, equality between both cost projections and late file/source
checks. Publish exclusive cost inspection metadata and rederive every byte on
readback. Extend the exact 97-source scored closure with three consumer modules.

Preserve raw method fields, all three checkpoint capacities and shared seed
context. Count wake, applied replay and rejected-executed replay separately.
Keep group storage and whole-run wall/RSS observations in their original units.
This inspection has no scored-bundle or statistical-report authority and opens
no data, model, training, final-release or prediction boundary.

## Alternatives

- Grouped costs alone cannot expose each arm's executed rejected replay or capacity.
- A second independent training-reader implementation duplicates the existing gates.
- Holding both complete checkpoint graphs increases memory without helping the join.
- Converting every family into estimated FLOPs or per-arm RSS invents measurements.

## Consequences

The next scored report can join costs by exact family/seed/arm with one pinned
training-result identity. Additional read-only validation is budgeted separately
from the original scientific runs. Deterministic repeats remain reproducibility
checks; they add no seed replications. No scientific source, seed, method,
metric, interval rule, resource cap or original artifact is changed.

Actual publication measured a 112,635,395-byte inspection: raw shared guard
context includes full proof states, so the prospective "small metadata"
assumption was incorrect. The 103,705,403-byte projection preserves that full
context and both first publications pass the 180-second validation budget in
approximately 74 seconds. Keep this literal evidence; the b2 report may retain
all per-arm cost fields while linking complete shared context by exact byte
identity. This measurement changes the size rationale, not the acceptance.
