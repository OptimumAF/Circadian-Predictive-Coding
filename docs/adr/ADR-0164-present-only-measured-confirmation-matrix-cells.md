# ADR-0164: Present only measured confirmation matrix cells

Status: accepted and implemented; complete correctness/publication/readback
evidence and original criterion/scope audit pass.
Date: 2026-10-01.

## Context

P6.11b supplies complete frozen seed/contrast/cost evidence. P6.9 asks for
an explicit stage/task accuracy matrix and transparent forgetting/retention/
transfer. The original protocol measured A-after-A, A-after-B and B-after-B,
with final access globally sealed until training completion. B-after-A was
not declared or measured. No retrospective evaluation is authorized by the
presentation task.

## Decision

Add a consumer over the unchanged complete report readback. Present every
560 arm/seed identity as a 2×2 stage/task matrix, preserving three measured
endpoint records and explicitly marking B-after-A unmeasured before arrival.
Keep raw correct counts, count/role/content hashes, checkpoint and exact
original JSON pointers. Keep measured endpoints when another endpoint fails;
original whole-cell derived metrics retain their existing failure/null policy.

Reuse the frozen public scored/report declaration validators to verify all
stored endpoint/cell/summary/cost links. Rebuilding the expected training
declaration for that pure validation grants no source/execution authority.
The infrastructure port must call the unchanged complete official report
reader, bind its returned bytes and recheck current source/input/output facts.
Backward transfer is a labeled descriptive restatement of negative signed
forgetting, not a new primary metric. Forward transfer stays unavailable.

Keep publication exclusive, failure/ownership handling explicit, and full
readback deterministic. The new consumer has its own prospective source and
request freeze and 180-second derivative budget, based on the prior four
160–163-second complete report operations. No original scientific cap or
frozen model/data/scoring source is changed. The original P6.8 contract defines
the full current two-task matrix with these three endpoints. The subsequent
original scope audit therefore closes P6.9 after complete matrix evidence;
it corrects the earlier inference that a fourth measured slot was required.
The original wording and measurement scope are unchanged. Immutable fully
measured four-slot flags remain false, and any future forward-transfer endpoint
protocol requires prospective independent evaluation gates.

## Alternatives

- Inferring B-after-A fabricates an unmeasured value.
- Evaluating it retrospectively changes the frozen endpoint protocol.
- Dropping the unknown slot hides matrix scope.
- Rewriting scored/metric validators duplicates already verified rules.
- Treating pure declaration reconstruction as source proof overstates authority.

## Consequences

The explicit matrix is useful and incomplete at the future-task slot. Null,
negative, failed and above-one retention observations stay visible. Existing
report inputs, scientific settings and seed replication remain unchanged.
The reporting acceptance audit determines each original parent status
individually; presentation completion does not close the full research plan.
All 110 new/556 related tests pass. Both actual publications and both complete
readbacks exit 0 under their 180-second budgets, with exact whole JSON/Markdown
repetition and all 24 scientific guards zero. The matrix report and session log
record exact source/input/output/test/producer identities and the linked
scope correction. Resource and hypothesis presentation remain open.
