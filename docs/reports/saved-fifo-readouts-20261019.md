# FIFO36 learns, retires, and recovers known tasks

The existing count-memory control now has all30 planned TRAIN readouts. The
original run exhausted180seconds after128updates and remains incomplete. A new
read-only invocation used its five full snapshots to finish missing cells with
zero training. Independent review checked all probabilities, first-argmax actions,
transitions, outcomes, preserved states and costs.

| Native updates | A successful cues | B successful cues | C successful cues |
|---:|---:|---:|---:|
| 0 | 0/2 | 0/2 | 0/2 |
| 32 | 2/2 | 0/2 | 0/2 |
| 64 | 2/2 | 2/2 | 0/2 |
| 96 | 0/2 | 2/2 | 2/2 |
| 128 | 2/2 | 0/2 | 2/2 |

Each successful cell also has exact teacher-trace fit. A was acquired, retained
through B, lost after C displaced its entries, then recovered when A returned.
B was later displaced; C remained available. Capacity is36entries, enough for two
18-key tasks. This demonstrates finite known-task count recall and recovery.
It establishes no neural plasticity, heldout performance or circadian advantage.
All neural matched negative/partial-acquisition results remain unchanged.

## What it costs

The original128countupdates/1152rows remain charged, together with its180.0138s
failed run. New readouts took89.2306s in the adapter/90.4369s outer supervision,
12raw prediction calls108rows and16games186decisions/steps. External RSS was
82,493,440bytes. All attempts returned; no teacher, replay, SGD or new training.

The censored original B-blue game at64 already spent8predictions/steps. Its new
whole game is separately charged. Across both attempts:420native prediction calls/
660rows,31live attempts390steps, plus original6teacher episodes54steps. Allfive
snapshots occupy305,504logical bytes; maximum retained snapshot87,280bytes.
Copies, record validation arithmetic, canonical projection, native/Python/report
storage and sampled RSS add costs. No equal-compute or equal-memory comparison.

## Delivered interface and limits

[Saved-state usage and extension](../saved-fifo-readouts.md) documents the pure
preview, finite/unique bounded JSON IO, exact snapshot/source admission, selected
existing evaluator and fresh zero-update budget. At most three new result groups
are guarded, including shutdown/time/RSS readbacks; preceding corrupt or replaced
records stop progress. Existing full-six evaluator behavior stays intact.

209focused tests and affected lint/format/Windows+Linux source types passed.
These are scoped checks. Repository-wide, release and full scientific requirements
remain open. Initial schema/RSS/static failures and review corrections are retained.

[Exact commands, budgets and failures](../../artifacts/runs/saved-fifo-readouts-20261019/RESULTS.md) ·
[Curves and accounting](../../artifacts/runs/saved-fifo-readouts-20261019/analysis.json) ·
[Independent SOURCE and ACTUAL review](../../artifacts/runs/saved-fifo-readouts-20261019/REVIEW.md).

Next substantive outcome: use existing isolated fresh-task machinery on owned
FIFO36 snapshots for a declared distinctF16→A16 probe with old-A readouts. Verify
policy/roles/ownership/budgets first and review a small local protocol before any
native experiment. No new checkpoint/audit sequence is needed for this step.
