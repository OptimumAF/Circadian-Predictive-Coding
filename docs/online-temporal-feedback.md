# Causal online temporal feedback

This optional local example connects consented raw TRAIN feedback to bounded
memory while the existing delayed temporal perceptron stream runs. It preserves
the twelve-batch/36-row objective, policy versions, candidate masks and A→B→A
schedule. It adds no learner/queue/checkpoint framework or dependency.

## Observation, publication and readout

The existing prearrival feature hook observes each raw request and explicitly
records TRAIN plus memory-reuse consent before its label arrives. This raw view
is distinct from the stream's projected array experience, whose original
training-only permissions stay unchanged. No historical permission is renewed.
Each label is admitted by `apply_decision_feedback` only with a matching typed
raw source, key/model version, allowed IDs, role, original consent and causal
observation/arrival/recording times.

The read-only `AttributionStream.applied_updates` property detaches current
receipts at a quiescent checked owner boundary. After each committed three-row
batch, the outer script binds its raw row labels to the real receipt and publishes
memory additions at those original ticks. A prearrival hook may observe the next
batch before the preceding batch's same-tick label arrives; observation alone
never adds its target. Published history is checked before more TRAIN work.

At fixed boundaries0/4/8/12 the example records actual weight and retrieval
outputs against the current committed prefix. Later memory additions cannot
backfill an already queried cutoff. Frozen-A memory retains cutoff4; current
requests retain the declared current policy. The v2 filter can make all v1
records unavailable, even when one rule stayed unchanged. This is explicit
policy eligibility, not ordinary forgetting or successful recovery.

All scoring runs after TRAIN finishes, using the saved raw readout cache. Scoring
labels never select work. Current weights, unchanged0 and frozen-A4 states share
one live fit; no separate control is trained. Memory has36 additions/zero edits,
top1/768-byte context and≤64KiB canonical storage. `decision_from_evidence` returns
an explicit unavailable value for no hit. The existing context/retrieval wrapper
retains its sticky-error/default behavior; this example never resumes an uncertain
model callback by treating it as a no-hit.

## Metrics

Report every declared case. Accuracy uses the full four-case denominator and
counts unavailability as non-success. Coverage is separate. NLL/Brier/ECE are the
existing metrics on answered cases only, with their denominator explicit; empty
views have no proper score. No probability or confidence is invented for refusal.
A wrong one-hot answer still has infinite NLL. Conditional scores cannot support
a full-coverage claim against a different answered set.

New parcel acquisition, its retention on the A return, stable-password answer
availability and current-rule old-domain return are reported separately. Concepts
and templates overlap. This example does not establish unseen-task generalization,
CPC benefit, native2B competence, deployed actor promotion or matched computation.
The original source and actor remain frozen while the candidate updates.

## Run and inspect

```powershell
python -B -m scripts.run_online_temporal_feedback --output fresh-online-feedback.json
```

The CLI refuses an existing output before constructing a model or store. Follow
[the frozen protocol](../artifacts/runs/online-temporal-feedback-20261022/PROTOCOL.md)
for the one local functional case; past execution allowances are not reusable.
Scoped checks and exact resource receipts are recorded beside that protocol.

```text
src/app/decision_evidence.py             typed feedback and selection helpers
src/app/coupled_attribution.py           existing runner, detached receipt access
scripts/run_online_temporal_feedback.py  live fixed chronology and cached scoring
tests/test_online_temporal_feedback.py   causal/isolation/refusal/integration checks
```

No new environment variables. Extend through existing typed feedback and runner
hooks; keep original consent/clock truth, evaluation separation and permanent
charges before expanding a scenario or choosing another feedback policy.

## Executed local result (2026-10-22)

Independent ACTUAL review accepted the bounded R5.7-A outcome.135 selected tests
pass per platform with scoped lint/format/types. The one frozen case uses12
updates/36 rows,80 affine model readouts and48 retrievals, in2.359712100 s wrapper.
Memory correct/answered counts are0/0→3/3→3/3→4/4 of four cases; learned weights
correct1→4→3→4. Memory therefore shows no full-case accuracy gain over the linear
learner. Parcel's current label is acquired at B and retained on return. Password
memory eligibility disappears under v2 until new return feedback; actual weight
error at B and return correction are reported separately.

Store36 additions/zero edits/22,213 canonical bytes;1,056 scans/384 comparisons/
480 token visits/5,046 rendered bytes. The four nominal coefficient snapshots
retain640 bytes, with owner/stream/copies additional. Sampled RSSmax42,393,600 B
from402 observations is not a true process peak. All no-hit/frozen-A/obsolete-rule
and after-A ECE negatives remain saved. No final/probe/replay optimizer work,
new native model, randomness or promotion occurred.

Ordinary scoped developer checks:

```powershell
python -m pytest -q tests/test_online_temporal_feedback.py tests/test_decision_evidence.py tests/test_temporal_decision_reference.py
python -m ruff check scripts/run_online_temporal_feedback.py src/app/coupled_attribution.py src/app/decision_evidence.py tests/test_online_temporal_feedback.py
python -m ruff format --check scripts/run_online_temporal_feedback.py src/app/coupled_attribution.py src/app/decision_evidence.py tests/test_online_temporal_feedback.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/run_online_temporal_feedback.py src/app/coupled_attribution.py src/app/decision_evidence.py tests/test_online_temporal_feedback.py
```

The recorded study used the metered qualify/execute scripts, not an unbounded
fixture allowance. No large native/model allowance is inherited.
[Results and exact costs](../artifacts/runs/online-temporal-feedback-20261022/RESULTS.md) ·
[Independent review](../artifacts/runs/online-temporal-feedback-20261022/REVIEW.md) ·
[Decision](adr/ADR-0535-bind-consented-raw-feedback-to-delayed-stream-receipts.md).
