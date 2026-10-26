# Contextual ticket queue rehearsal

The existing recurring runner now supports complete contextual TRAIN queue replay:

```powershell
python -m scripts.recurring_ticket_workflow_example --context-return --replay recent --output recent.json
python -m scripts.recurring_ticket_workflow_example --context-return --replay coverage --output coverage.json
```

Each output must be new. Default/no-replay and original D7/policy-change behavior
remain supported. Contextual replay currently uses the existing perceptron only;
unsupported native combinations fail before work or output reservation.

## A queue preserves both real tickets

The fixed contextual TRAIN packet has two three-step tickets. The compiler
validates each ticket through the real sandbox, preserving exact features, labels
and ordering. Its replay episode is the ordered six-decision queue. After the
first correct commit, the queue advances to the second sealed ticket; the next
observation means the next public policy decision. Reward1 records each commit;
only the queue's last transition is terminal. This explicit handoff does not
change the original complete-episode codec or reinterpret an individual ticket's
terminal state. Neither rewards nor task outcomes become training targets.

The original A(v1invoice/password)→B(v2invoice/parcel)→sameA stream still has twelve
wake updates,72TRAIN rows and delay1. Every two wake updates, the existing
StreamRehearsal selects one arrived six-row queue. Six events add six passes and
36row exposures. Both modes retain at most two whole queues:12native transitions,
3616encoded array bytes including dummy labels, excluding Python metadata and
selected-event reports. Coverage uses declared TRAIN context keys and rotates the
latest eligible queue per key; it has stronger metadata than recent FIFO.

## Executed finding: no final improvement at this dose

Issue76/R4.1-G/R2.1-V executed one frozen case per mode, in order recent,coverage.
The saved no-replay75case was reused after exact complete report parity was
checked by a separately charged correctness fixture. Tasks/metrics/zero source
and four checkpoint readouts were fixed before execution.

| Recipe / wake point | v1invoice | v1password | v2invoice | v2parcel |
|---|---|---|---|---|
|All /0|fail|fail|fail|fail|
|All /4 after A|pass|pass|fail|fail|
|No replay or recent /8 after B|fail|pass|pass|pass|
|Coverage /8 after B|pass|pass|fail|pass|
|All /12 after A returns|pass|pass|fail|pass|

Recent produces exactly the original success table despite extra exposure.
Coverage retains old invoice at the B checkpoint, but new invoice fails there;
this does not prove it never succeeded between readouts. Coverage's old skill
was not observed lost, so recovery is null. Recent repeats acquired-old-invoice
loss/recovery. Both finish at3/4ordinary success,3/4stale success and4/4correct
commit revocation denials. Permission refusal is not task success. Each records
8/16ordinary and3/4stale successes,24total episodes. No aggregate or endpoint
task-success improvement is observed at these readouts.

Each replay case costs18passes108rows versus12passes72rows for no replay (+50%),
with13mistakes for recent and18for coverage. Parameter capacity remains60float64
coefficients, while replay adds storage and computation. Nominal TRAIN affine
terms grow4320→6480;92typed reads cost4600nominal affine terms/case. Four retained
model checkpoints use1920coefficient bytes. Physical costs/case:9learner/zero
initializations,1224snapshot copy/validation entries73440logical coefficient slots,
2teachers13packet constructions260encodes188sandboxdispatches;18recent versus
30coverage codec encode/decode entries. Copy/validation/Python overhead additional.
Meter times13.88985s/13.66774s; sampled RSS peaks45,301,760B/44,732,416B. Fixed order,
startup and validation differ; no timing efficiency ranking is established.

No seed/dose/metric selection, retries, final/probe updates, native SGD/PC,
controller tuning, promotion or external actions. These are known-template
descriptive results, not generalization, matched compute against no replay,
statistical evidence, algorithm ranking or CPC advantage. Both final gaps remain.

## Verification and extension

89selected checks pass per Windows/WSL platform, plus four-file Ruff, formatting
and fresh default/Linux mypy. Old D7recent behavior and no-replay saved parity
pass; codec roundtrip, authenticity, prior receipt/counter refusals, bounded
arrived-only selection and source/actor/RNG/budget separation are exercised.
Source/procedure/result review is self-review, not independent acceptance.
Full suite, native families, physics/math, packaging, clean-clone and CI unrun.

Raw results, frozen protocol, counts, source pins, zero-model readback and exact
commands: `artifacts/runs/context-queue-rehearsal-20261023/RESULTS.md`.
Live status: [issue76](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/76).

Subsequent issue77implemented the [separate-head conventional control](context-head-control.md)
through the existing stream. It retains4/4known tasks while increasing capacity
96versus60coefficients and requiring explicit known context. This original
replay-negative evidence remains unchanged; no matched-capacity or CPC claim.
