# Editable temporal decision evidence

This optional local memory path stores explicitly consented TRAIN decisions and
later label corrections. It retrieves a bounded causal view for a typed request.
It does not train a learner or admit final/probe/evaluation labels.

## API and time semantics

`DecisionEvidence` binds an evidence ID, observation/label-availability/recording
ticks, policy version, source group, typed request and allowed target. Only role
`train` with both training and replay consent can enter. Replay consent authorizes
reuse here; the store performs zero replay optimizer updates. These are trusted
local declarations, not authentication of the physical source of a label.

`DecisionEvidenceStore.add` creates revision0. `edit` requires the exact current
revision and changes only the target/availability/recording time. Observation,
policy, source group, request and consent remain unchanged; a new policy
observation requires another evidence record. Corrections cannot move label
availability backwards and require a later recording time.

Every retrieval attempt seals its cutoff. Later writes, including new IDs, must
be recorded strictly after the largest queried cutoff. Initial records may be
batched at one tick before reads. Historical reads remain available; querying a
future cutoff deliberately seals that future time too, so callers should pass
their authoritative observation clock. This rule prevents backfills or equal-time
edits from silently rewriting an already observed view.

Retrieval first chooses the latest recorded revision per ID at/before the cutoff
and under the requested policy. It then filters currently allowed targets. A
new inactive target never revives an older allowed target. Positive Jaccard
overlap ranks evidence queries, breaking ties by newest recording then evidence
ID. Top-k rendering is deterministic JSON containing data only. Instructions and
candidate definitions remain separate protected fields. This does not establish
semantic prompt-injection immunity or contain arbitrary Python callbacks.

## Capacity and failures

Policy caps events (including revisions), canonical UTF-8 storage bytes, rendered
context bytes and top-k. Storage is the exact compact JSON representation of all
retained events, not process memory. Python objects, copies and scoring work cost
extra. Full storage refuses the next write without eviction or partial publication.
Context overflow fails without truncation. No unlimited revision chain or silent
retention policy is added. Store counters retain failed attempts and rendering.

`EvidenceDecisionPredictor` has three explicit modes:

- `retrieval`: nearest eligible label becomes a deterministic one-hot typed
  decision; absence of evidence is an explicit refusal, not a hidden model call.
- `context`: append JSON evidence to a detached request's query, then return the
  supplied predictor's actual typed result. Candidate definitions/identity remain
  fixed. The wrapper detects request/evidence mutation and preserves born callback
  returns before refusal. It stops after uncertain work and bounds total calls.
- `fallback`: use eligible memory first; if the selection is empty, invoke the
  supplied frozen predictor once with an owned, unchanged original request.
  No empty context is appended and no probability threshold selects the branch.
  `choice_source` records memory or fallback. Both branches share one finite call
  allowance; callback/store work and failed returns remain paid.

An edited evidence label must not replace a context-model choice after inference.
One-hot retrieval scores are a deterministic rule, not empirically calibrated
model confidence. Weight immutability belongs to the supplied frozen predictor;
the functional driver explicitly checks its state before and after readouts.

For a caller-owned store and frozen typed predictor:

```python
from src.app.decision_evidence import EvidenceDecisionPredictor

predict = EvidenceDecisionPredictor(
    store,
    cutoff=current_tick,
    policy_version=current_policy,
    mode="fallback",
    predict=frozen_predictor,
    max_calls=4,
)
result = predict(request)
report = predict.report
```

`store`, `current_tick`, `current_policy`, `frozen_predictor` and `request` are
supplied by the application. The frozen predictor retains its own native work
limits; this wrapper never renews them. A version mismatch, future evidence,
inactive target or no lexical overlap counts as unavailable. A valid adverse
fallback choice is returned without correction or retry. Caller request/candidate
or store-write mutation, malformed returns and unreturned calls stop the owner.

Windows79/WSL79 checks pass for this new mode and the57existing scalar cases,
with scoped Ruff/format/default and Linux mypy. This demonstrates typed composition
with mocked callbacks, not native hybrid quality or broad recovery. The separate
native CLM memory observation has memory answer/correct counts0/3/3/4 and an
unchanged head trajectory; it did not execute this later fallback policy.
[Fallback evidence](../artifacts/runs/evidence-fallback-20261022/RESULTS.md) ·
[Native memory observation](../artifacts/runs/clm-train-evidence-memory-20261022/NATIVE_RESULTS.md).

The [CLM temporal runner](clm-catalog-learning.md#live-typed-memory-with-a-frozen-clm-fallback)
now composes this mode with a completed, separately paid TRAIN-memory view and
the existing frozen head binding. It has literal end-to-end checks for known
memory and one unlabeled unknown fallback, retaining original phase counters.
This does not retroactively establish native hybrid performance.

## Local controlled example

```powershell
python -B -m scripts.run_decision_evidence --reference artifacts/runs/local-ticket-workflow-20261022/actual.json --output fresh-decision-evidence.json
```

The reference must match the accepted local source8 artifact. No fitting or RNG
is performed. Twelve new evidence-case records use explicit reuse consent. The
ready-parcel label is deliberately wrong (`inspect`) at10 and corrected to
`defer` at20. Original TRAIN/native artifacts remain unchanged. Two stale-parcel
retrieval episodes and six frozen affine readouts run under a separate small
protocol; all negative outcomes and ignored-context results are retained.

This is not the four-arm same-information R5.7 comparison, native Context-LM
qualification, broad adaptation/generalization, CPC benefit or repair of the
previous native parcel failure. Those claims remain open.

The executed controlled case retains13 revisions/10,287 canonical bytes. Before
correction, retrieval commits the wrong parcel workflow; after correction it
completes the stale-recovery episode. An old cutoff still returns the old target,
and the pre-recording view is empty. The unchanged linear baseline and all four
context readouts already choose defer. Thus the edit changes retrieval behavior
but demonstrates no context-model benefit; extra memory is unnecessary for this
already-correct linear choice. The linear vocabulary does not encode these action
labels as features, limiting conclusions about other model families.

All57 scalar checks pass/platform, with scoped lint/format/type checks. The sole
case pays six frozen readouts/240 affine terms and zero new fitting, about.444s
inner/1.334s wrapper; sampled RSSmax42,041,344B across14 observations (not true
peak). [Full result/costs](../artifacts/runs/editable-decision-evidence-20261022/RESULTS.md).

## Structure and focused checks

```text
src/core/decision_evidence.py       records, capacity, revisions, causal retrieval
src/app/decision_evidence.py        bounded retrieval/context/fallback typed calls
scripts/run_decision_evidence.py    frozen snapshot and controlled local case
tests/test_decision_evidence.py     scalar isolation, time, storage and failures
tests/test_evidence_fallback.py     availability routing and paid callback failures
```

```powershell
python -m pytest -q tests/test_decision_evidence.py tests/test_evidence_fallback.py
python -m ruff check src/core/decision_evidence.py src/app/decision_evidence.py scripts/run_decision_evidence.py tests/test_decision_evidence.py
python -m ruff format --check src/core/decision_evidence.py src/app/decision_evidence.py scripts/run_decision_evidence.py tests/test_decision_evidence.py
python -m mypy --explicit-package-bases --follow-imports=silent src/core/decision_evidence.py src/app/decision_evidence.py scripts/run_decision_evidence.py tests/test_decision_evidence.py
```

New mode's exact scoped static commands:

```powershell
python -m ruff check src/app/decision_evidence.py tests/test_evidence_fallback.py
python -m ruff format --check src/app/decision_evidence.py tests/test_evidence_fallback.py
python -m mypy --explicit-package-bases --follow-imports=silent src/app/decision_evidence.py tests/test_evidence_fallback.py
python -m mypy --explicit-package-bases --follow-imports=silent --platform linux src/app/decision_evidence.py tests/test_evidence_fallback.py
```

No new dependencies or environment variables. Extend via existing typed ports;
preserve TRAIN consent, observation-time truth and final/probe isolation before
adding a separately frozen retrieval/adapter/hybrid comparison.
[Protocol](../artifacts/runs/editable-decision-evidence-20261022/PROTOCOL.md) ·
[Review](../artifacts/runs/editable-decision-evidence-20261022/REVIEW.md).
