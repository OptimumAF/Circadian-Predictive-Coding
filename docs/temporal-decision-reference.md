# Temporal tool and workflow reference

This conventional perceptron example learns through the existing delayed TRAIN
stream, then scores fixed snapshots under the policy effective at each phase.
Tool choices are typed local data; no external tool or action is executed.

```powershell
python -m scripts.run_temporal_decision_reference --output temporal-routing.json
```

The output must be fresh. The fixed example makes12 updates/36 TRAIN-row
exposures, followed by80 typed readouts (40 original and40 reverse-order
controls). There is no seed, dose, learning-rate or rule-selection option.
The lightweight NumPy environment suffices; no Torch, SDK or download is needed.

## Declared temporal truth

| Policy | Active tools | invoice | legacy | parcel | password |
|---|---|---|---|---|---|
| v1: A |inspect,retire,route|route|retire|route|inspect|
| v2: B and old-domain return |defer,inspect,route|inspect|defer|defer|inspect|

The four-slot union catalog and vocabulary are known from the start. This tests
adding/removing allowed tools, not expansion to previously unknown output slots.
The subset compiler keeps the existing full-union multiclass mistake rule;
readout masks inactive candidates before softmax. An active TRAIN target must
be allowed by its own request. Default compiler calls still require the complete
catalog; opt-in is `build_decision_training_batch(..., allow_subsets=True)`.

TRAIN A and return contain invoice/legacy/password; B contains
invoice/legacy/parcel. Four passes per phase are fixed before execution.
Queries are concept tokens; development adds ` please` under a separate template
group. Concepts overlap by design, so this is not unseen-source generalization.
Policy/domain/version/request IDs do not enter the feature encoder. Labels are
bound when their requests are observed, before the delayed stream applies them.
Later rules do not relabel historical TRAIN packets.

Returning to the old domain preserves **v2** truth. Old invoice→route is now an
error even though route is still allowed. That distinguishes obsolete-rule
recall from retired-tool masking. Legacy→retire cannot be selected after removal;
choosing a newly allowed fallback alone does not establish learned acquisition.

## Controls and interpretation

All TRAIN finishes before any development score is read. Retain states0/4/8/12.
Each checkpoint evaluates the learned state and unchanged state0 under that
checkpoint's current rules/allowed tools. B and return also evaluate frozen A4
under the same current rules. Frozen controls add no TRAIN but their readouts
are charged. No final/probe data, replay, promotion or tuning is involved.

The report retains per-case targets/selections/obsolete errors, original and
reverse-order results, accuracy/rawNLL/Brier/five-bin ECE, old/new domains and
all signed comparisons. Positive accuracy and negative loss/error differences
are favorable. A→B changes some ground truth, so aggregate phase differences are
not old-rule forgetting. Unchanged password truth is reported separately.
Recovery requires observed loss and later recovery; return success alone is
insufficient. The [native temporal bridge](native-temporal-workflow.md) now
executes this task with separately accounted native updates and frozen controls;
broader scientific and release claims remain open.

The frozen local case completed: learned1/4→4/4→3/4→4/4; unchanged v2 gives2/4,
frozen A under v2 gives1/4 with two obsolete route errors. Password is lost after
B and recovered on return (1→0→1), so retention failure stays visible. Revised
invoice succeeds beyond either control; unchanged masking already solves parcel
and legacy, so their final success alone is not fresh acquisition evidence.
ECE worsens versus unchanged despite accuracy/NLL/Brier gains. This is a useful
bounded conventional example, not a modern-model/CPC or generalization result.
[Observed results, costs and limitations](../artifacts/runs/temporal-decision-stream-20261022/RESULTS.md).

Work records committed updates/rows, attempted/returned/validated readouts,
mistakes, affine multiply/column-add counts, nominal retained coefficients and
RSS observations. Array/stream/actor copies, softmax, validation and Python costs
are additional. A partial failure preserves paid counters/captured state and
marks unobserved attempted work unknown, with no automatic retry.

## Structure, tests and extension

```text
src/core/temporal_decision_sequence.py       fixed rules/requests/role validation
src/adapters/perceptron_decision.py          opt-in subset TRAIN projection
scripts/run_temporal_decision_reference.py   existing stream/evaluator composition
tests/test_temporal_decision_reference.py    semantics, boundaries, integration
```

```powershell
python -m pytest -q tests/test_linear_decision_reference.py tests/test_temporal_decision_reference.py
python -m ruff check src/core/temporal_decision_sequence.py src/adapters/perceptron_decision.py scripts/run_temporal_decision_reference.py tests/test_temporal_decision_reference.py
python -m ruff format --check src/core/temporal_decision_sequence.py src/adapters/perceptron_decision.py scripts/run_temporal_decision_reference.py tests/test_temporal_decision_reference.py
python -m mypy --explicit-package-bases --follow-imports=silent src/core/temporal_decision_sequence.py src/adapters/perceptron_decision.py scripts/run_temporal_decision_reference.py tests/test_temporal_decision_reference.py
```

No new dependency or environment variable. Extend via these task/learner ports
after freezing roles, effective rules, comparisons and costs. Keep changed-policy
truth separate from old-context identity. Do not reuse obsolete labels as current
targets or call reserved/evaluation labels TRAIN. A released-model comparison
needs a separate reviewed budget; this conventional slice grants none. The native
bridge has its own frozen protocol and evidence, linked above.

[Protocol](../artifacts/runs/temporal-decision-stream-20261022/PROTOCOL.md) ·
[Decision](adr/ADR-0531-bind-temporal-tool-rules-at-observation.md).
