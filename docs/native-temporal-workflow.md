# Native temporal workflow

The native runner uses the [same temporal task](temporal-decision-reference.md)
as the conventional reference through the existing Open-Jev learner/inbox.
It scores typed local choices; no workflow tool or external action is executed.

```text
scripts/run_open_jev_temporal_sequence.py   native stream/evaluator composition
scripts/run_open_jev_sequence.py           explicit task validator/policy option
scripts/run_open_jev_native_update.py      explicit readout instructions option
tests/test_native_temporal_sequence.py     scalar integration/failure boundaries
```

Use an already qualified offline native source/checkpoint environment:

```powershell
python -m scripts.run_open_jev_temporal_sequence --source SOURCE --checkpoint CHECKPOINT --output native-temporal.json
```

The output must be fresh. The fixed workflow instruction contains no target or
rule table. TRAIN labels and policy versions bind at observation. v1 permits
inspect/retire/route; v2 addsdefer/removesretire and revises invoice/legacy/parcel
truth. Returning A keeps v2. Development templates remain separate from TRAIN,
with overlapping concepts and a known union catalog/vocabulary.

All36 native single-request updates finish before development. Four existing
owner snapshots at0/12/24/36 supply learned and unchanged controls at each phase,
plus frozenA12 at B/return, all under the same current catalog/instruction.
Ten evaluations each make eight original/reverse readouts:80 prediction calls.
Successful execution restores final36. Snapshot restores never renew update
allowances. On failure, retain the paid prefix and selected source, release
private snapshots, and stop; latest profiler totals may be only a lower bound.
Native response fingerprints describe the loaded birth package; the selected
in-memory source update is recorded separately. No portable trained checkpoint
is saved or reloaded.

Reports include raw native replies, probabilities, provider margin confidence,
max-probability calibration, accuracy/rawNLL/Brier/five-binECE, candidate-order
sensitivity, domains, obsolete errors and signed learned-minus-control changes.
Stable password loss/recovery is distinct from changed-target A-to-B differences.
No labels reach prediction wires; no final/probe/replay/promotion is involved.

The native36updates and conventional12three-row passes expose the same36TRAIN
rows with different math/optimization/compute. Neither is a CPC comparison.
Snapshot storage, restores, training, serving, tokens, wall time and process RSS
are separately bounded and reported. The local frozen protocol admits one
offline2B CPUfloat32 graph, seed17/rank8,36TRAIN/80predict/276topforward calls,
four snapshots/11restores/400MiB retained numeric storage,600s/20GiB process limit.
See the protocol for complete prospective allowances and qualification gates.

## Observed bounded result

The sole native case completed: learned correct counts were2/4→3/4→2/4→4/4.
Both unchanged and frozen-A controls score0/4 under v2, with two obsolete route
errors. Learned legacy/parcel defer choices survive return; parcel was absent
from return TRAIN, but its target probability falls from.782394 to.507942.
Password is0→0→1 after A/B/return, so final success is late acquisition, not
observed loss/recovery. ECE worsens versus unchanged after A; all metrics and
negative endpoints remain reported. This is useful local native adaptation,
not a CPC, matched-compute, generalization or empirical calibration claim.

36 updates/80 readouts cost421.98s wrapper time and13,990,428,672B process peak.
Four snapshots retained130,479,340 numeric bytes before release; eleven restores
preserved permanent work. Focused checks:166PASS/platform, scoped lint/format/
type checks passed. [All results, costs and scope](../artifacts/runs/native-temporal-stream-20261022/RESULTS.md).

## Scoped verification and extension

```powershell
python -m pytest -q tests/test_native_temporal_sequence.py tests/test_native_choice_sequence.py
python -m ruff check scripts/run_open_jev_temporal_sequence.py scripts/run_open_jev_sequence.py scripts/run_open_jev_native_update.py tests/test_native_temporal_sequence.py
python -m ruff format --check scripts/run_open_jev_temporal_sequence.py scripts/run_open_jev_sequence.py scripts/run_open_jev_native_update.py tests/test_native_temporal_sequence.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/run_open_jev_temporal_sequence.py scripts/run_open_jev_sequence.py scripts/run_open_jev_native_update.py tests/test_native_temporal_sequence.py
```

No new environment variables or dependencies. Keep new task validation explicit,
preserve routing defaults, and freeze a new task/protocol before extending the
corpus. Existing final/probe examples must not become TRAIN or controller input.

[Protocol](../artifacts/runs/native-temporal-stream-20261022/PROTOCOL.md) ·
[Review](../artifacts/runs/native-temporal-stream-20261022/REVIEW.md) ·
[Decision](adr/ADR-0532-use-native-owner-for-temporal-workflow-controls.md).
