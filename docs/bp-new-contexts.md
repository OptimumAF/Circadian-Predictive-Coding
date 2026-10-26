# New-context learning, old-task retention and return with qualified BP

This fixed recipe reuses the [qualified BP state](ticket-bp-fit-qualification.md)
and tests both previously untrained input contexts. It does not repeat warmup.

```powershell
python -B -m scripts.run_bp_new_contexts --input artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --output fresh-new-contexts.json
```

Requires existing CPU Torch. This performs native TRAIN; use the fixed
[protocol](../artifacts/runs/bp-new-contexts-20261023/PROTOCOL.md), no automatic
retry, sweep, early stop or seed/context/dose selection. Output must be fresh;
the fixed source hash rejects changed/replaced prior evidence.

## What runs and what stays separate

Small `adapters.torch_bp_record.bp_snapshot_from_record` converts exact bounded
CPUfloat32 scalar records and canonical optimizer IDs to the existing validated
TorchFeatureSnapshot. It preserves weights, SGD momentum, gradients, policy and
native96update/576row counters. It reads no old budget, runtime cursor, receipts,
consent or authority. This BP-only reader supports small existing heads and is
not a generic model loader; no pickle or executable deserialization.

The new isolated runtime has16new update slots. Existing AttributionStream and
delay1 apply eight6-row legacy/password TRAIN batches, then eight6-row invoice/
parcel return batches:96new row exposures. Native counters at local0/8/16 must
equal96/104/112 and576/624/672. The old96fit cost remains charged, separately
reported with its immutable source. Source/seed/fresh local stable actor/full
frozen readout/RNG and original source bytes are guarded around work.

Both untrained contexts were frozen before observing them. They use known action
classes and vocabulary; this is new input-context learning, not new classes or
unseen-task statistical generalization. All four ordinary tasks run at every
point; stale/refusal tests cover both new contexts at8and old contexts at16.
Teacher12-row readouts are separately charged, with exposure counts. No final/
reserved-probe/replay/promotion/controller selection, no changed native algorithm.

## Observed result

| New updates / native clock | Invoice | Parcel | Legacy | Password |
|---|---|---|---|---|
|0 /96|success|success|wrong workflow|success|
|8 /104|success|success|success|success|
|16 /112|success|success|success|success|

`legacy` is acquired; `password` is initial transfer, not newly learned success.
Both old tasks are retained and still succeed when their context returns.
New-context gain+1, old forgetting0, old return change0, new change during return0.
No old task was lost, so this does not demonstrate recovery from forgetting.

Both scheduled new-context and old-context stale cases recover in5steps; all
four revoked commits are refused in3steps. Refusal is the desired constraint
outcome, not a committed task. Ordinary tasks take3steps, including the original
legacy failure that committed a permitted but wrong workflow.

Teacher global meanCE.30325→.24717→.20501, correct11/12→12/12→12/12. Exact metrics,
all task traces and negative baseline preserved in [results](../artifacts/runs/bp-new-contexts-20261023/RESULTS.md).
One fixed synthetic stream/seed supports scoped task behavior, not broad
retention/reliability/generalization, a causal mechanism, CPC or efficiency gain.

New actual work16updates96rows/28native components1092initializer scalar slots,
87native forwards68typed,999snapshotrecords143280logical tensor elements,
80local transitions incl12teacher; outer15.7464s, meter10.6500s, recipe8.83382s,
sampled peakRSS331522048B. Historical warmup96updates576rows and costs remain
additional. Counts exclude some allocation/validation/IO/Python work; no
single-observation efficiency ranking. Nothing restores the old spent budget.

## Structure and verification

```text
src/adapters/torch_bp_record.py    Bounded BP payload readback, native validation
scripts/run_bp_new_contexts.py    Fixed recipe over existing stream/task ports
tests/test_bp_new_contexts.py     Full state/continuation and tiny context/fault checks
```

```powershell
python -m pytest -q tests/test_bp_new_contexts.py
python -m ruff check src/adapters/torch_bp_record.py scripts/run_bp_new_contexts.py tests/test_bp_new_contexts.py
python -m ruff format --check src/adapters/torch_bp_record.py scripts/run_bp_new_contexts.py tests/test_bp_new_contexts.py
python -m mypy --explicit-package-bases --follow-imports=silent src/adapters/torch_bp_record.py scripts/run_bp_new_contexts.py tests/test_bp_new_contexts.py
```

20checks/platform and scoped statics pass. Tiny tests use one batch/phase plus
two live continuation calls, not96warmup or the scientific16update observation.
Private native/actor access in the recipe is trusted synchronous composition,
not a hostile-process or concurrency protection boundary.

The next outcome should be a small frozen matched PC/neutral+gated CPC control
study using the same96old-joint +8new +8return data/seed/endpoints, reusing saved
BP evidence. Keep peer warmup/adaptation in one live owned stream to avoid more
record codecs. Compare native task behavior and costs before new mechanisms;
equal dose/geometry is not equal computation or objectives, and no control is
tuned to beat BP. No further run is implied by this handoff.
