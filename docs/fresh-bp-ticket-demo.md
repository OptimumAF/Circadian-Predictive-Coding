# Run a fresh native BP task demonstration

In an existing CPU Torch environment:

```console
python -m scripts.run_native_ticket_controls --fresh-bp --output fresh-bp.json
python -m scripts.build_key_door_inspector --source-artifact fresh-bp.json --output fresh-bp.html
```

Both output paths must be fresh. This explicitly performs native training;
saved `--example-ticket` inspection remains read-only and requires no Torch.
The fresh command needs no historical artifact or input checkpoint. Existing
`--method pc|neutral_cpc|gated_cpc` control commands keep their original behavior;
`--method bp` stays refused, and source mode selection is mutually exclusive.

## What runs

The original native BP policy starts from seed17, seven fixed features, three
hidden units, six classes, lr.03/SGD momentum.9. The existing delayed-label stream
owns96joint invoice/parcel TRAIN batches, eight legacy/password batches and eight
old-context return batches, six rows each. All112updates/672row exposures belong
to one new budget; there is no restore of prior runtime authority or charges.

This repeats twelve synthetic teacher-state templates, not672independent examples.
Both new contexts and all task slots are fixed in advance. Native checkpoints
96/104/112 (local0/8/16) use separate frozen readout owners. Twelve teacher rows
per endpoint and20complete task episodes are charged separately; stale revision
and revoked-commit cases stay distinct from ordinary task success. The stable
actor/seed remain initial and unchanged: this is candidate learning, not live
deployment or promotion.

Source/actor/seed/full frozen readout/RNG/native-clock/receipt guards remain.
Partial failures retain attempted/returned work, committed prefix and unknown
returns. Native budget112updates/60s/512MiB sampled process RSS; output512KiB.
Failures do not renew budgets or automatically retry. No replay, reserved probe,
final evaluation, controller selection, new algorithm or changed baseline.

## Reading the result

`fresh_bp_ticket_demo_v1`, method`bp`, role`fresh_train_demo` distinguish this
from historical saved-state adaptation and the peer comparison. The inspector
shows fresh ownership, exact source cells, complete action/effect/result states,
original counters and inactive/unknown mechanisms. Snapshot bytes, sampled RSS
and nominal affine counts are partial cost views, not all allocations or FLOPs.

The prior accepted fixed case acquired legacy, transferred password initially
and retained both old tasks; no old task was lost, so it did not demonstrate
recovery from forgetting. Exact reproduction in an engineering validation is
not a second independent scientific replicate. Known-class synthetic task ability
does not establish generalization, continuous deployment, safety or CPC benefit.
All old peer failures, source hashes and scientific protocols remain unchanged.

Why this: a fresh user needs an actual training-to-task path without a developer's
historical JSON hash. Reusing the existing live recipe avoids another state reader
or orchestration framework. Original defaults, data, metrics and peer commands
stay fixed; this is not a new comparative study or an invitation to tune the dose.

Focused checks:

```console
python -m pytest -q tests/test_fresh_bp_ticket_demo.py tests/test_native_ticket_controls.py tests/test_ticket_workflow_inspector.py
python -m ruff check scripts/run_native_ticket_controls.py src/app/key_door_inspector.py src/app/ticket_workflow_inspector_projection.py tests/test_fresh_bp_ticket_demo.py
python -m ruff format --check scripts/run_native_ticket_controls.py src/app/key_door_inspector.py src/app/ticket_workflow_inspector_projection.py tests/test_fresh_bp_ticket_demo.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/run_native_ticket_controls.py src/app/key_door_inspector.py src/app/ticket_workflow_inspector_projection.py tests/test_fresh_bp_ticket_demo.py
```

Fixtures use one batch in each phase, not a full warmup. Optional Torch setup,
clean SDK/GPU installation, broad fresh-run robustness and release readiness
remain separate requirements; no dependency is downloaded by this command.

Observed installed validation: all112updates returned. Legacy changes from failure
to success, password transfers initially, old tasks remain correct and all four
are correct after return. All scheduled stale cases recover and revoked commits
are refused. No old forgetting occurred. Full native states, teacher readouts
and task traces match the saved accepted BP case, checked only after completion.
This is engineering consistency, not independent replication or new science.
41focused checks/platform and scoped statics pass; original peer/bundled results
are unchanged. [Commands, costs, skips and evidence](../artifacts/runs/fresh-bp-ticket-demo-20261023/RESULTS.md).
