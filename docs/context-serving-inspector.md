# Inspect guarded contextual learning

Render an existing guarded learning report with the ordinary inspector CLI:

```console
python -m scripts.build_key_door_inspector --source-artifact fresh-context.json --output context.html
```

Use the JSON from [the installed CPU learner](context-example.md), or another
saved `guarded_context_serving_v1` report. Output must be new. Open the standalone
HTML locally; no server, network, model or simulator is needed to view it. The
browser automation qualification used a local server because its automation
policy blocks file URLs; ordinary browser viewing remains offline.

The existing table/source dialog shows every arm, phase, public policy context,
task success/failure, atomic actor read and validated action/observed/effect
state. Rejected and accepted guards, private candidate counters, serving actor
observations, original TRAIN receipts, historical charges and rollback remain
distinct. Click any recorded value for its exact source pointer; the full report
includes original model snapshots and raw pre-validation returns.

The source's initial v2legacy success is not new acquisition. v1legacy is the
new task in this example; earlier failure and restored failure on rollback
remain visible. Guard values are repeated known TRAIN-template evidence, not
independent final evaluation. Missing data stays unknown; null, zero and false
remain recorded values. Separately supplied costs have UNVERIFIED association.
The viewer does not authenticate source claims, infer success, train, promote or
recompute scientific metrics. No CPC/efficiency/safety/generalization claim.

## Verification and extension

54 distinct scoped metadata checks per platform pass, including the unchanged
older ticket routes, source pointers, partial/error states, array/type bounds,
inert strings and exclusive publication. Scoped Ruff/format/mypy pass. Desktop
and narrow Chrome checks qualify source dialogs, all42outcomes/126reads/126steps,
guard decisions, rollback, unknown/null/false/zero, keyboard table scrolling and
the old PC view. Four screenshots were inspected. All model counters are zero.
Initial fixture/tooling/browser assertion failures are preserved in the evidence.

```console
python -m pytest -q tests/test_context_serving_inspector.py tests/test_ticket_workflow_inspector.py
python -m ruff check src/app/context_serving_inspector_projection.py src/app/key_door_inspector.py tests/test_context_serving_inspector.py
python -m ruff format --check src/app/context_serving_inspector_projection.py src/app/key_door_inspector.py tests/test_context_serving_inspector.py
python -m mypy src/app/context_serving_inspector_projection.py src/app/key_door_inspector.py tests/test_context_serving_inspector.py
```

New pure module: `src/app/context_serving_inspector_projection.py`; existing
dispatch and generic table renderer handle the rest. Extend explicit recorded
fields/bounds here, retaining detached source pointers and old family routes.
Do not introduce execution or fill absent metrics with computed assumptions.
No dependencies or environment variables added. Installed-wheel requalification
of this new viewer, clean-clone and full release remain outside this scoped check.

[Issue91](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/91),
[commands and evidence](../artifacts/runs/context-serving-inspector-20261023/RESULTS.md).
