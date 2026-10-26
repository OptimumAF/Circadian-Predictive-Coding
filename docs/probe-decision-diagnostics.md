# Reserved-probe decision and class diagnostics

`run_fresh_task_probe` supports explicit `decision_diagnostics=True`. It describes
the predictions already returned by each reserved `probe_test` scoring call.
The default is `False`, preserving the existing report fields and scoring.
No additional learner update, prediction, fork, budget callback or timer reading
is requested. The additional NumPy work and Python report objects still cost
time and memory.

```python
report = run_fresh_task_probe(
    source, owned_checkpoint, probe_training, probe_evaluation,
    budget=separate_fresh_probe_budget,
    checkpoint_updates=(0, 4, 8, 16),
    decision_diagnostics=True,
)
details = report["checkpoints"][0]["decision_diagnostics"]["E-red"]
```

The existing input, role, checkpoint and original budget requirements apply.
Probes train disposable owned forks; they neither change the source/deployed
learner nor select benchmark training, promotion or final evaluation. Check the
report status before consuming all checkpoints. A diagnostics failure retains
preceding complete checkpoints and actual native charges; it publishes no
partial checkpoint and starts no later work.

## Reading the report

Each context contains:

- `class_counts`, `predicted_class_counts`, and `confusion` with true classes
  on rows and predicted classes on columns.
- `per_class`: support, predicted count, correct count, accuracy and mean loss.
  An absent target class has `accuracy=None` and `mean_loss=None`.
- `decisions`: row index, original target, target class, predicted class, loss
  and correctness. `row_ids` retains the corresponding reserved row identities.
- `episode_id` and `sequence_kind`. An index in `complete_episode` follows the
  validated complete episode order. `batch_row_order` is not a temporal claim.
- `num_classes`, `rows`, `loss_definition`, and `target_class_rule`.

Binary reports retain soft targets, inclusive `>=0.5` target/prediction classes
and the original clipped binary cross entropy with `log1p`. Categorical reports
retain hard labels, first-argmax ties and original categorical cross entropy.
The existing aggregate scores remain authoritative and unchanged. No additional
aggregate ranking or model-selection score is introduced.

## Bounds and cost accounting

The pure helper accepts exact float64 arrays with 1–128 rows and 2–16 classes;
binary probabilities have one column. It rejects an oversized input envelope
before value-check arrays are allocated. The app admits at most 17 checkpoints,
four evaluation contexts, 128 decisions per context and a 16×16 confusion matrix.
All returned values are detached Python primitives.

`reserved_data_bytes` still counts only detached feature/target payloads. It
excludes the new Python report and temporary NumPy arrays, as well as existing
episode/native-state objects. Measure total invocation time around the call;
the inner elapsed clock starts after preflight and detachment. Sampled process
RSS observes whole-process memory rather than assigning bytes to the report.

## Structure and checks

```text
src/core/probe_decision_diagnostics.py   pure bounded report construction
src/app/fresh_task_probe.py             opt-in existing probe integration
tests/test_probe_decision_diagnostics.py core semantics, bounds and ownership
tests/test_fresh_task_probe_diagnostics.py app defaults, costs and failure behavior
tests/test_probe_diagnostics_integration.py native off/on engineering equivalence
```

Run the fake scope with the repository virtual environment:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_probe_decision_diagnostics.py tests/test_fresh_task_probe_diagnostics.py tests/test_fresh_task_probe.py tests/test_categorical_fresh_task_probe.py tests/test_key_door_learnability_pilot.py
```

The native integration test performs two one-update disposable probes on a
fresh recurrent checkpoint, under its explicit local engineering caps. It is
an equivalence check, not a learnability or adaptation experiment. Use the
saved [scope and commands](../artifacts/runs/r21-probe-decision-diagnostics-20261016/SCOPE.md)
when interpreting its evidence. No environment variable is needed. The existing
key/door learnability CLI now exposes the opt-in with
`--scenario torch-key-door-learnability --probe-decision-diagnostics`;
[`--describe` and usage](key-door-learnability.md) retain its fixed protocol.

Why this: aggregate accuracy previously matched a majority rule, and the matched
warm-start comparison lowered CE while lowering accuracy. Existing predictions
can reveal class and decision failures before changing learners or running more
experiments. The existing checkpoint-probe adapter now forwards the option while
retaining original roles, updates, checkpoints and isolation checks.
