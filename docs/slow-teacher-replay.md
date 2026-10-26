# Slow-teacher replay

The opt-in coherent NumPy route rehearses arrived labeled rows against a mixture
of their original targets and a slow parameter teacher. It uses the existing
attribution runner, retention selector and native BP/PC/CPC implementations.

```powershell
python -m scripts.run_stream_rehearsal_pilot --describe --scenario slow-teacher
python -m scripts.run_stream_rehearsal_pilot --run --scenario slow-teacher
python -m pytest -q tests/test_slow_teacher_learner.py tests/test_slow_teacher_rehearsal.py
```

The run is a fixed, small exploratory protocol: seed23, A/B/A, four learners,
72 wake and 48 replay calls, 624 training-row exposures, CPU1, 60 seconds, 256MiB RSS,
384 retained replay-array bytes per arm and 2MiB report allowance. Review its
descriptor and declare a fresh local resource envelope before another execution;
saved experimental results are not a reason to tune its seeds or mixture.
No new environment variables or dependencies are required. The CLI fixes BLAS
thread counts before importing NumPy.

## Responsibilities and interfaces

- `core/slow_teacher.py`: validated owned policy and target records, plus the
  affine target mixture. No models, data access, budgets or recovery authority.
- `adapters/slow_teacher_learner.py`: complete owned ordinary NumPy student and
  teacher state, explicit wake/replay updates, pure teacher forwards and restore.
- `app/stream_rehearsal.py`: optional teacher-port orchestration, selected arrived
  rows and permanent replay/forward accounting. No adapter import or final data.
- `adapters/slow_teacher_pilot.py`: frozen native/data binding, resources and
  reporting. Historical comparison results are read only by post-run analysis.

`SlowTeacherLearner(student, policy)` owns its ordinary matched BP or coupled
PC/CPC student. `train_batch` performs the original wake update, then prepares
`teacher = decay*old_teacher + (1-decay)*post_wake_student`. Replay never advances
the teacher. Native CPC chemistry and its native update count still advance on
replay; those fields are not averaged into teacher weights. Prediction always
uses the student. State and EMA publish together after validation.

`make_teacher_targets(x,y,role="train",feature_schema=...,output_schema=...)`
returns detached probabilities with teacher ID, schema, teacher revision, total
student update count and existing content IDs of the original labeled rows.
`train_distillation_batch(x,y,packet)` refuses incompatible/stale packets before
native work, then calls the original native update against
`(1-lambda)*y + lambda*q`. The returned diagnostic has an explicit mixed-objective
ID. At fixed latent states binary BCE and its derivatives are affine in the
target; the native hidden penalties retain their original coefficient. This
does not equate independently relaxed hard-label and teacher trajectories.

Setting `mixture_weight=0` uses the exact native target and diagnostic path, with
no teacher weights, forward or EMA. Default rehearsal without a teacher keeps
its original schema and behavior. Snapshots bind complete native family,
geometry, rates/inference policy, precision, CPC config/chemistry/gates and
pre-wrapper native clock. Forks own their arrays; invalid restore or staged work
cannot partially publish the student/teacher pair. Snapshots convey no consent,
budget or restoration authority.

## Cost and failure interpretation

Each positive arm retains 840 bytes of teacher parameter arrays in addition to
the replay arrays. It recomputes one probability for each selected replay row;
no unbounded success-record chain or target cache is retained. Forward attempts
are recorded before the call; returned forwards, generated probability bytes,
returned mixed-target bytes and peak probability payload are separate fields.
The fixed family adds 48 teacher forwards and 72 wake EMA updates. Logical payload
bytes exclude copies, snapshot/checkpoint states, event metadata and Python
overhead; measured process RSS is separate. Student and teacher parameters plus
a full replay buffer total 2064 logical bytes per arm. Native CPC also retains
128 bytes of chemistry/gates, giving 2192 bytes for that subtotal; these totals
exclude staged copies and all the overheads listed above.

Native replay exposure remains charged before the native update, including
exceptions. Reports distinguish attempted, returned and committed work. A failed
teacher forward can incur computation without a native replay charge. A failed
wake may return from native training before EMA/publication fails: the completed
wake counter alone does not measure those native attempts. Failure reports must
preserve unknown partial work rather than impute zero. Budget clocks/charges are
external and are never restored or refunded by a learner snapshot.

Final scoring starts only after all training for that arm. Neither final data
nor final scores select targets, replay rows, mixture weights or checkpoints.
Teacher helper continuation, Torch distillation and trajectory replay are
explicitly unsupported. This probability ablation is not DER/DER++, an equal
compute comparison, a general efficiency result or evidence of CPC advantage.

For extensions, retain the existing runner and selector; add a separately
versioned native objective/port only when needed, with disabled-path parity,
stale-schema refusal, complete state isolation and prospective cost limits.
