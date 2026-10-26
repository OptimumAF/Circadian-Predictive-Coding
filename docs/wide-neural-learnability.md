# H32 isolated fresh-task and A-return diagnostic

## Optional prior-gradient diagnostics on original TRAIN pairs

Add `--prior-gradient-geometry` to the original `--wide-neural --joint-catalogue`
route for BP, PC, neutral CPC or active CPC. This uses existing detached mean-CE
prior gradients at TRAIN checkpoints0/32/64/96/128, three blue/red task pairs per
point. Native54-row clocks and complete model/optimizer/adaptive/RNG state remain
bound. It performs15pair calls,270forward/backward rows and639360projectionbytes
per method, with zero diagnostic optimizer steps or published updates.

This option excludes fixed64 controls, PC-update diagnostics, extra F exposure,
rehearsal, games and other diagnostics. H8 joint recipes still refuse it; the older
H8 non-joint diagnostic and recurrent gradient route keep their existing behavior.
Prior-CE cosine is descriptive and does not measure the native PC update direction.
No final/probe data enters TRAIN diagnostic selection. Completed evidence stays
sealed through final publication callbacks; corrupted values/owners/costs refuse.

[Runnable describe command and verification scope](reports/h32-prior-gradient-integration-20261019.md).
No acquired-task diagnostic study or learning benefit is yet established.

The existing MiniGrid checkpoint-learnability runner has an explicit `wide_neural`
recipe for its four native learners. Why this: the preceding H32 BP checkpoint
can complete red A but fails blue A, making separate new-task fit and both old-A
monitors useful. Those games establish neither complete acquisition nor retention.
This option reuses the isolated F16 then A16 runner and keeps both cue outcomes.

## Usage

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --joint-catalogue --distinct-probe --old-task-readouts --return-to-a --wide-neural --describe
```

`qualification_method` is one of `bp`, `pc`, `neutral_cpc`, `gated_cpc`.
The API is `minigrid_learnability.run_pilot(public_history=True,
invariant_history=True, qualification_method="bp", joint_catalogue=True,
distinct_probe=True, old_task_readouts=True, return_to_a=True, wide_neural=True)`.
Use `describe_protocol` with the same arguments for a resource-free preview.
Actual comparative execution needs a separately frozen and reviewed local protocol;
this increment's budget covers synthetic engineering checks only.

The flag requires this exact route. Other diagnostic, weighting, relaxation,
rehearsal, memory, cue-relative and TRAIN-game options refuse. Explicit
`probe_rollouts=True` adds [fixed target-free A/F games](wide-neural-probe-games.md)
after each paid F16/A16 endpoint. It never
infers permission from a snapshot's width. Omitting the flag preserves original
H8/default descriptors, native algorithms, callback arguments and routes.

## Sequence and isolation

Fixed D296/H32/K7/seed17/LR.03/momentum.9/PC S2/eta.04, unchanged native objectives.
The existing stream performs128 complete54-row joint TRAIN updates. It reads all
six original TRAIN traces at0/32/64/96/128 using owned frozen readers. Separate
disposable cohorts start from the genuine0 and128 full native checkpoints.

Each cohort learns the fixed F-blue/red `probe_train` sequence for16 updates;
F `probe_test` and original A-blue/red TRAIN monitors are read at0/4/8/16. That
same disposable learner and optimizer then receive16 original A TRAIN updates,
again reading both F and A at0/4/8/16. Return0 must reproduce the preceding F16
and A-monitor endpoint. No best endpoint or favorable cue is selected.

Original source, benchmark candidate, serving actor, native optimizer/adaptive/
gradient/RNG state, supplied packets, monotonic clocks and permanent charges stay
guarded. The four original F/return budget objects remain separate; neither phase
renews the other or the original enclosing time/RSS allowance. Probe updates never
enter deployed or benchmark learners, and probe_test/final data never train,
promote or select a model. Paid failures and potentially unobserved work remain
explicit. Structural snapshot validation does not prove historical training;
the public composition additionally binds checkpoints to its completed stream.

## Reports and costs

Existing `readouts`, `probes`, nested `return_phase`, both-A `training_readouts`,
budgets and work records stay separate. `joint_catalogue_acquired` means exact
teacher-forced TRAIN traces, not game success. Fullfit false or negative fresh-task
gains are valid. The already accepted game results remain separate evidence.

Per complete selected-method case:128 benchmark+32 F+32 return=192 native calls;
6912+288+288=7488 rows;94 prediction calls/846 rows. H32 has9735 trainable float32
parameter elements/38940 array-equivalent bytes. BP derives192 SGD/backward calls;
each PC family derives192 parameter updates and14976 nominal latent-row steps.
These are code-derived primitive costs, not an external operation meter.

`protocol.wide_neural` describes this recipe. `work.wide_neural` includes the
observed paid-prefix totals and derived native costs. After unobserved failure,
affected derived counts are `null`, rather than invented zeros. Full state and
copies, history/projection/validation, Python/report/backend allocations, elapsed
time and sampled RSS add costs. Parameter bytes are not total memory; equal row
exposure is not equal computation. Original work fields retain their definitions.

## Structure and checks

```text
src/adapters/minigrid_learning_pilot.py    explicit policy/counter/stream admission
src/adapters/minigrid_learnability.py      existing factory/readers/F16/A16 composition
scripts/run_coupled_attribution.py         explicit CLI route and preview
tests/test_minigrid_wide_probe_counters.py policy/clock/default refusal checks
tests/test_minigrid_wide_learnability.py   CLI/ownership/public synthetic chain
```

Focused: `python -m pytest -q tests/test_minigrid_wide_probe_counters.py
tests/test_minigrid_wide_learnability.py`. Use `python -m ruff check` and
`python -m ruff format --check` on those five paths. Types: `python -m mypy
--python-version 3.11 --follow-imports=silent --no-incremental --no-sqlite-cache
--cache-dir=NUL --platform win32` on the three source files; repeat for linux.
No dependency or product environment variable was added. Future recipe changes
must bind explicit policies, counters, roles and budgets and retain old defaults.

[Scoped commands, outcomes, failures, costs and independent review](../artifacts/runs/wide-neural-learnability-20261018/RESULTS.md).
Engineering correctness does not establish useful adaptation, acquired retention,
return-context recovery, heldout/RL/release readiness or circadian advantage.

## Observed scoped qualification

354distinct cases pass (149new/205regression), six-path Ruff/format and3source
Win/Linux types pass after the preserved initial18-error typing failure. The
synthetic public BP192update case completes both isolated cohorts with unchanged
source/actor/RNG/inputs/budgets. Independent SOURCE/PROTOCOL/ACTUAL GO verifies
270rawprobability+576decisionrows/128receipts/64diagnosticblocks. No actual
physics or original-source comparative probe study ran. Synthetic constant-action
F16fits, oldAdoesnotfit, and A16destroysFfit; all curves and failures remain.
