# Four-phase post-training policy evaluation

The explicit route preserves the accepted A64/B64/C_train64/A64 training recipe,
then evaluates all five owned checkpoints on complete final traces and actual
live episodes. No final packet exists during training. Final observations,
labels, returns and failures never train, promote, tune or select a learner.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-four-phase-policy --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-four-phase-policy
```

The description creates no native learner or environment. Execution is a fixed
local protocol, not a sweep. No new dependencies/environment variables. Existing
train-only and three-phase evaluation/replay/probe routes keep their defaults.

## Boundaries and behavior

```text
src/adapters/train_acquisition_pilot.py   existing stream and final guards
src/adapters/return_policy_evaluation.py  explicit fixed five-point readout
src/adapters/key_door_pilot.py           shared layout descriptor/live runner
scripts/run_coupled_attribution.py       explicit CLI route
tests/test_four_phase_policy_*.py        boundary and real-stream fake-port tests
```

Why: train accuracy can differ from navigation success after a learner encounters
its own actions. Reuse the existing cached/live evaluator and episode activation
boundary instead of another runner. Fixed checkpoints0/64/128/192/256 include
every A/B/C_train/D cue (40 live episodes). Known A/B/C_train repeat training
geometry. D is held out from this invocation's training but was evaluated in
prior protocols; this is not a newly unseen-domain claim.

C_train uses the same shared rotation2/branch3/swapped/appearance−.75 descriptor
as the accepted training route. A new explicit declared-layout tuple is validated
and copied before learner work; defaults continue using the original layout set.
Every live episode gets a fresh owned recurrent activation session with frozen
parameters/optimizer/gradients. Raw first-tie argmax is used without masking,
action repair, retries or teacher help, at most12 decisions per episode.

All256 updates and stream.finish precede final collection. Before and after
external calls, original source/actor, completed candidate, selected checkpoints,
full receipts and budgets remain binding. Missing/corrupt evidence fails without
more work. Returned cached calls and partial live transitions remain charged;
failed and unreached slots remain visible. No replay/probe/native optimizer
change is part of this route.

## Verification

Prefix Python commands with `.venv/Scripts/python.exe -X utf8 -B`:

```text
-m pytest -q tests/test_four_phase_policy_boundary.py tests/test_four_phase_policy_pilot.py tests/test_return_policy_evaluation_pilot.py tests/test_return_evaluation_boundaries.py
-m ruff check src/adapters/train_acquisition_pilot.py src/adapters/return_policy_evaluation.py src/adapters/key_door_pilot.py scripts/run_coupled_attribution.py tests/test_four_phase_policy_boundary.py tests/test_four_phase_policy_pilot.py
-m ruff format --check <same six paths>
-m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 <same six paths>
```

Repeat mypy with `--platform linux`. Actual execution outcomes and limitations
are recorded in stage RESULTS after checks/run, not inferred from authored tests.
Do not infer circadian benefit or recovery savings from component correctness.
Extend through existing stream/evaluation boundaries with separately frozen
protocols, keeping final evaluation unavailable during any learning or selection.

## Observed frozen case

153 focused tests, six-path statics and one frozen local native case passed.
Complete old train-field projection and known cached final/train readouts match
exactly. All40 actual episodes remain: C_train at192 and A at256 succeed on both
cues (return1); the other36 fail illegal actions (return0). C is learned then
forgotten on A return. All B/D attempts fail; original A acquisition was
incomplete. Useful retention, recovery savings and generalization remain unmet.

The case used256SGD/2048trainrows/3288unroll/246predictions (104live),40sessions,
54environments/180steps and13.2841parent seconds; sampled recipeRSS318042112B
does not certify later serialization peak. No CPC comparison or advantage.
[Exact commands, costs, failures/skips and next action](../artifacts/runs/r42-four-phase-policy-20261016/RESULTS.md),
[independent review](../artifacts/runs/r42-four-phase-policy-20261016/REVIEW.md).
