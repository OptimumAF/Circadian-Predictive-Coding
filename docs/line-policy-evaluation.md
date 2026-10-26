# Live policy evaluation from recorded Torch checkpoints

The existing attribution CLI can evaluate actual LineNavigation episodes from
all saved checkpoints of its accepted complete-demonstration recipe. The JSON
visibly separates recorded training/parameters from current live transitions.
Original R4.1 is independently accepted; broader MiniGrid/RL and useful adaptation
remain unfinished. See [all outcomes](reports/line-policy-evaluation-20261016.md).

## Run and inspect

From the repository root, using the existing environment/dependencies:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --describe --scenario torch-trajectory-rollouts
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-trajectory-rollouts --source-artifact artifacts/runs/r24-trajectory-rehearsal-20261016/pilot.json
```

The second command executes a new bounded evaluation, with no retraining. Reuse
the [accepted output](../artifacts/runs/r41-line-rollouts-20261016/pilot.json) to
inspect this case without spending another evaluation allowance. The source is
required explicitly, at most2MiB UTF8 JSON, with exact fixed protocol, geometry,
task sequence, checkpoint clocks, methods and float32-representable weights.
`--describe` reads no source. Source arguments with other scenarios are refused.
No new environment variables or dependencies were introduced.

Every episode displays nested `identity` (method, seed, local actor version,
task sequence, wake checkpoint, replay mode), `phase=evaluation`,
`learning_active=false`, `checkpoint_origin=recorded_parameters` and
`execution_origin=live_environment`. The actor version is a local evaluation
identity, not a promoted historical actor. Top-level training origin is recorded.
No illustrative frames are reported as measured behavior.

## Policy, failures and costs

The fixed case uses seed17 and all32 method/replay/checkpoint copies. Four logged
parameter tensors replace those in a fresh valid native template. All64 original
A/B cached scores must match before any live episode: accuracy exact, BCE absolute
tolerance1e-12/relative0. Native `predict` supplies the probabilities. This bridge
does not recover historical optimizer, RNG, adaptive state or publication history.
Prediction checks its complete local native state before and after each call.

Each copy acts in both actual A(start−1, goal+1) and B(start+1, goal−1). Native
class1 probability≥.5 chooses right, otherwise left; at most4 decisions. Legal
moves change integer position by one; reaching the goal terminates with reward1.
The report records observations, probabilities, action, current legal actions,
whether applied, reward, next observation and terminal flag for every decision.
An illegal action stops with unchanged position/zero reward; max steps is another
negative outcome. There is no clipping, retry or action replacement.

Unexpected prediction, budget or environment faults preserve an owned partial
episode, attempted/returned work and failed/unreached remaining rows, then stop.
Predictor state faults permanently stop that copy. Invalid source metadata is
rejected before native model construction. App `LineEvaluationFailed.report`
retains the original cause through exception chaining.

Limits: CPU/BLAS1, GPU0,60s body,768MiB process RSS,2MiB JSON;32 copies,
64 cached forwards,64 live episodes/at most256 live forwards and256 legal steps.
Training, replay, probes, promotion and feedback are disabled. The saved case used
192 total forwards/107 legal steps in7.6162161s, RSS312,217,600B. All43 successes
and21 illegal-action failures remain. All final B episodes fail; no circadian
advantage or robust retention follows. Final data never selects training,
promotion, tuning, probes or checkpoints; the complete frozen set is evaluated.

## Structure and extension

```text
src/core/line_navigation.py                 actual environment and legal actions
src/app/line_policy_evaluation.py           identity, bounded episode, partial errors
src/adapters/frozen_torch_predictor.py      validated weights, native frozen predict
src/adapters/line_rollout_pilot.py           fixed source binding and budgeted recipe
scripts/run_coupled_attribution.py          existing CLI scenario
tests/test_line_policy_evaluation.py        episode and identity behavior
tests/test_line_rollout_pilot.py            bridge, recipe, source and CLI boundaries
```

The app consumes a prediction callback; core imports neither native learners nor
IO. The adapter composes the existing native learner and existing budget/RSS
utilities. Why this: demonstrate actual behavior from accepted weights without
retraining or creating another orchestration framework. No training loss changes.
See [ADR-0322](adr/ADR-0322-evaluate-recorded-weights-through-live-policy-episodes.md).

Focused commands (the saved budgeted invocation also includes legacy trajectory
and Torch recipe tests):

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_line_policy_evaluation.py tests/test_line_rollout_pilot.py
```

Exact affected Ruff/format/Windows+Linux mypy commands, initial test typing errors,
all133 passing focused cases and native receipts are in
[results](../artifacts/runs/r41-line-rollouts-20261016/RESULTS.md).

The [categorical stream path](categorical-trajectory-rehearsal.md) now supports
explicit action counts through existing native/episode/attribution ports.
[Actual key/door imitation and live episodes](key-door-navigation.md) now execute
with separate final-role data and declared held-out layouts. All320 learned
episodes fail illegal; binary defaults/native objectives remain. Trainable
recurrence with episode resets is next; separately evaluated RL remains open.
