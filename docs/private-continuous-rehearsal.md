# Explicit old-TRAIN rehearsal in a private full-network probe

The existing `run_continuous_probe` API accepts a separate fixed TRAIN bank and
original replay budget. This opt-in control currently requires exact
`NumpyContinuousBCLearner` and `optimizer="full-gauss-newton"`. Default probes and
their reports remain unchanged. PC/CPC, SGD, nearest, velocity/gradient-diagnostic
and corrective-query combinations are refused. There is no new CLI mode, runtime,
optimizer, model export or deployment path.

Why this: new-task-only neural fits lost old-task behavior. The existing replay
helper binds a TRAIN runtime while the private runtime admits `probe_train`.
Joint fits already serve the affine/context controls. Reuse their role-preserving
pattern plus the existing fixed-bank validation and native fitter; keep runtime
role checks intact. The new path is a conventional control, not a CPC mechanism.

```python
from time import monotonic
from src.adapters.continuous_probe import ContinuousProbePlan, run_continuous_probe
from src.app.continuous_stream_rehearsal import ContinuousReplayPlan
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget

# Caller supplies qualified source, original reserved episodes, already released
# old TRAIN tuple and a prospectively frozen dose. This is API shape, not a study grant.
old_rows = sum(len(e.transitions) for e in old_train_episodes)
replay_plan = ContinuousReplayPlan(
    every_updates=1, max_replay_examples=frozen_updates * old_rows,
)
replay_budget = ToyBudgetSession(
    ToyExecutionBudget(max_replay_examples=replay_plan.max_replay_examples), monotonic,
)
result = run_continuous_probe(
    qualified_source,
    ContinuousProbePlan(probe_train, probe_test, updates=frozen_updates,
                        optimizer="full-gauss-newton"),
    training_readouts=old_train_episodes,
    replay_plan=replay_plan,
    replay_episodes=old_train_episodes,
    replay_budget=replay_budget,
)
```

Replay requires all three explicit arguments. `training_readouts` alone grants
no training authority: default mode still only scores them. Each replay episode
must equal an already declared old TRAIN readout, have matching action spec and
an ID distinct from both reserved inputs. The current BP route retains its old
maximum of three readouts; the shared bank validator remains1..4whole episodes,
128rows/1MiB defaults. The original budget must be fresh, separate and have exactly
the declared replay allowance; no implicit budget creation/renewal. Only already
arrived TRAIN packets are eligible; this API never selects future data or fetches
referenced files. The caller must preserve released input provenance/consent.

Every private release fits new `probe_train` rows followed by the explicit old
TRAIN bank in one native full-network step. Row spans retain roles and episode
IDs. Probe rows are transient and never enter the TRAIN bank. `probe_test` is
score-only; final evaluation remains unavailable. Source, serving actor and
benchmark learner stay unchanged, with original full-state/owner/receipt guards.
No private child is adopted or promoted. Native kernel math/loss/normalization is
unchanged; extra old rows change the training population and computation.

Read `rehearsal` for admitted bank/row spans, retained payload, per-fit joint array
payload, original replay limits, charged exposures, native attempts/returns and
unknown work. `work.training_rows_attempted/completed` count physical joint rows;
`work.probe_training_rows_completed` separates new-packet rows. The ordinary wake
progress still reports private wake updates and zero wake-budget replay; the
separate permanent replay charge is `rehearsal.charged_exposures`. Do not add this
subset to physical training rows again. Native diagnostics/work in `native_updates`
cover the joint fit; costs for copies, workspace, containers/RSS remain additional.
Outer `elapsed_seconds` includes all probe work; original replay wall/RSS ceilings
are checked without resetting its clock. Charge covers failures, not only commits.
A returned native result is captured before post-native observers can reject it.

New TRAIN and reserved score-only losses are separate readouts; all declared old
losses remain visible. Lower residual error is better; acquisition/new-fit gains
are before minus after, old retention loss is after minus before (positive worse).
A charged return without observed forgetting is not recovery. Equal wake updates
are not matched computation or exposure. No native usefulness, reliable retention,
generalization, CPC advantage or release claim follows from tiny correctness checks.

Focused checks (physical D: with TEMP/TMP/cache on D:):

```powershell
.venv/Scripts/python.exe -B -m pytest -q tests/test_private_continuous_rehearsal.py
.venv/Scripts/python.exe -B -m ruff check src/adapters/continuous_probe.py src/adapters/continuous_probe_rehearsal.py tests/test_private_continuous_rehearsal.py
.venv/Scripts/python.exe -B -m mypy --follow-imports=silent --platform win32 src/adapters/continuous_probe.py src/adapters/continuous_probe_rehearsal.py tests/test_private_continuous_rehearsal.py
```

[Scoped evidence, costs and review](reports/private-continuous-rehearsal-20261102.md).
A new actual comparison needs a separately frozen/reviewed small protocol. Reuse
qualified conventional/no-replay evidence where valid; preserve all failed/spent
runs. No seed/dose/metric/mixture search to manufacture a win. Extend other learner
modes only with explicit role/cost/isolation contracts and meaningful checks.
