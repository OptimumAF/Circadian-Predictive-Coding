# Isolated A-only learning before a frozen robot return

The existing learned-robot composition accepts `a_only_control=True` and CLI
`--a-only-control`. It forwards the existing continuous learner control: every
training phase updates from warm A, while new B remains read-only TRAIN diagnostics.
The default A/B/A path, omitted/False call arguments and output schema remain.

## Inputs, output and ownership

Use the same two qualified teacher inputs or owned complete TRAIN episodes. Both
must pass the existing input binding; A-only does not convert probe/guard/final
roles or omit the B diagnostic cost. `--readiness-teachers` admits the separately
qualified full-phase demonstrations without changing their definition/IDs.
Camera/lighting/state/RGB restrictions stay with the existing input routes.

Only the outer robot script changes; no model, objective, optimizer or orchestration
framework is added. It validates the actual A-only receipt identities, original
arrival clocks, permanently charged counts, successful history and exposure before
handing off the detached candidate. Exact counts reject float/bool aliases. Full
parameter/momentum/wake/source/actor guards and paid failure reports remain.
BP context-work maps remain; PC uses actual native per-packet rows and work counts.
A-only candidate cannot satisfy an unrequested default mode even with equal lengths.

Opt-in output adds a truthful A-only protocol/scope and `a_only_control: true`.
Training source IDs still identify both inputs; only warm-source receipts update.
Original acquisition receipts remain authoritative; reading saved inputs is new
IO/validation work, not new teacher collection. Output files are exclusive.

## Usage

Running this example spends learning and simulator work. Freeze a local protocol
first; do not repeat the accepted case as a correctness check:

```powershell
.venv/Scripts/python.exe -m scripts.run_continuous_robot_learning --output fresh-a-only.json --warm-teacher readiness-17.json --new-teacher readiness-19.json --readiness-teachers --visual-observation --a-only-control --learner-family bp --phase-updates 8
```

Python callers use `run_learning_rollout(output, warm, new, a_only_control=True)`;
nonboolean mode refuses before fitting or output. All four existing learner families
are supported; a new family comparison requires a separately frozen fair protocol.

## Verification and interpretation

193 focused cases pass perplatform with scoped statics and independent combined
acceptance. The new four-family fixture captures actual A features/labels and A-only
fit, read-only B diagnostics, unchanged source/actor and exact candidate serving.
Metadata forgery refuses before native while paid returned training remains.

The six existing metric names remain for compatibility. In A-only mode, B residual
changes describe effects of A updates; they are not evidence of B acquisition.
No new-task learning or old-task retention claim follows from this control. Equal
update counts with A/B/A do not equalize row exposure, computation or information.
The preceding full-phase A/B/A case failed native task performance; final A fit was
better than postA8, so B is not demonstrated as the failure cause. Isolating base
acquisition is the declared question, not a promised fix or circadian advantage.

[Exact checks, frozen native case, all outcomes/costs/failures](../artifacts/runs/readiness-a-only-learning-20261020/RESULTS.md)
· [Independent reviews](../artifacts/runs/readiness-a-only-learning-20261020/REVIEW.md).

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_continuous_robot_a_only.py tests/test_continuous_robot_learning.py tests/test_readiness_stream_input.py
.venv/Scripts/python.exe -m ruff check scripts/run_continuous_robot_learning.py tests/test_continuous_robot_a_only.py
.venv/Scripts/python.exe -m ruff format --check scripts/run_continuous_robot_learning.py tests/test_continuous_robot_a_only.py
.venv/Scripts/python.exe -m mypy --platform linux --python-version 3.12 --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL scripts/run_continuous_robot_learning.py tests/test_continuous_robot_a_only.py
```

Extend through existing recipe/admission composition and independently test actual
source exposure. Do not loosen clocks or counts, refit from B/final/probe data, add
silent updates, or infer native success from supervised error. These checks are
scoped developer evidence, not repository-wide or release readiness.

## Executed native result (2026-10-20)

The independently accepted one-use BP control trained24updates480Arows and made
50 native controls/250knowncontrol-physics steps on TRAIN17. Return/any/finalsuccess
were zero/false/false; truncation ended the run. No observed violation in51sampled
joint readings, which is not a safety guarantee. Prior ABA11violations remain.

Final A half-MSE decreased to.02004705 (priorABA.04197594); B read-only error was
.34109855 (priorABA.22805609). Exactreset actionerror improved.38053→.09339 but stored
support departure still beganstep1. This did not demonstrate base-task acquisition.
All checkpoints/sixlegacy metrics and costs/signs remain; policy/checkpoints0/8
exactly reproduce the savedcontrol.480vs496rows/24A vs16A+8B differ in information
and computation; no Bcausal or comparativebenefit claim. Original launch-wrapper
wording failure happened before any subprocess/model/native work and is preserved;
exact actual command ran once, no experimentretry.

## Executed official state42 diagnostic (2026-10-20)

The existing collector defaults to official state42 when `--visual-observation`
is omitted. Inputs must be qualified in that same mode; RGB demonstrations cannot
be converted into full-state records. The existing robot command also omits that
flag for state42 and retains the input-role, source and frozen-state guards.

Two new full-phase state42 teachers qualified on TRAIN17/19 in20/22controls.
One unchanged BP A-only phase8 learner trained24updates480Arows, then made50native
TRAIN17 controls/250knownphysics with zero return/any/finalsuccess and truncation.
Independent review accepted this negative result. A half-MSE decreased to.0486502,
yet exact-reset maximum actionerror was.3891560. Five accepted source snapshots
were unchanged, so existing focused correctness evidence was reused.

State42 changes information/capacity/compute; it neither completes the RGB goal
nor proves why RGB failed. All earlier negative results and costs remain.
Any fit-depth follow-up needs its own prospectively reviewed budget/protocol;
no additional model/native work remains under this spent diagnostic.
[Exact commands, all costs and preserved metadata-check failure](../artifacts/runs/state-readiness-acquisition-20261020/RESULTS.md)
· [Independent review](../artifacts/runs/state-readiness-acquisition-20261020/REVIEW.md).

A separate frozen192-update/3840Arow follow-up improved A half-MSE to.0120957 and
exact-reset actionerror to.0861986, but native success stayedzero;34sampledjoint6
upper violations were recorded. No further dose was selected. The explicit saved
[state42 diagnostic](learned-control-diagnosis.md#official-state42-mode) preserves
those outcomes while describing visited-state support without new model/native
work. [Larger-dose commands/costs/negative](../artifacts/runs/state42-fit-dose-20261020/RESULTS.md).
