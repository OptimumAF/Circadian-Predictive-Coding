# Public21 mixed-gripper control at matched exposure

## Accepted observed outcome, 2026-10-19

The single arm completed within budget. Final physical half-MSE is
A.026426856148490524/B.04180924567280607, improving by
.0007021232858169212/.003861306440065347 versus the read-only MSE arm.
Four of six metric changes improve, while new acquisition and warm retention
worsen. A dx/dz and B dy translation errors worsen, and maximum gripper errors
still exceed1. One B-phase pre-update loss increase remains; all losses retained.

58cases pass on each platform, scoped statics and independent actual acceptance
pass. Full final96/2816 parameters/momentum are saved. This is a modest offline
fit result with harms; native utility remains untested. Next is explicit mixed
policy restoration and a separately frozen zero-SGD native17/19 check, using
existing readers/readouts/rollouts and unchanged MSE endpoints.

The existing continuous MLP, public-vector projection and stream runner already
support the binary-gripper objective together. This case composes those APIs to
test one fixed objective hypothesis after the MSE and corrective native failures.
It introduces no new learner, objective, fitter or runner.

## Fixed comparison

One fresh A32/B32/A32 arm uses public21 inputs, seed17, hidden64, learning rate.01,
momentum.9 and the accepted A-only statistics. It changes only the existing
`binary_gripper_axis=3` selection. The source packets,96 updates,2,816 training
row exposures, four checkpoints and six physical acquisition/retention/recovery
metrics match the saved p32 MSE arm. That MSE evidence is read-only.

Why this: lower mean MSE and added on-policy provisional corrections did not
produce native success, while large gripper errors remain. Testing the existing
alternative objective is an unresolved baseline question. This does not establish
that the gripper objective, observability or correction quality caused the prior
failures. All earlier state/RGB mixed-loss and native negative results remain.

Primary outcomes are mixed-minus-MSE final physical-action half-MSE for TRAIN A
and B separately; negative is better. All four endpoints, five fields for each
of four axes and all six metric differences remain visible, including harms.
The unlike MSE and mixed training objective values are retained with their own
definitions and are never ranked numerically against each other.

This is matched data and training exposure. Dense MACs are equal; added binary
cross-entropy elementwise work means computation and latency are not identical.
These are known TRAIN contexts, not new tasks, heldout generalization or a CPC
comparison. No native case is admitted by this offline protocol.

## Structure and guards

```text
artifacts/runs/public-vector-mixed-control-20261019/
  run_case.py       existing APIs, explicit comparison and bounded state archive
  run_check.py      existing serial command recorder
  PROTOCOL.md       frozen sources, endpoints, budgets and gates
  REVIEW.md         independent source, gates and actual readback
  RESULTS.md        exact commands, outcomes, costs, failures and next action
  case.json         all new paid work and unchanged MSE comparator
  final-state.npz   eight final parameter/momentum arrays,96/2816 counters in report
tests/test_public_vector_mixed_control.py  synthetic zero-training checks
```

The case reuses qualified TRAIN readers, complete raw789 validation/public21
projection and accepted p8/p32 admission. Before the new fit or training it checks
the saved MSE policy, statistics, progress, physical measurements and work. Two
fresh zero-update constructor templates establish current numeric initialization
parity across all eight arrays. They are not an archived historical initial
snapshot; source constructor identity and initial physical reports provide the
separate historical consistency evidence.

The existing runner performs one A-only fit and requires exact accepted statistics
before construction/SGD. It preserves source and stable actor, checks preceding
success and original charges before more work, and retains all96 native receipts.
The new case preserves the returned report before comparison or archive checks.
Mixed receipt/event validation binds the correct objective explicitly; the old
MSE validator is unchanged. The common physical initial summaries and frozen actor
outputs must agree within absolute1e-12/relative0, using already returned readouts.
Conditioning objective values and gradients intentionally differ.

The final typed snapshot checks policy, all parameter/momentum arrays and96/2816
counters before one exclusive NPZ write. No serving publication occurs. The
comparison helper's derived endpoint tag is explicitly96; observed source and
baseline counters are never rewritten.

## Commands and limits

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m pytest -q tests/test_public_vector_mixed_control.py
.\.venv\Scripts\python.exe -X utf8 -B -m ruff check artifacts/runs/public-vector-mixed-control-20261019/run_case.py artifacts/runs/public-vector-mixed-control-20261019/run_check.py tests/test_public_vector_mixed_control.py
.\.venv\Scripts\python.exe -X utf8 -B -m ruff format --check artifacts/runs/public-vector-mixed-control-20261019/run_case.py artifacts/runs/public-vector-mixed-control-20261019/run_check.py tests/test_public_vector_mixed_control.py
.\.venv\Scripts\python.exe -X utf8 -B -m mypy artifacts/runs/public-vector-mixed-control-20261019/run_case.py
```

The focused checks use manual tiny reports and snapshots, with no fitting, SGD or
forward passes. Accepted mathematical, count, projection and old baseline tests
are reused. Exact Windows/Linux cases, static work and independent acceptance are
recorded in [RESULTS.md](../artifacts/runs/public-vector-mixed-control-20261019/RESULTS.md).

Only after the prospective gates pass, one actual module call is admitted:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m artifacts.runs.public-vector-mixed-control-20261019.run_case
```

This performs96 new updates, not a continuation of the historical96. It is a
single30-second local allowance with one A fit,2 parity templates, CPU1/GPU0,
case<=512KiB, final archive<=128KiB and log<=8MiB. There is no retry, alternate
seed/checkpoint, sweep, native simulation, new teacher label, replay, guard/probe/
final access, baseline retraining or promotion.

Work is11,680,512 dense MACs, plus3,052 binary-gripper loss rows from2,816 training
and236 conditioning rows. Loss rows are not a complete elementwise operation
count. Projection, normalization, diagnostics, gradient payloads, fit, constructor
templates and final archive costs remain separate. Numeric payload/dense counts
exclude copies, validation, temporary snapshots and process overhead; no process
memory bound is claimed. A negative or mixed result is valid evidence.

Future native utility must be tested separately from offline fit using the saved
state and existing frozen rollout. Do not infer success from a favorable loss or
add another count, capture, provenance or audit increment automatically.
