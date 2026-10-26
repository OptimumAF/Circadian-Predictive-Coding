# Bounded coupled latent inference

Run the fixed local trace:

```powershell
python -m scripts.run_coupled_inference
```

It prints one fixed-step PC update, early-stop PC/neutral-CPC updates and guarded
fallback PC/neutral-CPC updates from exact shared tensors. This is an engineering
control with training data only. It does not measure speed or learning benefit.
R1.5's declared allocated-time comparison is now accepted in a separate
[engineering-preset study](reports/engineering-preset-comparison-20261017.md).
The original failed timing studies remain failed; general benefit is unresolved.

The retained trace has fixed PC energy `.616200` after six latent steps, early
PC/neutral-CPC energy `.643086` after two, and fallback energy `.660378` after
zero accepted steps. Early stopping gives a less-relaxed training energy here;
the lower evaluation count is not evidence of equal-quality efficiency.

## Policy and native calls

### Recurring stream integration

```powershell
python -m scripts.run_coupled_inference --stream
```

This fixed engineering demo uses the existing delayed attribution stream:
seed23, A6/B6/A6, eight rows per wake, label delay2, checkpoints0/6/12/18.
PC, neutral CPC and active CPC share original initialization and policy
`CoupledInferencePolicy(min_steps=2, gradient_tolerance=1.0)`, cap4/rate.2,
parameter rate.03. The tolerance comes from the original engineering example;
it was not selected from these final scores. BP's accepted baseline is retained
without retraining. This is not a matched-compute comparison.

Each original18update session starts before learner work and shares the global
30second deadline/RSS ceiling. Final64A/64B readouts occur after all18 wakes in
each case. Failures preserve completed update charges and partial final scoring;
unknown native attempts/costs remain null. Source learners stay unchanged. The
CLI prints incomplete results and exits1; the original no-argument trace keeps
its call shape and output. PC/neutral semantic parity is required for completion.

The report includes signed BCE acquisition/retention/return differences, whose
formulas are frozen in the [small native protocol](../artifacts/runs/r15-bounded-stream-20261017/PROTOCOL.md).
New-B gain uses B6−B12, distinct from the older shift report's B0−B12. Positive
gain is better; positive old-A loss or return gap is worse. Endpoints can show
further fitting without proving useful skill or recovery.

`build_learner(..., inference_policy=policy)` now accepts an independently owned
policy for ordinary/tracked PC and CPC. Invalid/corrupted policies, minimum>4,
and policies supplied to BP fail before native construction. `None` preserves
the old fixed loop and constructor arguments. Generic training receipts expose
energy, not inference counters: the demo reports those costs as unavailable.
The saved native qualification separately instruments returned diagnostics,
candidate checks, norms/comparisons, predictions and whole observed overhead.

Why this: reuse the existing runner, factory and numerical kernel for a small
runnable integration. The separate matched-time runner already exists; both
its10ms and30ms qualifications failed, with adaptive updates reaching cap6.
This demo neither repairs those failures nor closes R1.5b or establishes savings.

```python
from src.core.coupled_inference import CoupledInferencePolicy

policy = CoupledInferencePolicy(
    min_steps=2,
    gradient_tolerance=1e-5,
    relative_energy_tolerance=None,
)
result = model.train_epoch(
    X_train, y_train, learning_rate=0.01,
    inference_steps=6, inference_learning_rate=0.1,
    inference_policy=policy,
)
print(result.inference)
```

Both coupled PC and CPC accept the keyword-only, per-call policy. Without it,
`inference_steps` remains the exact fixed count and `result.inference` is `None`.
With it, that argument is the hard proposal cap. Model initialization, objective,
precisions, parameter updates and target-free prediction are unchanged.

`CoupledInferencePolicy` is frozen and validates native integer `min_steps` in
1–10000 and optional native finite nonnegative tolerances. `None` disables a
criterion; both `None` means guarded cap-only inference. The call requires a
native integer cap in 1–10000, minimum no greater than cap, and positive finite
native inference rate. Booleans and NumPy scalar controls are refused. Policy
validation runs again on use. All arrays retain existing float64/shape/finite
rules and binary training targets must lie in `[0,1]`.

## Exact stopping and fallback behavior

The pure `infer_coupled` kernel holds parameters, input, training targets and
precisions fixed. It initializes owned hidden latents from target-free
feedforward values and evaluates the existing coupled energy. Every proposal
simultaneously updates all latents from the same accepted evaluation's partials.

1. Reject nonfinite latent proposals before evaluation.
2. Evaluate a finite candidate. A numerical evaluation failure or **any** finite
   energy increase rejects that candidate. There is no line search or retry.
3. Accept a finite candidate with nonincreasing energy.
4. After at least `min_steps` accepted updates, check gradient infinity norm
   first, then relative energy drop. Reaching the cap gives `max_steps`.

The residual is the maximum absolute latent partial across every hidden layer,
including the energy kernel's `1/B` batch normalization. It is a stationarity
residual, not the hidden prediction-error norm. Relative energy drop is
`(old_energy-new_energy)/max(1,abs(old_energy))`. Thresholds use `<=`; if both
criteria hold, the reason is `gradient_tolerance`. Equal finite energies are
accepted, so exact or rounded plateaus can trigger the energy criterion.

On an unsafe proposal, return the last accepted latent/evaluation pair with
`used_fallback=True` and reason `nonfinite_candidate`, `numerical_failure` or
`energy_increase`. Safety overrides the minimum: zero accepted steps is possible.
An invalid or numerically invalid **initial** problem has no valid fallback and
raises. Schema/programmer `ValueError` is not converted to a numerical fallback.
The inputs are borrowed and must be quiescent; output arrays are owned. This is
a synchronous numerical boundary, not concurrent or untrusted execution.

Why this: a bounded proposal that worsens the represented energy should not
replace a valid state. Returning the last accepted state supplies an explicit,
budgeted fallback without hidden extra iterations or learning-rate tuning.
Strict rejection can also respond to floating-point rounding; it is not a proof
of convergence or a guarantee that the minimum is always reached.

Native training uses the returned state's gradients for the ordinary atomic
parameter step. CPC chemistry/gates use that same returned state, including a
feedforward fallback. A valid fallback can therefore still produce a parameter
update. Candidate parameter energy/finite checks and CPC chemistry checks must
all pass before publication; failure leaves all committed model fields intact.
An energy-nonincreasing latent prefix does not guarantee parameter-step descent.

## Work diagnostics and state boundaries

`result.inference` is an immutable scalar diagnostic with:

- `attempted_steps`, `accepted_steps`, `reason`, `used_fallback`;
- `energy_evaluations`: initial plus every attempted candidate evaluation,
  including a candidate evaluation that fails;
- `gradient_norm_evaluations`: initial norm plus one for every accepted step;
- `energy_comparisons`: one proposed-versus-current monotonicity comparison per
  successfully evaluated candidate, including a rejected energy increase;
- initial/final energy and final latent gradient infinity norm.

Gradient/relative-energy threshold comparisons and relative-drop arithmetic are
not separately counted by `energy_comparisons`. These counters identify kernel,
norm and monotonicity work; they do not enumerate every scalar stopping operation.

Stopping reuses the accepted evaluation; it adds norm reductions/comparisons,
not an extra gradient kernel solely for stopping. Parameter candidate validation
adds one more energy/gradient evaluation **outside** these inference counts.
The fixed path performs `inference_steps+1` inference evaluations and one parameter
candidate evaluation. Counters are neither FLOPs nor measured time savings.
Any future comparison must include these reductions, comparisons, initialization,
candidate validation and other orchestration overhead.

The native policy is per training call, so native snapshot format/state is
unchanged. `CoupledLearner(..., inference_policy=policy)` owns a copied fixed
adapter policy and copies it again when forking. Restoring a native snapshot
does not replace adapter settings, budgets or authority. The established
`TrainingDiagnostic` port still returns only its native energy; detailed
inference work is available from direct native training results. No durable
decoder, promotion criterion or new runtime resource guarantee is introduced.

Only currently available training-role labels may enter inference/stopping. Final
evaluation data must remain outside it. Inference is not performed at prediction
time by the existing target-free classifier.

## Modules and checks

```text
src/core/coupled_inference.py                 policy, guarded pure inference, counters
src/core/coupled_predictive_coding.py         optional per-call PC integration
src/core/coupled_circadian_predictive_coding.py shared CPC inference and atomic wake
src/adapters/coupled_learners.py              owned fixed adapter policy and forks
scripts/run_coupled_inference.py              fixed five-update engineering trace
tests/test_coupled_inference.py               independent exact stopping/fallback controls
tests/test_coupled_inference_integration.py   native parity, chemistry, ownership and failures
tests/test_coupled_inference_cli.py           actual CLI and scalar work records
```

```powershell
python -m pytest -q tests/test_coupled_inference.py tests/test_coupled_inference_integration.py tests/test_coupled_inference_cli.py
python -m ruff check src/core/coupled_inference.py src/core/coupled_predictive_coding.py src/core/coupled_circadian_predictive_coding.py src/adapters/coupled_learners.py scripts/run_coupled_inference.py tests/test_coupled_inference.py tests/test_coupled_inference_integration.py tests/test_coupled_inference_cli.py
python -m ruff format --check src/core/coupled_inference.py src/core/coupled_predictive_coding.py src/core/coupled_circadian_predictive_coding.py src/adapters/coupled_learners.py scripts/run_coupled_inference.py tests/test_coupled_inference.py tests/test_coupled_inference_integration.py tests/test_coupled_inference_cli.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/core/coupled_inference.py src/core/coupled_predictive_coding.py src/core/coupled_circadian_predictive_coding.py src/adapters/coupled_learners.py scripts/run_coupled_inference.py tests/test_coupled_inference.py tests/test_coupled_inference_integration.py tests/test_coupled_inference_cli.py
```

The CLI sets OMP/MKL/OpenBLAS threads to one before NumPy imports. No dependency
or persisted environment configuration is added. The separate accepted comparison
retains all seeds, original failures and stopping/instrumentation overhead.
Its quality interval crosses0; no large sweep or threshold tuning on final
scores is authorized by these engineering checks or saved comparison.
