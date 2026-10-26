# Bounded continuous PC latent inference

`src/core/continuous_pc_inference.py` infers target-conditioned hidden free states
with the accepted continuous energy. It leaves all parameters unchanged. It does
not train or publish a policy, select data, authorize roles, or access a simulator.
Use only labels released for the intended training operation. Final evaluation
must remain unavailable to training, promotion, tuning and probe selection.

## API

```python
result = infer_continuous_pc(
    parameters, features, targets, precisions, spec,
    max_steps=4, learning_rate=0.05,
    policy=CoupledInferencePolicy(gradient_tolerance=None),
)
```

Inputs follow the [continuous energy contract](continuous-pc-energy.md), with
native integer `max_steps` in `1..10000`, native finite positive learning rate,
and the existing exact `CoupledInferencePolicy`. Minimum steps must not exceed
the cap. Policy, parameters, features, targets, precisions and action geometry
are owned for the call. Callers must supply quiescent inputs while they are
copied. Targets/precision/settings are validated before initialization.

The frozen `ContinuousPCInferenceResult` contains owned mutable `latents`, their
matching `ContinuousPCEvaluation`, and `CoupledInferenceDiagnostics`. It retains
no trajectory or optimizer state. Results may stop before the requested minimum
when safety requires fallback. Invalid initial inputs/arithmetic raise because
there is no valid state to return; programmer errors also propagate.

## Dynamics and stopping

One target-free feedforward initialization precedes the initial energy and latent
gradient norm. Each attempt proposes all layers simultaneously from the previous
accepted state: `Z_l - rate * dE/dZ_l`. It does not change parameters or clip
latents, retry proposals, backtrack, or tune the rate.

- A nonfinite candidate returns the last accepted state before another energy
  evaluation.
- Candidate arithmetic failure/nonfinite energy returns that same valid state.
- A finite energy increase is refused after comparison. Equality may be accepted.
- After acceptance and the minimum steps, gradient tolerance takes precedence
  over relative energy tolerance. Relative drop is
  `(old_energy - new_energy) / max(1, abs(old_energy))`.
- Disabled tolerances leave the cap as the ordinary stopping condition. Even an
  initially zero gradient performs the minimum accepted work before stopping.

Why this: both binary and continuous likelihoods require identical guarded
descent/accounting. The existing binary loop was extracted into the private
`coupled_inference._run_inference`; the public binary entrypoint, validation,
ordering, math and dynamic test hooks remain compatible. No second stream runner
or public callback mechanism is introduced.

## Cost and use boundaries

Diagnostics count attempted/accepted proposals, actual energy entries (including
failed candidates), gradient norm evaluations, and finite energy comparisons.
Norm evaluations are `1 + accepted_steps`; rejected candidates are still paid
work. The final accepted evaluation is reused without another energy call.

Counters do not measure initialization, validation, array copying, wall time or
memory. The wrapper copies the problem once and public energy calls copy their
inputs again. Every energy call computes parameter partials as well as latent
partials. Storage is bounded by model/problem dimensions and a constant number
of states/evaluations, without a retained chain. These are not measurements of
whole-process peak memory or deep-network scaling.

`result.evaluation.predictions` are **target-conditioned**. They are not deployable
predictions for an unlabeled task. The existing `predict_continuous_pc` remains
target-free and sees unchanged parameters. A bounded learner update and an owned
adapter through existing stream interfaces are still needed before this can
learn. Neutral CPC equivalence, retained skills, returned-context recovery and
comparative benefit remain separate scientific requirements.

## Example and focused checks

```powershell
python -m scripts.continuous_pc_inference_example
python -m pytest tests/test_continuous_pc_inference.py tests/test_coupled_inference.py tests/test_coupled_inference_integration.py
python -m ruff check src/core/coupled_inference.py src/core/continuous_pc_inference.py tests/test_continuous_pc_inference.py scripts/continuous_pc_inference_example.py
python -m ruff format --check src/core/coupled_inference.py src/core/continuous_pc_inference.py tests/test_continuous_pc_inference.py scripts/continuous_pc_inference_example.py
python -m mypy --follow-imports=silent src/core/coupled_inference.py src/core/continuous_pc_inference.py tests/test_continuous_pc_inference.py scripts/continuous_pc_inference_example.py
```

The fixed two-row/four-action example separates inferred and target-free actions
and reports all stopping counters. It performs no random draws, model creation,
parameter learning, native work or experiment selection. Its energy-decrease
gate demonstrates only this declared latent calculation.

[Protocol](../artifacts/runs/continuous-pc-inference-20261020/PROTOCOL.md) ·
[Results and exact next action](../artifacts/runs/continuous-pc-inference-20261020/RESULTS.md) ·
[Independent review](../artifacts/runs/continuous-pc-inference-20261020/REVIEW.md).
