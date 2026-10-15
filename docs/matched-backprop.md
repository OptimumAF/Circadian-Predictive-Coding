# Matched feedforward backprop baseline

Run the same four delayed-label training arrivals with BP, coupled PC and its
neutral CPC control:

```powershell
python -m scripts.run_native_stream_demo --scenario matched-baselines --seed 23
```

The JSON records exact shared initial weights/biases, topology `(3,2)`, per-method
predictions/diagnostics, four admitted updates per learner, stable serving actors
and latent step counts. This is an integration example. The probes include
unlabeled inputs before their training arrival, so it is not a strict online
evaluation protocol or a comparative study.

## What is matched

`MatchedBackpropNetwork` takes explicit `CoupledParameters`; construction copies
and validates every array and makes no RNG draws. Initialize a single coupled
model and pass its snapshot to BP and to CPC's `restore_parameters`. Matching is
by exact tensors, rather than an assumption that seeds mean the same thing in
different initializers. Predictions use the same target-free feedforward kernel.

```python
import numpy as np
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.matched_backprop import MatchedBackpropNetwork
from src.adapters.matched_backprop_learner import MatchedBackpropLearner

pc = CoupledPredictiveCodingNetwork(2, (3, 2), seed=23)
bp = MatchedBackpropNetwork(pc.snapshot_parameters())
x = np.array([[0.3, -0.2], [-0.5, 0.4]], dtype=np.float64)
y = np.array([[1.0], [0.0]], dtype=np.float64)
np.testing.assert_array_equal(bp.predict_proba(x), pc.predict_proba(x))
learner = MatchedBackpropLearner(bp, learning_rate=0.03)
saved = learner.snapshot_state()
diagnostic = learner.train_batch(x, y)
assert diagnostic.definition == "numpy_matched_binary_bce_preupdate_v1"
learner.restore_state(saved)
```

Both models use tanh hidden predictions, a linear scalar logit, soft binary
targets in `[0,1]`, batch mean normalization and the same stable, unclipped
sigmoid/logit likelihood. All inputs and parameters are exact nonempty finite
rank-2 float64 NumPy arrays. Hidden layers may have different widths.

Why this: the legacy `BackpropMLP` uses its historical initialization and clipped
sigmoid/BCE diagnostics. A separate opt-in baseline preserves that behavior while
making new objective and initialization comparisons explicit.

## Derivatives and diagnostic meaning

With feedforward `h_0=X`, `h_l=tanh(h_(l-1)W_l+b_l)`, and
`o=h_L V+c`, define:

```text
loss = mean(logaddexp(0,o) - y*o)
q = (sigmoid(o)-y)/B
d_L = (q V.T) * (1-h_L^2)
d_l = (d_(l+1) W_(l+1).T) * (1-h_l^2)

dW_l = h_(l-1).T d_l           db_l = sum_rows(d_l)
dV = h_L.T q                   dc = sum_rows(q)
```

`evaluate_matched_backprop` returns the loss and full feedforward parameter
gradient. `train_epoch` returns that **preupdate** evaluation, then performs one
simultaneous gradient step. The diagnostic ID is
`numpy_matched_binary_bce_preupdate_v1`.

The feedforward likelihood matches the task term in the coupled objective when
its latents are initialized feedforward. PC subsequently relaxes hidden latents
and differentiates a total energy containing residual penalties. BP instead
differentiates through the entire feedforward graph. The complete training
objectives and updates therefore differ; PC/CPC energy diagnostics must not be
treated as BP losses or used to rank the methods.

Each CLI learner uses rate `0.03` and the same four synthetic training rows and
arrivals. PC/CPC perform two latent steps at rate `0.2`; BP performs none. Both
implementations also evaluate candidate validity before publishing updates.
Latent step counts are not total FLOP or wall-time measurements, and this is not
an equal-compute comparison. No metric, seed or baseline is selected to favor CPC.

## State and runtime ownership

The native model owns only fixed-topology parameters. There is no continuing RNG,
optimizer momentum, traffic counter, precision or chemical state. Format-1
`MatchedBackpropSnapshot` contains detached parameters. Complete restore requires
the exact snapshot family, native integer version 1 and compatible topology.
Parameter restore is an explicit separate API for tensor transfer between models.
Neither operation grants runtime clock, budget, receipt or serving authority.

Training validates the entire prospective parameter bundle and its feedforward
evaluation before publishing. Failed input/rate/candidate checks leave model
state unchanged. Invalid schema/domain inputs raise `ValueError`; nonfinite
arithmetic raises `FloatingPointError`. Extremely large finite rates need not
produce a valid or descending update. These are synchronous single-owner
guarantees, not thread safety or an untrusted file decoder.

`MatchedBackpropLearner` owns its native model and fixed learning rate; forks own
independent tensors and preserve that rate. The existing actor/shadow runtime
enforces delayed labels, final-role exclusion, budgets and stable serving state.
No runtime port signature or legacy model contract changes.

## Structure and verification

```text
src/core/matched_backprop.py                 full feedforward derivatives/model
src/adapters/matched_backprop_learner.py      owned learner/fork ports
src/adapters/native_stream_demo.py           existing runtime composition
scripts/run_native_stream_demo.py            scenario registration
tests/test_matched_backprop.py               independent Torch/state oracles
tests/test_matched_backprop_ports.py         native parity, actor/arrival/CLI
```

Independent tests use actual CPU Torch autograd and simultaneous GD at 1/2/4
hidden layers, including nonzero biases, soft labels and saturated logits. Tests
check exact initial tensors and predictions, repeated updates, snapshot/restore
continuation, normalization, detached arrays, schema refusal and atomic refusal
when finite candidate tensors produce nonfinite logits.

```powershell
python -m pytest -q tests/test_matched_backprop.py tests/test_matched_backprop_ports.py
python -m ruff check src/core/matched_backprop.py src/adapters/matched_backprop_learner.py src/adapters/native_stream_demo.py scripts/run_native_stream_demo.py tests/test_matched_backprop.py tests/test_matched_backprop_ports.py
python -m ruff format --check src/core/matched_backprop.py src/adapters/matched_backprop_learner.py src/adapters/native_stream_demo.py scripts/run_native_stream_demo.py tests/test_matched_backprop.py tests/test_matched_backprop_ports.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/core/matched_backprop.py src/adapters/matched_backprop_learner.py src/adapters/native_stream_demo.py scripts/run_native_stream_demo.py tests/test_matched_backprop.py tests/test_matched_backprop_ports.py
```

No new dependency or environment variable is required. The existing Torch test
installation supplies the independent oracle. The separate
[R1.7 pilot report](reports/coupled-attribution-20261014.md) now supplies a small
stationary/recurring comparison with isolated scoring, intervals and explicit
work differences. It does not establish equal compute or general benefit.
Follow-up comparisons require a new protocol; preserve the negative outcomes.
