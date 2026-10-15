# A genuine neutral control for coupled CPC (R1.3)

`CoupledCircadianPredictiveCodingNetwork` adds the existing basic circadian
activity-trace gate to the opt-in [coupled PC learner](coupled-predictive-coding.md).
It shares all-latent inference, energy partials, initialization and candidate
validation with that learner. The legacy PC/CPC implementations are separate.

## Actual wake behavior

For the final hidden layer only, compute prospective chemistry from the final
inferred latent and then the plasticity gate:

```text
c_new = chemical_decay * c_old
        + chemical_buildup_rate * mean_rows(abs(z_last))
g     = clip(exp(-plasticity_sensitivity * c_new), min_plasticity, 1)
```

Chemistry starts at zero and remains a finite nonnegative vector of final hidden
width. Default basic settings are decay `0.995`, buildup `0.02`, sensitivity
`0.7`, and floor `0.2`. They preserve the basic legacy formula's interpretation,
not a claim of complete legacy-algorithm equivalence. A dedicated frozen
`CoupledCircadianConfig` contains exactly these supported settings; it does not
silently accept the legacy configuration's unsupported feature flags.

Use `g` to multiply columns of the last hidden weight gradient, the corresponding
hidden bias gradient, and rows of the output weight gradient. Earlier hidden
gradients and the output bias gradient are unchanged. The shared energy and
latent inference are unchanged. Gated weight updates are diagonal preconditioned
steps using these partials, not a claim that the chemistry is differentiated as
part of the static energy.

The new trace, gates, complete candidate parameters and candidate coupled energy
must all validate before parameters, chemistry or the completed-wake count change.
Errors retain all prior model state. This is synchronous single-owner failure
atomicity, not a thread-safe serving or hard resource-allocation guarantee.

The exponential may safely saturate to zero for a large negative exponent before
clipping to the positive floor; a nonfinite chemistry vector itself is refused.
Configuration accepts native finite numbers, excludes booleans, and bounds decay
to `[0,1]`, buildup and sensitivity to `[0,1e6]`, and floor to `[1e-6,1]`.

## Neutral preset

```python
import numpy as np
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig, CoupledCircadianPredictiveCodingNetwork,
)

x = np.array([[0.3, -0.2], [-0.5, 0.4]], dtype=np.float64)
y = np.array([[1.0], [0.0]], dtype=np.float64)
pc = CoupledPredictiveCodingNetwork(2, (3, 2), seed=23)
cpc = CoupledCircadianPredictiveCodingNetwork(
    2, (3, 2), seed=23, config=CoupledCircadianConfig.matched_pc_control(),
)
pc_result = pc.train_epoch(x, y, 0.03, 3, 0.1)
cpc_result = cpc.train_epoch(x, y, 0.03, 3, 0.1)
assert pc_result == cpc_result
np.testing.assert_array_equal(pc.predict_proba(x), cpc.predict_proba(x))
assert np.any(cpc.get_chemical_state() > 0)
np.testing.assert_array_equal(cpc.get_plasticity_state(), np.ones(2))
```

`matched_pc_control()` sets the plasticity floor to exactly one. Chemistry still
updates through the real wake path, but every gate is one. This makes initialized
parameters, repeated wake parameters, energy values/identifiers and predictions
match the ordinary coupled PC reference exactly under matched seeds, topology,
precisions, batches and rates. Internal chemistry/counters are extra CPC state,
so the complete model objects are not asserted identical.

The independent test also uses an active gate and verifies the intended update
scaling; a no-op alias would fail this control. This is a functional control,
not evidence that nonneutral gates improve learning. Extra chemistry computation
is not automatically compute-matched; studies must account for its overhead.

## State and supported scope

`get_chemical_state()` and `get_plasticity_state()` return detached vectors;
`wake_updates` counts completed wakes. Inherited parameter snapshots are detached.
`restore_parameters` restores parameters only and preserves chemistry/counters;
it is not a complete CPC checkpoint or a runtime/budget reset.
For complete compatible in-memory continuation, use the versioned
`snapshot_state()`/`restore_state()` ports described in the
[coupled runtime adapter guide](coupled-learner-ports.md). They include chemistry,
gates and wake counts while leaving runtime budgets and permissions external.

This implementation supports coupled binary **wake training only**. Replay,
sleep, structural adaptation, reward modulation, dual chemistry and durable
restore are not implemented. Owned native runtime adapters are now available
through the guide above. There are no placeholder sleep
operations or aliases presented as complete algorithms. Existing shallow legacy
neutrality remains the separate ADR-0023 scope.

## Structure and checks

```text
src/core/coupled_predictive_coding.py          shared inference/validated update
src/core/coupled_circadian_predictive_coding.py basic gate and neutral preset
tests/test_coupled_neutral_control.py          independent 1/2/4-layer parity
tests/test_coupled_circadian_state.py          chemistry/config/failure behavior
```

Run the affected math and learner checks:

```powershell
python -m pytest -q tests/test_coupled_energy.py tests/test_coupled_predictive_coding.py tests/test_coupled_neutral_control.py tests/test_coupled_circadian_state.py
```

Tests use fixed synthetic training batches and declared seeds. No held-out final
data, scoring, hyperparameter selection, comparative experiment or sweep is
needed to establish this software parity. R1.1 reproduction, R1.4 scaling, R1.6
structural changes and R1.7 comparative evaluation retain their separate gates.
