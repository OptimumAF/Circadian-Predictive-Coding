# Class-weighted BP arrays

`src.adapters.torch_array_weighted.train_class_weighted_batch` applies one explicit
class-weighted cross-entropy/SGD update to an owned generic `TorchFeatureLearner`
BP candidate. It uses the existing CPU Torch environment and native policy,
optimizer staging and complete state validation.

```python
from src.adapters.torch_array_weighted import train_class_weighted_batch

# Caller has admitted and released this TRAIN batch, owns candidate and budget,
# and has declared a finite positive weight for each catalog class.
diagnostic = train_class_weighted_batch(
    candidate, arrived_train.features, arrived_train.targets, class_weights
)
```

Arrays are exact float64, features1..128rows at the policy's bound width, targets
Bx1 finite hard class indices. Weights are an exact K-tuple accepted by existing
`validate_class_weights` (finite positive native numbers1e-6..1e6). The caller
owns TRAIN roles, label release, weight derivation, original budget and attempted
work charges. Arrays alone confer no data-role or continuation authority. Derive
any future class-balancing policy only from currently arrived TRAIN labels.

The objective is `sum(weight[target] * CE) / sum(weight[target])`. It normalizes by
observed target mass, including repeated labels. All-one weights use the original
unweighted kernel and preserve exact parameter/gradient/momentum continuation.
The returned diagnostic has its own definition:
`torch_feature_array_class_weighted_cross_entropy_v1`. A weighted loss should not
be ranked against ordinary CE or PC energy as though their scales were equal.

The helper stages one independent native head/optimizer, validates the complete
candidate, proves the donor and inputs still match, then assigns one payload.
Policy, payload, optimizer/state/group/parameter-list/Parameter/gradient identities
and full values are checked, along with Torch/NumPy/Python RNG and grad/inference
mode. Initial optimizer parameters must be exactly bound to the native head.
Failed candidate work is not committed. Foreign changes are refused and remain
visible; the helper never rolls them back. Caller charges attempts even when no
diagnostic returns; a failed return is not a committed update or free work.

Only native BP is supported. PC/CPC and invalid input are refused before candidate
construction. The original generic `train_batch` and raw296 invariant-game
weighted episode port are unchanged. This separate adapter keeps their boundaries
and avoids an import back into the native learner. No new policy/snapshot format,
loader, scheduler, replay, final/probe access or automatic selection is added.

[Executed correctness evidence](reports/torch-array-weighted-20261029.md) includes
exact all-one continuation, an independent weighted-loss/gradient/momentum
reference, corruption/failure cases and two delayed TRAIN updates through the
existing stream with source/actor unchanged. It establishes operator correctness,
not improved task acquisition, retention, recovery or a circadian advantage.
The recurring ticket command does not yet expose class balancing.

Developer checks (native fixtures incur real optimizer work):

```powershell
$env:TEMP = 'D:\Dev\_Temp'
$env:TMP = 'D:\Dev\_Temp'
.venv/Scripts/python.exe -B -m pytest -q tests/test_torch_array_weighted.py
.venv/Scripts/python.exe -B -m ruff check src/adapters/torch_array_weighted.py tests/test_torch_array_weighted.py
.venv/Scripts/python.exe -B -m ruff format --check src/adapters/torch_array_weighted.py tests/test_torch_array_weighted.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent src/adapters/torch_array_weighted.py tests/test_torch_array_weighted.py
```

Use physical D working directories and Linux `--capture=sys` where D-backed
file-descriptor capture is incompatible. Extend through the existing delayed
TRAIN runner and record objective/weights/exposures explicitly before any new
development qualification; preserve all earlier negative outcomes.
