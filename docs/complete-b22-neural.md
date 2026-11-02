# Complete220-row neural TRAIN coverage

The conventional mixed-SGD continuation now explicitly supports the same set of
220TRAIN examples used by the working fixed-input memory control. It adds the
complete original22-row B19 demonstration to the six existing198-row groups,
preserving their separate episode identities. One actual cold H64 fit has now
completed64updates/14080row exposures; TRAIN error decreased in all seven groups.
Push17 completed50controls with zero success. A subsequent inference-only
continuation completed Pick17/19/20, each with zero success over50controls.
No training or Push repeat occurred; the original failed observer capture remains.
[Executed study, retained failure and limits](reports/complete-b22-neural-study-20261101.md).
[Complete native observations and continuation costs](reports/complete-b22-pick-continuation-20261101.md).

```python
import numpy as np
from scripts.push_cube_correction_weights import (
    complete_b22_training_groups, build_complete_b22_correction_weights,
)
from scripts.push_cube_training_readouts import run_training_readout
from src.adapters.continuous_train_continuation import run_continuous_train_continuation

# Inputs must already have qualified source/release/receipt/role/budget authority.
# original198 is the existing six-group tuple; b22 is original TRAIN19 TS12.
groups = complete_b22_training_groups(original198, b22)
raw_weights, recipe = build_complete_b22_correction_weights(groups)
result = run_continuous_train_continuation(
    admitted_source, groups[0][1], references=tuple(e for _, e in groups[1:]),
    updates=declared_updates, update_rule="bp_mixed",
    row_weights=np.asarray(raw_weights, dtype=np.float64), maximum_rows=220,
)
# Only a complete result with a candidate can proceed to separate readout gates.
# run_training_readout(candidate, groups, label=..., guard=original_owner_guard,
#                      corrective=True, aggregated=True, complete_b22=True)
```

Neither snippet authorizes data reads, computation or a native rollout. #399's
original study allowance is spent. Its fit/Push observations are partial evidence:
the inner owner's time/RSS fields were null despite the protocol's assertion;
measured outer fit gates passed. Qualification remains open. Any continuation
needs explicit prospective limits; preserve the original failed protocol.
Do not rerun the spent198/affine/memory controls or reset their budgets.

| Neural group order | Rows | Weight rule |
| --- | ---: | --- |
| current_corrections |50|49 then49ones, raw mass98|
| corrections |50|same, preserving original values|
| push |18|uniform raw mass98|
| a |15|uniform raw mass98|
| b (derived B50) |50|uniform raw mass98|
| train20 |15|uniform raw mass98|
| b22 (original B19) |22|uniform raw mass98|

All198raw weights remain an exact prefix. Adding the seventh group changes each
ideal normalized group mass from1/6 to1/7, each correction's initial row to1/14.
Why this: a complete teacher group uses the same existing mass98, with no new
priority tuned from outcomes. Every row remains positive and present. The memory
control uses unweighted nearest retrieval and its own append order
`a,b,train20,push,corrections,current_corrections,b22`. The neural order deliberately
preserves its old198prefix. Same coverage is not matched order, objective,
exposure or all-in computation; memory's earliest-row tie policy stays unchanged.

The B22 group requires the original `pick-cube-readiness-teacher-19` identity,
`pick-cube-train-19` context, exact TRAIN role,22rows and the original four-axis
TS12 spec. It remains distinct from derivedB50. The helper re-owns data but does
not authenticate external files or create release/consent/clock authority. The
existing `diagnose_pick19_memory.read_pick19_inputs` qualification reader supplies
the accepted198bank and original teacher binding, including source hashes, native
receipt checks and physical history projection. It orders the bank differently;
any future caller must explicitly map original named groups without changing rows.
No source files were read by that reader during this software increment.

Only explicit `maximum_rows=220` with mixed SGD admits the new total. Mixed default
148, explicit198, head-GN130,64-update mixed ceiling,8-update head ceiling and all
128-row frozen-hidden/categorical contracts remain intact. State/weight/scope
guards survive every private facade fork. Actor/source/optimizer ownership,
original budgets/clocks and paid/lost-return behavior use the existing runtime.
The readout needs all three flags; older call shapes and bounds are unchanged.

An additional default-off `command_fidelity=True` option reuses the existing
TRAIN residual predictions to describe native32 command errors, translation
directions and analog gripper endpoint-sign margins. It adds no model pass and
keeps primary MSE/native-success definitions unchanged.
[Contract and cost accounting](train-command-fidelity.md).
The [inference-only command diagnosis](reports/train-command-diagnosis-20261101.md)
returned all7TRAIN groups with unchanged published residuals and remaining action
disagreements. Its outer phase capture failed; final file qualification is
incomplete and the allowance spent. This does not establish a causal explanation
or repaired native control, and no model retry was performed.

Focused verification, failure and review scope:
[complete220software evidence](reports/complete-b22-neural-software-20261101.md).
The full H64seven-group readout also passes on literal model state within the
existing64KiB phase-payload gate; that software increment used no actual dataset.
The later actual study is linked above. Existing memory220success and198neural/
affine failures remain valid. Lower TRAIN error establishes neither joint native
success nor retention, recovery, CPC benefit, generalization or repository readiness.
