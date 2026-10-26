# Held-out D readout from the frozen ordinary affine endpoint

The original native game evaluator now has an explicit final-only affine route:
`evaluate_heldout_context_linear_checkpoint(source, state, *, budget, check_boundary)`.
It restores the full saved ordinary10-fit/252-row endpoint into a detached scorer,
then plays Dblue followed by Dred without fitting or selecting a checkpoint.
The successful original G/A result stays separate from these final outcomes.

The fixed local result is mixed:

| Final cue | Success | Return | Steps | Ending |
| --- | --- | ---: | ---: | --- |
| Blue | No | 0 | 16 | Truncated |
| Red | Yes | 0.26875 | 13 | Terminated |

Both environments closed. All29predictions/actions/steps returned, model/source/
checkpoint and NumPy RNG guards passed, and no training, teacher, replay, probes,
promotion or target-score calls ran. This is limited held-out behavior, not reliable
generalization. D was absent from this model's TRAIN history but its public family
was studied earlier; it is not a globally untouched benchmark. Earlier FIFO D
failure and neural failures remain valid. There is no controlled causal comparison
or circadian benefit claim, and these final outcomes must not guide further policy,
seed, replay, probe, controller, dose or metric selection.

## Data, interface and isolation

The source is the actual accepted ordinary TRAIN workflow in issue124:
`artifacts/runs/ordinary-fourth-task-20261024/actual.json`,1809283bytes,
SHA`fb34053cd338fc6a501ee6ad1a0170ec3b59d5ab480105b0010ad9587f87638a`.
Use only its fixed last10/252 state, not the best intermediate or a privately
adapted model. The existing `_context_linear_training_snapshot` converter reads
data into owned dataclasses; it does not fit or reconstruct missing parameters.
The model has two36-row/rank24 heads,297x7coefficients each, with the original
public-cue/canonical296 encoding. Numeric coefficient storage is33264bytes plus
Python wrappers, fit evidence, arrays and native/report/copy overhead.

The public binding requires an exact untrained `ContextLinearLearner` source,
default policy and complete10/252/two36-row-head state with no failed work. A
separate original `ToyBudgetSession` must bind0updates/0replay/150seconds/512MiB,
including unchanged clock, progress, sampler and permanent charges. The native
loop preserves full input/source/scorer states; no training authority follows
saved counters. Intermediate, probe-trained, foreign or corrupted states and
changed budgets are refused before more work.

Only original `predefined_layouts()[3]` enters the plan: Drotation3/branch4,
unswapped/yellow, blue thenred, `final_test`, seed17/max16steps. Raw all-seven-action
probabilities use firstargmax, without masks, repairs, teachers, targets or online
updates. Original TRAIN, private and FIFO bindings and presets remain unchanged.
The shared `_run_checkpoint` still owns native steps, frozen-state checks and
cleanup. Final-only frame storage is explicitly labeled on success and failure.
No runner, loader, dependency, CLI flag or configuration layer was added.

Why this: known-task learning is already demonstrated by the simpler affine
control. Evaluating its saved state answers a separate missing held-out-layout
question without retraining a failed neural case or adding another mechanism.
This readout is not a fresh learning or matched-compute comparison.

## Reproduction and verification

Inspect the saved result without running the model:

```powershell
.venv/Scripts/python.exe artifacts/runs/affine-heldout-d-20261024/verify_result.py
```

The recorded one-case allowance is spent; do not relaunch it or adapt the model
to D. API use on other work needs a separately frozen final-evaluation protocol.
The result JSON retains every public frame/probability/action/native transition,
return, ending, cleanup and work fact. Artifacts are local research evidence and
are not implicitly distributed in a wheel or a clean checkout.

Fifteen scoped literal metadata checks qualify on each platform; Windows retained
one fixture identity mistake and only its affected repair. These fixtures perform
no real predictions or physics. Ruff/format/fresh Windows mypy and source/procedure/
result self-review pass. Self-review is not independent acceptance. Full suite,
clean clone, installation, Linux native games and release readiness were not tested.

Sole native readout:36.6951661seconds,75771904bytes sampled RSS,29adapter/core
predictions29rows,1489model copies1689fit validations58projections,2emptyowners/
environments/resets/Gym seeds,0fits/LAPACK/SGD/Torch/teacher. Original sunk training
10fits252rows (90arrivals+162reuse,72distinct TRAIN rows), original confidence
negatives and known four G/A games are preserved, not rerun or renewed. Equal
exposure would not imply equal compute.

[Protocol and exact costs/results](../artifacts/runs/affine-heldout-d-20261024/RESULTS.md),
[issue126](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/126),
[ordinary workflow](ordinary-fourth-task.md), ADR-0637. Full R4.2 recurrent RL,
reliable acquisition/retention/return, broader held-out tasks, science and release
requirements remain open. Issue125's lost neural endpoint remains unqualified.
