# Frozen TRAIN observation standardization

The continuous BC learner optionally applies a frozen `(raw - mean) / scale`
transform before prediction and training. Raw inputs remain the default. Action
units, bounds, normalized half-MSE objective, initialization, optimizer and public
learner ports are unchanged.

## Files and boundaries

```text
src/core/observation_standardization.py         Pure statistics and application
src/adapters/numpy_continuous_bc.py             TRAIN fitting and owned policy
tests/test_continuous_bc_standardization.py     Independent numeric/state tests
tests/test_continuous_bc_metadata_compatibility.py Legacy raw metadata boundary
```

The core takes exact finite float64 arrays with 1..256 rows and 1..1024 columns.
It returns `ObservationStandardization(means, scales, fitted_rows)` with immutable
owned native tuples. Statistics are population mean/std, not sample std. Fitting
uses per-column magnitude scaling to avoid overflowing physical sums/squares.
Constant columns explicitly use scale1 and their observed mean: later deviations
remain visible. Nonconstant unrepresentable scales are refused. Applying the
transform refuses unrepresentable subtraction/division without clipping or refit.

The pure array function grants no data authority. The adapter's
`fit_training_standardization(episode)` revalidates a complete TRAIN record before
fitting; probe_train/probe_test/final roles are refused before statistics. Raw
array callers retain release responsibility, as with existing BC batch methods.

## Usage and ownership

```python
from src.adapters.numpy_continuous_bc import (
    ContinuousBCPolicy, NumpyContinuousBCLearner, fit_training_standardization,
)

# episode is a complete, released TRAIN ContinuousEpisode.
transform = fit_training_standardization(episode)
model = NumpyContinuousBCLearner(ContinuousBCPolicy(
    episode.spec, observation_standardization=transform,
))
diagnostic = model.train_episode(episode)
# Prediction and TRAIN diagnosis apply the same frozen statistics.
child = model.fork()
checkpoint = model.snapshot_state()
model.restore_state(checkpoint)
```

The optional policy field is last, preserving previous positional construction.
Full policy copies/snapshots/forks/restore include the transform and fitted-row
metadata. There is no implicit fitting or in-place normalization update in any
training, prediction, diagnostic or rollout call. Restore refuses mismatched
transforms before publication. The existing frozen robot entry point works with
this policy and checks complete transform state through inference and cleanup.

`asdict(policy)` now includes `observation_standardization`, null for raw models.
The earlier native-case reader was narrowly adapted to default only that missing
legacy field to null; all other fields remain strict. Its exact prechange source
is preserved with the current evidence. Old results/protocols were not rewritten
or rerun. This is additive metadata compatibility, not a new disk codec.

## Executed checks and comparison

425 combined cases pass, including independent population-statistic and manual
pretransform update oracles, TRAIN-before-fit refusal, overflow/corruption,
ownership, full continuation/restore and joined fake-native frozen inference.
Scoped lint/format and two-source types pass. Independent source and actual
readback accepted the fixed matched study and its saved states.

Both fresh policies used the same initial eight arrays, TRAIN data, H64/seed17/
LR.01/momentum.9 and 256-update budget. The only configuration difference was the
frozen input transform. All512 returned losses and eight complete diagnostics
are retained; the raw256 endpoint exactly reproduces the accepted reference.

| Total updates | Raw TRAIN half-MSE | Standardized | Standardized minus raw |
|---:|---:|---:|---:|
| 0 | 0.265632 | 0.292825 | +0.027192 |
| 64 | 0.062273 | 0.022520 | -0.039753 |
| 128 | 0.042838 | 0.020309 | -0.022529 |
| 256 | 0.032012 | 0.019050 | -0.012962 |

The standardized start is worse. Four increases in the raw pre-update loss curve
are retained. No monotonicity/improvement gate or best checkpoint selection.
These observations show better fitting on one TRAIN demonstration under this
fixed budget, not generalization or robot skill.

## Costs and closed-loop negative

One statistics fit reads29rows/1218values, with84 numeric statistics/672 raw bytes
plus metadata. The matched study spends512updates/14848training-row exposures,
eight diagnostic forwards/232rows/928values/7424 cumulative raw prediction bytes.
The standardized policy applies the frozen transform260times/7540rows, separate
from the fit. Measured raw/standardized method times .199078/.214874s and fit
.000682s are one local observation, not a speed claim. Both final states contain
eight arrays/48192 raw bytes/50102 file bytes; the report is203129B. Process
final-self peak38973440B excludes a full-lifetime process-tree guarantee.

A separately frozen standardized-final256 TRAINseed17 rollout also completed50
steps with **zero sparse reward and no any/final success**, ending at the time
limit. It exactly matched the teacher's initial state and preserved original/
private full model state. It spent50predictions/rows/1600rawB,50controls/250known
physics and zero optimizer updates; reset physics are unknown. Inference adds50
frozen-transform applications/50rows/2100feature values beyond the TRAIN study.
The accepted raw
TRAIN negative was reused as the matched task control, without rerunning it.
Earlier held-out probe and all other negative results remain unchanged.

Useful robot BC, new-task acquisition/old-task retention/context recovery, visual
features, adaptation and CPC benefit remain open. No deployment/promotion or
final-data access occurred.

## Fixed continuation and next representation question

The separate matched continuation to1024 is now independently accepted. It added
768updates per policy, preserving full owners and fitted statistics. Final TRAIN
half-MSE is0.025183 raw and0.016886 standardized. All1536newlosses and sixfull
reports remain; no simulator/probe/final work occurred.

Standardized descend-stage loss worsened from0.017011 at256 to0.017537 at1024,
also worse than raw1024 at0.012674. Grasp still has a wrong-sign gripper error.
Better overall fit does not establish robot skill; the256native negatives remain
valid and1024policies have no native evidence. [Full curves, costs and review](../artifacts/runs/continuous-bc-1024-20261019/RESULTS.md).

Next implement a bounded public episode-step feature and matched zero-clock
control through existing continuous records and the rollout entry point. The
teacher explicitly uses its step schedule; this tests whether an observable
phase feature helps, without claiming it is necessary. Preserve the unclocked
route and test role/spec/step/configuration and full state before a separately
frozen equal43D/equal-initialization TRAIN comparison. No automatic more epochs,
native retry, probe/final selection or new runner.

[Exact protocols, commands, costs, failures and review](../artifacts/runs/continuous-bc-standardization-20261019/RESULTS.md).
