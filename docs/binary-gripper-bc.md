# Optional binary-gripper training loss

The continuous BC learner can explicitly treat one normalized gripper action as
binary while regressing the other active actions. The default remains all-axis
normalized half-MSE. Prediction uses the same bounded MLP and parameter shapes;
there is no thresholding or discretization of predicted controls.

## Objective and boundaries

Set `ContinuousBCPolicy(..., binary_gripper_axis=3)` for the current four-action
robot. The index must be an exact native integer, its action unit must be
`normalized_gripper`, and its bounds must be `(-1,+1)`. Targets on that axis must
be exact endpoint labels. Other labels, inferred axes and implicit conversion
of continuous targets to classes are refused.

For the existing gripper output `tanh(z)`, the implied probability of label+1 is
`sigmoid(2z)`. With label `s` in `{-1,+1}`, its loss is
`softplus(-2*s*z)` and derivative with respect to `z` is
`-2*s*sigmoid(-2*s*z)`. The factor two is explicit. Remaining active axes retain
their half-squared normalized error. Sum these terms and divide by rows times
the original number of active axes; constant axes remain excluded. There is no
adjustable classification weight. `logaddexp` evaluates the classification term
without taking a log of saturated probabilities. Unrepresentable logits, losses
or gradients are refused before an update is published.

The named mixed training objective is distinct from the original training loss.
Existing diagnostics always report the original normalized half-MSE, with the
same per-axis/stage summaries. Compare policies with those common metrics and
task endpoints; their different training objectives are not a common score.

Why this: the teacher's gripper labels have two levels and repeated fixed TRAIN
diagnostics retain wrong-sign predictions. Classification is an objective
hypothesis, not a demonstrated explanation of prior native failures.

## Usage and ownership

```python
from src.adapters.numpy_continuous_bc import (
    ContinuousBCPolicy, NumpyContinuousBCLearner, fit_training_standardization,
)

# episode is a complete, released TRAIN record with endpoint gripper labels.
statistics = fit_training_standardization(episode)
model = NumpyContinuousBCLearner(ContinuousBCPolicy(
    episode.spec,
    observation_standardization=statistics,
    binary_gripper_axis=3,
))
diagnostic = model.train_episode(episode)
snapshot = model.snapshot_state()
child = model.fork()
model.restore_state(snapshot)
```

The optional field is last, preserving existing positional construction. Full
policy/snapshot/fork/restore state owns and validates it alongside normalizer,
parameters, momentum and counters. Restore refuses a different objective
configuration. Endpoint refusal precedes gradient work; role/spec and atomic
failure behavior remain in the existing adapter. Raw batch callers retain data
release responsibility. Probe-training updates must still use a separate learner
with matching released role; final evaluation is not training input.

The core owns validation, forward math and analytic derivatives. The existing
NumPy adapter owns policy and updates; neither owns simulator, IO, replay,
promotion or evaluation selection. Four existing stage readers narrowly default
only a missing `binary_gripper_axis` to null in copied legacy default metadata.
Other fields remain strict, and source JSON/state artifacts are unchanged. Their
exact prechange reader sources are preserved with this increment.

Evidence, exact commands and the frozen matched protocol are under
[continuous-bc-binary-gripper-20261019](../artifacts/runs/continuous-bc-binary-gripper-20261019/PROTOCOL.md).
Correct implementation or better one-demonstration TRAIN fitting cannot establish
robot skill, adaptation/retention/recovery, generalization or CPC advantage.

## Executed verification and fixed TRAIN result

604 focused cases pass (70new+534prior), with independent handloss/all15parameter
finite differences, saturated±1000, state/role/atomic and legacy-reader checks.
Windows30math cases pass with the Linux metadata module skipped. Scoped lint,
formatting and types pass. Independent source and actual-study review accepted
all512training objectives, eight common reports and both final states.

A single frozen normalized42D comparison used identical initial eight arrays,
data/defaults and256updates per policy. The original MSE control reproduced its
accepted value exactly. Common final normalized half-MSE is .01905001047 for
MSE and .01853506719 for binary-gripper loss, difference -.00051494328. But mixed
align/grasp and all three translation-axis errors are worse, as is worst grasp
gripper error1.49395843 versus1.48602887. Wrong-sign grasp remains. Different
training objectives are retained separately and are not compared as a score.

One fit29rows/1218values/672numericB,512updates14848row exposures,8diagnoses232rows;
each policy260frozen transform calls7540rows. Two final states50102fileB/48192rawB
each. No native, probe, final, refit or deployed/benchmark work. The implementation
and modest offline improvement do not qualify useful robot BC or CPC benefit.
[Exact commands, costs, tradeoffs and next action](../artifacts/runs/continuous-bc-binary-gripper-20261019/RESULTS.md).
