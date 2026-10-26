# Explicit native momentum-only control

## Purpose and boundaries

The separate control applies native SGD with dense zero gradients to the original
observations of a TRAIN policy episode. Existing optimizer momentum can move
weights despite zero current gradient. This provides a component for separating
additional optimizer updates from reward-gradient effects. It does not itself
establish adaptation, retention, recovery or a circadian advantage.

```text
src/adapters/torch_recurrent_zero_loss.py              staged native objective
src/adapters/torch_recurrent_learner.py                additive native method
src/adapters/torch_demonstration_reward_learner.py     explicit MomentumOnlyUpdate
tests/test_torch_recurrent_zero_loss.py              fake boundary/dispatch tests
tests/test_torch_recurrent_zero_loss_native.py       small native numerical unit
```

Inputs are an exact TRAIN PolicyGradientEpisode and the owned native learner.
Output is diagnostic `torch_recurrent_observed_episode_zero_loss_sgd_v1`, exactly
zero, and one committed update with the episode's original row count. Geometry,
finite data and native counters must validate. All original actions, rewards,
behavior probabilities and ending remain unchanged; only observations enter the
control objective. This is not an on-policy REINFORCE update and does not require
old behavior probabilities to match a control learner that has since diverged.
Existing CE and reward objectives retain their validation and math.

Why this: skipping backward or using a detached constant can leave gradients
`None`, which causes SGD to skip parameters and their momentum. The explicit
finite-logit objective `(logits * 0.0).sum()` connects all five parameters and
zeros before reduction to avoid overflow of otherwise finite logits. All five
gradients must be dense, finite and exactly zero. Native SGD runs on staged
copies; complete state validation precedes publishing the new payload. Failure
leaves the live parameters, gradients, optimizer and counters unchanged, although
attempted work remains paid. Zero momentum leaves weights unchanged; carried
momentum follows the ordinary native recurrence.

## Existing typed runtime port

```python
from src.adapters.torch_demonstration_reward_learner import MomentumOnlyUpdate

# control is a separate TorchDemonstrationRewardLearner fork; observed is the
# unchanged released TRAIN PolicyGradientEpisode. The caller owns admission.
diagnostic = control.train_batch(MomentumOnlyUpdate(observed), observed.episode_id)
```

The matching identifier is required, but grants no provenance or release
authority. The caller must retain original observation provenance and completed
receipt validation, budgets, permanent charges, resource checks and evaluation
isolation. Reusing the reward arm's observed rows is repeated exposure by the
control; account for its rows, optimizer work, memory and compute explicitly.
Do not relabel rewards or fabricate behavior observations to fit old reward APIs.

## Qualification and remaining integration

134 fake/dispatch cases and affected Ruff/format/Windows+Linux-target mypy pass.
The initial invalid test fixture and typing failure remain preserved in
[component evidence](../artifacts/runs/r21-minigrid-momentum-control-20261016/).
The separately reviewed three-case native unit passes: exact momentum0/.9
recurrence and full live-state preservation after a staged capture refusal.
Eight physical optimizer steps include one failed staged step; seven commits
use14rows plus2failed-staged rows. This is synthetic numerical/atomicity evidence,
not a comparative MiniGrid experiment. Independent native ACTUAL acceptance is recorded in the component REVIEW.md.

## Existing runner composition

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_reward_smoke --warm-start --train-readouts --momentum-control --describe
# Native runs require a separate frozen local protocol and a new output file.
```

The option requires both warm start and TRAIN readouts. Reward training completes
and its original45second charges/clock freeze before the separate control branch.
The existing ActorShadowRuntime forks the same warm128 state and applies the
same eight original observed TRAIN episodes through delayed matching-ID releases.
The independent control budget caps8updates/128repeatedrows/45seconds AND whole
remaining time. The ledger labels repeated observation exposure explicitly;
there is no control collection, reward relabel, optimizer reset or final access.

The CLI composes three TRAIN-only readouts after ALLtraining: sharedwarm128,
reward136 and distinct `recurrent_warm_momentum_control`136. Their separate
zero-update90second budget remains within whole180; old two-readout route keeps
60seconds. This is conditional on reward-arm observations, not equal independent
exploration or exact FLOPs. Signed comparisons preserve CE and goal/return metrics
separately and require complete reports; no selection from partial endpoints.

Native eight-reward/eight-control behavior is independently accepted, with122
originalrows per branch. The first v1 full native comparison is INCOMPLETE: diagnostics
hit90seconds during the second readout; the control readout never ran and all
comparisons are null. Preserve the paid144updates1396rows and spent attempt.
[Exact commands, costs, failure and later tested narrow repair](../artifacts/runs/r21-minigrid-momentum-compose-20261016/RESULTS.md).
The repaired tree passes185focused cases/scopedstatics but has no new native run.
Future engineering allowances must be separately versioned/frozen/reviewed before
execution; do not renew the failedcase or infer scientific benefit.

## Explicit v2 engineering allocation

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_reward_smoke --warm-start --train-readouts --momentum-control --extended-control-readouts --describe
```

This additional option selects distinct protocolv2:210second diagnostics within
300whole/330parent seconds, keeping reward/control45 and every scientificchoice
unchanged. It is a prospective separatecase; v1/default caps and spent failures
remain. One frozen reviewed v2 run completes all8+8updates/all3readouts. The
control has lowerCE than reward but still worsens fromwarm and fails both-cue
acquisition; no usefuladaptation/CPCbenefit follows. [Endpoint results/costs](reports/minigrid-momentum-control-20261016.md).

## Focused commands

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_torch_recurrent_zero_loss.py tests/test_torch_demonstration_reward_learner.py tests/test_torch_reward_update_learner.py tests/test_class_weighted_demonstration_update.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check --no-cache src/adapters/torch_recurrent_zero_loss.py src/adapters/torch_recurrent_learner.py src/adapters/torch_demonstration_reward_learner.py tests/test_torch_recurrent_zero_loss.py tests/test_torch_recurrent_zero_loss_native.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check --no-cache src/adapters/torch_recurrent_zero_loss.py src/adapters/torch_recurrent_learner.py src/adapters/torch_demonstration_reward_learner.py tests/test_torch_recurrent_zero_loss.py tests/test_torch_recurrent_zero_loss_native.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/torch_recurrent_zero_loss.py src/adapters/torch_recurrent_learner.py src/adapters/torch_demonstration_reward_learner.py tests/test_torch_recurrent_zero_loss.py tests/test_torch_recurrent_zero_loss_native.py
# Same mypy command also passes with --platform linux; this is not a native Linux run.
```

Native unit invocation is separately budgeted, not implied by these fake checks;
see the saved SCOPE, review and exact execution receipt. No new dependency,
environment variable or default training behavior is introduced.
