# Legal-action-conditioned reward trajectories

The existing train-only collector has an explicit option:

```python
from src.app.key_door_training import collect_key_door_reward_episode
from src.adapters.torch_reward_update_learner import LegalRewardUpdate

report = collect_key_door_reward_episode(
    collector, layout, cue, episode_id="legal-train-0", sampler=private_rng,
    condition_on_legal_actions=True,
)
update = LegalRewardUpdate(report["episode"], report["behavior_state"])
release_id = update.episode.episode.episode_id
# Release update through the existing runtime using this exact identifier.
```

`collector` is an owned frozen-session policy source. `private_rng` must be the
caller’s `Generator(PCG64)`; an invocation budget belongs to the surrounding
runtime. The default `False` keeps the old unmasked sampling, report and record.
The explicit game recipe below composes this option through the existing runner.

Why this: prior train-role reward episodes all sampled illegal actions with zero
return. The environment already exposes legal actions from visible dynamics,
independent of the hidden required key. Conditioning provides that extra action
information explicitly, with no teacher repair, resampling or fabricated reward.
Its availability must match across any future comparison’s controls.

## Runnable game recipe

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-legal-rl --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-legal-rl
```

The separate `actual_key_door_legal_on_policy_recurrent_rl_v1` protocol fixes
native seed17/private sampler29, eight actual train-only episodes A+/A-/B+/B-/
C+/C-/A+/A-, at most12 decisions each, and checkpoints0/2/4/6/8. The existing
rotated layouts are used. Every typed packet retains raw probabilities and legal
support; the base episode supplies row counts without dropping that evidence.

All eight validated commits finish before final teacher packets are constructed.
Raw categorical teacher CE/accuracy remain diagnostics of raw policy output.
Live evaluation chooses the first greedy maximum of the same conditional law
used for training collection and likelihood; it does not sample or repair actions.
Thus CE is not a conditional policy likelihood or a replacement for live outcomes.
Source, actor, receipts, private sampler and the original budget are checked at
existing boundaries. A post-commit refusal retains its receipt and permanent charge.

The original `torch-key-door-rl` and warm-start routes retain unmasked behavior.
This new cold reward route has no demonstration phase, replay, probe, promotion,
new orchestration framework or baseline rerun. Legal action information would need
to match across controls in a future separately frozen comparison. A legal path
can still loop or truncate without reward; useful adaptation remains an open goal.

## Probability law and ownership

At each step, let raw probabilities be `p` and the observed legal support be `L`.
The sampling law is `q(a) = p(a) / sum(p(b) for b in L)` on `L`, and exactly zero
outside. One private random draw chooses from `q`. Zero legal mass is a refusal
before a draw or physical step; no epsilon or uniform fallback is inserted.

`LegalPolicyGradientEpisode` owns three fields: the existing reward episode,
per-step legal supports and raw probabilities. The nested episode records `q`.
The old seven-field `PolicyGradientEpisode` and two-field `RewardUpdate` remain
unchanged. Supports must be nonempty sorted unique native indices; the selected
action must have positive conditional mass, excluded behavior must be exactly
zero and recorded normalization must agree within `1e-12`. Copying revalidates
frozen-record bypasses. This validates declared evidence, not physical provenance.

Native admission requires the complete original behavior checkpoint, train role,
geometry and current raw probabilities. It recomputes conditioning from the
current public probabilities, refusing zero legal/selected mass, and checks the
stable masked-logit law against recorded behavior with the existing absolute
`1e-6` native tolerance. Float32 underflow disagreement is refused explicitly.

The objective remains the sum `-sum(gamma**t * G_t * log(q_t(action_t)))` with
discounted reward-to-go `G_t`. It differentiates masked log-softmax, without a
mean, baseline, advantage normalization or bootstrap. The diagnostic is
`torch_recurrent_legal_reward_logprob_sum_v1`. Native parameter, gradient and SGD
state publication stays atomic. Full-support conditioning has the original
objective; singleton support has zero policy gradient, while carried optimizer
momentum can still move weights.

## Boundaries and extension

```text
src/core/legal_policy_gradient.py        owned support/normalization evidence
src/app/key_door_training.py             actual train-only collection and RNG
src/adapters/torch_recurrent_reward.py   conditional likelihood and atomic SGD
src/adapters/torch_reward_update_learner.py  explicit LegalRewardUpdate
src/adapters/torch_demonstration_reward_learner.py  continued demo/reward owner
```

Both existing typed learner ports accept `LegalRewardUpdate` with its exact
nested episode release ID. Ordinary supervised and unmasked reward methods keep
their semantics. The component adds bounded per-step raw probability/support
records and extra normalization/law checks; it does not claim equal computation.
Copies, native state and Python storage belong in whole-process time/RSS costs.

The game recipe declares train/final roles and action information in its distinct
frozen protocol. Preserve old unmasked results when extending it. Component correctness and a small
engineering smoke do not establish useful RL, retention or circadian advantage.

## Executed evidence

221 focused cases, nine-path Ruff/format/win32+linux mypy, independent SOURCE/GO/
ACTUAL and one existing-runtime two-update engineering case pass. Both actual
12-step collections used only legal actions but truncated with return0; there was
no reward-gradient signal or weight learning. B is a default-geometry identifier
in that smoke, not the historical rotated task. No comparative benefit claimed.
[Commands, costs, failures and remaining work](../artifacts/runs/r42-legal-reward-20261016/RESULTS.md).

## Game integration evidence

215 focused cases and six-path Ruff/format/win32+linux mypy pass; independent
SOURCE/GO/ACTUAL accept the sole frozen native CLI case. All eight training
episodes used12legal steps and truncated with return0; held keys were observed
in6/8 and door adjacency in3/8. No reward-gradient signal; all checkpoint parameters
stayed initial. All40conditional greedy live episodes reached the step limit with
return0. Extra legal information has not established useful learning or advantage.
Native cost8SGD/96rows/992unroll/616predict/56env/640steps,19.9546531parent seconds;
fake gate RSS was not measured. [Detailed evidence](../artifacts/runs/r42-legal-game-20261016/RESULTS.md).

Next: same-legal-information warm-start RL/BC-only pair through the existing paired
recipe, motivated by train-only zero rewards. Freeze/review before a small run;
account for carried optimizer momentum and unequal extra updates.

## Paired warm-start recipe

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-legal-warm-start-control --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-legal-warm-start-control
```

This explicit pair reuses the existing BC-only versus warm-start reward recipe.
BC-only receives8train-only demonstrations, then freezes. The reward arm receives
the identical8demonstrations; its complete parameters/gradients/optimizer/policy/
clocks must match at8 before8actual conditional reward episodes. Both live
evaluators receive the same visible legal-action information. Overall training
exposure/computation remain unequal. Demonstration mean CE stays unchanged.

Exactly one shared set of8final_test packets is collected after BOTH arms finish
all24updates. Existing completion callbacks validate the deferred control before
more peer work, and both arms around final collection/scoring/live. Owned final
packets and their serialized evidence remain unchanged between evaluators.

Protocol `actual_key_door_legal_warm_start_comparison_v1` fixes the existing native
seed17/private sampler29, rate.03/momentum.9/gamma1 and task sequence. Comparison
directions remain BC8CE minus RL16CE; RL16accuracy/success/return minus BC8.
Teacher CE/accuracy describe raw probabilities; live argmax uses the conditional
law. All context/cue outcomes remain reported, including failed or zero-return ones.

Additional updates can move parameters using inherited BC momentum even when
rewards are all0. Motion or a performance change alone does not establish a
reward-gradient benefit. This pair does not equalize computation, prove useful
learning or compare circadian models. Old unmasked scenarios remain separate.

### Executed paired readout

107 latest focused cases and scoped Ruff/format/Windows+Linux mypy pass. Sole
frozen pair:24SGD224rows1546unroll874predict96env1002physicalsteps,
72.4083152parent seconds. All8rewardtrain episodes truncate/return0; no reward
gradient signal. BC8 and RL16 endpoints both1/8success/meanreturn0, with all A/B/C
cues at the step limit, Dred wrongkey and Dblue success. CE improved .0201153;
accuracy fell .03125. The extra updates provided no observed live benefit; BC-only
uses fewer updates here. Carried momentum prevents reward-learning attribution.
[Full evidence and preserved failures](../artifacts/runs/r42-legal-warm-pair-20261016/RESULTS.md).

The next diagnostic uses the existing fixed64-update train-only A±acquisition
runner: owned0/64closed-loop rollouts versus its fixedtrain-prior withsamelegal
information, no final_test packets. Its purpose is operational train fit before
more RL features; no change of dose/seed from these final outcomes.
