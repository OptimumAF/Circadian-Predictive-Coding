# Demonstration-to-reward phases

The separate `torch-key-door-warm-start-rl` scenario runs eight actual train-role
demonstrations followed by eight sampled reward episodes through the existing
actor/inbox runtime. The pure-RL `torch-key-door-rl` route remains available.
This feature makes the objective boundary explicit; useful navigation, retained
skill and comparative benefit remain unproven.

```powershell
# Native-free protocol preview; this does not train or evaluate.
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --describe --scenario torch-key-door-warm-start-rl
```

Omitting `--describe` executes the fixed exploratory recipe. That full sixteen-
update arm is now measured in a [matched BC-only comparison](warm-start-control.md).
The earlier phase increment was engineering-only; the new frozen comparison
preserves its negative observations and explicitly accounts for momentum.
Further comparisons require a new prospective protocol. Do not use earlier final-scored weights or
select seeds, metrics or additional episodes from final outcomes.

## Boundaries and continuation

`DemonstrationUpdate` owns an exact complete train-role demonstration;
`RewardUpdate` continues to own a distinct sampled episode and full behavior
checkpoint. Both require matching episode-ID releases. The new
`TorchDemonstrationRewardLearner` owns one native learner and dispatches to its
unchanged mean-CE or gamma1 reward-logprob objective. Array labels, nontraining
roles, forged packet shapes and class mismatches are refused.

The existing recipe owns the fixed phase order. A single original sixteen-update
budget covers both phases; native parameters, gradients, momentum and clocks
continue without resetting or renewing them at the boundary. Eight BC updates
expose 64 teacher rows; sampled reward rows and private PCG64 draws are accounted
separately. Zero-reward steps can still move weights through carried momentum.
Phase summaries count committed work; unknown partial costs are never zero-filled.
Corrupted prior evidence refuses more work and a secondary summary fault preserves
the primary refusal.

Checkpoints are 0/8/10/12/14/16; final classification and 48 fresh live episodes
are deferred until all sixteen commits. Reward-stage CE directions use update8
as baseline and refer to already demonstrated contexts. They do not measure
acquisition of an unseen task. Final data never train, promote or select a probe.

## Structure and verification

```text
src/adapters/torch_demonstration_reward_learner.py   typed dual-objective native port
src/adapters/key_door_reward_pilot.py               existing recipe, explicit phases
src/adapters/key_door_pilot.py                      existing opt-in budget/helper reuse
scripts/run_coupled_attribution.py                 separate explicit CLI choice
tests/test_torch_demonstration_reward_learner.py    fake-native admission/ownership
tests/test_demonstration_reward_pilot.py           runtime order/isolation/faults
tests/test_demonstration_reward_integration.py     actual train-only continuation
```

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_torch_demonstration_reward_learner.py tests/test_demonstration_reward_pilot.py
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_demonstration_reward_integration.py
```

The integration check exercises eight native BC updates, one sampled reward
update and an owned branch/resumed next update. It uses no final evaluation and
did not execute the full comparative recipe. RSS was sampled during its training
window after source/runtime construction, excluding earlier import/construction
transients. Exact commands, initial failures, work counts and independent review
are recorded in [the results](../artifacts/runs/r42-demonstration-reward-phase-20261016/RESULTS.md).

Accepted scoped evidence: 171 fake cases plus one actual native case qualify
(516 latest passing test phases); final seven-path statics and independent
SOURCE/GO/ACTUAL review pass. Two generic-guard false refusals remain preserved;
ordinary pytest completed within the same remaining engineering allowance.
Native work is 11 SGD/69 train rows/74 unrolled steps/five predictions/66 physical
steps. All three sampled training episodes return zero; parameters still move
through carried BC momentum. No final data, acquisition or reward benefit claim.
The matched control is now measured above. Next add bounded per-decision/class
diagnostics to the existing reserved probe, retaining original metrics and budgets.

Why this: the generic runtime already supports typed opaque packets. Reusing one
native owner preserves optimizer continuation without another runner or a change
to either objective. [ADR-0332](adr/ADR-0332-preserve-native-state-across-demonstration-and-reward-phases.md).
