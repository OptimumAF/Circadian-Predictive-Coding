# NumPy replay side-effect audit (P4.3a)

## Scope and method

This is a deterministic core audit, not an outcome experiment. The fixture
in `tests/test_replay_side_effect_audit.py` trains two wake batches on two
labeled rows, freezes a valid pending prune and cooldown state, then
compares direct `_run_replay_consolidation()` with a replay-only component
sleep starting from the same model state. Chemical reset, homeostasis,
split, and new prune selection are disabled for that comparison so the
sleep wrapper's own clock changes can be distinguished from the replay
training call. Reward modulation and dual chemistry are enabled to expose
their state paths. The row source is train-only NumPy data; no guard,
outer-selection, or final-test source exists in the fixture.

| State | Direct replay observed behavior | Boundary for P4.3b |
|---|---|---|
| Weights | Updated by the replay optimizer. Pending-prune weights also decay before that update. | Keep replay learning explicit; test whether prune decay belongs to replay or wake. |
| Fast/slow and combined chemistry | Updated during latent replay training. | Test an opt-in wake-only adaptive-state policy against current behavior. |
| Importance EMA and traffic sum/steps | Updated. | Decide whether replay examples count as usage evidence for structural decisions. |
| Reward-error EMA and last scale | Updated when modulation is enabled. | Separate replay difficulty from the wake difficulty baseline in the opt-in policy. |
| Split/prune cooldowns | Each replay optimizer call decrements both. | Keep cooldown units explicit; replay must not silently consume wake epochs. |
| Pending-prune TTL | A marked neuron's TTL decrements; a later replay call can finalize removal. | Decide whether only wake calls should advance gradual pruning. |
| Neuron age | Unchanged by direct replay. | Preserve this explicit wake-only unit unless a separate rationale is tested. |
| Wake batches/examples/since-sleep | Unchanged by direct replay. | Preserve separate wake and replay clocks. |
| Replay updates/exposed IDs | Increase by one successful train-row replay call. | Retain truthful applied exposure accounting. |
| Sleep-event count/since-sleep reset | Unchanged by direct replay; the surrounding successful sleep increments the event count and resets since-sleep. | Do not attribute wrapper scheduling effects to replay training. |
| Energy history | Unchanged by direct replay. | Preserve its wake-only definition. |
| Adaptive sleep readiness | Chemistry variance changes; a deterministic threshold between pre/post variances flips `should_trigger_sleep()` while wake clock/history stay fixed. | Make this coupling explicit in the opt-in policy and ablation. |
| Retained examples and order | Unchanged; the exposure ledger gains only selected observed train IDs. | Preserve no recursive refill and the train-only source rule. |

## Decision boundary

Keep historical behavior and protocol/checkpoint identities unchanged.
For P4.3b, evaluate a versioned, opt-in policy that lets replay update
weights and its exposure ledger while leaving wake adaptive state and
cooldown/prune progress untouched. Compare it with the current policy
under the same fixed train-only replay schedule and accepted-work budget.
The replay optimizer should use the adaptive state present before each
row; mutating that state temporarily and restoring it after the update
would still change the replay gradient and must be treated as a different
policy. No new behavior is selected by this audit.
The policy choice should use development evidence and report any negative
result; the P4.4 matched final scores must not be used to tune it.

## P4.3b resolution

`wake_only_adaptive_v1` now opts in before wake training. It updates replay
weights and exposure using the pre-row chemistry, importance, and supervised
error baseline while leaving wake adaptive/progress state fixed. Historical
behavior remains the default and retains its old snapshot field set. The
fixed v10 two-seed/two-retention-policy ablation uses the P4.4 shared
schedule and globally sealed final roles. It found no aggregate score change
from this policy choice: circadian 0.20 under both policies, compared with
PC 0.625 and backprop 0.65/0.70. The result remains a null observation;
the ablation did not change its seeds, metrics, baselines, or guard threshold
after observing P4.4. See ADR-0109 for the policy and work contract.
