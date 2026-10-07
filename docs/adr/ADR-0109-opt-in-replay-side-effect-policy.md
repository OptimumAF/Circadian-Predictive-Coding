# ADR-0109: Keep replay adaptive state wake-only in a versioned ablation

## Context

The P4.3a audit found that historical NumPy replay changes chemistry,
importance, traffic, the supervised-error baseline, cooldowns, and pending
prune TTL. It does not advance wake age, wake clocks, or energy history. Those
mixed units can change a later adaptive sleep decision without a wake batch.
P4.4 established a fixed, matched replay schedule and recorded a negative
circadian result before any side-effect change.

## Decision

Keep historical replay as the default with its existing config fields,
snapshot field set, v9 training/result IDs, and checkpoint identity. A caller
may select `wake_only_adaptive_v1` once, before training. Only that model
stores an additional policy field in its snapshot; restore requires the same
policy. Under this policy, replay still updates model parameters, replay
update counts, and observed train-row exposure. It computes plasticity and
reward scaling from the state present before each replay row, but leaves
chemistry, importance, traffic, the reward-error baseline and last wake
scale, cooldowns, pending-prune decay/TTL, age, wake clocks, energy history,
and retained rows unchanged. A successful sleep wrapper still resets its
since-sleep clock and increments sleep events. Guard rejection or a sleep
exception restores the model before PC/backprop receive any replay work.

The opt-in training result uses `continual_replay_side_effect_training_v10`
and a digest bound to the unchanged v9 schedule digest and side-effect name.
The v10 ablation manifest binds the existing two seeds (17, 19), FIFO and
seeded bottom-k retention (seed 53), four-example/96-byte caps, newest-row
selection, two replay updates per sleep, and separate PC/circadian inference
work. It limits runs to two seeds and four wake epochs, trains all eight
side-effect/retention/seed trials, rederives every train-only schedule,
compares applied replay boundaries and baseline model state across
side-effect policies, then releases and compares every final A/B role before
scoring. The result retains all scores and chooses no winner.

## Alternatives

- Mutate adaptive state during replay and restore it afterward. That would
  still use the replay-mutated plasticity gate for the parameter update.
- Add a default field to `CircadianConfig`. That would change old config
  digests and snapshot/checkpoint identities even for historical runs.
- Change the existing v9 outcome runner in place. That would recast its
  recorded negative result under a different protocol.

## Consequences

The fixed v10 artifact has eight scored rows and 32 matched boundaries, with
eight applied row/update calls per method per trial. Its historical rows
equal the existing v9 rows exactly; repeating the v10 run gives the same
bytes. Mean balanced scores are unchanged by the wake-only policy in this
small study: circadian 0.20 for both retention policies, PC 0.625, and
backprop 0.65/0.70. This is a null side-effect ablation and does not support
a general claim about replay or deeper models. The different PC (2) and
circadian (3) inner inference iterations remain visible in the work ledger.
No v10 durable checkpoint or broad hyperparameter sweep is added.
