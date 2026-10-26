# ADR-0325: Evaluate actual hidden-cue key/door episodes

## Context

Categorical packets now traverse the existing native stream, but synthetic records
do not demonstrate a game policy. R4.2 requires partial observation, remembered
cues, actual legal interactions, recurring contexts and held-out layouts before
any useful adaptation or recurrent-policy claim.

## Decision

Implement a small deterministic local key/door game, with eight local features
and five physical actions. Erase the required-color cue after the first legal
step. Legality never reveals the target. Collect actual eight-step teacher
demonstrations and use the existing stream/replay/native learners unchanged.
After training, complete owned checkpoints execute live categorical episodes with
the common evaluation identity and all negative outcomes retained.

Why this: a paired-cue observation-alias test makes the need for memory explicit,
while actual movement, pickup, wrong-key failure and exit return distinguish live
behavior from classification. One final-only changed geometry separates known-room
recall from held-out layout evidence. A shared target-free probability validator
preserves the already frozen numerical admission rules without dummy labels.

## Alternatives

- Exposing the required color after the first step would remove the memory task.
- Masking illegal actions or inserting teacher choices changes the evaluated policy.
- Training from final outcomes or choosing favorable layouts would violate isolation.
- Another stream, retention system or checkpoint framework duplicates accepted work.

## Consequences

This is actual local MiniGrid-like navigation, not official MiniGrid integration
or recurrent RL. Original native objectives, neutral controls, binary behavior
and original budget/actor authority remain. The fixed feedforward case can fail
honestly; memoryless policies face an explicit information limit. Future memory
or recurrent learners must use only observed history, reset at episode boundaries
and report their extra cost. Full R4.2 and broader scientific/release goals remain open.
