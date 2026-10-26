# Matched H32 fresh-task and return games

## Result and limits

The fixed four-method deferred-game protocol is complete:64 known-task game
observations, with historical BP reused and exactly one new run of each PC-family
control. Independent SOURCE/ACTUAL review accepted the costs, isolation and
qualification. This is a single-seed exploratory study, not independent replicates,
heldout generalization, a full acquired-task retention result or a release gate.

PC and neutral CPC match exactly across270 TRAIN prediction rows,32 F/A monitor
checkpoints and16 raw game traces each. Active CPC has no goal-success advantage.
All prior inline failures remain failed under their original ordering/budgets.

| Method | Source0 after F16 | Source0 after A16 | Source128 after F16 | Source128 after A16 |
|---|---:|---:|---:|---:|
|BP (reused)|0/4|0/4|1/4|0/4|
|PC|0/4|0/4|0/4|0/4|
|Neutral CPC|0/4|0/4|0/4|0/4|
|Active CPC|0/4|0/4|0/4|0/4|

Entries are goal successes out of four fixed A-blue/A-red/F-blue/F-red games.
Only BP source128 after F16 succeeds on F-red: return+.49375 in9steps. Its
F-blue game fails with−1 in7steps. Every other game returns0 and lasts16steps.
After A16 return, BP loses that F-red success and neither A cue succeeds.
There are no pre-F0 live games here, so causal live-learning gains cannot be
derived. Neither full new-task acquisition nor full old-task retention/recovery
is demonstrated. Teacher-forced fit is reported separately below.

All80 frozen per-slot signed contrasts are retained in analysis.json. Each PC-
family method minus BP has return+1 on source128/F16 F-blue (both still fail),
and success−1/return−.49375 on F-red. Other success/return contrasts are0,
including active-minus-neutral. A higher failed-game return does not establish
goal acquisition. Step contrasts are descriptive, with no preference for a fast
failed termination. No blanket winner or circadian benefit is established.

## All teacher-fit endpoint summaries

Correct teacher decisions out of18, both cues combined. Sequences show
before F updates → after F16 → after A16 return. All original CE/accuracy/exact
curves at0/4/8/16 remain in the raw reports and analysis; no best endpoint chosen.

| Method | Source | F correct /18 | A correct /18 |
|---|---:|---|---|
|BP (reused)|0|1 → 10 → 14|2 → 10 → 14|
|BP (reused)|128|12 → 17 → 16|17 → 15 → 16|
|PC|0|1 → 10 → 10|2 → 10 → 10|
|PC|128|10 → 10 → 10|10 → 10 → 10|
|Neutral CPC|0|1 → 10 → 10|2 → 10 → 10|
|Neutral CPC|128|10 → 10 → 10|10 → 10 → 10|
|Active CPC|0|1 → 10 → 10|2 → 10 → 10|
|Active CPC|128|10 → 10 → 10|10 → 10 → 10|

At source128, BP new-F fit improves12→17/18 after F16, while old-A fit falls
17→15/18. Return improves A to16/18 and reduces F to16/18. PC-family source128
accuracy remains5/9 per cue; CE changes are fully retained and PC-neutral exact.
Active CPC source128 endpoint CE is higher than neutral for both F and A.
All PC-family games choose `forward` on all256 decisions per method. This action
histogram is a descriptive read of already returned predictions, not a new
primary metric or a selected probe. It helps explain zero goals despite CE gains.

## Observed computation, memory and wall time

| Method | Native updates / rows | Predictions / rows | Resets / steps | Body seconds | External peak RSS B |
|---|---:|---:|---:|---:|---:|
|BP (reused)|192/7488|334/1086|26/330|76.820866|460640256|
|PC|192/7488|350/1102|26/346|64.632206|381038592|
|Neutral CPC|192/7488|350/1102|26/346|92.775610|384786432|
|Active CPC|192/7488|350/1102|26/346|99.700297|379432960|

Each method has94 fixed score forwards846rows; game predictions are240 for BP
and256 for each PC-family method. Each includes10 teacher episodes90steps plus
16 frozen games. BP has192 SGD calls; each PC-family method has192 native PC
updates and nominal14976 latent row-steps, with0 SGD. Counts/exposure are matched
where declared; objectives and primitive computation differ, so no equal-FLOP
claim. Wall times include validation/copies; this is not a causal speed comparison.

| Method | Serialized capture B | Logical native B | Preparation copies | Reader snapshots |
|---|---:|---:|---:|---:|
|BP (reused)|485759|467280|2444|2428|
|PC|165823|156272|2588|2572|
|Neutral CPC|215350|182128|2588|2572|
|Active CPC|215342|182128|2588|2572|

Each case keeps four actual typed states, two readers/four restores;1MiB serialized
and620544logical byte bounds unchanged. Every attempted count returned; no paid
work unknowns. Full source/actor/input/Python-NumPy-Torch RNG, original budgets,
native optimizer/gradient/adaptive/RNG states, receipts and post-clock proofs pass.
Final evaluation, replay, game updates and source promotion remain0/unavailable.

## What to build next

The evaluator/runtime works; repeated capture/profiling work is no longer the
critical path. Existing fixed old-A rehearsal is a concrete candidate control,
currently restricted to H8 by build_minigrid_old_a_rehearsal and wide-probe
admission. Extend that existing composition to explicit H32 with fixed TRAIN-bank
membership, separate replay charges and role/state isolation tests. Keep current
no-replay controls unchanged. Any replay experiment must freeze its additional
exposure/computation controls first; replay is not a promised remedy or CPC win.
Full R4.2/R2.1/R2.3/R2.7 and the repository goal remain open.

[Frozen protocol, exact commands, tests, costs, failures and handoff](../../artifacts/runs/deferred-h32-controls-20261018/RESULTS.md) ·
[Independent review](../../artifacts/runs/deferred-h32-controls-20261018/REVIEW.md)
