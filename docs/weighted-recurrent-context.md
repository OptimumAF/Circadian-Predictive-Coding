# Fixed class-balanced recurring TRAIN case

The existing public-cue recurrent pilot has an explicit class-balanced recurring
mode. It uses the original raw148 A32/B32/C_train32/A32 schedule, two H8 heads,
seed17,128updates/1152training rows and delayed labels. The existing weighted
episode adapter supplies the fixed current-episode class-count rule.

Describe it without reading data or constructing a model:

```powershell
.venv/Scripts/python.exe -m scripts.run_minigrid_paired_bc_smoke --describe --public-cue --recurring-training --class-balanced-training
```

The corresponding Python entry is
`run_pilot(episodes, recurring_training=True, class_balanced_training=True)` in
`minigrid_public_cue_recurrent_pilot`. The CLI uses the original packaged TRAIN
example. Outputs are compact, finite and exclusive. Existing fixed-A, ordinary
recurring and replay defaults retain their original protocols; mixed weighted/
replay or implicit-source modes are refused before native work.

All128fits finish before any scores or games. Six TRAIN teacher episodes are
scored at0/32/64/96/128 using unchanged unweighted CE/accuracy/exact sequences.
Native Ablue/red games run at32/64/96, then A/B/C_train blue/red at128:12games,
max16steps each, genuine recurrent sessions and all-seven first-argmax actions.
Middle A observations check any forgetting premise; recovery needs observed
acquisition, intervening loss and later regain. B64/C96 native games remain
unmeasured. Teacher fit alone cannot establish native skill.

The pilot retains the original weighted wrapper and its inner native source.
Both remain under full-state, identity, receipt, clock and original-budget guards
during readonly evaluation; the evaluator still requires an exact native owner.
Source/actor isolation and paid failures retain their existing contracts. No
probe/final labels, replay, promotion, teacher fallback or action masking enters
this case.

The new treatment reuses accepted132meanCE evidence where data/config/chronology/
metric assumptions remain unchanged. It does not retrain or rescore that control.
Equal updates/rows are not equal compute: weighted scheduling, changed validation
overhead and four extra middle A games have explicit costs. Comparisons use
common native endpoints; old unobserved points remain null.

Software qualification:64focused cases/platform, scoped available statics and
lead self-review. Includes actual packaged source admission, byte-identical old
protocols, strict mode/reader bindings, preserved paid prefixes, CLI and genuine
readonly kernel with fake physics. Fakes are not learned-task evidence. Accepted
native mathematics/stream parity from139 is reused.

The sole prospective study protocol and process receipts live in
`artifacts/runs/weighted-recurrent-context-20261025/`. Earlier incomplete/spent
and negative runs remain unchanged. No circadian, causal, heldout/reliability,
RL or release advantage follows from this implementation.

The sole native run is incomplete and spent: all128updates returned, then the
original200s readout budget expired at64. Both A32games failed after16steps;
A64blue has2partial steps/nulloutcome, red and96/128readouts are unrun. No
acquisition/retention/recovery benefit is established. The partial result and
original charges are preserved; no retry or renewed budget follows. See the
[observed results and limits](reports/weighted-recurrent-context-20261025.md).
