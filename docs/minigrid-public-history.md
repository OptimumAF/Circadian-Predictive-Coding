# Shared causal public history

`torch-minigrid-history-learnability` is an explicit new condition of the existing
MiniGrid checkpoint diagnostic. The original `torch-minigrid-learnability` recipe
keeps148 inputs and its original evidence.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability
```

The new condition gives BP, PC, neutral CPC and active CPC the same first public
frame followed by current public frame (296 normalized coordinates). Recurrent
BC keeps its148 inputs and existing ordered recurrent state. Each complete
episode starts its own history; no preceding episode, next frame, semantic cue,
context, label, reward, hidden state or final evaluation enters the features.
The initial public image itself contains the publicly visible cue.

The pure online encoder supports `reset`, `encode`, and `clear`; encoding before
reset fails. A malformed reset clears the prior episode's retained frame.
`encode_public_history_episode` detaches original148 episode records, preserving
roles/actions/rewards/terminal flags and next-observation evidence. Replay still
accepts only train episodes. Probes retain the original65536 aggregate-entry cap.

Why: accepted TRAIN data include equal current148 observations with different
turn targets. History removes that particular input ambiguity without selecting
seeds, changing targets or modifying native objectives. This is a recorded-data
finding, not a proof that a memoryless closed-loop policy cannot reach the goal.

## Structure

```text
src/core/minigrid_public_history.py       bounded public encoder/episode transform
src/core/demonstration_episode.py         additive exact296 episode/replay geometry
src/adapters/torch_feature_learner.py      additive exact296 native input geometry
src/app/fresh_task_probe.py               same roles/storage/isolated budget
src/adapters/minigrid_learnability.py      opt-in existing stream/probe composition
scripts/run_coupled_attribution.py         existing CLI scenario
tests/test_minigrid_public_history.py     causal/reset/codec/admission/native checks
tests/test_minigrid_history_route.py       mixed296/148 route and refusal behavior
```

## Costs and interpretation

The fixed single-seed stream/probe dose, metrics, comparison directions and limits
are unchanged:800 SGD calls/7200 rows,230 score forwards/2070 rows,10teacher
episodes/90 expected physics steps (maximum160),180body/210parent seconds,
768MiB RSS,32MiB output,CPU1/inter-op1/GPU0. No finalD or learner live episodes.
All five benchmark streams finish before any readout/probe. Probe updates remain
separate from benchmark/source/actor weights, with independent original budgets.

FF parameters grow1255 to2439; RNN remains1319. Derived probe arrays384912B plus
original193104B remain retained; detached stream/state/report copies and Python
overhead add to time/RSS. Per-episode initial memory stores148 numeric values
(1184 float64-equivalent bytes, excluding Python overhead). Same updates/rows
do not mean matched computation/capacity or justify a ranking against the original
condition. Correctness, teacher-forced fit, useful live skill and scientific
benefit are separate claims. Preserve every failed/inconclusive/negative result.

Scoped commands and exact evidence are in
[the prospective protocol](../artifacts/runs/r21-minigrid-history-20261016/SCOPE.md).
Extend via existing ports with a separately reviewed protocol; preserve the
original conditions, final isolation and matched information across controls.

## Executed evidence

441 scoped cases pass with no skips, including four native one-update boundaries;
scoped Ruff/format and both mypy platforms pass. Independent SOURCE/GO accepted
the frozen condition. Its sole native case completes in87.59parent seconds with
all800updates/7200rows,230score forwards/2070rows and90teacher steps.

The conflicting recorded E TRAIN inputs disappear under public history. Every
probe reduces CE, but all initial both-cue acquisition premises still fail. BP
and recurrent post-stream probes end with one of two exact sequences; PC variants
have zero. RNN already starts at its endpoint sequence accuracy. Active CPC has
smaller CE gains than PC/neutral. This establishes neither useful live skill nor
circadian benefit, and does not show improvement over the original representation.

[Commands, all curves, failures, costs and review](../artifacts/runs/r21-minigrid-history-20261016/RESULTS.md).
The [native activation readout](native-activation-health.md) is now implemented
and independently accepted with unchanged learning outputs. Next:296 public-history
live support in the existing evaluator before a separately frozen acquisition case.
