# Public-cue contextual MiniGrid CPU control

The existing diagnostic CLI can run a conventional NumPy learner with two
owned categorical perceptron heads. The blue/red cue visible in the episode's
initial public image chooses a head. All canonical spatial/color coordinates
remain model inputs. Context names, teacher labels and current/next views never
select the head. These are cue heads; they do not identify A/B/C layouts.

```powershell
python -m scripts.run_minigrid_linear_fit_control --context-perceptron --describe
python -m scripts.run_minigrid_linear_fit_control --context-perceptron --source artifacts/runs/joint-catalogue-study-20261018/bp.txt --output fresh-context-game.json
```

Use a fresh output path and the original six complete TRAIN source episodes.
Only `training_sources` is decoded; saved weights and probe/final results do not
enter the learner. The first command needs no source/models/environment. The
second requires optional pinned dependencies in `requirements-minigrid.txt`.
Other diagnostic options/defaults retain their existing behavior. This option
has a fixed schedule and rejects fit-dose/time/gradient/encoding overrides.

## Fixed behavior and boundaries

The existing delayed `AttributionStream` owns32 nine-row updates: A8,B8,C8,A8,
blue/red alternating,delay1. One original32-update/120s/768MiB budget and clock
cover training and frozen readouts. No live publication, replay, probes or final
evaluation. All updates finish before outcomes are read. Each checkpoint
0/8/16/24/32 retains six TRAIN score vectors and two actual known-A games with
seed17/max16steps/raw seven-action first argmax. Success/return/steps and full
failures remain separate from imitation CE/accuracy.

The source/actor remain at0. The evaluator receives owned private checkpoints,
checks original exhausted training authority and preserves source/actor/input/
budget bindings. Previous paid results remain guarded before further work.
Incomplete readouts stop later work and retain returned prefixes/unknowns.
Snapshots bind encoding, both complete heads and original cumulative clocks.
Mixed TRAIN cues, corrupt/foreign state and malformed/missing/ambiguous public
cues refuse before mutation. Readonly mixed-cue rows retain their original order.

The internal298-vector contains296 canonical history coordinates plus2cue bits.
The original unit-mistake rule consumes296 coordinates in each head; there are
4158float64 coefficients/33264numeric bytes total. Python/snapshot/stream/report
copies add memory and time. Initializer RNG is zero; environments keep their
original seeded RNG behavior. Caller NumPy state is checked unchanged.

Why this: the contextual CPU learner already works on ordinary ticket tasks.
An explicit projection and encoding-bound adapter reuse that implementation
without adding a learner objective, scheduler or source-loader framework.

## Evidence and interpretation

Issue93 and `artifacts/runs/contextual-minigrid-control-20261023/` contain frozen
protocol, commands, all costs/failures, software receipts and real-game outcome.
43 focused cases qualify on each platform (36 original plus7 reporting cases;
Windows initial failures and affected repair are retained). Scoped Ruff/format/
mypy and finite independent source/procedure review pass. Original FIFO36
success and native/standardized failures
remain unchanged. The case adds capacity and uses different exposure from pooled
BP and FIFO controls, so no relative or matched-compute benefit follows.

Known-A gameplay can demonstrate acquisition/retention/return. B/C TRAIN fit
alone cannot demonstrate their closed-loop acquisition. Recovery requires an
observed prior deficit; continued success alone is retention. No generalization,
CPC/controller/robustness/safety/latency-SLO or release claim is established.
FinalD stays outside all development choices.

The sole real-game attempt is **incomplete and spent**: all32updates/288rows
completed, but the original120s lifetime expired during the first trained A-blue
game after12steps. Only the two initial games completed (both0success/return).
No complete learned-policy game, retention or recovery was demonstrated.
Checkpoint8 TRAIN fit is5/9 actions for every source, no exact trace; later
checkpoint readouts are unavailable. No retry or dose/seed/metric change.

The original saved report has a stale checkpoint `running` status and top-level
work-unknown flag false despite nested game uncertainty true. Those immutable
bytes remain failed evidence. Future reports mark the checkpoint failed and
propagate game/missing/null uncertainty; seven metadata tests/platform pass with
all model/physics counters0. This reporting repair does not recover outcomes.

## Structure and extension

```text
src/core/minigrid_context_encoding.py         public initial cue projection
src/adapters/minigrid_context_perceptron.py   owned two-head learner adapter
src/adapters/minigrid_context_control.py      fixed existing-stream composition
src/adapters/minigrid_policy_evaluation.py    narrow frozen known-A facade
tests/test_minigrid_context_control.py        routing/state/stream/CLI boundaries
```

Run focused tests with `python -m pytest -q tests/test_minigrid_context_control.py`.
Physical fixture costs and scoped lint/format/type commands are in RESULTS.md.
Any future recipe or game readout must declare input roles, authority, exposure,
copy/physics/time/storage caps and outcome interpretation before execution.
Keep raw296 and internal298 interfaces distinct, and retain owned snapshots
through the existing stream/evaluator ports. Do not use final layouts to tune.
