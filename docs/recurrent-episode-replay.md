# Bounded complete-episode recurrent replay

The existing public-cue recurrent pilot now supports `recurrent_replay="recent"`
or `"coverage"` with explicit typed input and `recurring_training=True`.
Use `describe_protocol(...)` to inspect one arm, or
`describe_replay_comparison()` for the fixed pair. Describing a recipe constructs
no learner. `run_replay_comparison(episodes)` fits both controls before any
readout; two sequential single-arm calls do not provide that comparison gate.

Both controls retain at most six complete episodes (54 native rows, 256 KiB
encoded-array cap). Every eight incoming wakes they perform two separate
nine-row recurrent updates. Recent selects the latest blue/red pair; coverage
uses the existing rotating bank with six declared TRAIN context/cue keys.
Coverage selection has stronger released context metadata; model routing still
uses only the actual initial public observation. No probe/final data enters the
bank. Each update starts a complete independent episode.

The incoming A32/B32/C_train32/A32 sequence remains 128 updates/1,152 rows.
Replay adds 32 updates/288 rows, 16 updates per public-cue head. Logical
checkpoints 0/32/64/96/128 therefore have actual native clocks 0/40/80/120/160.
The evaluator checks both clocks and each head. Independent original wake,
replay and readout budgets retain their permanent charges and lifetime clocks.
Returned JSON is recorded evidence, not a resumable native checkpoint or a new
training authority. Full owned states remain private to the running pilot.

Why this: the preceding no-replay observation learned C while losing exact B
recall. Equal extra fitting on recent episodes helps distinguish retaining old
data from merely doing more SGD. The two new controls match steps/rows/per-head
exposure, with different selection and copy costs. Neither matches the saved
no-replay control's total computation.

`adapters/minigrid_recurrent_replay.py` composes the existing application
`StreamRehearsal`, buffers and episode codec. It checks causal selections,
original authority identities and the bounded preceding event/diagnostic
prefix. It provides no optimizer, persistence or alternate runtime. The
optional application boundary callback defaults to `None`; guarded replay
explicitly refuses continuation capture until a contract exists for its external
owner. Existing unguarded pause/resume remains supported.

Focused software checks are in `tests/test_recurrent_replay.py`; their scripted
learner demonstrates wiring and refusal behavior, not native mathematics or
game success. Native results, limitations and paid failures belong to
[issue 133](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/133)
and its fixed local protocol/artifacts. Unchanged native math evidence and the
saved no-replay result are reused. No circadian advantage, generalization or
release readiness follows from this integration.

For extensions, keep whole-episode boundaries, both clocks and matched per-head
exposures explicit; add a new fixed protocol before changing selection or dose.

The first paired native case failed at indented report publication after320
native updates/2880rows and196physics returns. Detailed outcomes and terminal
app preservation are unknown; it is not rerun. See the [evidence report](reports/balanced-recurrent-replay-20261025.md).
The existing paired CLI now emits compact bounded JSON through
`infra.local_result_json.encode_bounded_local_json`, preserving all values and
the original8/16MiB caps. `--describe` and the original pretty local writer
remain unchanged. Over-cap errors include the measured byte count; encoding
fails before opening the output path. This output repair has only model-free
fixture evidence and cannot recover the lost cohort.
