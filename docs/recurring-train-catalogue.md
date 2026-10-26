# Bounded recurring TRAIN catalogue

The existing pure `compose_arrived_cue_catalogue` can plan an explicitly bounded
recurring schedule. It allocates independent raw observation/target arrays and
records original source spans, arrival clocks, replay exposures and memory.
The existing learner and delayed stream still own fitting and label release.

```python
from src.adapters.minigrid_train_catalogue import compose_arrived_cue_catalogue

# episodes: an owned tuple of complete, distinct, nine-row raw296/K7 TRAIN episodes
plan = compose_arrived_cue_catalogue(
    episodes, retained_per_cue=3, max_arrivals=10
)
```

`max_arrivals` accepts exact integers6 or10. The default6 retains the original
admission and permits retention1 or2; explicit10 additionally permits3 earlier
packets per public cue. Default retention remains1. Booleans and ambiguous
bounds fail before source admission. Existing callers and returned schema are
unchanged. There is no new command, environment variable or dependency.

Why this: four contexts followed by a return require ten cue-specific arrivals
and three previous packets per cue. A finite opt-in enables that schedule while
preserving the original six-arrival protocols and default resource allowance.

Each wake joins only its current TRAIN packet and previously arrived packets
for the same visible public cue, oldest first. Labels arrive one tick after
observation. Source context/name strings do not route the bank. After the wake,
FIFO eviction retains at most three packets/cue. Role validation rejects
`probe_train`, `probe_test` and `final_test`; callers must supply original TRAIN
data, never relabel probe/final records or promote a private learner. The entire
finite schedule is validated privately during planning; it is not a live replay
service and does not expose future labels to an earlier wake. Existing source
witnesses reject changed or replaced original records during composition.

The opt-in plans at most36 rows per fit and54 bank rows. Ten balanced
A/B/C_train/G_train/A arrivals plan90 original plus162 repeated exposures:
252 fit rows,598752 bytes of owned input arrays,85536 peak joint-template bytes
and256176 bank observation/target float64 equivalent bytes. Ten single-cue
arrivals plan306 fit rows,216 replay exposures and27 peak bank rows. These are
planned exposures, not executed fits. The bank equivalent counts both raw296
views plus one target per row; it excludes Python metadata, rewards, native
learner copies and allocator overhead. Template, input and bank costs remain
separate. Cumulative owned input arrays are checked against the existing16MiB
limit before allocating the next joint template; the learner's128-row limit is
unchanged.

Focused verification:

```powershell
.venv/Scripts/python.exe artifacts/runs/recurring-train-catalogue-20261024/launch.py tests windows-tests
.venv/Scripts/python.exe artifacts/runs/recurring-train-catalogue-20261024/launch.py tests windows-repair -k bound_single_public_cue_fifo
.venv/Scripts/python.exe artifacts/runs/recurring-train-catalogue-20261024/launch.py tests linux-tests
.venv/Scripts/python.exe -m ruff check src/adapters/minigrid_train_catalogue.py tests/test_recurring_train_catalogue.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/minigrid_train_catalogue.py tests/test_recurring_train_catalogue.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir NUL src/adapters/minigrid_train_catalogue.py tests/test_recurring_train_catalogue.py
```

Those are the recorded commands, not permission to rerun a spent allowance.
Windows qualified39 distinct cases across an initial38-pass/one-invalid-fixture
failure and the corrected affected case; Linux final39 passed. No fits, model
predictions or native games ran. See [results](../artifacts/runs/recurring-train-catalogue-20261024/RESULTS.md)
and [issue122](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/122).
This is software planning acceptance with self-review, not independent review,
demonstrated adaptation, comparative advantage or release qualification.

## Ordinary chronological TRAIN integration

The existing runner now accepts a separate explicit recurring mode:

```python
from scripts.run_minigrid_linear_fit_control import run_context_linear_stream

report = run_context_linear_stream(
    episodes, retain_arrived=True, recurring_catalogue=True
)
```

Here `episodes` contains eight original complete TRAIN sources in exact order:
A blue/red, B blue/red, C_train blue/red, G_train blue/red. IDs are
`minigrid-source-{context}-{cue}`. Each source has nine raw296/K7 transitions;
the composer enforces alternating visible public cues. `recurring_catalogue`
must be an exact bool; True requires retention and cannot combine with
`abc_catalogue=True`. No role conversion or private-model promotion is performed.
These names identify supplied sources; the API does not establish their native
layout provenance or collect them. In particular, recorded F probe data cannot
be substituted or relabeled as G_train.

The schedule is fixed before learning: A/B/C_train/G_train/A, ten cue-specific
updates, label delay1. Native fit/row counters reconcile with the paid receipt
prefix; the original stream owns delayed application. Checkpoints are
0/2/4/6/8/10, corresponding to0/18/54/108/180/252 fitted rows. All ten fits finish
before48 detached TRAIN monitor calls (432 rows). Scores cannot change the
schedule, choose probes, trigger more learning or select a favorable endpoint.
The final observer receives the actual10-update/252-row endpoint only after
successful receipt capture. A later failure preserves earlier receipts and
unscored states, attempted work and unknown failure costs; it does not renew the
original ten-update/90-second/512MiB lifetime allowance.

The reported `catalogue.committed_rows` separates90 arriving from162 replayed
rows. These162 rows are fitted jointly inside ordinary updates; the separate
consolidation-event replay counter remains0. Do not interpret that0 as no replay.
Input/template/bank memory costs retain the scopes above. Reports expose TRAIN
acquisition, old-task retention and return separately from native success and
from heldout/generalization or matched-compute claims.

Existing A/B/A replacement/retention modes, ABC mode and first-party command
defaults remain unchanged. This is an API option; no new command, environment
variable or dependency. The current first-party command still uses six arrivals.
The saved affine viewer has no claimed support for this new report definition.

Focused commands and exact costs/failures:
[issue123 evidence](../artifacts/runs/recurring-affine-stream-20261024/RESULTS.md),
[protocol](../artifacts/runs/recurring-affine-stream-20261024/PROTOCOL.md),
[ADR-0634](adr/ADR-0634-recurring-affine-stream.md).
The literal software fixtures do not demonstrate native fourth-task adaptation.
The separate [original G workflow](ordinary-fourth-task.md) now qualifies that
source and its ordinary learning/native G/A observation through existing ports.
This pure/integration fixture result remains distinct from that actual case.
Reserved F/probe and D/final separation and all failed120/108 evidence remain;
no native/comparative allowance or retry follows from this API alone.
