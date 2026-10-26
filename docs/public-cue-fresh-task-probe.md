# Isolated probes for the public-cue recurrent learner

The existing `src.app.fresh_task_probe.run_fresh_task_probe` can use the current
two-head `PublicCueRecurrentLearner` directly. No new adapter, learner factory
or probe algorithm is required. Its opaque full checkpoint is copied before
external calls, restored on an owned fork, and updated only there. The caller
keeps its benchmark/deployed learner and original budgets separate.

Use caller-owned live state from the existing stream runner. The JSON reports
from earlier experiments are not resumable checkpoints. In particular, the
failed133cohort must not be reconstructed, refitted, rescored or retried.

```python
from src.app.fresh_task_probe import run_fresh_task_probe

# arm is the existing runner's protected live source/state/budget owner.
# Fix this checkpoint and all batches before reading probe results.
state = dict(arm["states"])[declared_checkpoint]
report = run_fresh_task_probe(
    arm["source"],
    state,
    probe_training,                 # 16 complete probe_train episodes
    probe_evaluation,               # two complete probe_test episodes
    budget=separate_probe_budget,   # fresh16-update lifetime, zero replay
    checkpoint_updates=(0, 4, 8, 16),
    training_readouts=old_train_pair,
    check_boundary=arm["guard"],
    decision_diagnostics=True,
)
```

Each packet must contain disjoint float64 raw148 feature and categorical K7
target arrays, complete episode metadata, unique row IDs and the exact declared
role. Evaluation contexts are unique. The first actual public observation
selects a head; application context names never route the model. Each native
batch is one complete episode, not a concatenation of different episodes.
Final evaluation has no route into this interface. Probe evaluation labels are
used only for probe diagnostics, not deployed updates or promotion.

Old TRAIN readouts perform no updates. The optional existing return contract
adds `return_training=two_train_templates` and `return_budget=separate_return_budget`.
It requires16fresh-task updates followed by16old-TRAIN updates on the SAME
disposable fork. Template values/transitions match their corresponding original
TRAIN monitors with distinct IDs and storage. Both phase budgets are born once,
retain their original clocks and charges, and are never reset. The top-level
status describes the fresh-task phase; inspect `return_phase.status` separately.

Logical probe/return counters start at0. Native counters keep the supplied
checkpoint's history. With16balanced updates per phase each native head gains8;
the full native state after return includes32additional updates. Preserve both
clocks and all rows. Caller boundary checks protect the original benchmark,
actor, checkpoint and budget bindings; the generic function does not invent
that outer authority. Native failures retain the completed prefix and known
unpublished work; rejected external mutations are not silently repaired.

The existing `endpoint_observer(learner, phase, boundary)` can read the SAME
disposable owner after each paid phase. Use its qualified synchronous read port
and final clock observation to check complete state before/after reading. It
grants no fitting, promotion, budget renewal or automatic game implementation.

Why this: existing generic ownership and role ports already cover this family.
Qualification is preferable to a duplicate probe wrapper. The existing MiniGrid
game reader now has [explicit public-cue endpoint admission](public-cue-probe-games.md)
through the same readonly kernel. Its sole native observation is incomplete;
software admission does not establish useful game performance.

Sixteen focused scripted cases passed on Windows/Linux, plus the affected return
case with qualified observer reads. They exercise full nonzero two-head state,
actor/benchmark isolation, role refusal, native/probe clocks and paid failures.
The literal packets and scripted uniform predictions do not demonstrate learning,
actual Tensor/native interoperability, game success, retention or recovery.
See [qualification evidence](../artifacts/runs/public-cue-probe-compatibility-20261025/RESULTS.md).
A separate fixed native protocol was subsequently executed once under136. Its
Freadout reached the original40-second deadline; return was not admitted. The
result remains failed-spent and cannot be rerun or reconstructed. Existing
negative results and broader requirements remain open.
