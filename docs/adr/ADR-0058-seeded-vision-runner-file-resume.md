# ADR-0058: Persist the seeded unmatched vision runner at training boundaries

## Context

The seeded unmatched vision benchmark trains three classifiers in a selected
order. Its circadian classifier can change width or reject a guarded sleep
event. P3.9c2a made one train loader replayable, but a process restart also
needs the completed classifiers, active classifier, retry gate, report counters,
and the exact development data used for those decisions. Final-test labels must
remain sealed until all training finishes.

## Decision

The public Python runner accepts an optional `VisionCheckpointStore`. The
application payload binds the full benchmark configuration, model order,
protocol, and raw train/guard/validation examples. It records detached complete
model snapshots and their hashes after each finished variant. During circadian
training it additionally saves the full classifier/backbone, seeded loader
cursor, retry gate, process and enclosing Torch random streams, learning
counters, and the wake/pre-sleep/post-sleep stage. Each save atomically replaces
one trusted local checksummed file through the infra store.

Resume validates the file, data roles, completed model reports and states, and
active circadian cursor/state on temporary objects before restoring process RNG
or entering training. The final-test loader is used only after all three models
have finished. Checkpoint writes and replay reconstruction are excluded from
the circadian active training duration; resumed wall time is not expected to
equal uninterrupted wall time. The file is local trusted pickle and its checksum
detects accidental corruption, not hostile modification.

At this decision point the file route was limited to seeded v3 on CPU. The
later ADR-0059 adds CPU v1/v2 continuation while retaining their original
shared streams. CUDA device-specific evidence remains open under P3.9c2b2b.

## Alternatives and consequences

Pickling live Torch classifiers failed because their runtime module references
are not serializable. Detached, typed model state reconstructs each classifier
and checks its structure before reuse. Saving only weights would lose the
circadian head's chemical, lineage, split-generator, and retry history. Saving
only the active model would retrain preceding variants and could re-open final
test labels. The complete ResNet-50 file can be large (about 270 MB in the
bounded CPU fixture), and a save after each wake batch adds local I/O. Repeated
sleep-boundary tests therefore use a tiny backbone while retaining the real
head and guarded-sleep logic; one integration case uses the real backbone.

## Evidence

`tests/test_vision_checkpoint_resume.py` covers real-backbone model-boundary
resume, reversed model order, terminal scoring without retraining, zero- and
two-worker stochastic augmentation, mid-wake and pre/post accepted/rejected
sleep, equal completed-model hashes/non-timing reports/next process draws,
sealed final test, and rejection of corrupt files or incompatible data,
configuration, order, model state, counters, loader cursor, and RNG before
training or process RNG mutation. The development log records the commands and
full quality gate. No baseline budget, seed-selection rule, or metric changed.
