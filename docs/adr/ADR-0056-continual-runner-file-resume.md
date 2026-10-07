# ADR-0056: Resume the continual NumPy runner across phases and seeds

## Context

The continual benchmark has a separate phase-A and phase-B training role,
local sleep intervals, a global sleep clock, three phase-A-frozen models
for retention scoring, and multiple seed reports accumulated into an
aggregate. The combined P3.9a circadian snapshot and P3.9c1a toy cursor
cannot recover all of those values. Repeating phase-A or a completed seed
would change update counts and could rescore held-out data.

## Decision

The public continual Python API accepts a checkpoint store and explicit
resume flag. The ordinary route keeps its existing phase helper calls.
The checkpointed route persists after each complete ordered model update,
before sleep, after sleep, at the A-to-B transition, and after committing
each seed result. The cursor names the seed, phase, phase-local epoch,
global epoch, next model, and sleep stage. The payload holds both mutable
baseline models, a full combined circadian/process snapshot, sleep report
counters, and detached phase-A copies of all three models. Prior seed
results and their data digests are carried forward without retraining.

Config and ordered seed-list identity bind the file. A current seed's
development-role digest covers both phases' exact training and validation
arrays; final-test arrays are not included in its training digest. Both
held-out tests are scored only after that seed completes A and B. A test
digest is recorded after scoring, then checked if a later resume reuses the
committed result. Previous completed seed data are regenerated and checked
before new training. The existing corrected/legacy protocol IDs and
per-seed split-hash reports are unchanged.

`TrustedLocalContinualCheckpointStore` uses a distinct type/header and the
same checksummed, replaceable local pickle mechanism as the fixed-feature
and toy adapters. Loading requires a trusted local file. The app validates
identity, counters, baseline arrays, frozen-A topology, and the current
circadian candidate before restoring process RNG or running an update.
Even a checkpoint after the final seed restores the saved process streams
before returning the committed aggregate.

## Alternatives and consequences

Replaying completed seeds would repeat learning and held-out scoring.
Saving only at the phase boundary would lose an in-progress sleep or an
ordered model update. The chosen transaction points recover exactly while
preserving the existing learning rules, metric definitions, seed list,
train/test roles, and report aggregation. Building phase-B development
roles before phase-A training on the checkpointed route permits their
identity to be checked at resume; its generators use local seeded RNG.
Checkpoint serialization is extra runtime overhead and is not part of an
equal-compute comparison. The CLI remains unchanged.

## Evidence

`tests/test_continual_checkpoint_resume.py` interrupts both phase-A and
phase-B structural/replay sleep in forward and reversed model order, a
partial B wake, the A-to-B transition, a committed seed, and a terminal
checkpoint. Resumed per-seed and aggregate metrics, baseline/frozen-A and
circadian state, work counts, and next process draws match uninterrupted
runs. Changed order, seed list, data, counters, frozen topology, and corrupt
bytes reject before training. A sealed role fixture proves both held-out
tests remain unopened until training finishes. The development log records
the full quality gate.
