# ADR-0055: Resume the NumPy toy comparison from a trusted file

## Context

The P3.9a combined checkpoint restores only the circadian model, process
random streams, and an outer cursor. The toy runner also owns two matched
baselines, three metric histories, sleep counters, a configurable model
order, and deterministic train/validation roles. Sleep can split the model
and use replay after the last model update in an epoch. A restart cannot
repeat any completed update or sleep without changing the comparison.

## Decision

The actual three-model toy runner accepts a checkpoint store and explicit
resume flag. It saves after each completed model update except the final
update of an epoch, immediately before sleep, and after sleep. A wake
cursor names the next model in the configured order; before/after sleep
positions determine whether the sleep transaction is pending or complete.
The payload includes detached backprop and ordinary PC models, all three
metric histories, sleep counters, and the combined circadian/model/process
snapshot. The identity binds all runner settings, the exact train and
validation arrays, protocol, and model order. It does not inspect the
final-test role when forming checkpoint identity. Baseline topology,
arrays, traffic steps, and report cursor are checked before restoring the
fresh circadian model or process RNG.

`TrustedLocalToyCheckpointStore` uses a distinct type/header and the same
checksummed, replaceable local pickle file mechanism as the fixed-feature
store. Its checksum catches accidental damage; only trusted local files
may be loaded. The durable route explicitly requires the stateless default
adaptation policy because arbitrary external policies may own unrecorded
state. The regular non-checkpoint route still supports those policies.

## Alternatives and consequences

Retraining the baseline models to the resume cursor would perform extra
updates and could silently change future results. Saving only at an epoch
boundary would fail to describe an interruption between the ordered model
updates. The chosen cursor makes both cases exact without changing the
learning rules, data roles, sleep schedule, metrics, or seed policy.

Toy checkpointing is exposed through the app API; the existing CLI remains
unchanged. Continual phase-order/frozen phase-A state and whole-image
classifier/loader state remain P3.9c1b and P3.9c2. P3.9c1 and P3.9 stay
open until their parent criteria pass.

## Evidence

`tests/test_toy_checkpoint_resume.py` interrupts the real runner at a
partial model-order wake cursor and before/after a structural replay sleep
in both forward and reverse order. Fresh-call results, all baseline and
circadian state values, sleep counts, and next Python/NumPy draws match
uninterrupted training. Changed order/data, corrupt bytes, malformed
counters/baseline arrays reject before training. Both validation and
legacy protocols resume; sealed final-test roles remain unopened through
interruption. The development log records the full quality gate.
