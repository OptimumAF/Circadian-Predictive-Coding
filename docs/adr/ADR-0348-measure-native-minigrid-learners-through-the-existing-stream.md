# ADR-0348: Measure native MiniGrid learners through the existing stream

## Context

Actual MiniGrid cue/key/door demonstrations and native148 learner ports are
accepted. Scripted success does not establish learned acquisition, retention
or recovery. Earlier custom-game negatives remain valid separate evidence.

## Decision

Compose six TRAIN public-teacher traces through the existing delayed stream
using existing BP, PC, neutral CPC, active CPC and recurrent BC. Freeze seed17,
A32/B32/C_train32/A32, original objectives and raw-first-argmax action law.
Retain owned0/32/64/96/128 checkpoints. Complete all training before raw/live
readouts; final D is constructed only afterward without demonstration labels.
Use fresh recurrent activation per live episode and guard complete native state,
original budgets/receipts and preceding observed evidence. Preserve failure work.

Why this: the smallest useful next outcome is an actual learned-behavior test
using the verified game and existing learner/stream boundaries. Independent
review and small fixed resource caps precede the sole native case.

## Alternatives

Another runner duplicates existing authority and accounting. Masking/repairing
actions changes the policy being tested. More seeds/doses or tuning from final
D would weaken the frozen comparison. None is part of this increment.

## Consequences

Raw teacher fit and autonomous success remain distinct. Acquisition premises
must hold before interpreting forgetting/recovery. Equal rows/updates are not
equal computation; recurrence adds parameters and unroll work. A single fixed
case establishes neither equivalence nor broad generalization/CPC advantage.
Future learning changes require a new TRAIN-only protocol and must preserve
this case and final isolation. See docs/minigrid-learning.md and the frozen
artifacts/runs/r42-minigrid-learning-20261016/SCOPE.md.

## Measured readout implementation follow-up

After the first180s budget failure, synthetic native measurement found repeated
Torch resolution in recursive equality and repeated serialization of accumulated
readouts. Resolve Torch once per comparison; hold completed readouts as private
immutable payloads and check their public slot identity at every existing boundary.
Decode only private captured bytes after work, keep decoded objects private through
the last readback/guard, then restore the original output schema. Untrusted public
replacement bytes are never decoded. Keep every native state/receipt/budget/
distinct-copy check and pending-record validation. No cross-call cache is added.

Why this: reduce repeated work without discarding fields or admitting changed
evidence.478 unique scoped cases pass. A fresh original-cap native case still
failed and covered fewer episodes; the microbenchmark improvement establishes
no whole-run speed benefit. Both failures remain, and full readout completion
stays open. See r42-minigrid-readout-cost-20261016/RESULTS.md.
