# ADR-0562 — Compare native task controls through one live stream

## Context

Saved BP learns a new input context while retaining both old tasks in the fixed
ticket sandbox. Original short native runs failed to fit the old tasks. Peer
behavior at the working BP exposure is still unknown; no CPC advantage follows
from implementation correctness.

## Decision

Reuse original teacher/task helpers and native PC/neutral/gated CPC adapters in
one live AttributionStream per peer, with the same 96 old +8 new +8 return
six-row batches, seed, geometry and endpoint metrics as saved BP. Reuse BP
evidence without training it again. Freeze a single serial cohort and retain
every poor fit and failure. Check common initial weights/data and exact
PC-neutral parameter, diagnostic, teacher and task parity.

Why this: composition gives the missing comparative task evidence without new
state readers, orchestration, promotion or another audit sequence.

## Alternatives

- Add generic CPC checkpoint decoding: unnecessary when peers can stay live.
- Repeat BP warmup: wastes already accepted training and diagnostic work.
- Tune peer schedules or select successful seeds: changes the fixed question.
- Infer retention from loss alone: full task failure can coexist with lower loss.

## Consequences

Native objectives and compute differ. The dose was informed by BP development;
it does not establish equally optimized models. BP used a restored fresh runtime
for adaptation; peers use one runtime, with distinct stable actor/event clocks
that are not learner inputs. Initial competence, transfer, acquisition and
return are reported separately. Scoped self-review is not independent review.
