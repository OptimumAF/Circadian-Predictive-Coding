# ADR-0541: Fixed disposable CLM checkpoint probes

## Context

The accepted owned CLM probe boundary isolates warm weights and a fresh native
optimizer, but does not exercise new-task learning through the existing temporal
stream. The current main-stream return error and earlier negative evidence remain.
A probe must not change the deployed/benchmark learner or select later work.

## Decision

Add a default-off reserved-task option to the existing temporal CLI. An explicit
checkpoint callback runs a separate six-update probe at baseline/A/B/return.
The core declares the fixed marker mapping and distinct TRAIN/suffix-test roles;
the app composes existing consent, delayed inbox, logical clock and budget ports.
An outer adapter binds frozen features, proves source preservation and scores
cached rows only after all four probes and the main stream finish. Each probe
has independent consent, budget, receipts and permanent native work charges.

Why this: explicit checkpoint operations make paid work visible; filesystem-save
callbacks only persist facts. Four bounded child reports preserve ownership and
values through later main updates without an unbounded success/state chain.
The encoder keeps its existing16-sequence ceiling. Probe plus catalog support
or comparison is refused before loading; standalone support recipes are unchanged.

## Alternatives

A new runner would duplicate working inbox/budget/stream behavior. Widening normal
TRAIN to probe roles would erase a useful isolation boundary. Continuing source
optimizer state would answer a different question from fixed fresh-optimizer
learnability. Searching markers or doses after native observations would bias this
small exploratory study. These alternatives were not selected.

## Consequences

One shared frozen encoder16-template batch pays for benchmark and probe inputs.
Four disposable warm copies add24updates48row exposures and24choice reads, beyond
original12updates36rows. All native copies, work, failures, time and memory are
accounted. No probe install/export, replay, promotion, final evaluation or CPC
benefit is implied. Bare/suffix groups are declared, not empirical independence;
previous model use of tavrix and unknown pretraining novelty are disclosed.
Initial correctness is a ceiling rather than acquisition. Full R2.1/generalization/
reliable return and release requirements remain open regardless of this pilot.

[Protocol](../../artifacts/runs/clm-reserved-probe-20261022/PROTOCOL.md).
