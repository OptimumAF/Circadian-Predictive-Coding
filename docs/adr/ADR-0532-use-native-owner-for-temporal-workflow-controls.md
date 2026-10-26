# ADR-0532 — Native temporal workflow with fixed snapshot controls

## Context

The accepted conventional temporal reference implements R5.5 observation-bound
tool/rule changes, but does not show behavior of the released native decision
model. Existing native delivery assumes support-routing examples/instructions.
Its learner already supports bounded owned snapshots without renewing work.

## Decision

Extend delivery with an explicit task validator and observation policy, and
readout with explicit instructions. Defaults retain the original routing task.
Compose the existing temporal corpus, delayed-label inbox, native learner and
probability evaluator in a new outer script. Train all36 single-request updates
before scoring. Capture four native states0/12/24/36 for learned/source0/frozenA
controls under each current catalog. Restore final36 only on successful scoring.
No new checkpoint format, owner, framework, dependencies or model objective.

Why this: current rules differ across phases, so reusing a v1 baseline score
under v2 would confuse masking/rule changes with learning. Fixed snapshot
controls answer that question without retraining or letting development scores
choose subsequent updates. The native36updates and perceptron12three-row passes
share row exposure, not mathematics, optimizer count or compute.

## Alternatives

Interleaved evaluation complicates causal isolation. New model instances add
unneeded load/copy cost. New replay, rank changes or dose search would change
the declared learning question before testing the existing implementation.

## Consequences

Permanent spent counters survive restores; private snapshots are bounded and
released at exit. Interrupted scoring stops without further mutation, records
the selected source and partial work, and does not claim final36 restoration.
Known union tools/vocabulary and concept-overlapping templates restrict claims.
Obsolete still-allowed choices count as errors; removed-tool masking alone is
not acquisition. All calibration/retention failures remain reported. Native
negative results are valid; no CPC or release conclusion follows from this case.

[Frozen protocol](../../artifacts/runs/native-temporal-stream-20261022/PROTOCOL.md).
