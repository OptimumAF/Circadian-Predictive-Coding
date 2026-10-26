# ADR-0577 — Keep TRAIN-standardized MiniGrid BP explicit

## Context

Accepted FIFO36 memory learns both known cue games. Native H32 BP's first blue
divergence is the cue-dependent turn; it fits51/54 TRAIN actions but fails that
game.210constant input channels dominate squared norm, suggesting a conventional
conditioning control. Earlier PC/CPC/encoding/kernel negatives remain valid.

## Decision

Reuse core population statistics fitted only on validated current TRAIN frames,
then bind immutable means/scales to an explicit new NativeLearner wrapper around
the original fixed BP head. Transform identically for training/prediction and own
the complete native state. Reuse the existing stream and actual game evaluator.
Keep original schemas/adapters/defaults unchanged. Do not refit on probe/final
data, clip values, tune scales, change doses or introduce an orchestration layer.

## Alternatives

Replacing default encodings would invalidate original comparisons. More width,
seed or dose searches do not isolate conditioning. New rollout frameworks would
duplicate owned frozen readouts. Existing FIFO memory remains a useful simpler
control; neural superiority is not required.

## Consequences

One fixed case improves TRAIN fit52/54 but loses the previously working red game;
both cue turns remain wrong. The variant stays opt-in and is not a successful
repair. An admission failure and one prospective same-case continuation retain
all costs within original cumulative resource caps. No robustness, generalization,
CPC, efficiency or release conclusion follows. Source/procedure/result self-review
is not independent acceptance. Next assess unfinished held-out game behavior with
the existing successful control rather than tune another normalization variant.
