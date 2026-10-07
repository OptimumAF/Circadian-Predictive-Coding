# ADR-0011: Separate attribution and practical vision tracks

## Context

The historical vision benchmark includes a linear backprop head and PC heads
on independent backbones. The matched fixed-feature route from ADR-0010
instead shares one frozen representation. Training throughput, capacity, and
accuracy from those routes answer different questions and must carry their
track identity into exported reports.

## Decision

The fixed-feature result and each head report explicitly record the
`frozen_shared_representation` track, frozen backbone, pretraining choice,
head type, and total/trainable parameter counts. A separate
`run_practical_backprop_benchmark` route trains an unfrozen ResNet and linear
head under `vision_end_to_end_backprop_v1`; it uses guarded validation and
opens final test only after training. Historical three-model reports retain
the `unmatched_reference` track. Multi-seed winner helpers require explicit
track metadata and reject mixed tracks; tuning exports retain the metadata.

For the fixed-feature track, each role's image-to-feature materialization has
its own seeded CPU RNG scope. This prevents augmentation or loader iteration
from depending on backbone initialization draws or head execution order. Head
initialization and circadian split noise use their existing explicit
generators. The route can run heads in a supplied order and records that
order. A tiny CPU reversal with forced sleep reproduced feature and trained
head hashes exactly; validation/test metrics matched within `1e-7` absolute
tolerance. This is a local reproducibility gate, not a model ranking.

## Alternatives considered

- Presenting end-to-end backprop as another fixed-feature head would mix
  feature learning with head learning and make attribution unclear.
- Renaming historical `BackpropResNet50` would break old report consumers;
  explicit track and head metadata preserve its identity without that change.
- Reseeding the process once before all loaders leaves augmentation tied to
  unrelated initialization draws. Role-specific scopes avoid that coupling.

## Consequences and open work

Frozen-feature timings exclude backbone extraction, whereas practical
backprop timing includes it. Both are reported descriptively; P1.8 still
needs fixed-data, wall-time, and capacity/compute budgets before a fair
comparison. P1.7 also remains open for the broader image-level reference
path, worker/GPU reproducibility, and any replay streams. No existing
historical output was rewritten.
