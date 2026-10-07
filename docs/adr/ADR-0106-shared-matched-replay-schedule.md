# ADR-0106: Freeze a prediction-independent schedule before replay-capable controls

## Context

The v8 FIFO/reservoir comparison changes circadian retention only. Its PC
and backprop baselines do not replay. Circadian can select replay rows using
its own prediction error and training-label class balance; copying that
selection independently into other models would expose different examples.
P4.4 requires the same retained rows, sampling policy, cap, and accounted
replay budget for all three methods without changing the existing v8 result.

## Decision

Introduce the separate `continual_matched_replay_schedule_v9` planning
protocol. It requires an arrived-role v6 setting, an ordered seed list, one
FIFO or seeded bottom-k retention policy, the same example and copied-array
byte caps as the circadian path, a positive per-sleep update limit, and an
explicit `newest_retained_v1` sampler. The new protocol requires circadian
priority sampling, replay class balancing, and adaptive sleep triggering to
be disabled; periodic sleep is forced. These choices remove model-specific
predictions and class labels from replay selection. Retention content IDs
still include the observed training label, as in the existing bounded
buffer; no guard, outer-selection, or final-test role is a source.

`SharedReplayBuffer` copies float64 labeled rows once, deduplicates by
content ID, applies the declared FIFO or seeded bottom-k eviction under both
caps, and selects the newest retained rows in retention order. The public
retention snapshot keeps its historical sorted-ID convention. A separate
ordered-ID field records the actual sampling order. Every method receives
the same selected IDs and can request private array copies.

The app session opens Phase A training at construction and opens Phase B
only after all declared A wake epochs have been acknowledged. It validates
the complete manifest before source access and before every schedule
advance, checks the arrived train-row digest before mutation, and rejects a
non-train source role. Each periodic boundary records selected IDs, retained
IDs/bytes, and **planned** examples, optimizer updates, and PC/circadian
inference iterations separately. No model update or score is performed in
this gate. Actual replay-capable training, checkpoint continuation, and
outcome comparison remain P4.4 work.

## Alternatives

- Reuse v8's circadian prediction-prioritized sampler. Its chosen rows can
  depend on circadian weights and need not be a matched baseline exposure.
- Give each baseline a private replay buffer. Even identical caps would not
  prove identical retained rows or sampling order without a shared schedule.
- Call equal epoch counts a replay control. That omits replay examples,
  optimizer updates, and latent-inference work.

## Consequences

Fixed A→B tests compare the shared buffer's retained and selected IDs to
the existing unprioritized circadian buffer for FIFO and seeded reservoir,
including repeated rows and independent count/byte limits. Arrived-role
tests cover both model orders, explicit B arrival, final and decision-role
sentinels, manifest/role changes before a schedule update, and detached
copies for each consumer. The local two-seed/two-policy schedule artifact
contains no model scores or winner. A later runner must account for actual
updates and separately report PC and circadian inner optimization work.
