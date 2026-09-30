# ADR-0141: Stage a matched gating pilot before the full mechanism matrix

## Context

P6.3 requires a controlled mechanism matrix with development seeds,
bounded cost, and reserved independent confirmation. The verified v14
periodic/no-sleep study changes replay exposure and final width together.
Its fixed manifest and prior final scores must not be rekeyed or used to
select a favorable new condition. The P6.2 corrected toy profiles are
historically tuned, descriptive, and lack a global final seal.

## Decision

Use a separate `continual_mechanism_gating_dev_v1` protocol for the first
small factor. Reuse v14's arrived source geometry, but train ordinary PC,
an exactly neutral shallow circadian PC control, and the same circadian
model with only its existing wake plasticity floor changed. Keep all arms
at width eight, with identical initial tensors, 24 full-batch wake updates,
matched train rows and latent work, and no sleep or replay. Require exact
ordinary/neutral parameter parity after every update. Score only
outer-selection development roles; reserve ten independent seeds and every
final role. Freeze a 240-update prelaunch cap, 120-second process limit,
source/manifest identities, and exclusive artifacts before publication.

Report explicit A-after-A, A-after-B, B-after-B, final mean task accuracy,
and signed forgetting. A better forgetting number caused by lower starting
A accuracy is not a retention benefit. Preserve every seed, null, and
negative result. Do not expand or choose a winner from final labels.

## Alternatives

- Treat fixed v14 periodic-minus-no-sleep as a gating ablation: rejected
  because it also adds replay and changes structural capacity.
- Change the v14 manifest in place: rejected because it would invalidate
  its frozen provenance and saved result identity.
- Begin with a broad multi-factor sweep: deferred until a narrow parity,
  role-seal, work, and reproducibility gate passes.

## Consequences

The first factor can establish whether gating acts and whether it changes
development outcomes under equal work/capacity. It cannot establish a
full circadian or final-test advantage. The remaining P6.3 matrix still
needs matched replay, structure, homeostasis, schedule, backprop, and
planned-width controls plus independent confirmation. The pilot app adds
no new learning rule and leaves all v9–v14 bytes and manifests unchanged.
