# Current backend capability matrix

Verified against HEAD `28e71ee` on `master`, including the retained local
checkout changes, on 2026-10-06. The Phase 0
[feature inventory](feature-inventory.md) describes the reviewed baseline
at `8793c49`; this matrix records the current research boundary. "Both"
means each backend has an implemented mechanism. Numerical comparison
requires the fixture and protocol scope described below.

| Capability | NumPy binary network | Torch multiclass head | Comparison boundary |
|---|---|---|---|
| Wake chemistry, adaptive plasticity, supervised-error reward scaling | Available in `CircadianPredictiveCodingNetwork` | Available in `CircadianPredictiveCodingHead` | Objectives, output shapes, default dtypes, and output updates differ. The controlled binary/two-logit fixtures in [test_backend_parity_boundaries.py](../tests/test_backend_parity_boundaries.py) cover selected forward/hidden/chemistry and separate structural operations; the free Torch output columns produce a different margin update. General numerical equivalence is unproven. |
| Periodic and current plateau/variance adaptive sleep | Available; successful `train_epoch` calls set the wake clock | Available; successful `train_step` calls set the wake clock | Runner epoch progress is separate from successful wake calls. A tick is a full epoch in the NumPy studies and a batch step in the Torch head; NumPy replay does not advance the wake clock. Adaptive energy and variance use each backend's own computations. Raw tick counts cannot establish equal trigger rates. |
| Chemical reset, homeostasis, split, and prune component switches; typed event and stable neuron IDs | Available in `sleep_mode="components"` | Available in `sleep_mode="components"` | Default `legacy` retains structural-budget gating; `disabled` refuses sleep. NumPy preflights external-policy proposals and can schedule gradual removal. Torch plans post-split prune candidates on detached state and removes selected neurons during sleep. Candidate sets and removal timing differ. |
| Sleep replay consolidation and opt-in FIFO/seeded bottom-k example and byte caps | Available in `circadian_predictive_coding.py` and `replay_retention.py` | Unavailable in the Torch head; its typed replay work and replay clock are zero | The v9/v14 shared replay and matched PC/backprop replay controls are NumPy-only application protocols. Retention policy names do not define a Torch implementation. |
| Four-role arrived A→B evaluation, guarded matched replay, global final-role seal | Available for the v14 NumPy study | No Torch v14 route | V14 trains binary NumPy backprop, ordinary PC, and circadian PC. All six trials pass the global preflight before final-role release. Its recorded result contains no Torch score or cross-backend conclusion. |
| Full-state snapshots and trusted local continuation | NumPy toy and continual routes, plus checked fixed-v14 trial-prefix resume | Torch head/classifier snapshots and fixed-feature/vision routes | Checkpoint formats, device scope, and runner protocols differ. V14 continues only from complete, unscored trial prefixes with checked source/environment/config/protocol/capture and checkpoint identities; it has no mid-trial cursor. |

The fixed v14 train-only and scored JSON files have no embedded `backend`
field. Their immutable SHA-256 values and explicit `array_backend: numpy`
scope are bound in [result-backend-metadata.json](result-backend-metadata.json).
This sidecar preserves the already recorded v14 artifact bytes and protocol
IDs. It is provenance metadata, not an additional score or a Torch run.

## Implemented NumPy v14 artifact and continuation scope

The completed Phase 5 contracts are available for the fixed NumPy study:

- [P5.1 run manifests](versioned-run-manifest.md) bind execution/source,
  resolved configuration, algorithm/protocol versions, seeds, dataset roles,
  runtime, hardware, precision, and the original raw payload hashes.
- [P5.2 recorded projections](structured-observation-audit.md) expose saved
  sleep, topology, replay, guard, role-access, and final records as JSONL and
  derived CSV. The raw v14 files lack per-epoch wake metrics. Separate,
  opt-in [measured wake observations](measured-wake-observations.md) supply
  432 genuine method/epoch diagnostic rows in the fixed study. Observation
  directories are published after the completed source bundle verifies;
  final scoring has already passed its global gate.
- [P5.3 atomic publication](atomic-artifact-publication.md) stages and checks
  complete bundles and observation directories before a same-volume rename.
  [Checked trial-prefix resume](v14-checked-resume.md), requested with
  `--resumable` or `--resume`, validates the saved prefix before further
  training and preflights all six trials again before final release.

Checkpoint loading is for trusted local files in the same environment.
An interrupted trial is retrained; cross-version checkpoint compatibility
is unclaimed. Atomic directory publication establishes completed-result
visibility; full filesystem-metadata crash durability is outside that
contract. These routes preserve the original baselines, seeds, metrics,
protocol IDs, and negative or mixed findings.

Execution attribution and saved-byte checks do not close the later
prospective seed/source/runtime admission gates. The
[model card](model-card.md) and [evaluation protocols](evaluation-protocols.md)
describe the interpretation and information-access limits.

**Why this scope:** The implemented Torch head already supports chemical
reset, homeostasis, splitting, and pruning. Torch replay and a Torch v14
application route remain absent. A later prospective port must specify
roles, matched baseline replay, work and capacity caps, backend semantics,
and a separate protocol identity before adding a mechanism or experiment.
