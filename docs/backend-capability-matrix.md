# Current backend capability matrix

Verified against the active checkout at `5134a17` plus the local P4.8b2
changes on 2026-09-29. The Phase 0 [feature inventory](feature-inventory.md)
describes the reviewed baseline at `8793c49`; this matrix records the
current research boundary. "Both" means each backend has an implemented
mechanism, not that its learning rule or numerical result is equivalent.

| Capability | NumPy binary network | Torch multiclass head | Comparison boundary |
|---|---|---|---|
| Wake chemistry, adaptive plasticity, supervised-error reward scaling | Available in `CircadianPredictiveCodingNetwork` | Available in `CircadianPredictiveCodingHead` | Their objectives, output shapes, default dtypes, and output updates differ; only the narrow fixture in `tests/test_backend_parity_boundaries.py` is a cross-backend numerical check. |
| Periodic and current plateau/variance adaptive sleep | Available; `train_epoch` calls set the wake clock | Available; `train_step` calls set the wake clock | A clock tick is a full epoch in the NumPy studies and a batch step in the Torch head. Trigger rates cannot be compared by raw tick count. |
| Chemical reset, homeostasis, split, and prune component switches; typed event and stable neuron IDs | Available | Available | Structural planners differ: NumPy preflights an external policy and can schedule gradual removal; Torch considers post-split prune candidates. A matching feature name does not imply the same candidate set. |
| Sleep replay consolidation and opt-in FIFO/seeded bottom-k example and byte caps | Available in `circadian_predictive_coding.py` and `replay_retention.py` | Unavailable in the Torch head; its typed replay work is zero | The v9/v14 shared replay and matched PC/backprop replay controls are NumPy-only application protocols. |
| Four-role arrived A→B evaluation, guarded matched replay, global final-role seal | Available for the v14 NumPy study | No Torch v14 route | V14 trains binary NumPy backprop, ordinary PC, and circadian PC. No Torch score or cross-backend conclusion is encoded by its result. |
| Full-state snapshots and trusted local continuation | NumPy toy and continual routes | Torch head/classifier snapshots and fixed-feature/vision routes | Checkpoint formats, device scope, and runner protocols differ. V14 does not claim checkpoint continuation. |

The fixed v14 train-only and scored JSON files have no embedded `backend`
field. Their immutable SHA-256 values and explicit `array_backend: numpy`
scope are bound in [result-backend-metadata.json](result-backend-metadata.json).
This sidecar preserves the already recorded v14 artifact bytes and protocol
IDs. It is provenance metadata, not an additional score or a Torch run.

**Why this scope:** The next planned work is the Phase 5 artifact contract,
not a predeclared Torch replay experiment. Porting replay or the v14 runner
would create a new mechanism and evaluation protocol without a matched
hypothesis. Torch consolidation remains intentionally absent until a later
prospective experiment specifies roles, baseline replay, work and capacity
caps, backend-specific semantics, and a separate protocol identity.
