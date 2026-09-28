# Phase 0 feature inventory

This inventory describes the fetched baseline at commit
`8793c49ee4f9f8b07649e8db6571ed53746a9a06`. It records existing behavior
before the corrected evaluation protocol in [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md).
“Implemented and tested” means the named behavior has relevant passing test
coverage in the Phase 0 environment; it does not establish a scientific
benefit or parity between backends.

## Verification scope

The default Python 3.9 environment ran the full suite with **37 passed, 0
skipped**. A Python 3.11 base install ran **24 passed, 2 skipped modules**
because Torch and torchvision are optional. A separate Python 3.11 CPU Torch
run passed the **7 head tests** in `tests/test_resnet50_variants.py`. Thus the
NumPy path is covered without Torch; the Python 3.11 base result alone does
not validate the vision benchmark. See [development-log.md](development-log.md)
for exact validation commands and environment details.

## Existing mechanisms

| Mechanism | Status and evidence | Boundary or limitation |
|---|---|---|
| Binary backprop and ordinary predictive coding, including multiple hidden layers | **Implemented and tested**: `tests/test_backprop_mlp.py` and `tests/test_predictive_coding.py` cover learning and multilayer construction. | NumPy only. Ordinary multilayer PC relaxes every hidden state. |
| NumPy circadian wake updates, chemical gating, bounded saturation, importance-sensitive plasticity, and supervised-error reward scaling | **Implemented and tested** for the named behaviors in `tests/test_circadian_predictive_coding.py`: `test_should_build_chemical_and_learn_with_circadian_predictive_coding`, `test_should_keep_chemical_bounded_with_saturating_updates`, `test_should_reduce_plasticity_for_high_importance_when_adaptive_sensitivity_is_enabled`, and `test_should_scale_learning_rate_by_reward_signal_for_easy_vs_hard_batches`. | “Reward” scales learning from supervised batch difficulty; it is not an environmental reward. NumPy circadian multilayer learning relaxes the final adaptive state and updates preceding layers by propagated gradients, unlike ordinary multilayer PC. |
| NumPy sleep trigger, adaptive split/prune budget, split and prune, zero-noise function-preserving split, cooldown/hysteresis, gradual prune, and external adaptation policy | **Implemented and tested** by corresponding direct tests in `tests/test_circadian_predictive_coding.py`: `test_should_trigger_adaptive_sleep_when_plateau_and_chemical_variance_are_high`, `test_should_expand_sleep_budget_when_plateau_and_chemical_variance_are_high`, `test_should_split_busy_neurons_and_prune_idle_neurons_during_sleep`, `test_should_preserve_function_on_split_when_split_noise_is_zero`, `test_should_respect_split_cooldown_and_hysteresis_between_sleep_events`, `test_should_gradually_prune_marked_neurons_when_decay_is_enabled`, and `test_should_apply_external_neuron_adaptation_policy_during_sleep`. `tests/test_numpy_proposal_preflight.py` and `tests/test_numpy_builtin_proposal_preflight.py` verify external and built-in preflight and overlap priority; `tests/test_numpy_neuron_lineage.py` verifies stable IDs across repeated shape changes and restore. | `NeuronAdaptationPolicy` is used by NumPy sleep. Structural adaptation is limited to the final hidden layer. Repeated noisy topology changes and removed-unit telemetry need later tests. |
| NumPy replay snapshot selection | **Implemented and tested** for prioritized, class-balanced selection by `test_should_select_balanced_high_priority_replay_snapshots_when_enabled`. | The deque holds whole input/target batches, with a batch-count limit. Replay training occurs during sleep, but its effect, old-task survival, distinct-example count, and byte use lack focused tests. |
| Torch multiclass predictive-coding and circadian heads | **Implemented and tested** for head split preservation, cooldown, importance-sensitive plasticity, adaptive trigger/budget, reward scaling, in-memory snapshot restoration, detached post-split proposal preflight, and stable lineage by `tests/test_resnet50_variants.py`, `tests/test_torch_builtin_proposal_preflight.py`, and `tests/test_torch_neuron_lineage.py`. A small benchmark/report smoke is in `tests/test_resnet50_benchmark.py`. | Optional Torch/torchvision dependencies. The smoke validates report production, not matched learning rules or held-out evaluation. Torch retains parent/child post-split prune eligibility and has no replay, gradual prune, or `NeuronAdaptationPolicy` integration. |
| Sleep-event stable identity evidence | **Implemented and tested** by `tests/test_sleep_event_lineage.py`: executed NumPy/Torch sleep results hold immutable lineage before and after the event, while skipped results leave the optional fields empty. The snapshots expose births and actual removals even when net width is unchanged. | Result indices remain positional. Explicit proposed/scheduled/actually removed status and richer trigger/guard telemetry remain P3.6/P3.10. |
| Isolated split function preservation | **Validated** by `tests/test_split_function_preservation.py`: NumPy float64 predictions and Torch float32 logits remain within dtype tolerance after one and repeated zero-noise splits, with duplicated incoming paths and conserved parent/child outgoing rows. Seeded nonzero output perturbations are checked separately. | Whole sleep may also run replay, pruning, and homeostasis; these isolated checks do not measure predictive benefit. |
| Prune metadata alignment and capacity | **Validated** by `tests/test_prune_metadata_alignment.py`: immediate NumPy/Torch masks retain distinguishable survivor values in every adaptive vector and downstream tensor; NumPy gradual marks reserve minimum width and finalize during wake, with clocks and dimensions checked. `tests/test_prune_outcomes.py` covers typed stable-ID requests, schedules, removals, pending IDs, and restore continuation. | Existing application `total_prunes` fields remain historical request counts. Richer application status aggregates remain P3.10. |
| Dual chemical state, adaptive threshold ranking, homeostatic weight controls, replay consolidation outcome, and the causal benefit of benchmark rollback | **Implemented; causal benefit unverified** by isolated ablations. Guard acceptance and atomic recovery have focused correctness tests in `tests/test_guarded_sleep_atomicity.py`. | A passing correctness suite does not show that each mechanism improves results. |
| Vision benchmark evaluation and model comparison | **Incomplete** for scientific comparison. `src/infra/vision_datasets.py` provides train/test loaders; `src/app/resnet50_benchmark.py` uses the test loader during early stopping and sleep rollback. Backprop has a linear head; PC/circadian have hidden heads, and each model constructs its own backbone. | Historical vision results are legacy/test-informed. The current benchmark is runnable but does not satisfy split isolation, matched-head, shared-backbone, or model-order-invariance gates. |
| Independent sleep components and atomic recovery | **Component selection, trigger clocks, in-memory snapshots, core/guard atomic recovery, and an operational retry gate implemented.** Opt-in `components` runs NumPy chemical reset/replay/homeostasis/split/prune and Torch chemical reset/homeostasis/split/prune independently, including nonstructural work with zero structural budgets. `legacy` preserves budget-gated behavior; `disabled` is a true no-op. A shared runner decision distinguishes attempts from performed events, typed epoch progress separates runner and core clocks, and changed-width component history restarts. Snapshot coverage is in `tests/test_numpy_full_snapshot.py`, `tests/test_torch_full_snapshot.py`, and `tests/test_torch_classifier_full_snapshot.py`; `tests/test_atomic_sleep_core.py`, `tests/test_guarded_sleep_atomicity.py`, `tests/test_sleep_retry_policy.py`, and `tests/test_sleep_retry_runners.py` cover rollback, cooldown, and seeded continuation. `tests/test_combined_circadian_checkpoint.py`, `tests/test_fixed_feature_checkpoint_resume.py`, `tests/test_toy_checkpoint_resume.py`, and `tests/test_continual_checkpoint_resume.py` cover combined state and trusted-file CPU fixed-feature, NumPy toy, and NumPy continual resume. | Torch has no replay. Legacy keeps historical trigger/history semantics. NumPy and Torch head snapshots cover model-owned state and local RNG; the Torch classifier has a separate full-state API while its frozen-backbone guard remains head-only. The circadian route owns no optimizer/scheduler/scaler. CUDA, memory, capacity, and whole-image durable resume remain open under P3.9. |
| Numeric learning contract and backend parity | **Incomplete**. NumPy reports binary cross-entropy plus a hidden penalty; Torch reports squared softmax residual plus a hidden penalty while its output update uses a cross-entropy-like residual. | No finite-difference gradient checks, common NumPy/Torch fixture, or matched deeper no-circadian control. Prediction APIs are feedforward; latent relaxation is a training operation. |
| Iterative serving inference, structural adaptation in earlier layers, and a biological phase oscillator | **Proposed** research extensions. | None is required for the first reliable evaluation protocol. |

## Configuration and artifact paths

At the baseline commit, `src/adapters/cli.py` maps **26 of 67** NumPy
`CircadianConfig` fields; all fields remain available through the Python API.
The continual-shift CLI exposes profile and scenario settings, with circadian
policy presets hardcoded in `scripts/run_continual_shift_benchmark.py`.
`src/adapters/resnet_benchmark_cli.py` maps all **99**
`ResNet50BenchmarkConfig` fields, and the app maps all **58** Torch
`CircadianHeadConfig` fields. These counts describe wiring, not behavioral
validation.

Existing exports are `--output-file` text from the continual-shift script,
`--output-prefix` JSON plus per-seed and summary CSV from the multi-seed
ResNet script, fixed JSON paths from the Pareto and policy-sweep scripts,
and PNG/GIF/HTML from the figure generators. The NumPy core and Torch head
expose in-memory snapshot/restore helpers. The CPU fixed-feature
matched-head, NumPy toy, and NumPy continual routes have trusted local
checkpoint files; whole-image and device/memory routes remain open under P3.9.
Historical exports and their limits are indexed in
[historical-benchmark-provenance.md](historical-benchmark-provenance.md).
