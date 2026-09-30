# Experiment entrypoint configuration audit (P5.4c)

This audit follows the experiment commands actively documented in the README
and `docs/modules/adapters.md` as of 2026-09-29. A fixed scientific protocol
has one resolved manifest and refuses setting overrides. A descriptive CLI
may be configurable, but must validate input before training and save every
setting it actually used. Operational projectors and verifiers consume saved
artifacts; they do not select experiment settings.

| Entrypoints | Configuration contract | Saved identity | Audit result |
|---|---|---|---|
| `run_versioned_v14_bundle.py` | Typed `fixed-v14` preset; explicit `--preset fixed-v14` or the same default. Argparse and the app resolver reject unknown presets/settings before training. Resume uses the same resolved manifest. | Existing `manifest.json` saves the complete `resolved_config`, its digest, protocol IDs, and raw file hashes. | Verified in P5.4c1. The preset maps to the historical single fixed manifest; no setting override is valid under v14. |
| `run_continual_trigger_replay_{schedule,training,outcomes}.py` | Each directly uses `fixed_trigger_replay_manifest()` and accepts only a new result path. | Each result saves the resolved manifest and/or digest under its existing fixed protocol. | Fixed v14 producers. The versioned bundle is the complete execution-provenance route. |
| `run_{difficulty_matched,structural_rank,sleep_trigger}_comparison.py` and `run_continual_replay_side_effect_ablation.py` | Fixed v11–v13 or replay-side-effect protocol; no setting flags. Unknown CLI switches fail parsing. | Typed fixed manifests and protocol IDs in result files. | Historical fixed studies. Do not turn them into tunable studies under the old IDs. |
| `run_continual_arrived_{selection_smoke,confirmation}.py`, matched-replay and replay-policy smokes, and the documented CIFAR selection/confirmation commands | Fixed construction or saved request/selection/manifest input, with explicit development versus final roles and digest verification before confirmation. Path and lifecycle flags choose artifacts, not scientific settings. | Requests/manifests and complete result/failure records, where applicable. | Frozen selection/confirmation routes; their existing protocol and role boundaries remain authoritative. |
| `run_continual_shift_benchmark.py` | `baseline`, `strength-case`, and `hardest-case` typed presets; strict allowlisted JSON overrides after legacy flags. | Full `continual_resolved_config_v1` artifact or existing result JSON with exact config and seeds. | Completed in P5.4a/b. Descriptive continual route only. |
| `run_multiseed_resnet_benchmark.py` | Configurable unmatched reference with many parser defaults and flags; builds a typed `ResNet50BenchmarkConfig`, but has no named CLI preset or strict resolved-config record before training. Unknown switches fail argparse. | Result JSON saves a dataset/runtime subset, seeds, and metrics; several model settings inherited from the typed config are omitted. | **Open P5.4c2 gap.** Move its existing defaults into a declared typed preset, validate explicit settings before a runner call, and save the complete resolved config without changing the historical unmatched label, baseline rates, metric selection, or old artifacts. |

The fixed command-line smoke scripts, diagnostic probes, projectors, and
verifiers in the README are supporting gates. They have no editable
experiment configuration contract. `run_pareto_hard_tuning.py` and
`run_circadian_policy_sweep.py` are old hardcoded sweep scripts and are not
documented as current reproduction commands; they must not be launched for
P5.4. This audit does not reclassify their historical output as matched
evidence.

Why this boundary: opening a fixed protocol to overrides would let a new
setting inherit an old study ID and raw hash expectations. The active
multi-seed ResNet CLI already exposes settings, so it is the next useful
place to finish the original P5.4 configuration contract.
