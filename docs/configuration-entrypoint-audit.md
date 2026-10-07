# Experiment entrypoint configuration audit (P5.4c)

This audit follows the experiment commands actively documented in the README
and `docs/modules/adapters.md` as of 2026-09-29, including root-level
wrappers that delegate to `src/adapters`. A fixed scientific protocol
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
| `run_multiseed_resnet_benchmark.py` | Typed `historical-unmatched` preset owns the former parser defaults. Legacy flags remain; repeatable allowlisted JSON overrides apply afterward. Invalid types/ranges, duplicate/unknown keys, and nonfinite values reject before a runner call. | New result JSON embeds complete `base_config` and ordered `trial_configs`, seeds, exact input tokens, and overrides under `resnet_multiseed_resolved_config_v1`; existing dataset/runtime/summary/CSV outputs remain. | Verified in P5.4c2 with pre-change default/flag config digests and mocked per-seed output. The route retains its unmatched status, baseline rates, and validation-only winner logic. |
| `predictive_coding_experiment.py` → `src/adapters/cli.py` | Typed `historical-toy` preset owns the old defaults, including environment precedence. Legacy flags remain; repeatable existing-field JSON overrides apply afterward. Malformed or unknown values reject before training. | New baseline CLI JSON retains its top-level report and adds `toy_resolved_config_v1`; `--resolved-config` saves the same full base/per-cell record for either mode. The record includes ordered seeds/noise, explicit inputs, and overrides. | Verified in P5.4c3 with pre-change config hashes, an actual bounded baseline, a mocked indepth grid, pretraining rejection, and full/static gates. The grid is built before scores and has no winner selection. |
| `resnet50_benchmark.py` → `src/adapters/resnet_benchmark_cli.py` | Typed `historical-single-unmatched` preset owns all 110 old defaults. The broad legacy flags remain; repeatable JSON overrides apply after them. Unknown, malformed, type-invalid, nonfinite, and range-invalid settings reject before the runner. | New `resnet_single_resolved_config_v1` artifact saves the full config, preset, seed, fixed model order, unmatched track, input tokens, and overrides; new result JSON embeds the same record beside its report fields. Both paths are exclusive. | Verified in P5.4c4 with pre-change default/README-flag hashes, mocked exact artifacts, pretraining rejection, and final full/static gates. The route remains unmatched; stdout and metric code are unchanged. |

The fixed command-line smoke scripts, diagnostic probes, projectors, and
verifiers in the README are supporting gates. They have no editable
experiment configuration contract. `run_pareto_hard_tuning.py` and
`run_circadian_policy_sweep.py` are old hardcoded sweep scripts and are not
documented as current reproduction commands; they must not be launched for
P5.4. This audit does not reclassify their historical output as matched
evidence.

Why this boundary: opening a fixed protocol to overrides would let a new
setting inherit an old study ID and raw hash expectations. The active
multi-seed ResNet CLI already exposed settings, so the new typed resolver
records those exact settings without changing this route into a matched
study. The root wrappers were found after the scripts-only pass and
completed as c3/c4 before the broad P5.4 audit was closed.
