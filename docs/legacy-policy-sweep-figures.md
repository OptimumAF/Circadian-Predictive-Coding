# Historical circadian policy reports

## Scope and limits

This completes only P9.5b18 saved policy-report validation/presentation after gates.
The whole 53,335-byte source benchmark_circadian_policy_sweep_results.json was
bound before parsing, SHA256 cb1b0a8ed0d766d2907e44ff2e45763dc5344eda0c4aa02668eeb922f10b2d2f.
Five whole metadata bodies, full original publication record, full parsed source
and all 1,091 typed leaves are retained. All 18 original trials and 24 repeated
ranking/winner records are checked; repeated copies are not additional experiments.
All 396 report/configuration table rows and 108 plotted saved values are retained.
Registered same-family paths contain only this JSON and no preexisting figure.
The Pareto JSON, its mismatched summary, older charts and canonical P6.10 aggregate
are separate sources and were not used to fill missing values.

Original declaration: hard difficulty, noise 0.08, 2,500 training / 700 test
samples, 10 classes, image size 96, CUDA, 14 epochs, ImageNet backbone weights.
These declarations do not establish dataset name, actual archive/weight bytes,
dependency inventory, code/build identity or complete execution environment.
No seed IDs/count or per-seed reports/std/uncertainty are recorded. There is one
report per configuration; this does not prove one seed or fresh independence.
No BP/PC comparator exists here, so no matched-baseline victory is inferred.
Historical test-label-informed stopping/sleep rollback limitations remain.
Actual defaults, complete attempt/failure history, RSS/allocator/resource sampling
and energy units are unknown. No failure field is not a full attempt ledger.
Original publication's Pareto-summary mismatch warning is preserved; it does not
prove that the separate summary belongs to this policy file.

Trial 1 params={} remains empty; no current-default backfill. Trial 7/8 dual-
chemical true and trial 9 dual-chemical false/adaptive-threshold true preserve
boolean type. All saved configurations, integer counts, exact float values and
missing fields remain unchanged. Trial 3 reports hidden 384->376 with 8 splits
and 16 prunes; this contraction is preserved. All recorded rollbacks are 0,
without inferring the absence of every historical failure or rollback opportunity.
The source contains no explicit null numeric values; absent provenance fields
remain absent rather than manufactured nulls, zeros or values from other runs.

Stored accuracy winner is trial 5 (0.93), training-speed winner trial 16
(2899.73672296097 samples/s), inference-speed winner trial 15
(4850.372532536333 samples/s), balanced winner trial 3 (0.8571817530350816).
Every stored top-10 record equals its original primary record; existing rank
metric order/cutoff and winner maxima are checked against all 18 saved records.
Only these existing claims were validated: no new selection, tie rule, balanced
formula, score, mean, uncertainty, confidence claim or experiment. Tie execution
rules and balanced-score execution provenance remain unproved. All trials are
presented in original trial order, including lower-accuracy/faster configurations.
Both complete PNG pages were visually checked; 12 panels have explicit zero
ranges and original labels/units, with energy units explicitly unrecorded.

## Every saved configuration and report field

| Trial | Original report field | Exact saved value/type |
|---:|---|---|
| 1 | params | {} |
| 1 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 1 | epochs_ran | 14 (int) |
| 1 | final_metric_name | "cross_entropy" (str) |
| 1 | final_metric_value | 0.2618305342538016 (float) |
| 1 | final_cross_entropy | 0.2618305342538016 (float) |
| 1 | final_energy | 0.002340208739042282 (float) |
| 1 | test_accuracy | 0.9257142857142857 (float) |
| 1 | train_seconds | 14.187422999999999 (float) |
| 1 | train_samples_per_second | 2466.9737414610113 (float) |
| 1 | mean_train_step_ms | 17.602486785714262 (float) |
| 1 | inference_latency_mean_ms | 13.554892857143079 (float) |
| 1 | inference_latency_p95_ms | 14.209330000000442 (float) |
| 1 | inference_samples_per_second | 4700.463986383332 (float) |
| 1 | total_parameters | 24311052 (int) |
| 1 | trainable_parameters | 803020 (int) |
| 1 | circadian_hidden_dim_start | 384 (int) |
| 1 | circadian_hidden_dim_end | 390 (int) |
| 1 | circadian_total_splits | 22 (int) |
| 1 | circadian_total_prunes | 16 (int) |
| 1 | circadian_total_rollbacks | 0 (int) |
| 1 | balanced_score | 0.672552895682281 (float) |
| 2 | params | {"circadian_adaptive_prune_percentile": 10.0, "circadian_adaptive_split_percentile": 90.0} |
| 2 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 2 | epochs_ran | 14 (int) |
| 2 | final_metric_name | "cross_entropy" (str) |
| 2 | final_metric_value | 0.2584695393698556 (float) |
| 2 | final_cross_entropy | 0.2584695393698556 (float) |
| 2 | final_energy | 0.00033900674316100776 (float) |
| 2 | test_accuracy | 0.9242857142857143 (float) |
| 2 | train_seconds | 12.8411927 (float) |
| 2 | train_samples_per_second | 2725.6035181218017 (float) |
| 2 | mean_train_step_ms | 15.85048660714281 (float) |
| 2 | inference_latency_mean_ms | 13.657671428571152 (float) |
| 2 | inference_latency_p95_ms | 14.500980000000041 (float) |
| 2 | inference_samples_per_second | 4665.091413826128 (float) |
| 2 | total_parameters | 24308993 (int) |
| 2 | trainable_parameters | 800961 (int) |
| 2 | circadian_hidden_dim_start | 384 (int) |
| 2 | circadian_hidden_dim_end | 389 (int) |
| 2 | circadian_total_splits | 21 (int) |
| 2 | circadian_total_prunes | 16 (int) |
| 2 | circadian_total_rollbacks | 0 (int) |
| 2 | balanced_score | 0.7990721597670916 (float) |
| 3 | params | {"circadian_adaptive_prune_percentile": 8.0, "circadian_adaptive_split_percentile": 92.0, "circadian_prune_cooldown_steps": 3, "circadian_split_cooldown_steps": 3} |
| 3 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 3 | epochs_ran | 14 (int) |
| 3 | final_metric_name | "cross_entropy" (str) |
| 3 | final_metric_value | 0.2573108564104353 (float) |
| 3 | final_cross_entropy | 0.2573108564104353 (float) |
| 3 | final_energy | 0.0004492849693633616 (float) |
| 3 | test_accuracy | 0.9285714285714286 (float) |
| 3 | train_seconds | 12.73554 (float) |
| 3 | train_samples_per_second | 2748.2148381615543 (float) |
| 3 | mean_train_step_ms | 15.718701249999699 (float) |
| 3 | inference_latency_mean_ms | 13.691621428572022 (float) |
| 3 | inference_latency_p95_ms | 14.800690000000216 (float) |
| 3 | inference_samples_per_second | 4653.523766098669 (float) |
| 3 | total_parameters | 24282226 (int) |
| 3 | trainable_parameters | 774194 (int) |
| 3 | circadian_hidden_dim_start | 384 (int) |
| 3 | circadian_hidden_dim_end | 376 (int) |
| 3 | circadian_total_splits | 8 (int) |
| 3 | circadian_total_prunes | 16 (int) |
| 3 | circadian_total_rollbacks | 0 (int) |
| 3 | balanced_score | 0.8571817530350816 (float) |
| 4 | params | {"circadian_adaptive_prune_percentile": 20.0, "circadian_adaptive_split_percentile": 85.0, "circadian_prune_hysteresis_margin": 0.03, "circadian_split_hysteresis_margin": 0.03} |
| 4 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 4 | epochs_ran | 14 (int) |
| 4 | final_metric_name | "cross_entropy" (str) |
| 4 | final_metric_value | 0.27546262877328054 (float) |
| 4 | final_cross_entropy | 0.27546262877328054 (float) |
| 4 | final_energy | 0.002085747430101037 (float) |
| 4 | test_accuracy | 0.9214285714285714 (float) |
| 4 | train_seconds | 12.793956799999997 (float) |
| 4 | train_samples_per_second | 2735.6665765824696 (float) |
| 4 | mean_train_step_ms | 15.772833928571306 (float) |
| 4 | inference_latency_mean_ms | 15.529500000000544 (float) |
| 4 | inference_latency_p95_ms | 16.979840000003676 (float) |
| 4 | inference_samples_per_second | 4102.790541503814 (float) |
| 4 | total_parameters | 24311052 (int) |
| 4 | trainable_parameters | 803020 (int) |
| 4 | circadian_hidden_dim_start | 384 (int) |
| 4 | circadian_hidden_dim_end | 390 (int) |
| 4 | circadian_total_splits | 22 (int) |
| 4 | circadian_total_prunes | 16 (int) |
| 4 | circadian_total_rollbacks | 0 (int) |
| 4 | balanced_score | 0.6604825543124745 (float) |
| 5 | params | {"circadian_prune_importance_mix": 0.5, "circadian_prune_weight_norm_mix": 0.2, "circadian_split_importance_mix": 0.3, "circadian_split_weight_norm_mix": 0.2} |
| 5 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 5 | epochs_ran | 14 (int) |
| 5 | final_metric_name | "cross_entropy" (str) |
| 5 | final_metric_value | 0.2647245965685163 (float) |
| 5 | final_cross_entropy | 0.2647245965685163 (float) |
| 5 | final_energy | 0.0004364221531432122 (float) |
| 5 | test_accuracy | 0.93 (float) |
| 5 | train_seconds | 12.841629100000006 (float) |
| 5 | train_samples_per_second | 2725.5108933180436 (float) |
| 5 | mean_train_step_ms | 15.911635535714375 (float) |
| 5 | inference_latency_mean_ms | 13.867785714282377 (float) |
| 5 | inference_latency_p95_ms | 14.906895000002152 (float) |
| 5 | inference_samples_per_second | 4594.409448414438 (float) |
| 5 | total_parameters | 24311052 (int) |
| 5 | trainable_parameters | 803020 (int) |
| 5 | circadian_hidden_dim_start | 384 (int) |
| 5 | circadian_hidden_dim_end | 390 (int) |
| 5 | circadian_total_splits | 22 (int) |
| 5 | circadian_total_prunes | 16 (int) |
| 5 | circadian_total_rollbacks | 0 (int) |
| 5 | balanced_score | 0.8479944302951853 (float) |
| 6 | params | {"circadian_importance_ema_decay": 0.98, "circadian_prune_importance_mix": 0.45, "circadian_split_importance_mix": 0.15} |
| 6 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 6 | epochs_ran | 14 (int) |
| 6 | final_metric_name | "cross_entropy" (str) |
| 6 | final_metric_value | 0.2703462736947196 (float) |
| 6 | final_cross_entropy | 0.2703462736947196 (float) |
| 6 | final_energy | 0.0012133859563618898 (float) |
| 6 | test_accuracy | 0.9228571428571428 (float) |
| 6 | train_seconds | 12.644095500000006 (float) |
| 6 | train_samples_per_second | 2768.0904498071836 (float) |
| 6 | mean_train_step_ms | 15.599166607142628 (float) |
| 6 | inference_latency_mean_ms | 13.645378571427257 (float) |
| 6 | inference_latency_p95_ms | 14.56306999999839 (float) |
| 6 | inference_samples_per_second | 4669.294104283795 (float) |
| 6 | total_parameters | 24311052 (int) |
| 6 | trainable_parameters | 803020 (int) |
| 6 | circadian_hidden_dim_start | 384 (int) |
| 6 | circadian_hidden_dim_end | 390 (int) |
| 6 | circadian_total_splits | 22 (int) |
| 6 | circadian_total_prunes | 16 (int) |
| 6 | circadian_total_rollbacks | 0 (int) |
| 6 | balanced_score | 0.808669938923929 (float) |
| 7 | params | {"circadian_dual_fast_mix": 0.6, "circadian_slow_chemical_decay": 0.9995, "circadian_use_dual_chemical": true} |
| 7 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 7 | epochs_ran | 14 (int) |
| 7 | final_metric_name | "cross_entropy" (str) |
| 7 | final_metric_value | 0.3302542087009975 (float) |
| 7 | final_cross_entropy | 0.3302542087009975 (float) |
| 7 | final_energy | 0.002607948612421751 (float) |
| 7 | test_accuracy | 0.8828571428571429 (float) |
| 7 | train_seconds | 12.464085400000002 (float) |
| 7 | train_samples_per_second | 2808.0680512667213 (float) |
| 7 | mean_train_step_ms | 15.373199642857266 (float) |
| 7 | inference_latency_mean_ms | 13.615271428572028 (float) |
| 7 | inference_latency_p95_ms | 14.56720499999733 (float) |
| 7 | inference_samples_per_second | 4679.6192091021785 (float) |
| 7 | total_parameters | 24311052 (int) |
| 7 | trainable_parameters | 803020 (int) |
| 7 | circadian_hidden_dim_start | 384 (int) |
| 7 | circadian_hidden_dim_end | 390 (int) |
| 7 | circadian_total_splits | 22 (int) |
| 7 | circadian_total_prunes | 16 (int) |
| 7 | circadian_total_rollbacks | 0 (int) |
| 7 | balanced_score | 0.391730773523999 (float) |
| 8 | params | {"circadian_dual_fast_mix": 0.8, "circadian_slow_buildup_scale": 0.15, "circadian_use_dual_chemical": true} |
| 8 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 8 | epochs_ran | 14 (int) |
| 8 | final_metric_name | "cross_entropy" (str) |
| 8 | final_metric_value | 0.3108819430215018 (float) |
| 8 | final_cross_entropy | 0.3108819430215018 (float) |
| 8 | final_energy | 0.0036831647157669067 (float) |
| 8 | test_accuracy | 0.8971428571428571 (float) |
| 8 | train_seconds | 12.512849799999998 (float) |
| 8 | train_samples_per_second | 2797.124600664511 (float) |
| 8 | mean_train_step_ms | 15.415797142857075 (float) |
| 8 | inference_latency_mean_ms | 13.643978571426121 (float) |
| 8 | inference_latency_p95_ms | 14.376729999999327 (float) |
| 8 | inference_samples_per_second | 4669.773217594994 (float) |
| 8 | total_parameters | 24311052 (int) |
| 8 | trainable_parameters | 803020 (int) |
| 8 | circadian_hidden_dim_start | 384 (int) |
| 8 | circadian_hidden_dim_end | 390 (int) |
| 8 | circadian_total_splits | 22 (int) |
| 8 | circadian_total_prunes | 16 (int) |
| 8 | circadian_total_rollbacks | 0 (int) |
| 8 | balanced_score | 0.5413280928205907 (float) |
| 9 | params | {"circadian_adaptive_prune_percentile": 15.0, "circadian_adaptive_split_percentile": 88.0, "circadian_use_adaptive_thresholds": true, "circadian_use_dual_chemical": false} |
| 9 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 9 | epochs_ran | 14 (int) |
| 9 | final_metric_name | "cross_entropy" (str) |
| 9 | final_metric_value | 0.27936020987374444 (float) |
| 9 | final_cross_entropy | 0.27936020987374444 (float) |
| 9 | final_energy | 0.0006330508622340858 (float) |
| 9 | test_accuracy | 0.9185714285714286 (float) |
| 9 | train_seconds | 12.47657740000001 (float) |
| 9 | train_samples_per_second | 2805.2565120944123 (float) |
| 9 | mean_train_step_ms | 15.355311785714168 (float) |
| 9 | inference_latency_mean_ms | 13.569814285716575 (float) |
| 9 | inference_latency_p95_ms | 14.386719999998832 (float) |
| 9 | inference_samples_per_second | 4695.295335128544 (float) |
| 9 | total_parameters | 24311052 (int) |
| 9 | trainable_parameters | 803020 (int) |
| 9 | circadian_hidden_dim_start | 384 (int) |
| 9 | circadian_hidden_dim_end | 390 (int) |
| 9 | circadian_total_splits | 22 (int) |
| 9 | circadian_total_prunes | 16 (int) |
| 9 | circadian_total_rollbacks | 0 (int) |
| 9 | balanced_score | 0.7879888019621064 (float) |
| 10 | params | {"circadian_max_prune_per_sleep": 1, "circadian_max_split_per_sleep": 1, "circadian_split_noise_scale": 0.0} |
| 10 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 10 | epochs_ran | 14 (int) |
| 10 | final_metric_name | "cross_entropy" (str) |
| 10 | final_metric_value | 0.2623045308249337 (float) |
| 10 | final_cross_entropy | 0.2623045308249337 (float) |
| 10 | final_energy | 0.00028836476849392056 (float) |
| 10 | test_accuracy | 0.9228571428571428 (float) |
| 10 | train_seconds | 12.534704200000007 (float) |
| 10 | train_samples_per_second | 2792.247781962017 (float) |
| 10 | mean_train_step_ms | 15.450315535714637 (float) |
| 10 | inference_latency_mean_ms | 13.677607142862971 (float) |
| 10 | inference_latency_p95_ms | 14.481135000011136 (float) |
| 10 | inference_samples_per_second | 4658.291837803813 (float) |
| 10 | total_parameters | 24304875 (int) |
| 10 | trainable_parameters | 796843 (int) |
| 10 | circadian_hidden_dim_start | 384 (int) |
| 10 | circadian_hidden_dim_end | 387 (int) |
| 10 | circadian_total_splits | 11 (int) |
| 10 | circadian_total_prunes | 8 (int) |
| 10 | circadian_total_rollbacks | 0 (int) |
| 10 | balanced_score | 0.8204176586959586 (float) |
| 11 | params | {"circadian_max_prune_per_sleep": 1, "circadian_max_split_per_sleep": 1, "circadian_sleep_interval": 2} |
| 11 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 11 | epochs_ran | 14 (int) |
| 11 | final_metric_name | "cross_entropy" (str) |
| 11 | final_metric_value | 0.3376092692783901 (float) |
| 11 | final_cross_entropy | 0.3376092692783901 (float) |
| 11 | final_energy | 0.008838141337037086 (float) |
| 11 | test_accuracy | 0.8842857142857142 (float) |
| 11 | train_seconds | 12.499911199999985 (float) |
| 11 | train_samples_per_second | 2800.0198913413114 (float) |
| 11 | mean_train_step_ms | 15.464056964285074 (float) |
| 11 | inference_latency_mean_ms | 13.614092857147446 (float) |
| 11 | inference_latency_p95_ms | 14.441005000003315 (float) |
| 11 | inference_samples_per_second | 4680.024323532911 (float) |
| 11 | total_parameters | 24300757 (int) |
| 11 | trainable_parameters | 792725 (int) |
| 11 | circadian_hidden_dim_start | 384 (int) |
| 11 | circadian_hidden_dim_end | 385 (int) |
| 11 | circadian_total_splits | 5 (int) |
| 11 | circadian_total_prunes | 4 (int) |
| 11 | circadian_total_rollbacks | 0 (int) |
| 11 | balanced_score | 0.4029522435927521 (float) |
| 12 | params | {"circadian_adaptive_prune_percentile": 10.0, "circadian_adaptive_split_percentile": 90.0, "circadian_head_hidden_dim": 512, "circadian_inference_learning_rate": 0.12, "circadian_inference_steps": 14, "circadian_learning_rate": 0.02} |
| 12 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 12 | epochs_ran | 14 (int) |
| 12 | final_metric_name | "cross_entropy" (str) |
| 12 | final_metric_value | 0.3203082466125488 (float) |
| 12 | final_cross_entropy | 0.3203082466125488 (float) |
| 12 | final_energy | 0.004148418549448252 (float) |
| 12 | test_accuracy | 0.8985714285714286 (float) |
| 12 | train_seconds | 12.614413500000012 (float) |
| 12 | train_samples_per_second | 2774.6038291831774 (float) |
| 12 | mean_train_step_ms | 15.618578571428186 (float) |
| 12 | inference_latency_mean_ms | 13.785007142859383 (float) |
| 12 | inference_latency_p95_ms | 14.545434999992324 (float) |
| 12 | inference_samples_per_second | 4621.998745012595 (float) |
| 12 | total_parameters | 24574604 (int) |
| 12 | trainable_parameters | 1066572 (int) |
| 12 | circadian_hidden_dim_start | 512 (int) |
| 12 | circadian_hidden_dim_end | 518 (int) |
| 12 | circadian_total_splits | 22 (int) |
| 12 | circadian_total_prunes | 16 (int) |
| 12 | circadian_total_rollbacks | 0 (int) |
| 12 | balanced_score | 0.5345218909250448 (float) |
| 13 | params | {"circadian_adaptive_prune_percentile": 8.0, "circadian_adaptive_split_percentile": 92.0, "circadian_head_hidden_dim": 512, "circadian_inference_learning_rate": 0.15, "circadian_inference_steps": 14, "circadian_learning_rate": 0.03} |
| 13 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 13 | epochs_ran | 14 (int) |
| 13 | final_metric_name | "cross_entropy" (str) |
| 13 | final_metric_value | 0.3364262730734689 (float) |
| 13 | final_cross_entropy | 0.3364262730734689 (float) |
| 13 | final_energy | 0.002796889515593648 (float) |
| 13 | test_accuracy | 0.8757142857142857 (float) |
| 13 | train_seconds | 12.63044050000002 (float) |
| 13 | train_samples_per_second | 2771.0830829692713 (float) |
| 13 | mean_train_step_ms | 15.614896249999635 (float) |
| 13 | inference_latency_mean_ms | 13.664357142859883 (float) |
| 13 | inference_latency_p95_ms | 14.423764999993693 (float) |
| 13 | inference_samples_per_second | 4662.808871881596 (float) |
| 13 | total_parameters | 24560191 (int) |
| 13 | trainable_parameters | 1052159 (int) |
| 13 | circadian_hidden_dim_start | 512 (int) |
| 13 | circadian_hidden_dim_end | 511 (int) |
| 13 | circadian_total_splits | 15 (int) |
| 13 | circadian_total_prunes | 16 (int) |
| 13 | circadian_total_rollbacks | 0 (int) |
| 13 | balanced_score | 0.2880448602111425 (float) |
| 14 | params | {"circadian_homeostasis_strength": 0.35, "circadian_homeostasis_target_input_norm": 1.2, "circadian_homeostasis_target_output_norm": 1.0, "circadian_homeostatic_downscale_factor": 0.995} |
| 14 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 14 | epochs_ran | 14 (int) |
| 14 | final_metric_name | "cross_entropy" (str) |
| 14 | final_metric_value | 0.21681346893310546 (float) |
| 14 | final_cross_entropy | 0.21681346893310546 (float) |
| 14 | final_energy | 2.2394493498723023e-05 (float) |
| 14 | test_accuracy | 0.9228571428571428 (float) |
| 14 | train_seconds | 12.554005199999978 (float) |
| 14 | train_samples_per_second | 2787.9548751501284 (float) |
| 14 | mean_train_step_ms | 15.475874821428807 (float) |
| 14 | inference_latency_mean_ms | 14.3565714285724 (float) |
| 14 | inference_latency_p95_ms | 15.179009999995685 (float) |
| 14 | inference_samples_per_second | 4437.987581594987 (float) |
| 14 | total_parameters | 24311052 (int) |
| 14 | trainable_parameters | 803020 (int) |
| 14 | circadian_hidden_dim_start | 384 (int) |
| 14 | circadian_hidden_dim_end | 390 (int) |
| 14 | circadian_total_splits | 22 (int) |
| 14 | circadian_total_prunes | 16 (int) |
| 14 | circadian_total_rollbacks | 0 (int) |
| 14 | balanced_score | 0.773734354185277 (float) |
| 15 | params | {"circadian_homeostasis_strength": 0.5, "circadian_homeostasis_target_input_norm": 1.0, "circadian_homeostasis_target_output_norm": 0.8, "circadian_homeostatic_downscale_factor": 0.998} |
| 15 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 15 | epochs_ran | 14 (int) |
| 15 | final_metric_name | "cross_entropy" (str) |
| 15 | final_metric_value | 0.25012935502188544 (float) |
| 15 | final_cross_entropy | 0.25012935502188544 (float) |
| 15 | final_energy | 3.889867366524413e-05 (float) |
| 15 | test_accuracy | 0.9128571428571428 (float) |
| 15 | train_seconds | 12.33291220000001 (float) |
| 15 | train_samples_per_second | 2837.9347418041275 (float) |
| 15 | mean_train_step_ms | 15.185831785714345 (float) |
| 15 | inference_latency_mean_ms | 13.13595714285652 (float) |
| 15 | inference_latency_p95_ms | 13.46373500000766 (float) |
| 15 | inference_samples_per_second | 4850.372532536333 (float) |
| 15 | total_parameters | 24311052 (int) |
| 15 | trainable_parameters | 803020 (int) |
| 15 | circadian_hidden_dim_start | 384 (int) |
| 15 | circadian_hidden_dim_end | 390 (int) |
| 15 | circadian_total_splits | 22 (int) |
| 15 | circadian_total_prunes | 16 (int) |
| 15 | circadian_total_rollbacks | 0 (int) |
| 15 | balanced_score | 0.7748243358380938 (float) |
| 16 | params | {"circadian_homeostasis_target_input_norm": 0.0, "circadian_homeostasis_target_output_norm": 0.0, "circadian_homeostatic_downscale_factor": 1.0} |
| 16 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 16 | epochs_ran | 14 (int) |
| 16 | final_metric_name | "cross_entropy" (str) |
| 16 | final_metric_value | 0.26381016867501395 (float) |
| 16 | final_cross_entropy | 0.26381016867501395 (float) |
| 16 | final_energy | 0.00027451664209365845 (float) |
| 16 | test_accuracy | 0.9171428571428571 (float) |
| 16 | train_seconds | 12.070061299999992 (float) |
| 16 | train_samples_per_second | 2899.73672296097 (float) |
| 16 | mean_train_step_ms | 14.864152321428342 (float) |
| 16 | inference_latency_mean_ms | 14.033335714285856 (float) |
| 16 | inference_latency_p95_ms | 14.590469999998845 (float) |
| 16 | inference_samples_per_second | 4540.2096131303215 (float) |
| 16 | total_parameters | 24311052 (int) |
| 16 | trainable_parameters | 803020 (int) |
| 16 | circadian_hidden_dim_start | 384 (int) |
| 16 | circadian_hidden_dim_end | 390 (int) |
| 16 | circadian_total_splits | 22 (int) |
| 16 | circadian_total_prunes | 16 (int) |
| 16 | circadian_total_rollbacks | 0 (int) |
| 16 | balanced_score | 0.7956615123008814 (float) |
| 17 | params | {"circadian_inference_learning_rate": 0.12, "circadian_inference_steps": 16, "circadian_learning_rate": 0.025, "circadian_prune_cooldown_steps": 3, "circadian_prune_importance_mix": 0.45, "circadian_split_cooldown_steps": 3, "circadian_split_importance_mix": 0.25} |
| 17 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 17 | epochs_ran | 14 (int) |
| 17 | final_metric_name | "cross_entropy" (str) |
| 17 | final_metric_value | 0.29051895686558316 (float) |
| 17 | final_cross_entropy | 0.29051895686558316 (float) |
| 17 | final_energy | 0.004225627053529024 (float) |
| 17 | test_accuracy | 0.9228571428571428 (float) |
| 17 | train_seconds | 12.641111899999999 (float) |
| 17 | train_samples_per_second | 2768.7437843185303 (float) |
| 17 | mean_train_step_ms | 15.695224642857296 (float) |
| 17 | inference_latency_mean_ms | 13.48685000000062 (float) |
| 17 | inference_latency_p95_ms | 14.601530000001617 (float) |
| 17 | inference_samples_per_second | 4724.178419296039 (float) |
| 17 | total_parameters | 24311052 (int) |
| 17 | trainable_parameters | 803020 (int) |
| 17 | circadian_hidden_dim_start | 384 (int) |
| 17 | circadian_hidden_dim_end | 390 (int) |
| 17 | circadian_total_splits | 22 (int) |
| 17 | circadian_total_prunes | 16 (int) |
| 17 | circadian_total_rollbacks | 0 (int) |
| 17 | balanced_score | 0.820059726445397 (float) |
| 18 | params | {"circadian_inference_learning_rate": 0.18, "circadian_inference_steps": 10, "circadian_learning_rate": 0.035, "circadian_prune_cooldown_steps": 1, "circadian_prune_hysteresis_margin": 0.01, "circadian_split_cooldown_steps": 1, "circadian_split_hysteresis_margin": 0.01} |
| 18 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| 18 | epochs_ran | 14 (int) |
| 18 | final_metric_name | "cross_entropy" (str) |
| 18 | final_metric_value | 0.35408414568219865 (float) |
| 18 | final_cross_entropy | 0.35408414568219865 (float) |
| 18 | final_energy | 0.0066289035603404045 (float) |
| 18 | test_accuracy | 0.89 (float) |
| 18 | train_seconds | 12.45785699999999 (float) |
| 18 | train_samples_per_second | 2809.471966165612 (float) |
| 18 | mean_train_step_ms | 15.322880714285754 (float) |
| 18 | inference_latency_mean_ms | 13.276257142856755 (float) |
| 18 | inference_latency_p95_ms | 13.87195999999875 (float) |
| 18 | inference_samples_per_second | 4799.1150690062495 (float) |
| 18 | total_parameters | 24311052 (int) |
| 18 | trainable_parameters | 803020 (int) |
| 18 | circadian_hidden_dim_start | 384 (int) |
| 18 | circadian_hidden_dim_end | 390 (int) |
| 18 | circadian_total_splits | 22 (int) |
| 18 | circadian_total_prunes | 16 (int) |
| 18 | circadian_total_rollbacks | 0 (int) |
| 18 | balanced_score | 0.49546563272237587 (float) |

Field units: training time=s, throughput=samples/s, mean train step=ms/step, inference mean/p95=ms, capacity and sleep fields=counts. Final energy retains its source scalar with units unknown. Accuracy/cross-entropy and balanced score remain original labels; the balanced formula is not recreated.

[Complete typed source view](../artifacts/runs/p95-policy-sweep-20261006/view.json) preserves all primary and repeated records, dataset declaration and original publication metadata.

## Every trial plotted

### Original trials 1–9

![Saved policy reports page 1](../artifacts/runs/p95-policy-sweep-20261006/page-1.png)

[SVG](../artifacts/runs/p95-policy-sweep-20261006/page-1.svg)

### Original trials 10–18

![Saved policy reports page 2](../artifacts/runs/p95-policy-sweep-20261006/page-2.png)

[SVG](../artifacts/runs/p95-policy-sweep-20261006/page-2.svg)

## Original ranking and winner references

```json
{
  "top10_by_accuracy": [
    5,
    3,
    1,
    2,
    10,
    14,
    17,
    6,
    4,
    9
  ],
  "top10_by_balanced_score": [
    3,
    5,
    10,
    17,
    6,
    2,
    16,
    9,
    15,
    14
  ],
  "best_accuracy": {
    "trial": 5,
    "params": {
      "circadian_split_importance_mix": 0.3,
      "circadian_prune_importance_mix": 0.5,
      "circadian_split_weight_norm_mix": 0.2,
      "circadian_prune_weight_norm_mix": 0.2
    },
    "report": {
      "model_name": "CircadianPredictiveCodingResNet50",
      "epochs_ran": 14,
      "final_metric_name": "cross_entropy",
      "final_metric_value": 0.2647245965685163,
      "final_cross_entropy": 0.2647245965685163,
      "final_energy": 0.0004364221531432122,
      "test_accuracy": 0.93,
      "train_seconds": 12.841629100000006,
      "train_samples_per_second": 2725.5108933180436,
      "mean_train_step_ms": 15.911635535714375,
      "inference_latency_mean_ms": 13.867785714282377,
      "inference_latency_p95_ms": 14.906895000002152,
      "inference_samples_per_second": 4594.409448414438,
      "total_parameters": 24311052,
      "trainable_parameters": 803020,
      "circadian_hidden_dim_start": 384,
      "circadian_hidden_dim_end": 390,
      "circadian_total_splits": 22,
      "circadian_total_prunes": 16,
      "circadian_total_rollbacks": 0,
      "balanced_score": 0.8479944302951853
    }
  },
  "best_train_speed": {
    "trial": 16,
    "params": {
      "circadian_homeostatic_downscale_factor": 1.0,
      "circadian_homeostasis_target_input_norm": 0.0,
      "circadian_homeostasis_target_output_norm": 0.0
    },
    "report": {
      "model_name": "CircadianPredictiveCodingResNet50",
      "epochs_ran": 14,
      "final_metric_name": "cross_entropy",
      "final_metric_value": 0.26381016867501395,
      "final_cross_entropy": 0.26381016867501395,
      "final_energy": 0.00027451664209365845,
      "test_accuracy": 0.9171428571428571,
      "train_seconds": 12.070061299999992,
      "train_samples_per_second": 2899.73672296097,
      "mean_train_step_ms": 14.864152321428342,
      "inference_latency_mean_ms": 14.033335714285856,
      "inference_latency_p95_ms": 14.590469999998845,
      "inference_samples_per_second": 4540.2096131303215,
      "total_parameters": 24311052,
      "trainable_parameters": 803020,
      "circadian_hidden_dim_start": 384,
      "circadian_hidden_dim_end": 390,
      "circadian_total_splits": 22,
      "circadian_total_prunes": 16,
      "circadian_total_rollbacks": 0,
      "balanced_score": 0.7956615123008814
    }
  },
  "best_inference_speed": {
    "trial": 15,
    "params": {
      "circadian_homeostatic_downscale_factor": 0.998,
      "circadian_homeostasis_target_input_norm": 1.0,
      "circadian_homeostasis_target_output_norm": 0.8,
      "circadian_homeostasis_strength": 0.5
    },
    "report": {
      "model_name": "CircadianPredictiveCodingResNet50",
      "epochs_ran": 14,
      "final_metric_name": "cross_entropy",
      "final_metric_value": 0.25012935502188544,
      "final_cross_entropy": 0.25012935502188544,
      "final_energy": 3.889867366524413e-05,
      "test_accuracy": 0.9128571428571428,
      "train_seconds": 12.33291220000001,
      "train_samples_per_second": 2837.9347418041275,
      "mean_train_step_ms": 15.185831785714345,
      "inference_latency_mean_ms": 13.13595714285652,
      "inference_latency_p95_ms": 13.46373500000766,
      "inference_samples_per_second": 4850.372532536333,
      "total_parameters": 24311052,
      "trainable_parameters": 803020,
      "circadian_hidden_dim_start": 384,
      "circadian_hidden_dim_end": 390,
      "circadian_total_splits": 22,
      "circadian_total_prunes": 16,
      "circadian_total_rollbacks": 0,
      "balanced_score": 0.7748243358380938
    }
  },
  "best_balanced": {
    "trial": 3,
    "params": {
      "circadian_adaptive_split_percentile": 92.0,
      "circadian_adaptive_prune_percentile": 8.0,
      "circadian_split_cooldown_steps": 3,
      "circadian_prune_cooldown_steps": 3
    },
    "report": {
      "model_name": "CircadianPredictiveCodingResNet50",
      "epochs_ran": 14,
      "final_metric_name": "cross_entropy",
      "final_metric_value": 0.2573108564104353,
      "final_cross_entropy": 0.2573108564104353,
      "final_energy": 0.0004492849693633616,
      "test_accuracy": 0.9285714285714286,
      "train_seconds": 12.73554,
      "train_samples_per_second": 2748.2148381615543,
      "mean_train_step_ms": 15.718701249999699,
      "inference_latency_mean_ms": 13.691621428572022,
      "inference_latency_p95_ms": 14.800690000000216,
      "inference_samples_per_second": 4653.523766098669,
      "total_parameters": 24282226,
      "trainable_parameters": 774194,
      "circadian_hidden_dim_start": 384,
      "circadian_hidden_dim_end": 376,
      "circadian_total_splits": 8,
      "circadian_total_prunes": 16,
      "circadian_total_rollbacks": 0,
      "balanced_score": 0.8571817530350816
    }
  }
}
```

## Structure and workflow

```text
artifacts/runs/p95-policy-sweep-20261006/
  run.py                 bounded command capture
  prepare.py             full checkout/docs/source/metadata freezing
  render.py              whole source/claim validation and saved figures
  audit.py               independent body/leaves/table/figures/controls
  validate_static.py     six-helper syntax/style/formatter AST gates
  finish.py              gated additive docs and reversible preservation
  metadata/              five frozen complete metadata bodies
  next-inputs/           exact original JSON
  view.json / all-reports.md / page-{1,2}.{png,svg}
  visual-review.json / readback.json / static-validation.json
  acceptance.json / terminal.json / final-accounting.json
  command-NNN.{json,stdout,stderr}
docs/legacy-policy-sweep-figures.md
```

Local helpers read saved bytes and draw existing values; no public module,
architecture boundary or dependency added. Extend with separately bound complete
source/metadata, exact null/type/default provenance and independent full readback.
Keep distinct families and write-once receipts; do not backfill absent defaults.

Exact argv/cwd/duration/stdout/stderr are in command-NNN receipts:
001 prepare exit0: full 1020-file checkout, 25 packages, HEAD182077 and full
AGENTS/plan/log read; five metadata bodies and one source bound before parsing.
002 render exit0: complete source/schema/rank/winner checks, full typed view,
396 table rows and two PNG/SVG pages. 003 audit exit0: independent whole body /
1,091 typed leaves / original boolean/empty configuration / 24 copied claims /
complete table / all SVG labels, zero ranges and bar geometry / PNG decode and
108 bar pixels. Separate full visual review accepted both PNGs. Eight negative
controls refused: duplicate JSON key, nonfinite JSON, wrong trial count, altered
ranked copy, reversed rank order, wrong existing winner, boolean numeric report,
unknown root field. 004 Ruff format exit0 (two formatted/four unchanged).
005 static exit0: six-helper Ruff check/format check/compile/full pre-post
formatter AST equality. No failed captured command before closing.
Required sequential gates: 006 finish prepare, 007 finish close, 008 scoped git
diff --check, 009 final source/output/metadata/checkout/task/AST/budget readback.
Their actual retained outcomes govern acceptance; all four must exit0.

PowerShell captured audit: `.venv/Scripts/python.exe -X utf8 artifacts/runs/p95-policy-sweep-20261006/run.py -X utf8 -m artifacts.runs.p95-policy-sweep-20261006.audit` (already accepted; write-once receipt prevents overwrite).

Prospective engineering scope: 600 aggregate seconds, 60-second hard child cap,
64 MiB owned stage; fixed 160-second manual/discovery/visual/closing reserve plus
every captured attempt/failure. Final accounting reserves its own full 60-second
cap. Not whole-session walltime or process RSS. Science 350.7925872/360 and
runtime 168.7993043/180 remain spent without reset/rekey.
Full pytest/native/Torch/CI/mypy/clean-clone and original scientific readers skipped
in this ignored helper/additive-document scope; original full gates remain open.
Pillow 12.3.0 already installed; no dependency/install/download/model/dataset/
archive/array/device/CI/sweep/algorithm/config/baseline/seed/metric change or guard
repair/publication/commit/push/merge/delegation/other-chat message.
P9.5/P9.5b/G0/R0.3/full R3.1 remain open; owning-with j6c repair remains
human-deferred; R0 publication remains separate. Unrelated user changes preserved.

## Exact next action

P9.5b19: freeze the full legacy-tuning-hardest publication/coverage metadata and whole benchmark_tuning_hardest_results.json before parsing under a fresh small engineering scope. Inspect every saved configuration/result/seed/resource/failure/environment/unknown limit and registered existing figure; independently validate complete bodies and present only uncovered saved values. Preserve original historical test-informed protocol and all parent acceptance. No model, original semantic reader, dataset, CI or scientific dispatch; no source borrowing or inferred independence/provenance.
