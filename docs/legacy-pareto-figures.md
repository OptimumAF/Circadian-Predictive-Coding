# Historical Pareto source and summary comparison

## Scope and original discrepancies

Only P9.5b17 saved JSON/summary comparison scope is complete after gates.
Two whole originals pinned before parsing: benchmark_pareto_hard_results.json
1,076,676 bytes/SHA0b5ffedfbee82c7de0a458246dd0cd0a41b0aefb26fa993126b246b639f30f10;
benchmark_pareto_hard_summary.md36,906 bytes/SHA2edd3ccf722b7f0315b7ae1e637bcc2cb9af69605b86360f614d54a511e4d067.
Full original publication bodies remain distinct. Full parsed JSON body and whole
summary text are retained, with independently checked20,210 typed leaves, all
34 primary trials,102 seed-report positions,136 duplicate ranked/front/best trial
copies and four global-report references. Duplicate presentations are not new
experiments. Stored means/std/nulls/energies/capacities/configs and all saved
fronts/rankings/winners are unchanged; no new score/front/selection/statistic.
Registered same-family paths contain those originals only, no existing figure;
older family figures/canonical P6.10 aggregate are separate and not reused.

JSON dataset declaration: hard/noise0.08/train2500/test700/classes10/image96/
CUDA/20epochs/seeds7,13,29. Summary declares14epochs and no seeds; it does not
identify the same run. Numeric seed IDs exist only in JSON dataset declaration;
individual seed reports have positions and no seed ID, so association or fresh
independence is unproved. Actual dataset name, dependency inventory, source/weight
identity, complete attempt/failure history, sampler/environment and corrected
baseline/isolation provenance unrecorded. Do not call this a corrected CIFAR run.
Both families retain historical test-informed provenance/unmatched capacities.
Original BP trainable count20490; PC527114/790666/1054218; CPC varies with source
adaptive state; copied reports preserve float aggregates and every seed-position
integer/null rather than converting types or claiming equal capacities.

All120 summary rows were compared.80 exact-parameter matches (40BP,40PC) disagree
on all320 displayed metric claims at the original summary precision.40CPC rows
have no exact JSON configuration match: summary thresholds/sleep intervals differ
from adaptive percentile/cooldown/dual-chemical/homeostasis JSON configurations.
Front sizesBP5=5,PC4=4,CPCsummary5 versusJSON4; equal counts do not prove equal
front membership. Every global-winner leaf/presence/type difference is recorded,
including summary14epoch/loss versusJSON20epoch/cross_entropy/aggregate fields.
Summary Best balanced score compared explicitly to source global_best_efficiency;
labels differ and identical score semantics are not inferred. JSON reports BP for
accuracy/train/inference globals and CPC for efficiency; summary reports BP for
all four. Preserve both claims rather than tune or select a preferred result.

## All original trials

| Family | Trial | Saved accuracy | Saved train SPS | Saved inference SPS | Trainable parameters | Exact original params |
|---|---:|---:|---:|---:|---:|---|
| backprop | 1 | 0.9347619047619048 | 3187.577672520138 | 4170.792749696685 | 20490.0 | {"backprop_learning_rate": 0.003, "backprop_momentum": 0.9} |
| backprop | 2 | 0.9457142857142857 | 3420.6606485435677 | 4372.643290184841 | 20490.0 | {"backprop_learning_rate": 0.005, "backprop_momentum": 0.9} |
| backprop | 3 | 0.9447619047619048 | 3451.3859800472906 | 4352.9720462449795 | 20490.0 | {"backprop_learning_rate": 0.01, "backprop_momentum": 0.9} |
| backprop | 4 | 0.9457142857142857 | 3425.7399174889138 | 4325.238627224683 | 20490.0 | {"backprop_learning_rate": 0.02, "backprop_momentum": 0.9} |
| backprop | 5 | 0.9452380952380951 | 3391.0251966873243 | 4361.165474136399 | 20490.0 | {"backprop_learning_rate": 0.03, "backprop_momentum": 0.9} |
| backprop | 6 | 0.9485714285714285 | 2833.217625048313 | 3552.2317401788146 | 20490.0 | {"backprop_learning_rate": 0.05, "backprop_momentum": 0.9} |
| backprop | 7 | 0.9404761904761904 | 2983.337059864926 | 3797.3912186129323 | 20490.0 | {"backprop_learning_rate": 0.01, "backprop_momentum": 0.85} |
| backprop | 8 | 0.9495238095238095 | 2995.5913245793963 | 3641.7250632036466 | 20490.0 | {"backprop_learning_rate": 0.01, "backprop_momentum": 0.95} |
| backprop | 9 | 0.9466666666666667 | 2880.7325868169332 | 3554.3747923229566 | 20490.0 | {"backprop_learning_rate": 0.02, "backprop_momentum": 0.85} |
| backprop | 10 | 0.9523809523809524 | 2927.516137138447 | 3799.8160467235334 | 20490.0 | {"backprop_learning_rate": 0.02, "backprop_momentum": 0.95} |
| predictive | 1 | 0.8919047619047619 | 2816.1570894466277 | 3601.2349423145747 | 527114.0 | {"predictive_head_hidden_dim": 256, "predictive_inference_learning_rate": 0.09, "predictive_inference_steps": 10, "predictive_learning_rate": 0.008} |
| predictive | 2 | 0.9028571428571429 | 2827.377471718457 | 3699.8666993166757 | 527114.0 | {"predictive_head_hidden_dim": 256, "predictive_inference_learning_rate": 0.1, "predictive_inference_steps": 10, "predictive_learning_rate": 0.01} |
| predictive | 3 | 0.9161904761904761 | 2790.6482968447344 | 3510.755194750816 | 527114.0 | {"predictive_head_hidden_dim": 256, "predictive_inference_learning_rate": 0.12, "predictive_inference_steps": 12, "predictive_learning_rate": 0.015} |
| predictive | 4 | 0.9238095238095237 | 2813.4615424411277 | 3739.988571535154 | 527114.0 | {"predictive_head_hidden_dim": 256, "predictive_inference_learning_rate": 0.12, "predictive_inference_steps": 12, "predictive_learning_rate": 0.02} |
| predictive | 5 | 0.9023809523809524 | 2797.721098285366 | 3735.69326897576 | 790666.0 | {"predictive_head_hidden_dim": 384, "predictive_inference_learning_rate": 0.09, "predictive_inference_steps": 10, "predictive_learning_rate": 0.008} |
| predictive | 6 | 0.9071428571428571 | 2888.714083187219 | 3902.7959770890316 | 790666.0 | {"predictive_head_hidden_dim": 384, "predictive_inference_learning_rate": 0.1, "predictive_inference_steps": 12, "predictive_learning_rate": 0.01} |
| predictive | 7 | 0.9114285714285714 | 2762.192670039774 | 3812.512061632188 | 790666.0 | {"predictive_head_hidden_dim": 384, "predictive_inference_learning_rate": 0.12, "predictive_inference_steps": 12, "predictive_learning_rate": 0.015} |
| predictive | 8 | 0.9152380952380952 | 2808.5194374361295 | 3556.4014001855667 | 790666.0 | {"predictive_head_hidden_dim": 384, "predictive_inference_learning_rate": 0.12, "predictive_inference_steps": 14, "predictive_learning_rate": 0.02} |
| predictive | 9 | 0.8947619047619048 | 2838.6541818421188 | 3573.6244251394987 | 1054218.0 | {"predictive_head_hidden_dim": 512, "predictive_inference_learning_rate": 0.09, "predictive_inference_steps": 10, "predictive_learning_rate": 0.008} |
| predictive | 10 | 0.9023809523809523 | 2767.405590584305 | 3682.3545029691013 | 1054218.0 | {"predictive_head_hidden_dim": 512, "predictive_inference_learning_rate": 0.1, "predictive_inference_steps": 12, "predictive_learning_rate": 0.01} |
| predictive | 11 | 0.9152380952380952 | 2845.877385052843 | 3989.7243246226644 | 1054218.0 | {"predictive_head_hidden_dim": 512, "predictive_inference_learning_rate": 0.12, "predictive_inference_steps": 12, "predictive_learning_rate": 0.015} |
| predictive | 12 | 0.9185714285714286 | 2806.772342121348 | 3756.243637459726 | 1054218.0 | {"predictive_head_hidden_dim": 512, "predictive_inference_learning_rate": 0.12, "predictive_inference_steps": 14, "predictive_learning_rate": 0.02} |
| circadian | 1 | 0.9204761904761906 | 2395.7120837009934 | 3619.8626967556556 | 794784.0 | {"circadian_adaptive_prune_percentile": 8.0, "circadian_adaptive_split_percentile": 92.0, "circadian_head_hidden_dim": 384, "circadian_inference_learning_rate": 0.15, "circadian_inference_steps": 12, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_cooldown_steps": 3, "circadian_sleep_interval": 1, "circadian_split_cooldown_steps": 3} |
| circadian | 2 | 0.9204761904761906 | 2380.5987433474647 | 3418.453374800532 | 796156.6666666666 | {"circadian_adaptive_prune_percentile": 10.0, "circadian_adaptive_split_percentile": 90.0, "circadian_head_hidden_dim": 384, "circadian_inference_learning_rate": 0.15, "circadian_inference_steps": 12, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_cooldown_steps": 3, "circadian_sleep_interval": 1, "circadian_split_cooldown_steps": 3} |
| circadian | 3 | 0.9204761904761906 | 2517.4623461463293 | 4086.930473719571 | 796156.6666666666 | {"circadian_adaptive_prune_percentile": 15.0, "circadian_adaptive_split_percentile": 88.0, "circadian_head_hidden_dim": 384, "circadian_inference_learning_rate": 0.15, "circadian_inference_steps": 12, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_cooldown_steps": 2, "circadian_sleep_interval": 1, "circadian_split_cooldown_steps": 2} |
| circadian | 4 | 0.9180952380952382 | 2694.233632680753 | 4024.3188326657814 | 794784.0 | {"circadian_adaptive_prune_percentile": 8.0, "circadian_adaptive_split_percentile": 92.0, "circadian_head_hidden_dim": 384, "circadian_inference_learning_rate": 0.12, "circadian_inference_steps": 14, "circadian_learning_rate": 0.025, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_cooldown_steps": 3, "circadian_sleep_interval": 1, "circadian_split_cooldown_steps": 3} |
| circadian | 5 | 0.9223809523809524 | 2686.8671896056862 | 4073.714143107057 | 794784.0 | {"circadian_adaptive_prune_percentile": 8.0, "circadian_adaptive_split_percentile": 92.0, "circadian_head_hidden_dim": 384, "circadian_inference_learning_rate": 0.18, "circadian_inference_steps": 10, "circadian_learning_rate": 0.035, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_cooldown_steps": 3, "circadian_sleep_interval": 1, "circadian_split_cooldown_steps": 3} |
| circadian | 6 | 0.9242857142857144 | 2545.469811501483 | 3959.496354138669 | 530545.6666666666 | {"circadian_adaptive_prune_percentile": 8.0, "circadian_adaptive_split_percentile": 92.0, "circadian_head_hidden_dim": 256, "circadian_inference_learning_rate": 0.15, "circadian_inference_steps": 12, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 768, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 96, "circadian_prune_cooldown_steps": 3, "circadian_sleep_interval": 1, "circadian_split_cooldown_steps": 3} |
| circadian | 7 | 0.9166666666666666 | 2407.507885968474 | 3541.368787281888 | 1059708.6666666667 | {"circadian_adaptive_prune_percentile": 8.0, "circadian_adaptive_split_percentile": 92.0, "circadian_head_hidden_dim": 512, "circadian_inference_learning_rate": 0.12, "circadian_inference_steps": 14, "circadian_learning_rate": 0.02, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_cooldown_steps": 3, "circadian_sleep_interval": 1, "circadian_split_cooldown_steps": 3} |
| circadian | 8 | 0.9219047619047619 | 2430.6096207563414 | 3768.3502726324527 | 1059708.6666666667 | {"circadian_adaptive_prune_percentile": 8.0, "circadian_adaptive_split_percentile": 92.0, "circadian_head_hidden_dim": 512, "circadian_inference_learning_rate": 0.15, "circadian_inference_steps": 14, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_cooldown_steps": 3, "circadian_sleep_interval": 1, "circadian_split_cooldown_steps": 3} |
| circadian | 9 | 0.9195238095238095 | 2435.084466955797 | 3877.3131630611856 | 795470.3333333334 | {"circadian_adaptive_prune_percentile": 8.0, "circadian_adaptive_split_percentile": 92.0, "circadian_head_hidden_dim": 384, "circadian_inference_learning_rate": 0.15, "circadian_inference_steps": 12, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_cooldown_steps": 3, "circadian_sleep_interval": 1, "circadian_split_cooldown_steps": 3, "circadian_use_dual_chemical": false} |
| circadian | 10 | 0.9204761904761906 | 2478.1154071714172 | 3701.2350228607233 | 794784.0 | {"circadian_adaptive_prune_percentile": 8.0, "circadian_adaptive_split_percentile": 92.0, "circadian_head_hidden_dim": 384, "circadian_inference_learning_rate": 0.15, "circadian_inference_steps": 12, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_cooldown_steps": 3, "circadian_prune_importance_mix": 0.5, "circadian_prune_weight_norm_mix": 0.2, "circadian_sleep_interval": 1, "circadian_split_cooldown_steps": 3, "circadian_split_importance_mix": 0.3, "circadian_split_weight_norm_mix": 0.2} |
| circadian | 11 | 0.9195238095238096 | 2433.4739013400917 | 3774.7628287798566 | 794784.0 | {"circadian_adaptive_prune_percentile": 8.0, "circadian_adaptive_split_percentile": 92.0, "circadian_head_hidden_dim": 384, "circadian_homeostasis_strength": 0.35, "circadian_homeostasis_target_input_norm": 1.2, "circadian_homeostasis_target_output_norm": 1.0, "circadian_homeostatic_downscale_factor": 0.995, "circadian_inference_learning_rate": 0.15, "circadian_inference_steps": 12, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_cooldown_steps": 3, "circadian_sleep_interval": 1, "circadian_split_cooldown_steps": 3} |
| circadian | 12 | 0.9199999999999999 | 2445.80239666065 | 3711.1981881624984 | 793411.3333333334 | {"circadian_adaptive_prune_percentile": 8.0, "circadian_adaptive_split_percentile": 92.0, "circadian_head_hidden_dim": 384, "circadian_inference_learning_rate": 0.15, "circadian_inference_steps": 12, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 1, "circadian_max_split_per_sleep": 1, "circadian_min_hidden_dim": 128, "circadian_prune_cooldown_steps": 3, "circadian_sleep_interval": 1, "circadian_split_cooldown_steps": 3, "circadian_split_noise_scale": 0.0} |

Exact original report bodies, every seed position, stored std/nulls and all repeated ranked/front/winner references are retained in [view.json](../artifacts/runs/p95-pareto-20261006/view.json). No seed identity is assigned to a report position.

## Saved figures

### backprop

![All backprop JSON trials](../artifacts/runs/p95-pareto-20261006/backprop.png)

[SVG](../artifacts/runs/p95-pareto-20261006/backprop.svg)

### predictive

![All predictive JSON trials](../artifacts/runs/p95-pareto-20261006/predictive.png)

[SVG](../artifacts/runs/p95-pareto-20261006/predictive.svg)

### circadian

![All circadian JSON trials](../artifacts/runs/p95-pareto-20261006/circadian.png)

[SVG](../artifacts/runs/p95-pareto-20261006/circadian.svg)

## Every summary row compared

The original full lines/parameters/four metric literals and exact JSON comparisons are preserved in the view. Summary-precision formatting is used only for comparison, not to change source statistics. Different parameters do not gain a result match through similar labels.

| Model | Original section | Rank | Exact JSON trial match | Displayed metric mismatches |
|---|---|---:|---:|---:|
| BackpropResNet50 | Top-10 Balanced Score | 1 | 3 | 4 |
| BackpropResNet50 | Top-10 Balanced Score | 2 | 6 | 4 |
| BackpropResNet50 | Top-10 Balanced Score | 3 | 10 | 4 |
| BackpropResNet50 | Top-10 Balanced Score | 4 | 2 | 4 |
| BackpropResNet50 | Top-10 Balanced Score | 5 | 8 | 4 |
| BackpropResNet50 | Top-10 Balanced Score | 6 | 4 | 4 |
| BackpropResNet50 | Top-10 Balanced Score | 7 | 5 | 4 |
| BackpropResNet50 | Top-10 Balanced Score | 8 | 7 | 4 |
| BackpropResNet50 | Top-10 Balanced Score | 9 | 9 | 4 |
| BackpropResNet50 | Top-10 Balanced Score | 10 | 1 | 4 |
| BackpropResNet50 | Top-10 Accuracy | 1 | 3 | 4 |
| BackpropResNet50 | Top-10 Accuracy | 2 | 8 | 4 |
| BackpropResNet50 | Top-10 Accuracy | 3 | 6 | 4 |
| BackpropResNet50 | Top-10 Accuracy | 4 | 10 | 4 |
| BackpropResNet50 | Top-10 Accuracy | 5 | 2 | 4 |
| BackpropResNet50 | Top-10 Accuracy | 6 | 4 | 4 |
| BackpropResNet50 | Top-10 Accuracy | 7 | 9 | 4 |
| BackpropResNet50 | Top-10 Accuracy | 8 | 7 | 4 |
| BackpropResNet50 | Top-10 Accuracy | 9 | 5 | 4 |
| BackpropResNet50 | Top-10 Accuracy | 10 | 1 | 4 |
| BackpropResNet50 | Top-10 Training Speed | 1 | 4 | 4 |
| BackpropResNet50 | Top-10 Training Speed | 2 | 7 | 4 |
| BackpropResNet50 | Top-10 Training Speed | 3 | 6 | 4 |
| BackpropResNet50 | Top-10 Training Speed | 4 | 5 | 4 |
| BackpropResNet50 | Top-10 Training Speed | 5 | 3 | 4 |
| BackpropResNet50 | Top-10 Training Speed | 6 | 8 | 4 |
| BackpropResNet50 | Top-10 Training Speed | 7 | 10 | 4 |
| BackpropResNet50 | Top-10 Training Speed | 8 | 2 | 4 |
| BackpropResNet50 | Top-10 Training Speed | 9 | 9 | 4 |
| BackpropResNet50 | Top-10 Training Speed | 10 | 1 | 4 |
| BackpropResNet50 | Top-10 Inference Speed | 1 | 10 | 4 |
| BackpropResNet50 | Top-10 Inference Speed | 2 | 5 | 4 |
| BackpropResNet50 | Top-10 Inference Speed | 3 | 2 | 4 |
| BackpropResNet50 | Top-10 Inference Speed | 4 | 4 | 4 |
| BackpropResNet50 | Top-10 Inference Speed | 5 | 7 | 4 |
| BackpropResNet50 | Top-10 Inference Speed | 6 | 6 | 4 |
| BackpropResNet50 | Top-10 Inference Speed | 7 | 3 | 4 |
| BackpropResNet50 | Top-10 Inference Speed | 8 | 9 | 4 |
| BackpropResNet50 | Top-10 Inference Speed | 9 | 1 | 4 |
| BackpropResNet50 | Top-10 Inference Speed | 10 | 8 | 4 |
| PredictiveCodingResNet50 | Top-10 Balanced Score | 1 | 5 | 4 |
| PredictiveCodingResNet50 | Top-10 Balanced Score | 2 | 8 | 4 |
| PredictiveCodingResNet50 | Top-10 Balanced Score | 3 | 1 | 4 |
| PredictiveCodingResNet50 | Top-10 Balanced Score | 4 | 7 | 4 |
| PredictiveCodingResNet50 | Top-10 Balanced Score | 5 | 10 | 4 |
| PredictiveCodingResNet50 | Top-10 Balanced Score | 6 | 6 | 4 |
| PredictiveCodingResNet50 | Top-10 Balanced Score | 7 | 2 | 4 |
| PredictiveCodingResNet50 | Top-10 Balanced Score | 8 | 4 | 4 |
| PredictiveCodingResNet50 | Top-10 Balanced Score | 9 | 3 | 4 |
| PredictiveCodingResNet50 | Top-10 Balanced Score | 10 | 9 | 4 |
| PredictiveCodingResNet50 | Top-10 Accuracy | 1 | 10 | 4 |
| PredictiveCodingResNet50 | Top-10 Accuracy | 2 | 8 | 4 |
| PredictiveCodingResNet50 | Top-10 Accuracy | 3 | 2 | 4 |
| PredictiveCodingResNet50 | Top-10 Accuracy | 4 | 5 | 4 |
| PredictiveCodingResNet50 | Top-10 Accuracy | 5 | 7 | 4 |
| PredictiveCodingResNet50 | Top-10 Accuracy | 6 | 1 | 4 |
| PredictiveCodingResNet50 | Top-10 Accuracy | 7 | 6 | 4 |
| PredictiveCodingResNet50 | Top-10 Accuracy | 8 | 4 | 4 |
| PredictiveCodingResNet50 | Top-10 Accuracy | 9 | 3 | 4 |
| PredictiveCodingResNet50 | Top-10 Accuracy | 10 | 9 | 4 |
| PredictiveCodingResNet50 | Top-10 Training Speed | 1 | 5 | 4 |
| PredictiveCodingResNet50 | Top-10 Training Speed | 2 | 4 | 4 |
| PredictiveCodingResNet50 | Top-10 Training Speed | 3 | 6 | 4 |
| PredictiveCodingResNet50 | Top-10 Training Speed | 4 | 7 | 4 |
| PredictiveCodingResNet50 | Top-10 Training Speed | 5 | 3 | 4 |
| PredictiveCodingResNet50 | Top-10 Training Speed | 6 | 1 | 4 |
| PredictiveCodingResNet50 | Top-10 Training Speed | 7 | 8 | 4 |
| PredictiveCodingResNet50 | Top-10 Training Speed | 8 | 9 | 4 |
| PredictiveCodingResNet50 | Top-10 Training Speed | 9 | 10 | 4 |
| PredictiveCodingResNet50 | Top-10 Training Speed | 10 | 2 | 4 |
| PredictiveCodingResNet50 | Top-10 Inference Speed | 1 | 3 | 4 |
| PredictiveCodingResNet50 | Top-10 Inference Speed | 2 | 5 | 4 |
| PredictiveCodingResNet50 | Top-10 Inference Speed | 3 | 6 | 4 |
| PredictiveCodingResNet50 | Top-10 Inference Speed | 4 | 4 | 4 |
| PredictiveCodingResNet50 | Top-10 Inference Speed | 5 | 8 | 4 |
| PredictiveCodingResNet50 | Top-10 Inference Speed | 6 | 9 | 4 |
| PredictiveCodingResNet50 | Top-10 Inference Speed | 7 | 1 | 4 |
| PredictiveCodingResNet50 | Top-10 Inference Speed | 8 | 7 | 4 |
| PredictiveCodingResNet50 | Top-10 Inference Speed | 9 | 11 | 4 |
| PredictiveCodingResNet50 | Top-10 Inference Speed | 10 | 10 | 4 |
| CircadianPredictiveCodingResNet50 | Top-10 Balanced Score | 1 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Balanced Score | 2 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Balanced Score | 3 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Balanced Score | 4 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Balanced Score | 5 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Balanced Score | 6 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Balanced Score | 7 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Balanced Score | 8 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Balanced Score | 9 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Balanced Score | 10 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Accuracy | 1 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Accuracy | 2 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Accuracy | 3 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Accuracy | 4 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Accuracy | 5 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Accuracy | 6 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Accuracy | 7 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Accuracy | 8 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Accuracy | 9 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Accuracy | 10 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Training Speed | 1 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Training Speed | 2 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Training Speed | 3 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Training Speed | 4 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Training Speed | 5 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Training Speed | 6 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Training Speed | 7 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Training Speed | 8 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Training Speed | 9 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Training Speed | 10 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Inference Speed | 1 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Inference Speed | 2 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Inference Speed | 3 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Inference Speed | 4 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Inference Speed | 5 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Inference Speed | 6 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Inference Speed | 7 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Inference Speed | 8 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Inference Speed | 9 | No exact configuration | Not compared across different configs |
| CircadianPredictiveCodingResNet50 | Top-10 Inference Speed | 10 | No exact configuration | Not compared across different configs |

## All global winner differences

### Best accuracy

43 full value/type/presence differences.

```json
[
  {
    "path": "/accuracy_per_million_trainable_params",
    "summary": 10.52778358781287,
    "JSON": 46.48028074089567,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/accuracy_per_million_trainable_params_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.19991924671832106
  },
  {
    "path": "/accuracy_per_train_second",
    "summary": 0.01811783693318995,
    "JSON": 0.055757799279399445,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/accuracy_per_train_second_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0007871061649355537
  },
  {
    "path": "/balanced_score",
    "summary": 0.9471485503269111,
    "JSON": 0.6484923823209432,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/circadian_hidden_dim_end_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": null
  },
  {
    "path": "/circadian_hidden_dim_start_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": null
  },
  {
    "path": "/circadian_total_prunes",
    "summary": 0,
    "JSON": 0.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/circadian_total_prunes_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/circadian_total_rollbacks",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/circadian_total_rollbacks_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/circadian_total_splits",
    "summary": 0,
    "JSON": 0.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/circadian_total_splits_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/epochs_ran",
    "summary": 14,
    "JSON": 20.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/epochs_ran_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/final_cross_entropy",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.1391670518829709
  },
  {
    "path": "/final_cross_entropy_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.017060595630511953
  },
  {
    "path": "/final_energy",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": null
  },
  {
    "path": "/final_energy_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": null
  },
  {
    "path": "/final_metric_name",
    "summary": "loss",
    "JSON": "cross_entropy",
    "summary_type": "str",
    "JSON_type": "str"
  },
  {
    "path": "/final_metric_value",
    "summary": 325.91778564453125,
    "JSON": 0.1391670518829709,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/final_metric_value_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.017060595630511953
  },
  {
    "path": "/inference_latency_mean_ms",
    "summary": 14.14512857142926,
    "JSON": 16.7722547619002,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/inference_latency_mean_ms_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.2770064793436425
  },
  {
    "path": "/inference_latency_p95_ms",
    "summary": 14.412194999997396,
    "JSON": 18.525628333319826,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/inference_latency_p95_ms_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.689637851266356
  },
  {
    "path": "/inference_samples_per_second",
    "summary": 4504.327082821833,
    "JSON": 3799.8160467235334,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/inference_samples_per_second_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 62.08239599815964
  },
  {
    "path": "/mean_train_step_ms",
    "summary": 14.104383749999938,
    "JSON": 17.278588958333643,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/mean_train_step_ms_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.2985975835863321
  },
  {
    "path": "/params/backprop_learning_rate",
    "summary": 0.01,
    "JSON": 0.02,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/params/backprop_momentum",
    "summary": 0.9,
    "JSON": 0.95,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/seed_count",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 3
  },
  {
    "path": "/test_accuracy",
    "summary": 0.21571428571428572,
    "JSON": 0.9523809523809524,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/test_accuracy_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.004096345365258395
  },
  {
    "path": "/total_parameters",
    "summary": 23528522,
    "JSON": 23528522.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/total_parameters_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/train_samples_per_second",
    "summary": 2939.649038828833,
    "JSON": 2927.516137138447,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/train_samples_per_second_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 53.948835933598865
  },
  {
    "path": "/train_seconds",
    "summary": 11.906183200000001,
    "JSON": 17.085097666666666,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/train_seconds_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.31326601967825246
  },
  {
    "path": "/trainable_parameters",
    "summary": 20490,
    "JSON": 20490.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/trainable_parameters_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  }
]
```

### Best training speed

42 full value/type/presence differences.

```json
[
  {
    "path": "/accuracy_per_million_trainable_params",
    "summary": 5.0895907411280765,
    "JSON": 46.10843849496851,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/accuracy_per_million_trainable_params_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.2629321736267373
  },
  {
    "path": "/accuracy_per_train_second",
    "summary": 0.008783024973999722,
    "JSON": 0.06521758283534879,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/accuracy_per_train_second_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0008682123628254908
  },
  {
    "path": "/balanced_score",
    "summary": 0.46075780981674863,
    "JSON": 0.7573667045799968,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/circadian_hidden_dim_end_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": null
  },
  {
    "path": "/circadian_hidden_dim_start_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": null
  },
  {
    "path": "/circadian_total_prunes",
    "summary": 0,
    "JSON": 0.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/circadian_total_prunes_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/circadian_total_rollbacks",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/circadian_total_rollbacks_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/circadian_total_splits",
    "summary": 0,
    "JSON": 0.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/circadian_total_splits_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/epochs_ran",
    "summary": 14,
    "JSON": 20.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/epochs_ran_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/final_cross_entropy",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.16713821138654436
  },
  {
    "path": "/final_cross_entropy_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.009846639263221251
  },
  {
    "path": "/final_energy",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": null
  },
  {
    "path": "/final_energy_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": null
  },
  {
    "path": "/final_metric_name",
    "summary": "loss",
    "JSON": "cross_entropy",
    "summary_type": "str",
    "JSON_type": "str"
  },
  {
    "path": "/final_metric_value",
    "summary": 3553.029296875,
    "JSON": 0.16713821138654436,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/final_metric_value_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.009846639263221251
  },
  {
    "path": "/inference_latency_mean_ms",
    "summary": 14.062157142857094,
    "JSON": 14.637030952383292,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/inference_latency_mean_ms_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.03182509937232432
  },
  {
    "path": "/inference_latency_p95_ms",
    "summary": 14.451199999999531,
    "JSON": 15.423163333326784,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/inference_latency_p95_ms_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.18919765353887172
  },
  {
    "path": "/inference_samples_per_second",
    "summary": 4530.904118551223,
    "JSON": 4352.9720462449795,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/inference_samples_per_second_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 9.453595384117358
  },
  {
    "path": "/mean_train_step_ms",
    "summary": 14.073830714285814,
    "JSON": 14.676991416666793,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/mean_train_step_ms_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.09606353981450289
  },
  {
    "path": "/params/backprop_learning_rate",
    "summary": 0.02,
    "JSON": 0.01,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/seed_count",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 3
  },
  {
    "path": "/test_accuracy",
    "summary": 0.10428571428571429,
    "JSON": 0.9447619047619048,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/test_accuracy_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.005387480237611803
  },
  {
    "path": "/total_parameters",
    "summary": 23528522,
    "JSON": 23528522.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/total_parameters_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/train_samples_per_second",
    "summary": 2947.72755976703,
    "JSON": 3451.3859800472906,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/train_samples_per_second_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 26.49897883373612
  },
  {
    "path": "/train_seconds",
    "summary": 11.873553199999996,
    "JSON": 14.487791966666663,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/train_seconds_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.11178105549499871
  },
  {
    "path": "/trainable_parameters",
    "summary": 20490,
    "JSON": 20490.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/trainable_parameters_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  }
]
```

### Best inference speed

43 full value/type/presence differences.

```json
[
  {
    "path": "/accuracy_per_million_trainable_params",
    "summary": 9.272816007808686,
    "JSON": 46.154918775709405,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/accuracy_per_million_trainable_params_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.35550578774264685
  },
  {
    "path": "/accuracy_per_train_second",
    "summary": 0.015904307631497843,
    "JSON": 0.06469288239032354,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/accuracy_per_train_second_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0008170373957084607
  },
  {
    "path": "/balanced_score",
    "summary": 0.8275778219799201,
    "JSON": 0.7794659357363976,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/circadian_hidden_dim_end_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": null
  },
  {
    "path": "/circadian_hidden_dim_start_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": null
  },
  {
    "path": "/circadian_total_prunes",
    "summary": 0,
    "JSON": 0.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/circadian_total_prunes_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/circadian_total_rollbacks",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/circadian_total_rollbacks_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/circadian_total_splits",
    "summary": 0,
    "JSON": 0.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/circadian_total_splits_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/epochs_ran",
    "summary": 14,
    "JSON": 20.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/epochs_ran_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/final_cross_entropy",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.19341518538338798
  },
  {
    "path": "/final_cross_entropy_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.013077583661598231
  },
  {
    "path": "/final_energy",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": null
  },
  {
    "path": "/final_energy_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": null
  },
  {
    "path": "/final_metric_name",
    "summary": "loss",
    "JSON": "cross_entropy",
    "summary_type": "str",
    "JSON_type": "str"
  },
  {
    "path": "/final_metric_value",
    "summary": 604.1658325195312,
    "JSON": 0.19341518538338798,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/final_metric_value_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.013077583661598231
  },
  {
    "path": "/inference_latency_mean_ms",
    "summary": 14.023942857142353,
    "JSON": 14.572747619046668,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/inference_latency_mean_ms_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.1547834183950821
  },
  {
    "path": "/inference_latency_p95_ms",
    "summary": 14.42687999999066,
    "JSON": 15.588494999999844,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/inference_latency_p95_ms_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.5100162794913486
  },
  {
    "path": "/inference_samples_per_second",
    "summary": 4543.2505225758405,
    "JSON": 4372.643290184841,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/inference_samples_per_second_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 46.13818650236871
  },
  {
    "path": "/mean_train_step_ms",
    "summary": 14.177403928571714,
    "JSON": 14.706940916666634,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/mean_train_step_ms_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.21306674777536047
  },
  {
    "path": "/params/backprop_learning_rate",
    "summary": 0.02,
    "JSON": 0.005,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/params/backprop_momentum",
    "summary": 0.95,
    "JSON": 0.9,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/seed_count",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 3
  },
  {
    "path": "/test_accuracy",
    "summary": 0.19,
    "JSON": 0.9457142857142857,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/test_accuracy_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.007284313590846836
  },
  {
    "path": "/total_parameters",
    "summary": 23528522,
    "JSON": 23528522.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/total_parameters_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/train_samples_per_second",
    "summary": 2929.7408794864446,
    "JSON": 3420.6606485435677,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/train_samples_per_second_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 59.4654075424384
  },
  {
    "path": "/train_seconds",
    "summary": 11.946449000000015,
    "JSON": 14.621425966666665,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/train_seconds_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.2511794888059488
  },
  {
    "path": "/trainable_parameters",
    "summary": 20490,
    "JSON": 20490.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/trainable_parameters_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  }
]
```

### Best balanced score

59 full value/type/presence differences.

```json
[
  {
    "path": "/accuracy_per_million_trainable_params",
    "summary": 10.52778358781287,
    "JSON": 1.160542930382283,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/accuracy_per_million_trainable_params_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0059312281172121525
  },
  {
    "path": "/accuracy_per_train_second",
    "summary": 0.01811783693318995,
    "JSON": 0.04956480367118765,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/accuracy_per_train_second_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 4.3522404848582156e-05
  },
  {
    "path": "/balanced_score",
    "summary": 0.9471485503269111,
    "JSON": 0.8526740088746357,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/circadian_hidden_dim_end",
    "summary": null,
    "JSON": 386.0,
    "summary_type": "NoneType",
    "JSON_type": "float"
  },
  {
    "path": "/circadian_hidden_dim_end_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/circadian_hidden_dim_start",
    "summary": null,
    "JSON": 384.0,
    "summary_type": "NoneType",
    "JSON_type": "float"
  },
  {
    "path": "/circadian_hidden_dim_start_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/circadian_total_prunes",
    "summary": 0,
    "JSON": 4.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/circadian_total_prunes_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 1.632993161855452
  },
  {
    "path": "/circadian_total_rollbacks",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/circadian_total_rollbacks_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/circadian_total_splits",
    "summary": 0,
    "JSON": 6.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/circadian_total_splits_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 1.632993161855452
  },
  {
    "path": "/epochs_ran",
    "summary": 14,
    "JSON": 20.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/epochs_ran_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/final_cross_entropy",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.2274686981382824
  },
  {
    "path": "/final_cross_entropy_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.015065094189063643
  },
  {
    "path": "/final_energy",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0009423577672957132
  },
  {
    "path": "/final_energy_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0008868445581316682
  },
  {
    "path": "/final_metric_name",
    "summary": "loss",
    "JSON": "cross_entropy",
    "summary_type": "str",
    "JSON_type": "str"
  },
  {
    "path": "/final_metric_value",
    "summary": 325.91778564453125,
    "JSON": 0.2274686981382824,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/final_metric_value_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.015065094189063643
  },
  {
    "path": "/inference_latency_mean_ms",
    "summary": 14.14512857142926,
    "JSON": 15.651702380940455,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/inference_latency_mean_ms_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.42188646248032496
  },
  {
    "path": "/inference_latency_p95_ms",
    "summary": 14.412194999997396,
    "JSON": 16.777673333297116,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/inference_latency_p95_ms_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.39279626616215446
  },
  {
    "path": "/inference_samples_per_second",
    "summary": 4504.327082821833,
    "JSON": 4073.714143107057,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/inference_samples_per_second_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 109.70728769340646
  },
  {
    "path": "/mean_train_step_ms",
    "summary": 14.104383749999938,
    "JSON": 17.66925249999768,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/mean_train_step_ms_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.16097430286816972
  },
  {
    "path": "/model_name",
    "summary": "BackpropResNet50",
    "JSON": "CircadianPredictiveCodingResNet50",
    "summary_type": "str",
    "JSON_type": "str"
  },
  {
    "path": "/params/backprop_learning_rate",
    "summary_present": true,
    "JSON_present": false,
    "summary": 0.01,
    "JSON": null
  },
  {
    "path": "/params/backprop_momentum",
    "summary_present": true,
    "JSON_present": false,
    "summary": 0.9,
    "JSON": null
  },
  {
    "path": "/params/circadian_adaptive_prune_percentile",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 8.0
  },
  {
    "path": "/params/circadian_adaptive_split_percentile",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 92.0
  },
  {
    "path": "/params/circadian_head_hidden_dim",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 384
  },
  {
    "path": "/params/circadian_inference_learning_rate",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.18
  },
  {
    "path": "/params/circadian_inference_steps",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 10
  },
  {
    "path": "/params/circadian_learning_rate",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.035
  },
  {
    "path": "/params/circadian_max_hidden_dim",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 1024
  },
  {
    "path": "/params/circadian_max_prune_per_sleep",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 2
  },
  {
    "path": "/params/circadian_max_split_per_sleep",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 2
  },
  {
    "path": "/params/circadian_min_hidden_dim",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 128
  },
  {
    "path": "/params/circadian_prune_cooldown_steps",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 3
  },
  {
    "path": "/params/circadian_sleep_interval",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 1
  },
  {
    "path": "/params/circadian_split_cooldown_steps",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 3
  },
  {
    "path": "/seed_count",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 3
  },
  {
    "path": "/test_accuracy",
    "summary": 0.21571428571428572,
    "JSON": 0.9223809523809524,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/test_accuracy_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.004714045207910332
  },
  {
    "path": "/total_parameters",
    "summary": 23528522,
    "JSON": 24302816.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/total_parameters_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  },
  {
    "path": "/train_samples_per_second",
    "summary": 2939.649038828833,
    "JSON": 2686.8671896056862,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/train_samples_per_second_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 15.923840660589606
  },
  {
    "path": "/train_seconds",
    "summary": 11.906183200000001,
    "JSON": 18.609687266666697,
    "summary_type": "float",
    "JSON_type": "float"
  },
  {
    "path": "/train_seconds_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.11040929849299003
  },
  {
    "path": "/trainable_parameters",
    "summary": 20490,
    "JSON": 794784.0,
    "summary_type": "int",
    "JSON_type": "float"
  },
  {
    "path": "/trainable_parameters_std",
    "summary_present": false,
    "JSON_present": true,
    "summary": null,
    "JSON": 0.0
  }
]
```

## Structure and commands

```text
artifacts/runs/p95-pareto-20261006/
  run.py                bounded commands and failed-attempt receipts
  prepare.py            complete checkout/docs/source/metadata freezing
  render.py             complete JSON validation, summary comparison, figures
  audit.py              independent whole-body/claim/figure/control readback
  validate_static.py    six-helper syntax/style/formatter AST gates
  finish.py             gated additive docs and reversible preservation
  metadata/             five complete frozen metadata bodies
  next-inputs/           two whole originals
  candidate-v1/ / candidate-v2/  failed/intermediate helpers retained
  view.json / all-trials.md / {backprop,predictive,circadian}.{png,svg}
  visual-review.json / readback.json / static-validation.json
  acceptance.json / terminal.json / final-accounting.json
  command-NNN.{json,stdout,stderr}
docs/legacy-pareto-figures.md
```

Helpers read saved bytes and render stored values; no models or original semantic
reader. A new source needs full frozen metadata/raw body, exact type/null preservation
and independent full claim readback. Keep write-once receipts and mismatched
run identities distinct. No new public module/architecture/dependency.

Exact argv/cwd/duration/stdout/stderr retained in command-NNN receipts:
001 prepare exit0 (full1019-file checkout/25packages/HEAD182077, full AGENTS/plan/
log, five metadata bodies and two full sources frozen before parsing).
002 render exit1: original circadian_use_dual_chemical=false exposed numeric-only
configuration assumption; retained candidate-v1.003 render exit0 after preserving
that exact original boolean, with numeric types kept strict. Candidate-v2 retains
the broader retry before the explicit named-boolean restriction; outputs unchanged.
004 audit exit1: malformed-summary control appended text to source without trailing
newline and raised SyntaxError; failure retained, fixture changed to an explicit
new line and SyntaxError recognized as refusal.005 audit exit0: independent whole
bodies/leaves/all120rows/320mismatches/four winner bodies/9panels/102saved labels;
seven negative controls (duplicate JSON key, nonfinite JSON, unknown/truncated
summary, wrong seed count, altered ranked copy, unknown root field) refused.
All three complete PNGs visually inspected separately; all labels/ranges readable.
006 Ruff format exit0 (three formatted/three unchanged);007 static exit0:
six helper Ruffcheck/formatcheck/compile and full pre-postformatter AST equality.
Required sequential closing008 finish prepare,009 finish close,010 scoped git
diff --check,011 final whole-checkout/task/source/metadata/output/AST/budget readback.
Actual captured outcomes govern acceptance; all four must exit0. Two failed
captured attempts remain charged to the original engineering scope.

PowerShell captured audit: `.venv/Scripts/python.exe -X utf8 artifacts/runs/p95-pareto-20261006/run.py -X utf8 -m artifacts.runs.p95-pareto-20261006.audit` (accepted write-once receipt; do not overwrite for a rerun).

Scope600 aggregate local engineering seconds/60-second hard child/64MiB owned;
fixed160-second manual/discovery/visual/closing reserve plus every captured
attempt/failure; final command reserves its full60-second cap. No whole-session
walltime/processRSS claim. Science350.7925872/360 and runtime168.7993043/180 remain
spent, no reset/rekey. Full pytest/native/Torch/CI/mypy/clean-clone and original
scientific readers skipped for ignored helpers/additive docs; original gates open.
Pillow12.3.0 already installed; no dependencies/install/download/model/dataset/
archive/array/device/CI/sweep/baseline/seed/metric/algorithm/config change or guard
repair/publication/commit/push/merge/delegation/other-chat message.
P9.5/P9.5b/G0/R0.3/fullR3.1 remain open; j6c owning-with repair human-deferred,
R0 publication separate. All unrelated changes preserved.

## Exact next action

P9.5b18: freeze complete legacy-policy-sweep publication/coverage metadata and whole benchmark_circadian_policy_sweep_results.json before parsing under a fresh small engineering scope. Inspect all saved configurations/results/seed/resource/failure/environment/unknown limits and registered existing figures; independently validate complete bodies and present only uncovered saved values. Preserve historical test-informed protocol and all parent acceptance. No old semantic reader/model/dataset/CI/scientific dispatch or inferred missing provenance.
