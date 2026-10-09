# Historical hardest-tuning source and claim validation

## Scope and negative evidence

Only P9.5b19 saved source/claim validation and presentation is complete after gates.
Whole original benchmark_tuning_hardest_results.json: 38,191 bytes, SHA256
0525882a061092d29b84e3c488c5d632930861f2ce06c64f18db4601bee5d905.
Five complete metadata bodies bound before parsing. Full original publication,
whole JSON body and all 794 typed leaves retained; 24 original configurations
(BP6/PC8/CPC10), three family-best copies and four global report copies checked.
All 480 report/configuration table rows preserve original values/types/nulls.
Registered same-family paths contain only this JSON and no preexisting figures.
Pareto/summary/policy and canonical P6.10 aggregate sources remain separate.

Dataset declaration: hard/noise0.08/train2500/test700/classes10/image96/CUDA.
Every saved report records14epochs; requested epoch configuration is absent.
Dataset name, requested weights/defaults, actual archive/code/build/dependency
identity, seed IDs/count, per-seed reports/std/uncertainty and complete attempts/
failure/rollback/resource sampling/environment are unrecorded. Historical
protocol remains test-label informed with unmatched baselines/backbone states.
BP trainable parameters20490; PC527114/790666/1054218; CPC551822–1078926.
BP final metric is loss; PC/CPC final metric is energy. Units/reduction and
comparability are not proved; those scalar values are not renamed cross-entropy.
Loss and energy panels retain distinct source labels/ranges. No corrected-protocol
fairness, independent replication or scientific/source/execution admission.

Family-best copies are accuracy-selected BP trial2 (0.19), PC trial1
(0.11714285714285715), CPC trial10 (0.13428571428571429).
Source global accuracy and accuracy-per-training-second labels hold among all24
stored reports. Source global training speed and inference speed labels do not:
training record BP2=1417.3156399752304 samples/s is exceeded by BP5 and BP6;
inference record PC1=2030.5368152032515 samples/s is exceeded by BP4, PC2, CPC4
and CPC7. Every counterexample/value is retained below. All four stored global
records are maxima within the three accuracy-selected family records. This
observed narrower scope does not prove the original algorithm or tie rule; no
source winner is replaced and no new policy is selected. Negative claim evidence
is accepted for this saved validation scope; parent scientific criteria unchanged.

Existing accuracy/time and accuracy/million-trainable-parameter values match
arithmetic checks for all24 reports (48 boolean checks, relative tolerance1e-12).
Original ratio values unchanged; no new reported score/mean/std/uncertainty.
Primary BP/PC hidden start/end values are null;40 null leaves across whole source
including copies remain null.14 plotted hidden-end nulls have no numeric axis or
bars. CPC trials7/10 retain contractions384->382/512->510 with26splits/28prunes.
Rollback fields are absent, not zero-filled.Three full PNGs visually checked:
18panels,130 saved numeric labels and14 unknown null labels, explicit zero ranges
only for numeric panels. All24 trials shown in original order.

## Every global claim and counterexample

| Original label | Exact stored report match | All24 maximum claim | Accuracy-selected family maximum | Every exceeding source report |
|---|---|---|---|---|
| global_best_accuracy | ['backprop', 2] / 0.19 | True | True | None |
| global_best_train_speed | ['backprop', 2] / 1417.3156399752304 | False | True | backprop T5=1421.0531352514643; backprop T6=1420.521720450838 |
| global_best_inference_speed | ['predictive', 1] / 2030.5368152032515 | False | True | backprop T4=2036.6705733295667; predictive T2=2040.6940006797163; circadian T4=2063.7533194723046; circadian T7=2355.020999815334 |
| global_best_accuracy_per_train_second | ['backprop', 2] / 0.007693999188436966 | True | True | None |

Validation evaluates existing claims; broad speed labels remain unchanged in source. [Full view](../artifacts/runs/p95-hardest-tuning-20261006/view.json) preserves every original report and claim.

## All original configurations/report fields

| Family | Trial | Original field | Exact saved value/type |
|---|---:|---|---|
| backprop | 1 | params | {"backprop_learning_rate": 0.005, "backprop_momentum": 0.9} |
| backprop | 1 | model_name | "BackpropResNet50" (str) |
| backprop | 1 | epochs_ran | 14 (int) |
| backprop | 1 | final_metric_name | "loss" (str) |
| backprop | 1 | final_metric_value | 103.54399108886719 (float) |
| backprop | 1 | test_accuracy | 0.10142857142857142 (float) |
| backprop | 1 | train_seconds | 25.08838 (float) |
| backprop | 1 | train_samples_per_second | 1395.068155058238 (float) |
| backprop | 1 | mean_train_step_ms | 32.41233732142861 (float) |
| backprop | 1 | inference_latency_mean_ms | 31.667414285714354 (float) |
| backprop | 1 | inference_latency_p95_ms | 35.85936999999895 (float) |
| backprop | 1 | inference_samples_per_second | 2011.982574245987 (float) |
| backprop | 1 | total_parameters | 23528522 (int) |
| backprop | 1 | trainable_parameters | 20490 (int) |
| backprop | 1 | circadian_hidden_dim_start | null (NoneType) |
| backprop | 1 | circadian_hidden_dim_end | null (NoneType) |
| backprop | 1 | circadian_total_splits | 0 (int) |
| backprop | 1 | circadian_total_prunes | 0 (int) |
| backprop | 1 | accuracy_per_train_second | 0.0040428505718014245 (float) |
| backprop | 1 | accuracy_per_million_trainable_params | 4.950149898905389 (float) |
| backprop | 2 | params | {"backprop_learning_rate": 0.01, "backprop_momentum": 0.9} |
| backprop | 2 | model_name | "BackpropResNet50" (str) |
| backprop | 2 | epochs_ran | 14 (int) |
| backprop | 2 | final_metric_name | "loss" (str) |
| backprop | 2 | final_metric_value | 118.26506042480469 (float) |
| backprop | 2 | test_accuracy | 0.19 (float) |
| backprop | 2 | train_seconds | 24.6945698 (float) |
| backprop | 2 | train_samples_per_second | 1417.3156399752304 (float) |
| backprop | 2 | mean_train_step_ms | 31.605496071428558 (float) |
| backprop | 2 | inference_latency_mean_ms | 33.61447857142861 (float) |
| backprop | 2 | inference_latency_p95_ms | 36.27618999999811 (float) |
| backprop | 2 | inference_samples_per_second | 1895.4417388595493 (float) |
| backprop | 2 | total_parameters | 23528522 (int) |
| backprop | 2 | trainable_parameters | 20490 (int) |
| backprop | 2 | circadian_hidden_dim_start | null (NoneType) |
| backprop | 2 | circadian_hidden_dim_end | null (NoneType) |
| backprop | 2 | circadian_total_splits | 0 (int) |
| backprop | 2 | circadian_total_prunes | 0 (int) |
| backprop | 2 | accuracy_per_train_second | 0.007693999188436966 (float) |
| backprop | 2 | accuracy_per_million_trainable_params | 9.272816007808686 (float) |
| backprop | 3 | params | {"backprop_learning_rate": 0.02, "backprop_momentum": 0.9} |
| backprop | 3 | model_name | "BackpropResNet50" (str) |
| backprop | 3 | epochs_ran | 14 (int) |
| backprop | 3 | final_metric_name | "loss" (str) |
| backprop | 3 | final_metric_value | 2624.687255859375 (float) |
| backprop | 3 | test_accuracy | 0.10571428571428572 (float) |
| backprop | 3 | train_seconds | 24.746065800000004 (float) |
| backprop | 3 | train_samples_per_second | 1414.3662383698986 (float) |
| backprop | 3 | mean_train_step_ms | 31.661466785714435 (float) |
| backprop | 3 | inference_latency_mean_ms | 32.31957857142881 (float) |
| backprop | 3 | inference_latency_p95_ms | 35.80893500000215 (float) |
| backprop | 3 | inference_samples_per_second | 1971.3835554344287 (float) |
| backprop | 3 | total_parameters | 23528522 (int) |
| backprop | 3 | trainable_parameters | 20490 (int) |
| backprop | 3 | circadian_hidden_dim_start | null (NoneType) |
| backprop | 3 | circadian_hidden_dim_end | null (NoneType) |
| backprop | 3 | circadian_total_splits | 0 (int) |
| backprop | 3 | circadian_total_prunes | 0 (int) |
| backprop | 3 | accuracy_per_train_second | 0.004271963332219286 (float) |
| backprop | 3 | accuracy_per_million_trainable_params | 5.15931116223942 (float) |
| backprop | 4 | params | {"backprop_learning_rate": 0.03, "backprop_momentum": 0.9} |
| backprop | 4 | model_name | "BackpropResNet50" (str) |
| backprop | 4 | epochs_ran | 14 (int) |
| backprop | 4 | final_metric_name | "loss" (str) |
| backprop | 4 | final_metric_value | 945.39892578125 (float) |
| backprop | 4 | test_accuracy | 0.08142857142857143 (float) |
| backprop | 4 | train_seconds | 24.73151229999999 (float) |
| backprop | 4 | train_samples_per_second | 1415.1985359989494 (float) |
| backprop | 4 | mean_train_step_ms | 31.51659892857191 (float) |
| backprop | 4 | inference_latency_mean_ms | 31.283550000000762 (float) |
| backprop | 4 | inference_latency_p95_ms | 35.986160000004475 (float) |
| backprop | 4 | inference_samples_per_second | 2036.6705733295667 (float) |
| backprop | 4 | total_parameters | 23528522 (int) |
| backprop | 4 | trainable_parameters | 20490 (int) |
| backprop | 4 | circadian_hidden_dim_start | null (NoneType) |
| backprop | 4 | circadian_hidden_dim_end | null (NoneType) |
| backprop | 4 | circadian_total_splits | 0 (int) |
| backprop | 4 | circadian_total_prunes | 0 (int) |
| backprop | 4 | accuracy_per_train_second | 0.0032925027164057196 (float) |
| backprop | 4 | accuracy_per_million_trainable_params | 3.97406400334658 (float) |
| backprop | 5 | params | {"backprop_learning_rate": 0.02, "backprop_momentum": 0.95} |
| backprop | 5 | model_name | "BackpropResNet50" (str) |
| backprop | 5 | epochs_ran | 14 (int) |
| backprop | 5 | final_metric_name | "loss" (str) |
| backprop | 5 | final_metric_value | 253.97256469726562 (float) |
| backprop | 5 | test_accuracy | 0.18714285714285714 (float) |
| backprop | 5 | train_seconds | 24.629620900000006 (float) |
| backprop | 5 | train_samples_per_second | 1421.0531352514643 (float) |
| backprop | 5 | mean_train_step_ms | 31.40656482142844 (float) |
| backprop | 5 | inference_latency_mean_ms | 33.66819285714127 (float) |
| backprop | 5 | inference_latency_p95_ms | 37.46553999998952 (float) |
| backprop | 5 | inference_samples_per_second | 1892.4177482478524 (float) |
| backprop | 5 | total_parameters | 23528522 (int) |
| backprop | 5 | trainable_parameters | 20490 (int) |
| backprop | 5 | circadian_hidden_dim_start | null (NoneType) |
| backprop | 5 | circadian_hidden_dim_end | null (NoneType) |
| backprop | 5 | circadian_total_splits | 0 (int) |
| backprop | 5 | circadian_total_prunes | 0 (int) |
| backprop | 5 | accuracy_per_train_second | 0.007598284110936401 (float) |
| backprop | 5 | accuracy_per_million_trainable_params | 9.133375165586 (float) |
| backprop | 6 | params | {"backprop_learning_rate": 0.05, "backprop_momentum": 0.85} |
| backprop | 6 | model_name | "BackpropResNet50" (str) |
| backprop | 6 | epochs_ran | 14 (int) |
| backprop | 6 | final_metric_name | "loss" (str) |
| backprop | 6 | final_metric_value | 1658.9383544921875 (float) |
| backprop | 6 | test_accuracy | 0.08142857142857143 (float) |
| backprop | 6 | train_seconds | 24.638834800000012 (float) |
| backprop | 6 | train_samples_per_second | 1420.521720450838 (float) |
| backprop | 6 | mean_train_step_ms | 31.5308235714278 (float) |
| backprop | 6 | inference_latency_mean_ms | 31.9490642857139 (float) |
| backprop | 6 | inference_latency_p95_ms | 35.33714500001395 (float) |
| backprop | 6 | inference_samples_per_second | 1994.2457514405423 (float) |
| backprop | 6 | total_parameters | 23528522 (int) |
| backprop | 6 | trainable_parameters | 20490 (int) |
| backprop | 6 | circadian_hidden_dim_start | null (NoneType) |
| backprop | 6 | circadian_hidden_dim_end | null (NoneType) |
| backprop | 6 | circadian_total_splits | 0 (int) |
| backprop | 6 | circadian_total_prunes | 0 (int) |
| backprop | 6 | accuracy_per_train_second | 0.0033048872679876645 (float) |
| backprop | 6 | accuracy_per_million_trainable_params | 3.97406400334658 (float) |
| predictive | 1 | params | {"predictive_head_hidden_dim": 256, "predictive_inference_learning_rate": 0.1, "predictive_inference_steps": 10, "predictive_learning_rate": 0.01} |
| predictive | 1 | model_name | "PredictiveCodingResNet50" (str) |
| predictive | 1 | epochs_ran | 14 (int) |
| predictive | 1 | final_metric_name | "energy" (str) |
| predictive | 1 | final_metric_value | 0.03646039590239525 (float) |
| predictive | 1 | test_accuracy | 0.11714285714285715 (float) |
| predictive | 1 | train_seconds | 27.676682699999986 (float) |
| predictive | 1 | train_samples_per_second | 1264.6024228908047 (float) |
| predictive | 1 | mean_train_step_ms | 36.64374214285791 (float) |
| predictive | 1 | inference_latency_mean_ms | 31.378049999998684 (float) |
| predictive | 1 | inference_latency_p95_ms | 36.63003000001197 (float) |
| predictive | 1 | inference_samples_per_second | 2030.5368152032515 (float) |
| predictive | 1 | total_parameters | 24035146 (int) |
| predictive | 1 | trainable_parameters | 527114 (int) |
| predictive | 1 | circadian_hidden_dim_start | null (NoneType) |
| predictive | 1 | circadian_hidden_dim_end | null (NoneType) |
| predictive | 1 | circadian_total_splits | 0 (int) |
| predictive | 1 | circadian_total_prunes | 0 (int) |
| predictive | 1 | accuracy_per_train_second | 0.004232546884777387 (float) |
| predictive | 1 | accuracy_per_million_trainable_params | 0.22223438789874136 (float) |
| predictive | 2 | params | {"predictive_head_hidden_dim": 256, "predictive_inference_learning_rate": 0.12, "predictive_inference_steps": 12, "predictive_learning_rate": 0.02} |
| predictive | 2 | model_name | "PredictiveCodingResNet50" (str) |
| predictive | 2 | epochs_ran | 14 (int) |
| predictive | 2 | final_metric_name | "energy" (str) |
| predictive | 2 | final_metric_value | 0.024577222764492035 (float) |
| predictive | 2 | test_accuracy | 0.08571428571428572 (float) |
| predictive | 2 | train_seconds | 28.336083599999995 (float) |
| predictive | 2 | train_samples_per_second | 1235.1742214651006 (float) |
| predictive | 2 | mean_train_step_ms | 37.73573375000023 (float) |
| predictive | 2 | inference_latency_mean_ms | 31.22187142857464 (float) |
| predictive | 2 | inference_latency_p95_ms | 35.25886500001576 (float) |
| predictive | 2 | inference_samples_per_second | 2040.6940006797163 (float) |
| predictive | 2 | total_parameters | 24035146 (int) |
| predictive | 2 | trainable_parameters | 527114 (int) |
| predictive | 2 | circadian_hidden_dim_start | null (NoneType) |
| predictive | 2 | circadian_hidden_dim_end | null (NoneType) |
| predictive | 2 | circadian_total_splits | 0 (int) |
| predictive | 2 | circadian_total_prunes | 0 (int) |
| predictive | 2 | accuracy_per_train_second | 0.0030249164607308587 (float) |
| predictive | 2 | accuracy_per_million_trainable_params | 0.16261052773078635 (float) |
| predictive | 3 | params | {"predictive_head_hidden_dim": 256, "predictive_inference_learning_rate": 0.15, "predictive_inference_steps": 10, "predictive_learning_rate": 0.03} |
| predictive | 3 | model_name | "PredictiveCodingResNet50" (str) |
| predictive | 3 | epochs_ran | 14 (int) |
| predictive | 3 | final_metric_name | "energy" (str) |
| predictive | 3 | final_metric_value | 0.015425844117999077 (float) |
| predictive | 3 | test_accuracy | 0.08571428571428572 (float) |
| predictive | 3 | train_seconds | 27.87870239999998 (float) |
| predictive | 3 | train_samples_per_second | 1255.4386318927104 (float) |
| predictive | 3 | mean_train_step_ms | 37.159637321428974 (float) |
| predictive | 3 | inference_latency_mean_ms | 33.423992857141876 (float) |
| predictive | 3 | inference_latency_p95_ms | 36.07554999998541 (float) |
| predictive | 3 | inference_samples_per_second | 1906.2439962397118 (float) |
| predictive | 3 | total_parameters | 24035146 (int) |
| predictive | 3 | trainable_parameters | 527114 (int) |
| predictive | 3 | circadian_hidden_dim_start | null (NoneType) |
| predictive | 3 | circadian_hidden_dim_end | null (NoneType) |
| predictive | 3 | circadian_total_splits | 0 (int) |
| predictive | 3 | circadian_total_prunes | 0 (int) |
| predictive | 3 | accuracy_per_train_second | 0.003074543588308679 (float) |
| predictive | 3 | accuracy_per_million_trainable_params | 0.16261052773078635 (float) |
| predictive | 4 | params | {"predictive_head_hidden_dim": 384, "predictive_inference_learning_rate": 0.12, "predictive_inference_steps": 12, "predictive_learning_rate": 0.02} |
| predictive | 4 | model_name | "PredictiveCodingResNet50" (str) |
| predictive | 4 | epochs_ran | 14 (int) |
| predictive | 4 | final_metric_name | "energy" (str) |
| predictive | 4 | final_metric_value | 0.022630535066127777 (float) |
| predictive | 4 | test_accuracy | 0.10285714285714286 (float) |
| predictive | 4 | train_seconds | 28.23002180000003 (float) |
| predictive | 4 | train_samples_per_second | 1239.8148413757144 (float) |
| predictive | 4 | mean_train_step_ms | 37.94026589285729 (float) |
| predictive | 4 | inference_latency_mean_ms | 34.308964285705606 (float) |
| predictive | 4 | inference_latency_p95_ms | 36.34614499996189 (float) |
| predictive | 4 | inference_samples_per_second | 1857.0740050241463 (float) |
| predictive | 4 | total_parameters | 24298698 (int) |
| predictive | 4 | trainable_parameters | 790666 (int) |
| predictive | 4 | circadian_hidden_dim_start | null (NoneType) |
| predictive | 4 | circadian_hidden_dim_end | null (NoneType) |
| predictive | 4 | circadian_total_splits | 0 (int) |
| predictive | 4 | circadian_total_prunes | 0 (int) |
| predictive | 4 | accuracy_per_train_second | 0.003643537493022508 (float) |
| predictive | 4 | accuracy_per_million_trainable_params | 0.13008924483554732 (float) |
| predictive | 5 | params | {"predictive_head_hidden_dim": 384, "predictive_inference_learning_rate": 0.15, "predictive_inference_steps": 12, "predictive_learning_rate": 0.03} |
| predictive | 5 | model_name | "PredictiveCodingResNet50" (str) |
| predictive | 5 | epochs_ran | 14 (int) |
| predictive | 5 | final_metric_name | "energy" (str) |
| predictive | 5 | final_metric_value | 0.013987332582473755 (float) |
| predictive | 5 | test_accuracy | 0.10285714285714286 (float) |
| predictive | 5 | train_seconds | 29.20057300000002 (float) |
| predictive | 5 | train_samples_per_second | 1198.6066163838625 (float) |
| predictive | 5 | mean_train_step_ms | 38.985129107143635 (float) |
| predictive | 5 | inference_latency_mean_ms | 32.08287857142572 (float) |
| predictive | 5 | inference_latency_p95_ms | 37.235675000025026 (float) |
| predictive | 5 | inference_samples_per_second | 1985.927963802855 (float) |
| predictive | 5 | total_parameters | 24298698 (int) |
| predictive | 5 | trainable_parameters | 790666 (int) |
| predictive | 5 | circadian_hidden_dim_start | null (NoneType) |
| predictive | 5 | circadian_hidden_dim_end | null (NoneType) |
| predictive | 5 | circadian_total_splits | 0 (int) |
| predictive | 5 | circadian_total_prunes | 0 (int) |
| predictive | 5 | accuracy_per_train_second | 0.003522435770597474 (float) |
| predictive | 5 | accuracy_per_million_trainable_params | 0.13008924483554732 (float) |
| predictive | 6 | params | {"predictive_head_hidden_dim": 512, "predictive_inference_learning_rate": 0.12, "predictive_inference_steps": 10, "predictive_learning_rate": 0.02} |
| predictive | 6 | model_name | "PredictiveCodingResNet50" (str) |
| predictive | 6 | epochs_ran | 14 (int) |
| predictive | 6 | final_metric_name | "energy" (str) |
| predictive | 6 | final_metric_value | 0.014594187960028648 (float) |
| predictive | 6 | test_accuracy | 0.10428571428571429 (float) |
| predictive | 6 | train_seconds | 28.326037199999973 (float) |
| predictive | 6 | train_samples_per_second | 1235.6123008974948 (float) |
| predictive | 6 | mean_train_step_ms | 37.516275892858374 (float) |
| predictive | 6 | inference_latency_mean_ms | 33.02332142857673 (float) |
| predictive | 6 | inference_latency_p95_ms | 37.227500000020086 (float) |
| predictive | 6 | inference_samples_per_second | 1929.3724240333547 (float) |
| predictive | 6 | total_parameters | 24562250 (int) |
| predictive | 6 | trainable_parameters | 1054218 (int) |
| predictive | 6 | circadian_hidden_dim_start | null (NoneType) |
| predictive | 6 | circadian_hidden_dim_end | null (NoneType) |
| predictive | 6 | circadian_total_splits | 0 (int) |
| predictive | 6 | circadian_total_prunes | 0 (int) |
| predictive | 6 | accuracy_per_train_second | 0.003681620325123148 (float) |
| predictive | 6 | accuracy_per_million_trainable_params | 0.09892234270873224 (float) |
| predictive | 7 | params | {"predictive_head_hidden_dim": 512, "predictive_inference_learning_rate": 0.15, "predictive_inference_steps": 12, "predictive_learning_rate": 0.03} |
| predictive | 7 | model_name | "PredictiveCodingResNet50" (str) |
| predictive | 7 | epochs_ran | 14 (int) |
| predictive | 7 | final_metric_name | "energy" (str) |
| predictive | 7 | final_metric_value | 0.010758217424154282 (float) |
| predictive | 7 | test_accuracy | 0.08571428571428572 (float) |
| predictive | 7 | train_seconds | 28.17999040000001 (float) |
| predictive | 7 | train_samples_per_second | 1242.0160370246253 (float) |
| predictive | 7 | mean_train_step_ms | 37.501828035715334 (float) |
| predictive | 7 | inference_latency_mean_ms | 32.337300000002806 (float) |
| predictive | 7 | inference_latency_p95_ms | 37.03115000000423 (float) |
| predictive | 7 | inference_samples_per_second | 1970.3032013891138 (float) |
| predictive | 7 | total_parameters | 24562250 (int) |
| predictive | 7 | trainable_parameters | 1054218 (int) |
| predictive | 7 | circadian_hidden_dim_start | null (NoneType) |
| predictive | 7 | circadian_hidden_dim_end | null (NoneType) |
| predictive | 7 | circadian_total_splits | 0 (int) |
| predictive | 7 | circadian_total_prunes | 0 (int) |
| predictive | 7 | accuracy_per_train_second | 0.003041671927407246 (float) |
| predictive | 7 | accuracy_per_million_trainable_params | 0.0813060351030676 (float) |
| predictive | 8 | params | {"predictive_head_hidden_dim": 512, "predictive_inference_learning_rate": 0.18, "predictive_inference_steps": 14, "predictive_learning_rate": 0.05} |
| predictive | 8 | model_name | "PredictiveCodingResNet50" (str) |
| predictive | 8 | epochs_ran | 14 (int) |
| predictive | 8 | final_metric_name | "energy" (str) |
| predictive | 8 | final_metric_value | 0.006538061425089836 (float) |
| predictive | 8 | test_accuracy | 0.10285714285714286 (float) |
| predictive | 8 | train_seconds | 28.970703200000003 (float) |
| predictive | 8 | train_samples_per_second | 1208.117033210295 (float) |
| predictive | 8 | mean_train_step_ms | 38.784524107142154 (float) |
| predictive | 8 | inference_latency_mean_ms | 33.87524285714351 (float) |
| predictive | 8 | inference_latency_p95_ms | 36.44559999997057 (float) |
| predictive | 8 | inference_samples_per_second | 1880.8510387062756 (float) |
| predictive | 8 | total_parameters | 24562250 (int) |
| predictive | 8 | trainable_parameters | 1054218 (int) |
| predictive | 8 | circadian_hidden_dim_start | null (NoneType) |
| predictive | 8 | circadian_hidden_dim_end | null (NoneType) |
| predictive | 8 | circadian_total_splits | 0 (int) |
| predictive | 8 | circadian_total_prunes | 0 (int) |
| predictive | 8 | accuracy_per_train_second | 0.0035503847506588256 (float) |
| predictive | 8 | accuracy_per_million_trainable_params | 0.0975672421236811 (float) |
| circadian | 1 | params | {"circadian_head_hidden_dim": 256, "circadian_inference_learning_rate": 0.12, "circadian_inference_steps": 10, "circadian_learning_rate": 0.02, "circadian_max_hidden_dim": 768, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 96, "circadian_prune_threshold": 0.65, "circadian_sleep_interval": 2, "circadian_split_threshold": 0.8} |
| circadian | 1 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| circadian | 1 | epochs_ran | 14 (int) |
| circadian | 1 | final_metric_name | "energy" (str) |
| circadian | 1 | final_metric_value | 0.03397253528237343 (float) |
| circadian | 1 | test_accuracy | 0.10428571428571429 (float) |
| circadian | 1 | train_seconds | 28.59748000000002 (float) |
| circadian | 1 | train_samples_per_second | 1223.884062511801 (float) |
| circadian | 1 | mean_train_step_ms | 38.22268428571423 (float) |
| circadian | 1 | inference_latency_mean_ms | 32.119685714284124 (float) |
| circadian | 1 | inference_latency_p95_ms | 35.20037499998523 (float) |
| circadian | 1 | inference_samples_per_second | 1983.6522150635794 (float) |
| circadian | 1 | total_parameters | 24063972 (int) |
| circadian | 1 | trainable_parameters | 555940 (int) |
| circadian | 1 | circadian_hidden_dim_start | 256 (int) |
| circadian | 1 | circadian_hidden_dim_end | 270 (int) |
| circadian | 1 | circadian_total_splits | 14 (int) |
| circadian | 1 | circadian_total_prunes | 0 (int) |
| circadian | 1 | accuracy_per_train_second | 0.003646674961769856 (float) |
| circadian | 1 | accuracy_per_million_trainable_params | 0.18758447725602456 (float) |
| circadian | 2 | params | {"circadian_head_hidden_dim": 256, "circadian_inference_learning_rate": 0.15, "circadian_inference_steps": 10, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 768, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 96, "circadian_prune_threshold": 0.65, "circadian_sleep_interval": 2, "circadian_split_threshold": 0.8} |
| circadian | 2 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| circadian | 2 | epochs_ran | 14 (int) |
| circadian | 2 | final_metric_name | "energy" (str) |
| circadian | 2 | final_metric_value | 0.031469184905290604 (float) |
| circadian | 2 | test_accuracy | 0.09571428571428571 (float) |
| circadian | 2 | train_seconds | 28.06326710000002 (float) |
| circadian | 2 | train_samples_per_second | 1247.1819434024478 (float) |
| circadian | 2 | mean_train_step_ms | 37.38916553571526 (float) |
| circadian | 2 | inference_latency_mean_ms | 32.303957142853996 (float) |
| circadian | 2 | inference_latency_p95_ms | 36.52203500000155 (float) |
| circadian | 2 | inference_samples_per_second | 1972.3368698308234 (float) |
| circadian | 2 | total_parameters | 24063972 (int) |
| circadian | 2 | trainable_parameters | 555940 (int) |
| circadian | 2 | circadian_hidden_dim_start | 256 (int) |
| circadian | 2 | circadian_hidden_dim_end | 270 (int) |
| circadian | 2 | circadian_total_splits | 14 (int) |
| circadian | 2 | circadian_total_prunes | 0 (int) |
| circadian | 2 | accuracy_per_train_second | 0.0034106608248148573 (float) |
| circadian | 2 | accuracy_per_million_trainable_params | 0.17216657501580335 (float) |
| circadian | 3 | params | {"circadian_head_hidden_dim": 256, "circadian_inference_learning_rate": 0.15, "circadian_inference_steps": 12, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 768, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 96, "circadian_prune_threshold": 0.65, "circadian_sleep_interval": 2, "circadian_split_threshold": 0.78} |
| circadian | 3 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| circadian | 3 | epochs_ran | 14 (int) |
| circadian | 3 | final_metric_name | "energy" (str) |
| circadian | 3 | final_metric_value | 0.028304046019911766 (float) |
| circadian | 3 | test_accuracy | 0.1 (float) |
| circadian | 3 | train_seconds | 28.485069399999986 (float) |
| circadian | 3 | train_samples_per_second | 1228.7138749256485 (float) |
| circadian | 3 | mean_train_step_ms | 38.13809035714298 (float) |
| circadian | 3 | inference_latency_mean_ms | 31.43062142857746 (float) |
| circadian | 3 | inference_latency_p95_ms | 35.94200999999657 (float) |
| circadian | 3 | inference_samples_per_second | 2027.14050242593 (float) |
| circadian | 3 | total_parameters | 24063972 (int) |
| circadian | 3 | trainable_parameters | 555940 (int) |
| circadian | 3 | circadian_hidden_dim_start | 256 (int) |
| circadian | 3 | circadian_hidden_dim_end | 270 (int) |
| circadian | 3 | circadian_total_splits | 14 (int) |
| circadian | 3 | circadian_total_prunes | 0 (int) |
| circadian | 3 | accuracy_per_train_second | 0.0035106110712161387 (float) |
| circadian | 3 | accuracy_per_million_trainable_params | 0.17987552613591395 (float) |
| circadian | 4 | params | {"circadian_head_hidden_dim": 256, "circadian_inference_learning_rate": 0.18, "circadian_inference_steps": 12, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 768, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 96, "circadian_prune_threshold": 0.68, "circadian_sleep_interval": 2, "circadian_split_threshold": 0.78} |
| circadian | 4 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| circadian | 4 | epochs_ran | 14 (int) |
| circadian | 4 | final_metric_name | "energy" (str) |
| circadian | 4 | final_metric_value | 0.026054078713059425 (float) |
| circadian | 4 | test_accuracy | 0.10428571428571429 (float) |
| circadian | 4 | train_seconds | 28.529255300000045 (float) |
| circadian | 4 | train_samples_per_second | 1226.8108519467714 (float) |
| circadian | 4 | mean_train_step_ms | 38.16958714285811 (float) |
| circadian | 4 | inference_latency_mean_ms | 30.873014285727354 (float) |
| circadian | 4 | inference_latency_p95_ms | 36.29132499997354 (float) |
| circadian | 4 | inference_samples_per_second | 2063.7533194723046 (float) |
| circadian | 4 | total_parameters | 24059854 (int) |
| circadian | 4 | trainable_parameters | 551822 (int) |
| circadian | 4 | circadian_hidden_dim_start | 256 (int) |
| circadian | 4 | circadian_hidden_dim_end | 268 (int) |
| circadian | 4 | circadian_total_splits | 14 (int) |
| circadian | 4 | circadian_total_prunes | 2 (int) |
| circadian | 4 | accuracy_per_train_second | 0.0036553955996781354 (float) |
| circadian | 4 | accuracy_per_million_trainable_params | 0.18898433604625092 (float) |
| circadian | 5 | params | {"circadian_head_hidden_dim": 384, "circadian_inference_learning_rate": 0.12, "circadian_inference_steps": 10, "circadian_learning_rate": 0.02, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_threshold": 0.65, "circadian_sleep_interval": 2, "circadian_split_threshold": 0.8} |
| circadian | 5 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| circadian | 5 | epochs_ran | 14 (int) |
| circadian | 5 | final_metric_name | "energy" (str) |
| circadian | 5 | final_metric_value | 0.029964497312903404 (float) |
| circadian | 5 | test_accuracy | 0.08571428571428572 (float) |
| circadian | 5 | train_seconds | 28.453073500000073 (float) |
| circadian | 5 | train_samples_per_second | 1230.0955817655308 (float) |
| circadian | 5 | mean_train_step_ms | 37.921805714284424 (float) |
| circadian | 5 | inference_latency_mean_ms | 34.286385714283696 (float) |
| circadian | 5 | inference_latency_p95_ms | 37.006334999949786 (float) |
| circadian | 5 | inference_samples_per_second | 1858.2969416850015 (float) |
| circadian | 5 | total_parameters | 24327524 (int) |
| circadian | 5 | trainable_parameters | 819492 (int) |
| circadian | 5 | circadian_hidden_dim_start | 384 (int) |
| circadian | 5 | circadian_hidden_dim_end | 398 (int) |
| circadian | 5 | circadian_total_splits | 14 (int) |
| circadian | 5 | circadian_total_prunes | 0 (int) |
| circadian | 5 | accuracy_per_train_second | 0.0030124789757523204 (float) |
| circadian | 5 | accuracy_per_million_trainable_params | 0.10459441423990193 (float) |
| circadian | 6 | params | {"circadian_head_hidden_dim": 384, "circadian_inference_learning_rate": 0.15, "circadian_inference_steps": 12, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_threshold": 0.65, "circadian_sleep_interval": 2, "circadian_split_threshold": 0.8} |
| circadian | 6 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| circadian | 6 | epochs_ran | 14 (int) |
| circadian | 6 | final_metric_name | "energy" (str) |
| circadian | 6 | final_metric_value | 0.02460632100701332 (float) |
| circadian | 6 | test_accuracy | 0.10428571428571429 (float) |
| circadian | 6 | train_seconds | 29.337850600000024 (float) |
| circadian | 6 | train_samples_per_second | 1192.998099185902 (float) |
| circadian | 6 | mean_train_step_ms | 39.404048214284764 (float) |
| circadian | 6 | inference_latency_mean_ms | 32.91160714285622 (float) |
| circadian | 6 | inference_latency_p95_ms | 37.72194000000013 (float) |
| circadian | 6 | inference_samples_per_second | 1935.921434578606 (float) |
| circadian | 6 | total_parameters | 24327524 (int) |
| circadian | 6 | trainable_parameters | 819492 (int) |
| circadian | 6 | circadian_hidden_dim_start | 384 (int) |
| circadian | 6 | circadian_hidden_dim_end | 398 (int) |
| circadian | 6 | circadian_total_splits | 14 (int) |
| circadian | 6 | circadian_total_prunes | 0 (int) |
| circadian | 6 | accuracy_per_train_second | 0.00355464739757432 (float) |
| circadian | 6 | accuracy_per_million_trainable_params | 0.12725653732521403 (float) |
| circadian | 7 | params | {"circadian_head_hidden_dim": 384, "circadian_inference_learning_rate": 0.15, "circadian_inference_steps": 12, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_threshold": 0.68, "circadian_sleep_interval": 1, "circadian_split_threshold": 0.8} |
| circadian | 7 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| circadian | 7 | epochs_ran | 14 (int) |
| circadian | 7 | final_metric_name | "energy" (str) |
| circadian | 7 | final_metric_value | 0.021584125235676765 (float) |
| circadian | 7 | test_accuracy | 0.10571428571428572 (float) |
| circadian | 7 | train_seconds | 29.43664709999996 (float) |
| circadian | 7 | train_samples_per_second | 1188.9941093189261 (float) |
| circadian | 7 | mean_train_step_ms | 39.491405892855724 (float) |
| circadian | 7 | inference_latency_mean_ms | 27.054657142879737 (float) |
| circadian | 7 | inference_latency_p95_ms | 35.593840000041155 (float) |
| circadian | 7 | inference_samples_per_second | 2355.020999815334 (float) |
| circadian | 7 | total_parameters | 24294580 (int) |
| circadian | 7 | trainable_parameters | 786548 (int) |
| circadian | 7 | circadian_hidden_dim_start | 384 (int) |
| circadian | 7 | circadian_hidden_dim_end | 382 (int) |
| circadian | 7 | circadian_total_splits | 26 (int) |
| circadian | 7 | circadian_total_prunes | 28 (int) |
| circadian | 7 | accuracy_per_train_second | 0.003591247513861246 (float) |
| circadian | 7 | accuracy_per_million_trainable_params | 0.13440284091280597 (float) |
| circadian | 8 | params | {"circadian_head_hidden_dim": 512, "circadian_inference_learning_rate": 0.12, "circadian_inference_steps": 10, "circadian_learning_rate": 0.02, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_threshold": 0.68, "circadian_sleep_interval": 2, "circadian_split_threshold": 0.82} |
| circadian | 8 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| circadian | 8 | epochs_ran | 14 (int) |
| circadian | 8 | final_metric_name | "energy" (str) |
| circadian | 8 | final_metric_value | 0.02753276936709881 (float) |
| circadian | 8 | test_accuracy | 0.11285714285714285 (float) |
| circadian | 8 | train_seconds | 28.33375190000004 (float) |
| circadian | 8 | train_samples_per_second | 1235.2758689893078 (float) |
| circadian | 8 | mean_train_step_ms | 38.15277125000144 (float) |
| circadian | 8 | inference_latency_mean_ms | 32.05266428573493 (float) |
| circadian | 8 | inference_latency_p95_ms | 34.63863500006141 (float) |
| circadian | 8 | inference_samples_per_second | 1987.7999889900518 (float) |
| circadian | 8 | total_parameters | 24584899 (int) |
| circadian | 8 | trainable_parameters | 1076867 (int) |
| circadian | 8 | circadian_hidden_dim_start | 512 (int) |
| circadian | 8 | circadian_hidden_dim_end | 523 (int) |
| circadian | 8 | circadian_total_splits | 14 (int) |
| circadian | 8 | circadian_total_prunes | 3 (int) |
| circadian | 8 | accuracy_per_train_second | 0.003983134434700217 (float) |
| circadian | 8 | accuracy_per_million_trainable_params | 0.10480137552468675 (float) |
| circadian | 9 | params | {"circadian_head_hidden_dim": 512, "circadian_inference_learning_rate": 0.15, "circadian_inference_steps": 12, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_threshold": 0.68, "circadian_sleep_interval": 2, "circadian_split_threshold": 0.82} |
| circadian | 9 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| circadian | 9 | epochs_ran | 14 (int) |
| circadian | 9 | final_metric_name | "energy" (str) |
| circadian | 9 | final_metric_value | 0.022670505568385124 (float) |
| circadian | 9 | test_accuracy | 0.1 (float) |
| circadian | 9 | train_seconds | 28.655722699999956 (float) |
| circadian | 9 | train_samples_per_second | 1221.3965205630655 (float) |
| circadian | 9 | mean_train_step_ms | 38.458492499998236 (float) |
| circadian | 9 | inference_latency_mean_ms | 34.83185000000536 (float) |
| circadian | 9 | inference_latency_p95_ms | 37.73689499996067 (float) |
| circadian | 9 | inference_samples_per_second | 1829.1961441690842 (float) |
| circadian | 9 | total_parameters | 24586958 (int) |
| circadian | 9 | trainable_parameters | 1078926 (int) |
| circadian | 9 | circadian_hidden_dim_start | 512 (int) |
| circadian | 9 | circadian_hidden_dim_end | 524 (int) |
| circadian | 9 | circadian_total_splits | 14 (int) |
| circadian | 9 | circadian_total_prunes | 2 (int) |
| circadian | 9 | accuracy_per_train_second | 0.0034897043444659017 (float) |
| circadian | 9 | accuracy_per_million_trainable_params | 0.09268476243968539 (float) |
| circadian | 10 | params | {"circadian_head_hidden_dim": 512, "circadian_inference_learning_rate": 0.18, "circadian_inference_steps": 12, "circadian_learning_rate": 0.03, "circadian_max_hidden_dim": 1024, "circadian_max_prune_per_sleep": 2, "circadian_max_split_per_sleep": 2, "circadian_min_hidden_dim": 128, "circadian_prune_threshold": 0.7, "circadian_sleep_interval": 1, "circadian_split_threshold": 0.8} |
| circadian | 10 | model_name | "CircadianPredictiveCodingResNet50" (str) |
| circadian | 10 | epochs_ran | 14 (int) |
| circadian | 10 | final_metric_name | "energy" (str) |
| circadian | 10 | final_metric_value | 0.015308640897274017 (float) |
| circadian | 10 | test_accuracy | 0.13428571428571429 (float) |
| circadian | 10 | train_seconds | 28.666159300000004 (float) |
| circadian | 10 | train_samples_per_second | 1220.951841986031 (float) |
| circadian | 10 | mean_train_step_ms | 38.61632160714379 (float) |
| circadian | 10 | inference_latency_mean_ms | 34.02201428571873 (float) |
| circadian | 10 | inference_latency_p95_ms | 36.722779999996646 (float) |
| circadian | 10 | inference_samples_per_second | 1872.7370219531879 (float) |
| circadian | 10 | total_parameters | 24558132 (int) |
| circadian | 10 | trainable_parameters | 1050100 (int) |
| circadian | 10 | circadian_hidden_dim_start | 512 (int) |
| circadian | 10 | circadian_hidden_dim_end | 510 (int) |
| circadian | 10 | circadian_total_splits | 26 (int) |
| circadian | 10 | circadian_total_prunes | 28 (int) |
| circadian | 10 | accuracy_per_train_second | 0.004684468291701507 (float) |
| circadian | 10 | accuracy_per_million_trainable_params | 0.12787897751234575 (float) |

Time=s, throughput=samples/s, train step=ms/step, inference mean/p95=ms, capacities/hidden/split/prune fields=counts. Source loss/energy units and reduction are unproved. Nulls are not zero. Ratio arithmetic checks verify existing values only.

## Every original trial plotted

### backprop

![Original backprop reports](../artifacts/runs/p95-hardest-tuning-20261006/backprop.png)

[SVG](../artifacts/runs/p95-hardest-tuning-20261006/backprop.svg)

### predictive

![Original predictive reports](../artifacts/runs/p95-hardest-tuning-20261006/predictive.png)

[SVG](../artifacts/runs/p95-hardest-tuning-20261006/predictive.svg)

### circadian

![Original circadian reports](../artifacts/runs/p95-hardest-tuning-20261006/circadian.png)

[SVG](../artifacts/runs/p95-hardest-tuning-20261006/circadian.svg)

## Structure and workflow

```text
artifacts/runs/p95-hardest-tuning-20261006/
  run.py                 bounded command/failed-attempt capture
  prepare.py             whole checkout/docs/source/metadata freezing
  render.py              complete body and true/false claim validation
  audit.py               independent leaves/claims/table/figures/controls
  validate_static.py     six-helper syntax/style/formatter AST gates
  finish.py              gated additive docs and reversible preservation
  metadata/ / next-inputs/  five complete metadata bodies and exact source
  candidate-v1/ / candidate-v2/  previous/failed helper versions retained
  view.json / all-reports.md / {backprop,predictive,circadian}.{png,svg}
  diagnostic-claim-control.json  full edited diagnostic body, not science
  visual-review.json / readback.json / static-validation.json
  acceptance.json / terminal.json / final-accounting.json
  command-NNN.{json,stdout,stderr}
docs/legacy-hardest-tuning-figures.md
```

Local helpers read saved bytes and draw originals; no public module/dependency or
architecture change. Extend with separately frozen source/metadata and an independent
whole-body/typed/claim readback. A negative claim result must remain visible; do not
repair source labels or infer a better winner. Keep write-once receipts.

Exact argv/cwd/duration/stdout/stderr retained in command-NNN receipts:
001 prepare exit0: full1021-file checkout/25packages/HEAD182077, full AGENTS/plan/
log read, five metadata bodies and exact source bound before parsing.
002 render exit0: complete body/claims/ratios and original view/table/three PNG/SVGs.
Null-schema tightening retained prior helper candidate; no source/figure change.
003 audit exit1: helper's tuple report coordinates differed from JSON list after
roundtrip; failed candidate retained. Return coordinates made explicit JSON lists;
serialized view/data/figures unchanged.004 audit exit0: whole794typed leaves/
24trials/7copied records/40nulls/48ratio checks/all6speed counterexamples/480rows/
18panels/SVGlabels/geometry/PNGdecode/pixels. Seven refusal controls pass:
duplicate key, nonfinite JSON, wrong count, altered family reference, null-to-zero,
boolean numeric report, unknown root. Complete edited diagnostic claim-control
body/validation explicitly separate: contradictory accuracy claim reported false
without source repair; original body revalidated. All three PNGs visually inspected.
005 Ruffformat exit0 (two formatted/four unchanged).006 static exit0: six-helper
Ruffcheck/formatcheck/compile/full pre-postformatter AST equality after explicit
logic repairs (not an equivalence claim with earlier candidate versions).
Required sequential007 finish prepare,008 finish close,009 scoped git diff
--check,010 final source/output/metadata/checkout/task/AST/budget readback.
Actual retained outcomes govern acceptance; all four must exit0. Failed audit
charged to the original scope; diagnostic edits are not scientific positives.

PowerShell captured audit: `.venv/Scripts/python.exe -X utf8 artifacts/runs/p95-hardest-tuning-20261006/run.py -X utf8 -m artifacts.runs.p95-hardest-tuning-20261006.audit` (accepted write-once receipt; do not overwrite to rerun).

Prospective engineering scope:600 aggregate seconds,60-second hard child cap,
64MiB owned stage; fixed160-second manual/discovery/visual/closing reserve plus
all captured attempts/failures. Final command reserves its whole60-second cap;
not whole-session walltime or processRSS. Science350.7925872/360 and
runtime168.7993043/180 remain spent, no reset/rekey. One failed audit retained.
Full pytest/native/Torch/CI/mypy/clean-clone and original scientific readers skipped
in this ignored helper/additive-doc scope; original full gates remain open.
Pillow12.3.0 already installed; no dependency/install/download/model/dataset/archive/
array/device/CI/sweep/algorithm/config/baseline/seed/metric change or guard repair/
publication/commit/push/merge/delegation/other-chat message.
P9.5/P9.5b/G0/R0.3/full R3.1 remain open; owning-with j6c human-deferred,
R0 publication separate. Source/test/architecture/unrelated changes preserved.

## Exact next action

P9.5b20: freeze full legacy-continual-strength publication/coverage metadata and whole docs/benchmarks/benchmark_continual_shift_strength_case_2026-02-28.txt before parsing under a fresh small engineering scope. Inspect the complete text, exact original configurations/results/protocol/seed/resource/failure/history/environment/unknown limits and registered figures; independently validate all saved values and present only uncovered views. Preserve the historical tuned profile versus corrected profile-repeat distinction. No current-default backfill, other-family borrowing, inferred independence/source/execution admission or original semantic reader/model/dataset/CI/scientific dispatch.
