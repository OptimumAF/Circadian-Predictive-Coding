# Multi-seed ResNet configuration (P5.4c2)

The documented `run_multiseed_resnet_benchmark.py` route is a descriptive,
unmatched reference. Its named typed preset is `historical-unmatched` and
preserves the previous no-flag defaults: CIFAR-100, seeds 7/13/29, 12
epochs, ImageNet backbone weights, CUDA, frozen backprop backbone, and
the existing inherited model settings and learning rates. The older
individual flags still work. Their values apply after the preset;
repeatable `--override FIELD=JSON` settings apply last.

For example, the existing historical command may add
`--preset historical-unmatched --override epochs=12`. This selects the
same settings; it still requires the normal dataset, weights, hardware,
and runtime budget. The CLI refuses an occupied output prefix before
training. It does not run a sweep automatically.

The override allowlist contains only fields already exposed by this CLI:
sample and subset counts, classes, image and batch size, epochs, dataset
name/root/download/augmentation/difficulty/noise, protocol ID, device,
target accuracy, evaluation/inference/warmup batches, backbone weights,
and backbone freeze. Use typed `ResNet50BenchmarkConfig` field names,
such as `num_classes` for `--classes` and `dataset_data_root` for
`--dataset-root`. `target_accuracy=null` disables target stopping;
legacy `--target-accuracy` negative values keep their old disabled
behavior. Model head settings and learning rates are inherited from the
preset and cannot be changed through `--override` on this reference
route. Unknown, duplicate, nonfinite, wrong-type, or out-of-range
values fail before a runner opens data.

The existing JSON result gains `resolved_config` with schema
`resnet_multiseed_resolved_config_v1`. It records the preset, ordered
seeds, exact input tokens, parsed overrides, the complete typed base
configuration, and one complete configuration per requested seed.
The older `dataset`, `runtime`, `winners`, `summary`, and `per_seed`
fields and the two CSV output names remain available. Winner selection
still uses validation and speed fields; the comparison retains its
`unmatched reference` status. Old result files are never rewritten.

Why this: the historical parser hid defaults while its result JSON
omitted inherited model settings. A complete resolved record makes
new local runs inspectable without upgrading them to matched evidence
or changing any baseline to favor circadian outcomes (ADR-0127).
