# Historical master-subset saved values

## Evidence and limits

P9.5b16 is a saved presentation scope. The original is 2,438 bytes, UTF-16
little-endian with BOM; whole CRLF text roundtrips exactly. SHA256:
d22ec86990860ab4a8535f93a8ac67ab221d1fd9672efd1a56f744c379ec9db2.
All 24 nonempty lines and 44 numeric literals are preserved, including the whole
decoded body/full publication record. Five metadata bodies bound before parsing.
Registered same-family paths contain only this text: no same-family existing
figure. Older multi-seed charts/P6.10 aggregate have different sources, not reused.

Original setup: CUDA, CIFAR100/root data/size96/batch32/augmentationTrue/subsets
20000/5000,12 epochs each. Trainable parameters BP204900 versus PC/CPC825316;
total BP23712932 versus PC/CPC24333348. README labels single-seed but source seed
ID is unrecorded. Historical runner used test-label-informed early stopping/sleep
rollback and unmatched head/backbone states. Historical command declares ImageNet
weights/frozen BP backbone; declarations are not actual version/weight-byte proof.
Dependency inventory, original source commit, exact weight archive, complete
execution environment, energy units, RSS/allocator/sampler, attempts/failure
history unknown. No source failure lines does not prove a complete failure ledger.

Circadian reported throughput delta -107.0 is unchanged; rounded displayed
874.2 minus981.3 is -107.1. This inconsistency is flagged without explaining it
with invented missing sources. PC delta -16.1 retained. Saved accuracy:
PC0.692>CPC0.685>BP0.678; cross-entropy CPC1.1082<PC1.1175<BP1.7144;
throughput BP981.3>PC965.2>CPC874.2; p95 PC20.77<BP23.03<CPC23.27ms.
All values/units and sleep counts/energies are retained. Unrecorded BP energy,
sleep or delta fields are not filled with zero. Six panels show exact saved
literals with explicit zero baselines/historical/unmatched limitations.
No new means/uncertainty/composite/independent replication or scientific admission.

## Saved figure

![Six saved-value panels](../artifacts/runs/p95-master-subset-20261006/saved-values.png)

[SVG](../artifacts/runs/p95-master-subset-20261006/saved-values.svg) · [Complete saved view](../artifacts/runs/p95-master-subset-20261006/view.json)

## Every original model field

| Original field | Backprop | Predictive | Circadian |
|---|---:|---:|---:|
| epochs | 12 | 12 | 12 |
| cross_entropy | 1.7144 | 1.1175 | 1.1082 |
| accuracy | 0.678 | 0.692 | 0.685 |
| training_seconds | 244.58 | 248.65 | 274.53 |
| training_samples_per_second | 981.3 | 965.2 | 874.2 |
| training_ms_per_step | 17.69 | 18.50 | 22.03 |
| inference_mean_ms | 21.33 | 17.88 | 21.68 |
| inference_p95_ms | 23.03 | 20.77 | 23.27 |
| inference_samples_per_second | 1500.4 | 1789.7 | 1475.9 |
| total_parameters | 23,712,932 | 24,333,348 | 24,333,348 |
| trainable_parameters | 204,900 | 825,316 | 825,316 |
| reported_accuracy_delta | Unrecorded / not applicable | +0.014 | +0.007 |
| reported_throughput_delta | Unrecorded / not applicable | -16.1 | -107.0 |
| energy | Unrecorded / not applicable | 0.0006 | 0.0005 |
| hidden_before | Unrecorded / not applicable | Unrecorded / not applicable | 384 |
| hidden_after | Unrecorded / not applicable | Unrecorded / not applicable | 384 |
| splits | Unrecorded / not applicable | Unrecorded / not applicable | 0 |
| prunes | Unrecorded / not applicable | Unrecorded / not applicable | 0 |
| rollbacks | Unrecorded / not applicable | Unrecorded / not applicable | 0 |

Training_seconds=s; throughput=samples/s; training_ms_per_step=ms/step;
inference mean/p95=ms; total/trainable_parameters=counts. Energy units unknown.
Accuracy/cross-entropy remain original scalar labels. Hidden/split/prune/rollback
counts are saved sleep records, not a new telemetry sequence.

## Complete decoded original

```text
ResNet-50 Speed Benchmark (Backprop vs Predictive vs Circadian)
---------------------------------------------------------------
Device: cuda
Dataset: cifar100 (root=data, size=96, batch=32, augmentation=True, subset_train=20000, subset_test=5000)

BackpropResNet50
  epochs=12, cross_entropy=1.7144, acc=0.678
  training: 244.58s total, 981.3 samples/s, 17.69 ms/step
  inference: mean=21.33 ms, p95=23.03 ms, 1500.4 samples/s
  params: total=23,712,932, trainable=204,900

PredictiveCodingResNet50
  epochs=12, cross_entropy=1.1175, acc=0.692
  training: 248.65s total, 965.2 samples/s, 18.50 ms/step
  inference: mean=17.88 ms, p95=20.77 ms, 1789.7 samples/s
  params: total=24,333,348, trainable=825,316
  vs backprop: acc_delta=+0.014, train_samples_per_second_delta=-16.1
  energy=0.0006

CircadianPredictiveCodingResNet50
  epochs=12, cross_entropy=1.1082, acc=0.685
  training: 274.53s total, 874.2 samples/s, 22.03 ms/step
  inference: mean=21.68 ms, p95=23.27 ms, 1475.9 samples/s
  params: total=24,333,348, trainable=825,316
  vs backprop: acc_delta=+0.007, train_samples_per_second_delta=-107.0
  energy=0.0005
  circadian sleep: hidden=384->384, splits=0, prunes=0, rollbacks=0
```

Raw BOM/CRLF bytes are preserved in the frozen source.

## Structure and commands

```text
artifacts/runs/p95-master-subset-20261006/
  run.py                 bounded command receipts
  prepare.py             whole checkout/docs/source/metadata freezing
  render.py              complete literal parser and saved-value figures
  audit.py               independent body/value/figure and refusal checks
  validate_static.py     six-helper syntax/style/formatter AST gates
  finish.py              gated reversible docs and preservation
  metadata/              five complete frozen metadata bodies
  next-inputs/           whole BOM-bound original
  view.json / saved-values.{md,svg,png}
  visual-review.json / readback.json / static-validation.json
  manual-doc-stage-failure.json
  acceptance.json / terminal.json / final-accounting.json
  command-NNN.{json,stdout,stderr}
docs/legacy-master-subset-figures.md
```

Local helpers parse saved bytes and draw values; no models or old scientific
readers. Extend with independently frozen whole sources/coverage metadata, exact
unknowns, a fresh scope and independent complete manual literal readback. Keep
write-once receipts and distinct families separate.

Captured exact argv/cwd/duration/stdout/stderr in command-NNN receipts:
001 prepare exit0: full1018 checkout files/25packages/HEAD182077, full AGENTS/
plan/log read; five metadata files plus one whole source bound before parser.
002 render exit0: complete BOM text/three models, view/table/six-panel SVG/PNG.
003 audit exit0: all44 literals/full body/roundtrip, exact table/SVG labels/bar
geometry, complete PNG decode/18 bar pixels; separate whole PNG visual review.
Six negative controls refused: unknown line, truncation, nonfinite token, missing
capacity, wrong dataset, model reorder. 004 Ruff format exit0, four changed/two
unchanged. 005 static exit0: six helpers Ruffcheck/formatcheck/compile/full
pre-postformatter AST equality. Manual doc staging initially exit1 SyntaxError,
retained separately; retry only changes staging text, no helper or source repair.
Sequential required gates006 finish prepare,007 finish close,008 scoped git diff
--check,009 final metadata/source/output/checkout/task/AST/budget readback.
Actual retained outcomes govern acceptance, all four must exit0.

Captured PowerShell workflow:
`.venv/Scripts/python.exe -X utf8 artifacts/runs/p95-master-subset-20261006/run.py -X utf8 -m artifacts.runs.p95-master-subset-20261006.audit`
(already accepted; do not overwrite write-once receipts to manufacture a rerun).

Prospective local engineering scope:600 aggregate seconds,60-second hard child,
64MiB owned stage; fixed160-second manual/discovery/visual/closing reserve plus
all captured attempts. The manual documentation SyntaxError is retained separately
and included in that original fixed reserve; no reset/rekey. Final accounting
reserves its own full60-second cap. Not whole-session walltime or process RSS.
Science350.7925872/360 and runtime168.7993043/180 remain spent.
Full pytest/native/Torch/CI/mypy/clean-clone and original semantic readers skipped
in this ignored helper/additive-document scope; original full gates remain open.
Pillow12.3.0 already installed; no new dependency or install/download/model/dataset/
archive/array/device/CI/sweep/algorithm/config/baseline/seed/metric change, guard
repair, publication/commit/push/merge/delegation/other-chat message.
P9.5/P9.5b/G0/R0.3/fullR3.1 remain open; owning-with j6c human-deferred;
R0 publication separate. Source/test/architecture boundaries unchanged.

## Exact next action

P9.5b17: freeze complete legacy-pareto and legacy-pareto-summary publication/coverage metadata and whole benchmark_pareto_hard_results.json plus benchmark_pareto_hard_summary.md before parsing. Independently compare every summary claim against the JSON; preserve known mismatches, all configurations/results/seed/resource/failure/environment limits and historical test-informed protocol. Inspect registered figures and present only uncovered validated saved values. No inferred missing provenance, corrected-protocol/source/execution admission, old semantic reader/model/CI/scientific dispatch.
