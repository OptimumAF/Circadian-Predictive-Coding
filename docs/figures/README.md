# Figures

This folder stores visual assets used by `README.md`.

## Generated Artifacts

- `benchmark_overview_compact.png`
- `benchmark_accuracy.png`
- `benchmark_train_speed.png`
- `benchmark_inference_latency_p95.png`
- `hardest_mode_dynamics.gif`
- `interactive_hardest_mode_dynamics.html`
- `circadian_sleep_dynamics.gif`
- `interactive_benchmark_overview.html`
- `interactive_benchmark_accuracy.html`
- `interactive_benchmark_train_speed.html`
- `interactive_benchmark_inference_latency_p95.html`

## Regeneration

Run:

```powershell
python scripts/generate_readme_figures.py --summary-csv benchmark_multiseed_cifar100_summary.csv --output-dir docs/figures
python scripts/generate_hardest_mode_dynamics.py --gif-output-path docs/figures/hardest_mode_dynamics.gif --interactive-output-path docs/figures/interactive_hardest_mode_dynamics.html
```

The GIF is illustrative and intended for communication in the README.
Plotly HTML charts are interactive when opened in a normal browser context
(local file, static host, or GitHub Pages).

The combined dashboard page is at `docs/index.html`.
When Pages is enabled, the live dashboard URL is:
`https://optimumaf.github.io/Circadian-Predictive-Coding/`.


## Current historical artifact export (2026-10-06)

The asset names and regeneration recipes above are preserved historical guidance;
legacy benchmark figures have not been reproduced under the new protocol. The
dynamics GIFs are illustrative. The new [historical artifact exporter](../historical-outcome-figures.md)
uses only the complete already accepted P6.10 presentation and runs no model or
scientific reader. Accepted output: `artifacts/runs/p95-historical-figures-20261006/export-v2/`
(18 SVGs, an 18-page PDF, report, exact JSON view/series and hashed manifest).
Original uncertainty, all 560 cells and negative/nulls remain. Work/owned/shared
storage/whole-process RSS scopes are distinct; per-arm timing/RSS is unmeasured.
The first export remains retained and unaccepted for report-column/secondary
interval presentation issues. Remaining assets/reports require full P9.5b
coverage reconciliation before parent P9.5 acceptance.


## 2026-10-06 - Complete presentation inventory and saved matched confirmation

See [research presentation coverage](../research-presentation-coverage.md) for
all62 original families,341 entry documents/12 legacy assets and the complete
saved matched confirmation figures. Original negative outcomes, uncertainty and
resource limits remain; P9.5/P9.5b remain unfinished. No new scientific run.


## 2026-10-06 - Fixed-v14 saved-repeat presentation complement

See [fixed-v14 repeat figures](../fixed-v14-repeat-figures.md): six original
repeat bodies/old dashboard chains and complete uncovered work/contrast/history
presentation. Original two seeds stay two; no new statistics or experiment.
P9.5b3 complete for this scope, parent P9.5/P9.5b remain open.


## 2026-10-06 - Saved CUDA order-control figures

See [saved order-control figures](../vision-order-control-figures.md): four exact
original seed149 bodies, three pages covering scores/capacity/both timing orders
and original control facts. Nine declared score deltas are zero at original
1e-6 tolerance; all test accuracies zero, unequal capacities, timings vary.
Single-seed unmatched reference only; no fresh isolation/scientific admission.
P9.5b4 complete for saved presentation; parent P9.5/P9.5b remain open.


## 2026-10-06 - Saved CIFAR loader-control figures

See [saved loader-control figures](../vision-loader-control-figures.md): two
exact original seed73 bodies, original counts and complete saved identity table.
Same-worker objects repeat; cross-worker batch IDs/role hashes equal but view
hashes differ. Only first file has a separate training_seal. Single-seed saved
metadata only, no fresh isolation/scientific admission. P9.5b5 complete for this
presentation; parents P9.5/P9.5b unfinished.


## 2026-10-06 - Saved CIFAR memory-control figures

See [saved memory-control figures](../vision-memory-control-figures.md): one
exact original seed83 body, three pages preserving sampled RSS, separate cached
feature bytes, sampling/capacity/native work and nine null CUDA fields. Equal
saved head/initial/feature identities and three recorded PIDs are descriptive;
no continuous peak, memory winner or fresh isolation/scientific admission.
P9.5b6 complete for saved presentation; parents P9.5/P9.5b unfinished.


## 2026-10-06 - Saved CUDA environment-smoke figures

See [saved environment-smoke figures](../vision-environment-smoke-figures.md):
four exact original seed109 failure/request/retry bodies, full failure and
synthetic telemetry. First failure preserved; retry is not independent seed
replication, dataset scoring, supported-baseline or fresh scientific evidence.
P9.5b7 complete for saved presentation; parents P9.5/P9.5b unfinished.


## 2026-10-06 - Saved development-only feature-profile figures

See [saved feature-profile figures](../vision-feature-profile-figures.md):
both exact original seed101 request/result bodies, development counts/payload/
setup/weight/hash/counter facts. Recorded zero test iterations is saved metadata,
not fresh whole-lifetime isolation or held-out construction proof. No training,
accuracy, RSS/allocator or supported-baseline conclusion. P9.5b8 complete for
saved presentation; parents P9.5/P9.5b unfinished.


### P9.5b9 saved feasibility presentation

[Guide](../vision-feasibility-figures.md) and export-v3/ in
artifacts/runs/p95-representative-feasibility-20261006/: five vector SVGs/PDF pages,
18 panels/43 original numeric entries/47 identity rows/200 complete original facts.
Single seed173; separate request/timing/payload/RSS/allocator/device/counter scopes.


### P9.5b10 saved selection presentation

[Guide](../vision-selection-figures.md); artifacts/runs/p95-selection-presentation-20261006/export/:
seven SVGs/PDF pages,24 panels/119 numeric/eight original null entries/61 identities/
all5218 facts. Saved selection seed179, future confirmations distinct.


### P9.5b11 saved synthetic smoke presentation

[Guide](../vision-synthetic-smoke-figures.md); artifacts/runs/p95-synthetic-smoke-20261006/export/:
seven vector SVGs/PDF pages,30 panels/208 numeric entries/26 original nulls/40
identity rows/all5868 leaf facts/nine original summaries. All27 recorded scopes
retained; full input/role/resource/config limits explicit.


### P9.5b12 random-backbone CIFAR saved presentation

[Guide](../vision-random-cifar-smoke-figures.md); artifacts/runs/p95-random-cifar-smoke-20261006/export/:
ten SVG/PDF pages,42 panels/272 numeric34null entries/66 identity rows/all9380
leaf facts/nine original summaries. Early/v2 declarations and failure separate.


### P9.5b13 saved pretrained CPU CIFAR presentation

[Guide](../vision-pretrained-cifar-smoke-figures.md); artifacts/runs/p95-pretrained-cifar-smoke-20261006/export/:
seven SVG/PDF pages,30 panels/208numeric26null entries/52identityrows/all6940
leaf facts/nine original summaries. Original source/request/confirmation limits retained.


### P9.5b14 saved pretrained CUDA CIFAR presentation

[Guide](../vision-pretrained-cuda-smoke-figures.md); artifacts/runs/p95-pretrained-cuda-smoke-20261006/export-v2/:
nine SVG/PDF pages,36panels/274numeric8null entries/68identityrows/all6977facts/
nine original summaries. Whole deferred/wrapper/GPU/resource/config scope retained.


## P9.5b15 — registered presentation reconciliation (2026-10-06)

This is saved presentation metadata and derived-artifact consistency validation.
All 62 original family rows and publication bodies are preserved; 13 have declared
accepted saved scopes across 14 stage groups and 15 task IDs. The canonical
P6.10 aggregate view (560 cells/626 vectors) remains separate. The other 49
families retain pending original coverage; no family receives fresh scientific
admission. 463 whole metadata files were bound before joins and 142 accepted
presentation files were checked by exact bytes/hash. This does not revalidate
all original scientific sources, unregistered binaries or private attempts.
Frozen original coverage flags are retained verbatim; current checks are separate receipts.

Concrete next validation executed: nine original legacy-multiseed-charts artifacts
bound before parsing: four PNGs, four HTMLs and docs/index.html. The four complete
PNG images were visually inspected; independent PNG checks validated every chunk
CRC and decompressed scanline extent. Full HTML bodies and exact payloads were
preserved and the three detail vectors matched the overview. Browser/CDN code
was not executed; hardest-case animation/dashboard scope remains unvalidated.
Both benchmark_multiseed_cifar100_summary.csv and benchmark_multiseed_cifar100.json
remain absent. Per-seed source filenames, seed N/IDs, uncertainty, original
execution environment and corrected-protocol provenance remain unknown.
Charts are unchanged, with nonzero throughput/latency axes explicitly unsuitable
for bar-length ratio claims. The dashboard warns about test-label-informed
stopping/sleep rollback and unmatched baselines. No recomputed means, uncertainty,
composite, chosen seeds, changed metrics or new experiment.

Evidence: `artifacts/runs/p95-coverage-reconciliation-20261006/{inputs,coverage-index,derived-chart-validation,readback,static-validation-v2,acceptance,terminal,final-accounting}.json`; guide `docs/research-presentation-index.md`. Six helper static gates pass; failed initial F401 candidate/receipt retained and charged. The v2 preformatter AST proves formatting preservation after the explicit unused-import/path repair, not equivalence with the failed candidate.

Budget: original 600 aggregate engineering seconds, 60-second hard child cap,
64 MiB owned stage, fixed 160-second manual/discovery/visual/closing reserve plus
all captured attempts including the failed lint gate. Final accounting reserves
its own full 60-second child cap; no whole-session walltime or process RSS claim.
Science 350.7925872/360 and runtime 168.7993043/180 remain spent, no reset/rekey.
Full pytest/native/Torch/CI/mypy/clean-clone and old semantic readers skipped in
this ignored helper/additive-document scope; original full gates remain required.
No installs, downloads, new dependencies, datasets, weights, arrays, device jobs,
sweeps, algorithm/config/baseline changes, publication, commit, push or merge.
G0/R0.3/full R3.1 and P9.5/P9.5b remain open. Owning-with j6c repair remains
human-deferred; R0 publication remains separate.

Why this plan change: full membership reconciliation reveals original-source coverage gaps even after accepted vision presentation scopes. Validate the already present legacy charts before selecting the next independently bound text family; retain parent acceptance and unfinished coverage. No rerender of covered scope or missing-source reconstruction.

Exact next P9.5b16: bind the frozen legacy-master-subset original docs/benchmarks/benchmark_master_cifar100_subset_2026-02-28.txt under a fresh small engineering scope before parsing; inspect the complete non-JSON text, historical test-informed protocol, baseline capacities, original units, environment/resource/failure limits and existing figures. Validate and present only uncovered saved values; preserve unknown fields and original acceptance. No model, old semantic reader, CI or scientific execution.


## P9.5b16 — historical master-subset saved presentation (2026-10-06)

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

Evidence: docs/legacy-master-subset-figures.md; artifacts/runs/p95-master-subset-20261006/{scope,inputs,view,visual-review,readback,static-validation,acceptance,terminal,final-accounting}.json and saved-values.{md,svg,png}.

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

Plan rationale: uncovered master text has UTF16 encoding, unmatched capacities and an original rounding inconsistency. Preserve rather than normalize or borrow another family. Only this saved scope complete after gates, parent criteria retained. The48-epoch README family has no named original artifact and remains unresolved. Next Pareto JSON and separately registered known-mismatched summary need whole-source comparison.

Exact next P9.5b17: freeze complete legacy-pareto and legacy-pareto-summary publication/coverage metadata and whole benchmark_pareto_hard_results.json plus benchmark_pareto_hard_summary.md before parsing. Independently compare every summary claim against the JSON; preserve known mismatches, all configurations/results/seed/resource/failure/environment limits and historical test-informed protocol. Inspect registered figures and present only uncovered validated saved values. No inferred missing provenance, corrected-protocol/source/execution admission, old semantic reader/model/CI/scientific dispatch.


## P9.5b17 — historical Pareto JSON and mismatched summary (2026-10-06)

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

Evidence: docs/legacy-pareto-figures.md; artifacts/runs/p95-pareto-20261006/{scope,inputs,view,visual-review,readback,static-validation,acceptance,terminal,final-accounting}.json, all-trials.md and three PNG/SVG pairs.

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

Plan rationale: known summary mismatch is now checked across all claims, not only a headline. Keep the two source identities/epochs/configs distinct; preserve missing provenance and unknown report-seed association. Boolean source flag is a supported exact type, not numeric zero; malformed text is refused rather than coerced. Only b17 saved scope complete after gates, all parent criteria retained. Legacy-policy-sweep is another independently bound family; no carryover of its source or claims. Full48epoch family still lacks a named original and stays unresolved.

Exact next P9.5b18: freeze complete legacy-policy-sweep publication/coverage metadata and whole benchmark_circadian_policy_sweep_results.json before parsing under a fresh small engineering scope. Inspect all saved configurations/results/seed/resource/failure/environment/unknown limits and registered existing figures; independently validate complete bodies and present only uncovered saved values. Preserve historical test-informed protocol and all parent acceptance. No old semantic reader/model/dataset/CI/scientific dispatch or inferred missing provenance.


## P9.5b18 — historical circadian policy saved reports (2026-10-06)

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

Evidence: docs/legacy-policy-sweep-figures.md; artifacts/runs/p95-policy-sweep-20261006/{scope,inputs,view,visual-review,readback,static-validation,acceptance,terminal,final-accounting}.json, all-reports.md and two PNG/SVG pairs.

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

Plan rationale: this independent policy family has no comparator or seed-report data; preserve partial configuration/default/boolean provenance and every stored policy, then validate existing claims across the full file. Presentation completion does not close baseline/isolation/scientific requirements. Next legacy-tuning-hardest is independently registered and requires its own whole source/metadata validation. No parent criteria weakened; unresolved full48epoch/missing-source work preserved.

Exact next P9.5b19: freeze the full legacy-tuning-hardest publication/coverage metadata and whole benchmark_tuning_hardest_results.json before parsing under a fresh small engineering scope. Inspect every saved configuration/result/seed/resource/failure/environment/unknown limit and registered existing figure; independently validate complete bodies and present only uncovered saved values. Preserve original historical test-informed protocol and all parent acceptance. No model, original semantic reader, dataset, CI or scientific dispatch; no source borrowing or inferred independence/provenance.


## P9.5b19 — hardest-tuning original claims and saved figures (2026-10-06)

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

Evidence: docs/legacy-hardest-tuning-figures.md; artifacts/runs/p95-hardest-tuning-20261006/{scope,inputs,view,visual-review,readback,static-validation,acceptance,terminal,final-accounting,diagnostic-claim-control}.json, all-reports.md and three PNG/SVG pairs.

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

Plan rationale: exhaustive source inspection found broad speed labels that fail over all24 records but hold over accuracy-selected family records. Preserve negative evidence and scope instead of correcting/tuning source. Acceptance is original saved validation/presentation, not a silently weakened scientific gate. Next historical continual-strength text is independently registered and distinct from corrected profile repeats. All parent criteria and unresolved missing full48epoch source retained.

Exact next P9.5b20: freeze full legacy-continual-strength publication/coverage metadata and whole docs/benchmarks/benchmark_continual_shift_strength_case_2026-02-28.txt before parsing under a fresh small engineering scope. Inspect the complete text, exact original configurations/results/protocol/seed/resource/failure/history/environment/unknown limits and registered figures; independently validate all saved values and present only uncovered views. Preserve the historical tuned profile versus corrected profile-repeat distinction. No current-default backfill, other-family borrowing, inferred independence/source/execution admission or original semantic reader/model/dataset/CI/scientific dispatch.


## P9.5b20 — original continual-strength text and saved presentation (2026-10-06)

Only this saved text validation/presentation scope is complete after its gates.
The exact original UTF-8 CRLF text is 735 bytes, SHA-256
`d2411f42266801d22f87ef4442b064c4f0f814460bd98c28091640acd886cdd4`.
Five complete metadata bodies and the original text were bound before parsing.
Nine nonempty lines, seven declared seed IDs, four setup literals, 15 original
center/+/- pairs and four Circadian sleep literals are preserved (45 numeric
literals total). The complete source body and publication record remain in the
view; 27 table rows include all values and explicitly unrecorded BP/PC sleep fields.
One PNG/SVG pair presents six panels; every label, bar, whisker coordinate and
PNG decode/pixel check passed, with the full PNG visually inspected.

The source declares seeds [3,7,11,19,23,31,37], rotation 40.0 degrees,
translation (0.90,-0.70), and Phase B train fraction 0.14. Translation units are
unrecorded. Seed IDs do not prove seven actual independent executions. Original
`+/-` is not defined as SD, SEM, CI or another statistic; figures reproduce the
notation without statistical reinterpretation. Retention and balanced formulas,
aggregation, full tuned configuration/baseline capacities, dataset/build/code/
dependencies/weights, per-seed results, execution environment, timing/memory/work
and attempt/failure history are unknown. No means, spreads, seeds or metrics were
recomputed, selected or substituted. BP/PC sleep fields remain unrecorded;
Circadian preserves sleep_events=5.00, splits=5.00, prunes=0.00, hidden_end=17.00.
The exact zero prune field has no drawn PNG bar; translation sign is retained.
Retention +/- may extend above 1; original center+/- values are retained on a
0–1.05 display range rather than clipped. Statistical meaning remains unknown.

Mixed outcomes are preserved: recorded B_post BP0.933/CPC0.930/PC0.927;
retention PC0.997/CPC0.993/BP0.985; balanced CPC0.949/PC0.947/BP0.946.
These rounded differences support no significance or fair performance ranking.
Registry kind is `historical_tuned`; corrected profile repeats are distinct.
This text does not independently establish exact stopping/test-label use.
Historical protocol limitations remain; no corrected evaluation isolation,
matched capacity, complete provenance, replication or scientific admission claim.
Original registry missing-path lists are empty, not evidence of complete execution
provenance. Only this family is covered; parent P9.5/P9.5b remain unchecked.

Evidence: [saved presentation guide](docs/legacy-continual-strength-figures.md),
`artifacts/runs/p95-continual-strength-20261006/` complete source/metadata copies,
view, all-values table, saved-summary PNG/SVG, visual-review, readback/readback-v2,
static-validation, acceptance, coverage-delta, terminal and final-accounting.
Seven malformed-text refusal controls and a signed-translation positive control
passed. No existing scientific result or publication was rewritten.

Budget: prospectively 600 aggregate engineering seconds, hard 60 seconds per
child, 64 MiB owned artifacts; fixed 160 seconds for manual/discovery/visual/
closing work plus every captured attempt. Final accounting reserves its entire
60-second command cap; this is not whole-session elapsed time or process RSS.
Science350.7925872/360 and runtime168.7993043/180 remain spent without reset.
Full pytest/native/Torch/CI/mypy/clean-clone and original scientific readers were
skipped in this ignored-helper/additive-document scope; their gates remain open.
Existing Pillow used, no new dependency/download/model/dataset/device/sweep.
Whole checkout, HEAD and installed packages preserved except the new guide and
seven reversible additive documentation edits. Owning-with repair j6c remains
human-deferred; G0/R0.3/fullR3.1 and all other unfinished tasks remain open.

Plan rationale: original text has undefined spread notation and incomplete
execution evidence. Present its exact values and unknowns without inventing
uncertainty semantics, favorable winners or corrected-profile equivalence.
This is a separate saved-presentation increment; no scientific acceptance
criterion is weakened. Remaining missing original sources/work is preserved.

Exact next P9.5b21: freeze complete legacy-continual-hardest publication/coverage metadata and whole docs/benchmarks/benchmark_continual_shift_hardest_case_2026-02-28.txt before parsing in a fresh small engineering scope. Inspect complete original text/config/results/protocol/seed/resource/failure/environment/unknowns and registered figures; independently validate saved values and present uncovered views. Preserve historical tuned versus corrected-profile-repeat distinctions; no default backfill, seed selection, source borrowing, inferred independence/admission or original semantic-reader/model/dataset/CI dispatch.


## P9.5b21 — original continual-hardest text and saved presentation (2026-10-06)

Completed scope: saved original text validation and uncovered presentation only.
Full original `docs/benchmarks/benchmark_continual_shift_hardest_case_2026-02-28.txt`
is UTF-8 CRLF, 861 bytes, SHA-256
`6f167aedc3ebbb3762612a127a8ec0341a715c37fbce2a2602fd8f02cda29f1a`.
Five complete metadata bodies and exact source bytes bound before parsing;
checkout reconciled exactly with the P9.5b20 terminal (1023 files/25 packages,
HEAD182077). Whole original publication and text retained. Ten nonempty lines,
seven seed-ID literals, eight partial setup literals, four transform/fraction
literals, 15 center/+/- pairs and four sleep literals yield 53 numeric literals.
All 35 table rows retain recorded notation, list positions and unknown fields.
One six-panel PNG/SVG pair validated by independent original values/full header,
labels, axis extents, all 15 bars/45 whisker lines/four sleep bars and PNG decode/
pixels; full PNG visually inspected for legibility and clipping.

Partial setup records hidden_dim=24, hidden_dims=[24,24,24], Phase A/B epochs
120/180, and noise0.80/1.45. Exact per-method assignment and complete configurations
are unrecorded; these declarations do not prove matched baseline capacities.
Transform rotation68.0deg, translation(1.60,-1.30), train fraction0.05 retained;
translation units unknown. Seed IDs[3,7,11,19,23,31,37] are declarations, not evidence
of seven actually executed independent runs. `+/-` definition is unknown: no SD,
SEM, CI or aggregation formula inferred. Retention/balanced formulas and per-seed
outputs, actual dataset/source/code/build/dependencies/weights/device/environment,
time/memory/work/attempt/failure/rollback history are unknown. Original fractional
Circadian scalars sleep_events24.43/splits48.57/prunes0.00/hidden_end72.57 remain
fractional; neither rounded to event counts nor assigned a new aggregation meaning.
BP/PC sleep fields unrecorded, not zero. The zero prune bar has width0 in SVG and
no colored PNG bar. No source statistic, seed, metric or baseline was changed.

Recorded mixed outcomes preserved: PC A_post0.793 > CPC0.784 > BP0.699;
PC retention0.815 > CPC0.804 > BP0.718; CPC B_post0.841 > PC0.823 > BP0.808;
CPC balanced0.812 > PC0.808 > BP0.753. These rounded summaries/undefined spreads
support no significance, independent replication or fair general ranking.
Registered kind `historical_tuned`, distinct from corrected profile repeats and
the earlier hardest animation. Exact stopping/test-label access is not established
by this text; historical protocol limitations remain. No corrected isolation,
matched capacity, complete execution/source provenance or scientific admission.
Registry empty missing-path lists do not establish execution completeness.

Evidence: `docs/legacy-continual-hardest-figures.md`;
`artifacts/runs/p95-continual-hardest-20261006/` whole source/metadata copies, view,
all-values table, saved-summary PNG/SVG, visual-review/readback/static-validation/
acceptance/coverage-delta/terminal/final-accounting receipts. Nine malformed-text
refusal controls and a signed-translation positive control passed. One failed
audit retained/charged; its helper expected10 lines per panel instead of9 (three
whisker lines times three methods). Small assertion repair changed no source,
view or figures. Failed helper candidate retained; final six-helper AST comparison
uses the complete snapshot after logic repair, not the earlier failed candidate.

Engineering budget: prospective600 aggregate seconds/hard60 per child/64MiB
owned stage, fixed160 manual/discovery/visual/closing reserve plus all captures
including failure; final accounting reserves its full60-second cap. Not entire
session elapsed time or processRSS. Science350.7925872/360 and
runtime168.7993043/180 remain spent, no reset. Full pytest/native/Torch/CI/mypy/
clean-clone and original scientific-reader gates skipped for this ignored-helper/
additive-document scope; original gates stay open. Existing Pillow, no dependency
installation/download/sweep/model/dataset/device dispatch or publication.
Whole checkout/source/test/architecture/unrelated user changes preserved except
new guide and seven reversible additive docs. Parent P9.5/P9.5b/G0/R0.3/fullR3.1
remain open; owning-with j6c human-deferred. Unfinished original-source work stays.

Why this plan increment: the hardest text includes additional partial setup and
fractional sleep fields; exact literals and unknown statistical/configuration
meaning need explicit preservation. No scientific criterion is weakened.
Next historical animation has original registered visuals requiring validation;
illustrative-circadian remains a separate uncompleted family, not experiment data.

Exact next P9.5b22: bind complete legacy-hardest-animation publication/coverage metadata and whole docs/figures/hardest_mode_dynamics.gif plus docs/figures/interactive_hardest_mode_dynamics.html before decoding under a fresh small engineering scope. Inspect every original GIF frame and complete static HTML data/configuration/label payload, exact original source/configuration/seed/protocol/environment/failure/missing/unknown provenance and registered figures. Validate and document original presentation without browser execution or inferred telemetry/current-default reconstruction. Keep earlier test-informed visualization, tuned hardest text, illustrative-circadian and corrected repeats separate; no model/scientific-reader/dataset/CI dispatch or admission.


## P9.5b22 — original earlier hardest visualization validation (2026-10-06)

Completed original visualization validation scope, not scientific execution or
browser functionality certification. Complete sources bound before decoding:
GIF3,530,592 bytes/SHA256 aa338c5051f3cb90447df0b2e972ea6b93a9e08a2e2707d89822e3bb11504ee8;
HTML21,352,286 bytes/SHA256 222abd3f0281ece0b6404fb4ed72afde470c2a055df02b0f6e55edf268d42b7e.
Five whole metadata bodies and exact source copies retained. Reconciled against
P9.5b21 terminal:1024 nonignored files/25 packages/HEAD182077, full AGENTS/plan/log.
Registry `legacy-hardest-animation`, kind `historical_visualization`, earlier
profile/test-informed frames, original invocation/dependencies unknown. Separate
from the tuned hardest text, illustrative-circadian and corrected profile repeats.

All76 GIF frames decoded at1180x680 RGB, all canonical pixel hashes/durations/
disposal values independently checked. Every duration120ms and loop0, recorded
encoding properties rather than training wall time. Four contact sheets cover all
76frames with every thumbnail pixel checked; full original RGB copies at indices
0/30/31/75 (epochs1/120/124/300) match exact decoded pixels. All contact sheets
visually inspected; those four full originals inspected for labels/structure.
Original metric footer extends past right edge and clips Circadian latency in all
four full samples. Original unchanged. Thumbnail overview does not certify every
small numeric GIF label. HTML contains all recorded per-frame scalars separately.

Complete embedded JSON span:21,338,385 bytes; full source retains every array
without copying a second21MB payload. View records exact byte offsets, raw/canonical
payload identity, all root values, all76 scalar records, and identities/shapes/
typed counts for every full frame/array. Independent whole-source parser checked
1,035,935 leaves (13,856ints/1,022,000floats/79strings), no boolean/null coercion.
Every decision map110x110;175 Phase-B test points/labels/predictions per frame,
24 adaptive input values and24-by-hidden weight matrices, hidden-length state/
activation/weight vectors. All912 original scalar cells in76 table rows preserved.
Every accuracy/latency frame field matches its original series position;76 existing
Circadian prediction/label accuracy rounding checks (absolute tolerance0.00005)
and76 hidden=24+splits-prunes checks passed. No new reported score/statistic/winner.

Epoch series[1,4,8,...,300], Phase B begins index31/epoch124. Final saved accuracies
BP0.8286/PC0.7486/CPC0.7771, latency(ms)0.1343/0.1434/0.2127; Circadian hidden74,
splits50,prunes0. These earlier-profile values are not substituted for the tuned
hardest text summaries. Saved normalized objectives are not raw loss/energy;
normalization procedure, comparability and execution provenance unknown.
Source explicitly displays Phase-B test accuracy/labels during Phase-A snapshots;
this is historical test-informed visualization, not corrected arrival/isolation.
No seed identity/count/per-seed independence, full original configuration/baseline
capacity, code/build/dependency/device/weights, original measurement method/hardware/
resources or full stopping/selection/rollback/attempt/failure history established.
No matched fairness, significance, replication or scientific admission asserted.

HTML inspected statically only: declares Plotly2.35.2 CDN URL (not fetched),
260ms playback interval, source numeric axis labels/bounds and all original code.
Its heatmap supplies z but no declared x/y coordinates, while scatter/layout use
physical bounds. Alignment is not certified; original source unmodified. Browser
rendering/controls/network/CDN availability not exercised. Same76frame count and
four sampled matching headers do not prove identical original GIF/HTML backing
execution. Complete HTML shell saved separately; no reconstructed plots/telemetry.
Nine refusal controls passed: nonfinite JSON,duplicate keys,unknown root,epoch
mismatch,boolean hidden size,accuracy-series mismatch,wrong decision map shape,
missing hidden activation values,wrong prediction count. Diagnostic mutations were
restored and whole canonical payload rechecked; original bytes remain unchanged.

Evidence: `docs/legacy-hardest-animation-validation.md`;
`artifacts/runs/p95-hardest-animation-20261006/` whole source/metadata copies,
view/reference/76-row table/complete HTML shell/eight inspection PNGs,
visual-review/readback/static-validation/acceptance/coverage-delta/terminal/
final-accounting. Only saved original coverage complete; P9.5/P9.5b/G0/R0.3/fullR3.1
remain open. Owning-with j6c repair remains human-deferred; missing sources retained.

Budget: prospective600aggregate engineering seconds/hard60 per child/64MiB owned,
fixed160manual/discovery/visual/closing reserve plus every captured attempt; final
accounting reserves entire own60second cap. Not whole-session time or processRSS.
Science350.7925872/360 and runtime168.7993043/180 remain spent without reset.
Full pytest/native/Torch/CI/mypy/clean-clone/scientific-reader gates skipped in this
ignored-helper/additive-doc scope and remain required. Existing Pillow used;
no new dependency/download/model/dataset/device/browser/CI/sweep/publication.
New guide/seven reversible additive docs only; unrelated checkout/source/tests/
architecture and all other task rows preserved.

Why this increment: existing GIF/HTML already present the saved visualization.
Validate originals and document clipping/coordinate/provenance limitations instead
of generating replacement telemetry or favorably selecting snapshots. Keep the
full large arrays in the exact original HTML to stay within64MiB owned storage;
all payload values still independently checked. This changes no scientific gate.

Exact next P9.5b23: freeze complete illustrative-circadian publication/coverage metadata and whole docs/figures/circadian_sleep_dynamics.gif before decoding under a fresh small engineering scope. Inspect all original frames/labels and full illustration provenance/missing/unknown limits; independently validate frame dimensions/durations/pixels and document existing views. Preserve illustration-not-experiment distinction; no invented telemetry, new experiment, scientific source/execution/replication admission or browser/model/original scientific-reader/dataset/CI dispatch.


## P9.5b23 — illustration validation (2026-10-06)

[Guide](../circadian-illustration-validation.md): all25 original GIF frames/labels/pixels validated, including two downward width steps. The original remains illustration-not-experiment; constant split/prune labels are not live counters, and chemical telemetry is absent. Plan/log hold complete evidence. P9.5/P9.5b remain open; next P9.5b24 arrived-v1 request/result inspection.


### Saved arrived-v1 confirmation fixture

See [the whole-source validation guide](../arrived-confirmation-figures.md) for all seeds/candidates/orders, negative outcomes, figures and explicit checkpoint/environment gaps. P9.5b24 completes saved presentation only; no new scientific or runtime admission. Next: replay-v8 source validation.


### Saved replay-v8 smoke fixture

See [the complete validation and figures](../replay-v8-smoke-figures.md) for both replay policies, all seeds/events, retention above 1 and explicit physical memory/runtime gaps. P9.5b25 completes saved presentation only. Next: replay-v9 continuation source validation.


### Saved replay-v9 continuation bodies

See [the complete three-source validation](../replay-v9-continuation-figures.md) for exact whole-file equality, all duration differences, figures and the original v8 protocol retained inside v9 filenames. P9.5b26 completes saved presentation only; physical continuation remains unverified. Next: matched-replay schedule/training/outcome source inventory.


### Matched replay planned schedules

See [the complete source validation guide](../matched-replay-schedule-figures.md) for all nine originals and five validated planned schedules. P9.5b27a completes inventory/schedules only; inference counts differ and the parent remains open. Next: P9.5b27b applied training and cross-schedule validation.


### Matched replay saved training records

See [the complete training validation guide](../matched-replay-training-figures.md) for both whole training bodies, all five schedule comparisons, applied work and clock limits. P9.5b27b completes saved training relations only; unequal inference work is retained. Parent remains open; next P9.5b27c outcome/telemetry validation.


### Matched replay complete saved outcomes

See [the complete outcome and family reconciliation guide](../matched-replay-outcome-figures.md) for all scores, telemetry and remaining physical/provenance gaps. P9.5b27c and saved-family P9.5b27 are complete; CPC's negative aggregate result is preserved. Broad scientific parents remain open. Next: P9.5b28 side-effects-v10 originals.


## Saved replay side-effect comparison (P9.5b28)

[Complete guide](../replay-side-effect-figures.md): both whole originals and
both conditions validated; scores/work identical, chemistry differs. CPC's
lower aggregate scores and the FIFO seed exception remain. Fourteen figures
and complete evidence are local saved presentation, with physical execution
and independent repetition unproved. Next: P9.5b29 difficulty-v11 full bodies.


## Saved difficulty modulation comparison (P9.5b29)

[Complete guide](../difficulty-modulation-figures.md): both whole difficulty-v11
originals, all conditions/backends/seeds/arms and diagnostics validated. Final
modulation/control differences are zero; negative forgetting and failures remain.
Twenty figures preserve every recorded value. Physical execution and independent
repetition are unproved; no superiority claim or new heuristic. Next: P9.5b30
whole structural-ranking-v12 training/outcome originals.


P9.5b30: [structural-ranking v12 figures](../structural-ranking-figures.md), 24 complete PNG/SVG pages with literal signed contrasts and ID labels.


P9.5b31: [sleep-trigger v13 figures](../sleep-trigger-figures.md), 42 complete PNG/SVG pages with all decisions, actual events and paired differences.
