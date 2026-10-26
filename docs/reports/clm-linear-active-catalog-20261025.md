# R2.1-CLM.LA — explicit active-catalog typed linear readout

Issue [#165](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/165),
2026-10-25. Implemented and qualified scoped software with single-agent
source/procedure/result self-review. Not independent acceptance, scientific
learning benefit, actual pretrained rollout or release readiness.

## Working behavior

The existing linear typed provider now accepts an explicit immutable
`active_candidates` declaration. A full fitted union catalog remains the source;
the current readout selects the original ordered columns **before softmax**.
Inactive extreme scores are excluded before normalization. This supports the
temporal v1/v2 three-tool contexts from one four-column model without confusing
`retire` and `defer` or creating new fit metadata.

Core `predict_wide_categorical_active` performs only read-only projection.
Existing role-bound `predict_clm_linear_control` has optional
`active_candidate_ids`; the positional catalog stays full-source. Original fit
rank/error/rows/class count/weights remain unchanged. No synthetic smaller fit,
training, gradient, optimizer, parameter export or history chain.

Provider mode requires original IDs, descriptions and relative order; request
catalog must equal its bound declaration. Invalid/unknown/duplicate/reordered/
singleton catalogs fail before encoding. Probe source still fails before action
access or encoding. Full source owners/values, including excluded coefficients,
active candidate declarations/storage, requests, encoder returns and paid-work
shadows are guarded. Corrupted or lost paid results retain their prefix and stop
further work. Shared encoder counters are never broadly ignored.

Default `None` retains full-catalog output/work/readout schema. Explicit mode adds
retained active action-copy elements, source/active class counts, projection
readout attempts/returns and active/source column metadata. A row gather already
owns storage; the redundant clone was removed during self-review. Row-view
validation avoids additional data gathers. Work fields are logical payloads,
not universal allocation/FLOP counters. No dependencies, environment variables,
new CLI or stream framework.

```text
src/core/wide_categorical_linear_fit.py      read-only selected-column helper
src/adapters/clm_linear_fit_control.py       optional role-bound active readout
src/adapters/clm_linear_decision.py          typed opt-in binding and original guards
tests/test_wide_categorical_active_readout.py independent numeric oracles
tests/test_clm_linear_active_catalog.py      context, role, ownership and failure cases
docs/clm-linear-active-catalog.md            usage, costs, limits and extension route
artifacts/runs/clm-linear-active-catalog-20261025/ commands/XML/counters/statics
```

## Commands, outcomes and costs

```powershell
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-active-catalog-20261025/launch.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-active-catalog-20261025/launch.py linux-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-active-catalog-20261025/statics.py statics-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-active-catalog-20261025/launch.py windows-2
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-active-catalog-20261025/launch.py linux-2
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-active-catalog-20261025/statics.py statics-final
```

Exact child commands/environment, terminal codes and XML/stdout remain in the
stage. CPU1, sequential platforms, existing cached WSL runtime, no installation.

| Cohort | Outcome | Body seconds | Parent seconds | Sampled RSS peak bytes |
| --- | --- | ---: | ---: | ---: |
| Windows1 | 32PASS | 7.151279800 | 7.747341700 | 264384512 |
| Linux1 | 32PASS | 22.437513185 | 27.825893000 | 296841216 |
| Windows2 | 30PASS, new read-only paths only | 3.330714700 | 3.752269300 | 255991808 |
| Linux2 | 30PASS, new read-only paths only | 14.384136124 | 19.694152200 | 288653312 |

Final32 distinct qualified cases/platform by reuse plus targeted checks,
62passing executions/platform, not a single final32-case run. Initial cohort
includes two existing default workflow checks sharing exactly one original
12-row literal fit; ordinary completion, stale-revision recovery, external
permission denial, unchanged source/RNG and query-only encoding costs pass.
After the explicit-active-only copy refinement, the unchanged default `None`
branch and its results were reused, not retrained. The final30 checks execute
the changed active branch and readonly core/adapter.

New tests use **hand-set literal coefficients**, zero fits: both context masks,
inactive10000 scores, full-column parity, invalid column/mask declarations,
TRAIN/probe readout roles, probe deployment refusal, exact request/description
binding, excluded source-weight changes, active storage owner/value changes,
changed columns, post-encoder declaration/counter corruption and lost readout.
They are software oracles, not acquired/native policies. A forged counter stops
provider return admission while the encoder's separate completed return stays
paid. Failed callbacks are expected test cases, not hidden successful work.

Per platform cumulative measured/charged software work: one existing default
LSE/12fitting rows;44readout calls/55rows including refused core calls;
37provider binding attempts;22fake encoder calls/32sequences/32visited IDs/
32padded slots;43literal module constructor/deepcopy objects; zero head forwards,
random parameter elements, optimizer/backward/scheduler or simulator work.
No new fitting in active tests or second cohort. Reserves8Mlogical copy elements
and32MiBother-copy memory remain separate from observed counters. Readout counts
include precondition rejections and do not assert that every call performed a
matrix multiplication. RSS covers sampled process memory after imports and is
not all-lifetime/import peak or model-attributed memory.

Totals: Windows10.481994500s body/11.499611000s parent; Linux36.821649309s body/
47.520045200s parent, below60/90s and two cohorts/platform. All original fit/read/
encoder/module/random/storage caps held. No behavioral qualification failures,
budget amendments or scientific attempt.

Initial Ruff/format PASS, mypy10errors in the qualifier's saved callable
dictionary/restoration types,24.386878400s. Added its `dict[str,Any]` annotation;
no numerical logic changed. Final eight-file Ruff/format/default+Linux-target
mypy all PASS22.509287900s, cumulative46.896166300s plus bounded format below120s.
Actual Linux statics unavailable/uninstalled. The original wide cold/update
arithmetic remains unchanged; no full4096 baseline or scientific cohort was
repeated. Current source file hashes differ from historical snapshots, but
those historical executions remain frozen; the relevant default mathematics
and assumptions are source-reviewed/qualified rather than retroactively edited.

Conservative touched whole cache109,748,240B (<160MiB), inherited global
596,569,557B separate, with20:48UTC cutoff. Stage42,041B before final review,
below16MiB. All seven expected product/test/usage/report files are present.

## Self-review and remaining work

Reviewed the actual combined three-module change, both new tests, original
workflow fixture, exact commands/terminal outcomes/counters and static results.
Core depends inward. Adapter maintains role/identity/full-source ID binding;
active selection precedes paid readout. Provider pins full source and original
action bank, one active declaration and its owned storage, limits/work/counter
shadows; output probabilities/confidence use only declared active classes.
Rank/error/fit-row metadata is never recomputed or relabeled as a new fit.
Source fields and role remain unchanged. There is no per-request policy inference,
invisible update, permission bypass, unlimited state chain or budget renewal.

No pretrained/model/bank/old scientific raw/detached-review access or reconstruction
occurred. Skipped unchanged fullsuite, actual live/native adaptation, CPC/matched
comparisons, final generalization, cleanclone/package, GPU/CI/browser, independent
and release checks. No install/download/commit/push/delete. All prior negatives
remain, including #164Aparcel error/calibration degradation/recoveryFALSE/order
ties and #163original engineering copy-budget failure/correction. Historical
plans/log stay frozen under current GitHub tracking; full goal remains ACTIVE.

Next implementation action: connect the new binding to the existing ticket
workflow's public-policy/context composition. Its teacher already exposes
`training_cases(policy_version=..., include_policy_context=True, ...)`, and
`workflow_request` selects active tools from the declared policy/capabilities.
Keep a full six-tool union across warm TRAIN updates, explicitly map per-policy
teacher targets, and bind each declared live episode's active catalog without
refitting during inference. Use the original episode runner/permission effects,
not a new orchestration layer. Verify a complete literal context sequence and
all cumulative exposure/encoder/copy costs before freezing any fresh actual
end-to-end case. No #164transient model reconstruction or renewed data allowance.
Broader R2/R4/R5.1/R5.3/G5/R7/release remain open.
