# R2.1-CLM.LSX — one cached-feature temporal learning case

Issue [#164](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/164),
2026-10-25. The sole scientific attempt completed successfully. This is scoped
executed evidence with single-agent source/procedure/result self-review, not
independent acceptance, a live workflow rollout or repository readiness.

The current-only linear learner fitted all three TRAIN rows at every scored
checkpoint. Development suffix accuracy reached3/4 after A,4/4 after B, and4/4
on return. Parcel failed after A, then remained correct from B to return without
parcel return-TRAIN. Stable password never failed; the original recovery flag
is **false**. Return success under current v2 truth is **true**. Changed-policy
invoice/legacy targets do not measure forgetting of a still-valid old rule.

## Fixed results and limitations

| Checkpoint | TRAIN | Development suffix | NLL | Brier | ECE, five bins |
| --- | --- | --- | ---: | ---: | ---: |
| Untrained baseline | — | 1/4 | 1.098612259 | 0.666666667 | 0.083333343 |
| After A | 3/3 | 3/4 | 0.737520292 | 0.408465256 | 0.449378088 |
| After B | 3/3 | 4/4 | 0.562232883 | 0.283344486 | 0.426014647 |
| After return | 3/3 | 4/4 | 0.615560119 | 0.318291229 | 0.457726777 |

Candidate recall1.0, finite NLL, zero target-probability zeros throughout. The
original raw cases, probabilities, domain metrics, calibration bins, signed
contrasts, twelve losses and all receipts remain in `actual.json`. Last-batch
rank3, squared error3.394301063705598e-32; that metadata is not lifetime work.

All four development predictions, original concept order:

| Phase | Invoice | Legacy | Parcel | Password |
| --- | --- | --- | --- | --- |
| Baseline | inspect ✗ | inspect ✗ | inspect ✗ | inspect ✓ |
| A | route ✓ | retire ✓ | retire ✗ | inspect ✓ |
| B | inspect ✓ | defer ✓ | defer ✓ | inspect ✓ |
| Return, v2 truth | inspect ✓ | defer ✓ | defer ✓ | inspect ✓ |

Negative endpoints matter. Parcel was not acquired under A's development query.
Return leaves accuracy unchanged but worsens NLL by0.053327235731, Brier by
0.034946742307 and ECE by0.031712129712 relative to B. ECE is worse than the
uniform reference after training. Small support and low confidence limit
calibration conclusions. No temperature, metric, seed, class ordering or dose
was changed to improve these numbers.

The untrained/uniform reference uses first-maximum selection and is order
sensitive under exact ties. Its reverse-order control is false throughout;
the learned baseline is also false. Learned A/B/return reverse-order controls
are all true. These failures are retained, not repaired by changing the reference
after observation. The uniform reference is not a trained matched baseline.

Password was already correct at the uniform baseline. Its continued correctness
therefore does not establish acquisition from failure or recovery. Parcel's
B-to-return correctness is one scoped retention observation, with no parcel
return-TRAIN. There are no obsolete-rule errors at B/return, where two old
targets are still-allowed error opportunities. Full current-rule old-domain
return succeeds. No general recovery/retention, final generalization, replay
benefit, native-head benefit, matched BP/PC/CPC or circadian advantage follows.

## Protocol, provenance and data isolation

Before any original bank access, issue164 and stage `PROTOCOL.md` froze this one
case: original12delayed TRAIN packets/36rows, fixed A/B/return data and labels,
four union columns `defer,inspect,retire,route`, original active-three candidate
orders and metric directions. First batch cold-fits one-hot least squares;
eleven warm minimum-change residual solves use the existing intercept/rcond1e-12.
No tuning, seed search, early stopping, replay or probe updates. TRAIN diagnostics
are post-training readouts. V2 remains truth on A return.

The new explicit data-access allowance read **once** each original161 bank and
origin record,295,104B total:

| File in original161 stage | Bytes | SHA256 |
| --- | ---: | --- |
| frozen-features.npy | 262272 | 691df63c66f297149fe8197921709c676c032bd2b1cfdb3ab4f558ea9117f043 |
| actual.json, origin metadata projection | 32832 | d8a4aa5867bcffe7c4b307f953dcce58ac32b0f52b24e9d47f3e6c456e014436 |

The entire origin JSON parse/physical input is charged. The immediate whitelist
keeps only identity, text/token order, extraction/bank metadata, timing/RSS and
load events. Old fit/readout/metric/source-weight fields do not enter this study's
learner or scoring. This is a separately declared dataset use, not a reset of
161's case/reader or a reconstruction of an old fitted model. Both original
allowances and all charges stay spent. No other historical scientific raw or
detached-review inputs were accessed.

The hash-bound16x4096 CPUfloat32 bank comes from Qwen/Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, same tokenizer revision, last pooling,
recipe `clm-plain-state-tail-action-head-v1:max_len=2048:cap=2047:l2-plus-1e-12`.
Exact original identity, sixteen texts, action order, token/extraction dimensions
and role mapping passed before fitting. Selected rows0..7 and12..15 give the
old bare TRAIN queries, development suffix queries and actions. Reserved-probe
rows8..11 are physically read with the bank but never leave admission for the
new learner or scorer. No probe labels are converted into TRAIN. Final data
are absent; no promotion, selection or external effects.

The unchanged #163 learner/app/inbox/scorer and #162 mathematics are reused.
26 source/script hashes bind the relevant dirty checkout at HEAD09ff811f;
the supervisor checked them before/after execution. This is scoped source
binding, not a whole-repository attestation. Product code was unchanged.

## Costs and exact execution

```powershell
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-stream-cached-20261025/execute.py
```

EXIT0, sole child0, permanently **SPENT**. Exclusive receipt precedes preflight
and input access. Fresh WSL available RAM32,601,575,424B, no competing large
Python process. Existing cached WSL Python3.12.3/Torch2.14.0+cpu/NumPy1.26.4,
CPU1 sequential/offline. Parent8.143070400s including startup/preflight; pipeline
0.770432736s after imports. Requested RSS sampling interval0.02s/56samples;
sampled peak264,994,816B. Import-phase peak is not measured by that segment.
Conservative43s child timeout+2s kill stays within45s, parent75s; no limit change.

Exactly12attempted/returned LSE solves/36fitting rows, designs3x4097/four targets;
8learner choices/29query rows, including the unfitted uniform baseline. Existing
cached evaluator performs88typed scalar decodes, including reverse-order checks;
TRAIN diagnostics decode nine cached margin rows. Zero module construction,
pretrained load or random initialization attempts, zero new encoder/head forward,
optimizer/backward/scheduler/simulator work. The actual learner work is complete,
without uncertainty. No extra solve, model restoration or retry.

Current coefficient payload131,104B (16,388float64 values); current control plus
its value-guard Python representation1,051,746B, deduplicated by object identity.
It is not whole-learner/process memory. Other reported logical payloads:

| Payload | Bytes |
| --- | ---: |
| Selected twelve-feature tensor | 196608 |
| Twelve bound TRAIN batch tensors | 1179648 |
| Four outer choice tensors | 458752 |
| Three TRAIN choice tensors | 294912 |
| Learner action and tensor guard | 131072 |
| Bounded choice NumPy guards | 753664 |

One current coefficient model and guard, no unlimited model history. The model
was transient and not exported. Copying/validation was prepaid separately with
32M logical elements and128MiB temporary-copy reserve; those are conservative
charges, not observed universal counters. Reported arrays, sampled RSS and
elapsed time do not measure every allocation, FLOP, import peak or partial-report
rewrite IO. Output footprint: actual232,506B, work357B, timing195B, log688B;
supervisor stage288,521B before final documentation, below8MiB.

The original bank was not free. Origin metadata preserves the original shared
case359.647538873s post-import/367.900255900s parent and sampled peak
14,856,208,384B, including its original model setup and diagnostic work. Its
one encoder forward covered16sequences/137visited IDs/176slots/65,536pooled values.
That whole source case is not an isolated encoder latency. These historical
setup costs remain charged alongside the new incremental case; serial process
peak RSS values must not be added. No8B extraction or head construction was
repeated. This is not a from-scratch8-second learning-cost claim or matched
native/CPC comparison.

## Preparation, failures, skips and self-review

Exact preparation commands:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-stream-cached-20261025/launch.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-stream-cached-20261025/launch.py linux-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-stream-cached-20261025/statics.py statics-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-stream-cached-20261025/statics.py statics-final
```

Eight synthetic cases PASS/platform before actual input access: full original
pipeline, hash/identity/text/role/extraction/shape rejection before fitting,
occupied evidence refusal before IO. Each pays12literalLSE36rows/8choices29rows,
width2,0module/load/random work. Windows3.940444000s body/4.365184100s parent,
RSS256,655,360B; Linux12.810399803/17.564485600s, RSS290,254,848B. Separate
read-only runtime metadata query took5.1276706s parentwall, conservatively charged
6s body/parent Linux. All45s/60s engineering limits hold. Earlier unchanged
mathematics/native-default/copy-budget evidence is reused, not broadly repeated.

Initial Ruff/format PASS, mypy9errors33.377528800s retained. Fixed receipt/test
annotations and equivalent method `setattr` bindings; removed an optional stdout
accuracy helper whose signature was wrong, retaining every score in actual.json.
Final eight-file Ruff/format/default+Linux-target mypy all PASS27.784371500s,
total61.161900300s plus bounded formatter commands below120s. The tested pipeline,
bank admission and numerical ports did not change. Actual Linux statics were
unavailable/uninstalled. No scientific or behavioral qualification failure,
budget amendment, renewed attempt, install, download, commit, push or deletion.

Conservative touched whole cache109,707,280B (<160MiB), inherited global
596,528,597B measured separately with20:24UTC cutoff. Stage288,566B at final
metadata inspection before the final review document; no scope or cap change.

Three postterminal stdlib inspections of current case JSON read698,070B in
0.005976100s measured parsing/extraction time, below4MiB/10s; process overhead
also below10s. They performed no scoring, fitting or model imports. No original
bank/origin reparse after its single producer read. These inspections establish
all declared endpoint directions, not an independent result acceptance.

Pre-execution `REVIEW.md` records source/procedure GO. Final self-review checked
the actual successful terminal, source pin checks, exact work/receipts, role and
identity binding, lack of parameter export, all positive and negative endpoints,
cost qualification and unchanged original spent histories. Scope distinguishes
literal correctness, this one actual cached dataset, and unmet live scientific
and release claims. Fullsuite, cleanclone/package, pretrained/live workflow,
CPC/matched/replay comparisons, final generalization, GPU/CI/browser and
independent/release checks were skipped as outside this case. Historical plans
and log remain frozen under the live GitHub issue workflow.

## Next useful outcome

The declared rule learner now has positive acquisition/return evidence on this
tiny cached development sequence. Do not start another nearly identical
checkpoint, reader, seed or metric-tuning study. The next concrete dependency
is typed use: the existing `CLMLinearDecisionPredictor` requires its entire fitted
catalog in every request, while the learned model owns four union columns and
each context permits only three tools.

Next implementation action: add explicit read-only active-catalog binding to
that existing linear typed provider, preserving source/probe role, identity,
candidate descriptions/order, ownership and paid encoder/readout guards. Verify
both context catalogs and inactive-column exclusion with literal tests, then
separately freeze an actual closed-loop adaptation demonstration. This case
exports no weights and grants no future data/model/fit allowance; future work
must not reconstruct its transient model from the summary or silently repeat
fits. Broader R2/R4/R5.1/R5.3/G5/R7 and release remain open; full goal ACTIVE.
