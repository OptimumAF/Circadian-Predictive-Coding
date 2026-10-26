# R2.1-CLM.LS — warm linear learner in the existing stream

Issue [#163](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/163),
2026-10-25. Implemented and qualified scoped software; single-agent source,
procedure and result self-review, not independent acceptance. No pretrained,
scientific, CPC or release result is claimed.

## Delivered behavior

`--catalog-training --train-linear-update` selects the new linear learner in the
existing temporal command. The original inbox runs all twelve delayed-label
TRAIN packets/36rows with original clocks, receipts, checkpoints and metric
directions. No new stream framework. Original native defaults remain unchanged.

The model has four fixed columns, `defer,inspect,retire,route`. Original active
v1/v2 candidates map by tool identity, never by their changing positions. The
first batch cold-fits all four union one-hot classes; the next eleven use #162's
minimum-change residual solve with fixed intercept/rcond. This is a distinct
union squared-error objective, not native active-three-class CE or matched
compute. Active coefficients are selected before stable softmax, so an inactive
extreme score cannot win or underflow all active probabilities.

Baseline/unchanged predictions are explicitly uniform and named
`untrained_uniform_union_linear`; they are not released native head predictions
or a trained matched baseline. Their kind/values enter the guarded stream prefix
and are validated before scoring. Arrived TRAIN labels only; no probe conversion,
final data, replay, tuning, promotion or serving. Unsupported native probe,
state-only, memory, typed-serving and comparison options fail before loading.

One current immutable control, at most eight explicit choice bindings, no fitted
history chain or replay row store. Separate exact solve work schema:12solves,
12commits,36rows and five readouts/20queries (or eight/29 with TRAIN diagnostics).
The learner performs zero encoder/head forward, optimizer, backward or scheduler
work. The original loader/feature extraction still costs work and loads the
unused released heads; this is disclosed rather than hidden by the learner's
zero counters. Last-batch metadata is not lifetime work. Lost/unadmitted solves
retain attempted/returned/uncertain work and terminally refuse another update.

```text
src/core/clm_linear_catalog.py                objective, policy, control, exact work schema
src/adapters/clm_linear_catalog_learner.py     fixed-union learner; existing stream composition
src/app/clm_temporal_sequence.py              explicit linear admission/objective/work branch
scripts/run_clm_temporal_sequence.py          CLI and existing score admission
tests/test_clm_linear_catalog_stream.py       sequence, identity, mask, prefix/failure checks
tests/test_clm_linear_catalog_cli.py          actual literal CLI and pre-loader rejection
docs/clm-linear-stream.md                     usage, boundaries, costs, extension guidance
```

README and the existing update guide link the usage document. No dependency or
environment-variable additions. Base revision09ff811f; inherited dirty changes
preserved. No commit/push/deletion/install/download/GPU/CI/browser operation.

## Executed checks, failures and amendments

Exact parent commands, in order (CPU1, sequential):

```powershell
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-stream-20261025/launch.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-stream-20261025/launch.py windows-2
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-stream-20261025/launch.py windows-3
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-stream-20261025/launch.py linux-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-stream-20261025/statics.py statics-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-stream-20261025/launch.py linux-2
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-stream-20261025/launch.py linux-3
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-stream-20261025/statics.py statics-final
```

Each parent receipt records the exact child command/environment and terminal
code. Stage `artifacts/runs/clm-linear-stream-20261025/` retains stdout, XML,
counters, timings, static failures and marked **software-only** CLI output.

| Cohort | Outcome | Body seconds | Parent seconds | Sampled peak RSS bytes |
| --- | --- | ---: | ---: | ---: |
| Windows1 | 15PASS,1FAIL,10fixtureERROR | 19.220657000 | 19.825984300 | 344,932,352 |
| Windows2 | 25PASS,1FAIL | 30.211460300 | 30.885735100 | 350,482,432 |
| Windows3 | 26PASS targeted | 18.606282600 | 19.040233300 | 263,880,704 |
| Linux1 | 32PASS,2fixtureERROR | 39.733158860 | 44.598005400 | 356,843,520 |
| Linux2 | 2fixtureERROR | 12.923083596 | 17.570706000 | 373,792,768 |
| Linux3 | 2PASS targeted native regression | 13.516291553 | 18.544532600 | 364,986,368 |

Final34 distinct qualified cases/platform by reuse plus targeted checks, not a
single final34-case Windows run. Actual public command uses one literal encoder
forward and completes12solves/36rows plus all cached endpoint/TRAIN scoring.
Complete original streams with and without diagnostics, correct delayed clocks,
union identity mapping, extreme inactive masking, final/input/source refusal,
lost-return paid prefix, corrupted receipt/work/objective/control rejection and
save-time prefix changes pass. Two affected native-default regression checks
share one twelve-update fixture and pass on both platforms. Existing #162
numerical/isolation evidence is reused; no full4096 solve was repeated here.

Windows1 exposed a real integration omission: app refresh still required the
native CE receipt objective after a linear solve committed. Corrected it to the
explicitly selected objective, keeping every clock/receipt criterion. The failed
solve prefixes and output remain. Windows2 completed actual streams/CLI; its
remaining failure was a test treating the existing TRAIN diagnostic **list** as a
dict. Corrected the assertion. Self-review also added guarded uniform-control
metadata/values and corruption checks before Windows3. The actual Windows2 CLI
output was reused only for software cached-score checks, never as scientific
evidence. Windows maximum cohorts2→3 was amended prospectively; other limits
were unchanged at that amendment.

Linux1 passed all new linear checks but its native fixture failed during Torch
import because the engineering profiler materialized unrelated frame locals in
Python3.12 inlined comprehensions. Narrowed the profiler to relevant named
functions/files before inspecting locals. Linux2 only reran the affected native
fixture; its improved module-copy coverage then hit the engineering module cap
before the fixture finished. Both failures are preserved.

The original32-object allowance was also undercounted in Windows: native
`_copy_bundle` creates two tiny working heads on **each update and each readout**.
A complete regression fixture costs2original+2owned+2*(12updates+5readouts)=38
heads, plus3modules for the literal CLI. Two Windows cohorts therefore charge82
objects, exceeding the original32 allowance despite constructor-only raw counts
of10. **The original engineering budget did not pass.** Old raw receipts were
not rewritten. Source inspection, followed by Linux3's observed38objects for one
native fixture, supports that accounting correction.

Prospective amendment [6402686070](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/163#issuecomment-6402686070)
increased the cumulative tiny-object allowance32→128/platform and Linux maximum
cohorts2→3, preserving all prior charges and failed attempts. The new profiler
counts constructor and deepcopy `__setstate__` paths. Windows charge82; Linux
observed71 plus2previously missed owned copies conservatively charges73. No
scientific limit or failed scientific run was renewed. Bodies stayed below90s
and parents below120s/platform: Windows68.038399900/69.751952700s,
Linux66.172534009/80.713244000s. No fourth cohort or wider rerun.

Per-platform cumulative observed work, including failures:

| Work | Windows | Linux |
| --- | ---: | ---: |
| real literal LSE / fitting rows | 79 / 237 | 42 / 126 |
| linear choice calls / query rows, including refusals | 47 / 179 | 25 / 94 |
| fake encoder forwards | 2 | 1 |
| native TRAIN attempt charges / row charges | 24 / 72 | 20 / 60 |
| native head forwards / projected rows | 68 / 214 | 56 / 176 |
| constructor scheduler initial calls | 2 | 2 |
| returned capture + Torch RNG elements observed | 875,152 | 657,879 |
| physical random parameter elements | 0 | 0 |

Native attempt charges include failed/uncertain work and are **not** assertions
that every attempted optimizer step returned. The final native fixture returned
all12updates; earlier interrupted attempts stay charged. The separate16MiB
other-copy reserve remains charged. No random parameter initialization,
pretrained load, simulator, bank/scientific raw or detached-review read. The
profiler/RSS/logical-array counters do not cover every allocation, operation,
FLOP or IO. Software fake encoder sequences were12 each; original scalar-token
literal tokenizer returns one ID per sequence,12slots each, below declared caps.

Initial nine-file statics: Ruff/format PASS, mypy10annotation errors,
21.223525500s. Fixed bindings annotation, non-None assertions after genuine
fits, and unused refusal-test Path arguments; no numerical behavior changed.
Final Ruff/format/default+Linux-target mypy all PASS,24.721608100s; cumulative
45.945133600s plus bounded formatter commands below180s. Actual Linux statics
unavailable/uninstalled. Type-only changes reuse passed behavior; no fourth
model cohort. One redundant unreachable qualifier selection branch was removed.
Two atomic patch-context failures made no edits; exact displayed context was
used thereafter. The native fixture retains its existing optimizer-wrapper
scheduler warning; no unrelated optimizer change was made.

Conservative touched whole cache109,707,280B (<160MiB), inherited global
596,528,597B separate, using20:04UTC cutoff. Stage571,723B before final report,
below16MiB. Command/XML/counters and all failure output remain local.

## Self-review and next action

Reviewed actual combined source, original inbox/clock/receipt flow, default
false branches, class mapping, active-softmax path, immutable current control,
bounded bindings, ownership/value checks, exact completed work and scorer
admission, all terminal receipts and observed costs. No fake optimizer work or
source-head parity. The uniform reference is guarded prospectively and again
before scoring. A failed solve/readout stops further admission; a later failed
report guard retains paid work. Core imports remain inward; the adapter reuses
the existing app, and CLI composition remains outside it.

Skipped unchanged fullsuite, native catalog/CPC comparative experiments, clean
clone/package/release, GPU/CI/browser, actual pretrained learning, scientific
bank analysis and independent review. The executed native-default regression is
the original paired fixture; unchanged catalog false-branch semantics and prior
catalog evidence were source-reviewed/reused, not rerun as a cohort. All older
negative results and spent #161/#153/#151/#159 budgets remain unchanged.

Next action: freeze and self-review one small **current-only cached-feature**
temporal study using the newly paid #161 bank with an explicit new data-access
allowance, original TRAIN/dev/probe role separation,12solves36rows, original
metrics and complete setup/incremental/peak-memory costs. Then execute that
single case through the existing runner, without another8B extraction. This
will measure acquisition, interference and return under the declared current v2
truth; it cannot by itself establish matched BP/PC/CPC benefit, replay benefit,
generalization or reliable native retention. No access/study allowance is granted
by this software closeout. Broader R2/R4/R5.1/R5.3/G5/R7/release remain open;
full repository goal ACTIVE. Historical plans/log stay frozen; GitHub is live.
