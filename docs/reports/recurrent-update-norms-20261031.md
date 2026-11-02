# Recurrent committed-update diagnostics — #374 / R2.1-RNN.UPDATE

[Scoped completion and exact handoff](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/374#issuecomment-6548127540).
Issue374closed; broader goal and acquisition/scientific/release parents stay open.

The existing recurrent gradient diagnostic module now explicitly summarizes
actual consecutive native or public-cue snapshots. It reads all five parameter
blocks, gradients and before/after momentum, reporting actual/nominal SGD step
norms and residuals. Existing diagnostic API bytes and learner/default/stream
semantics are preserved. No new forward, backward, optimizer, random draw or
update runs in the summaries. Snapshot copies and scalar reductions are paid.
This is usable through the existing stream checkpoint callback, not a new runner.

Actual native stream correctness: four tiny literal TRAIN updates produce four
summaries with heads0/1/0/1 and local update clocks1/1/2/2; source and serving actor
remain unchanged. The direct native two-step fixture has nonzero gradients in
every block and SGD residuals below1e-7. Literal known-buffer cases check first
and continued momentum, zero momentum, exact residual values, corruption,
clock/policy/failed/otherhead refusal and detached outputs. Consecutive metadata
does not prove TRAIN provenance or objective; the caller must own that operation.

Stage: artifacts/runs/recurrent-update-norms-20261031/. Preimage retained. Physical
D: HEADd9119e8/inherited dirty tree preserved. Frozen software scope/caps are in
issue374body/start6547953101; prospective extra affected Windows slot6547993075.

## Executed checks and original failures

Windows1:20PASS/1FAIL,6.507415300002322s/RSS330366976B; wrong fixture row IDs correctly
refused before stream work. Windows2:1FAIL,7.478202300000703s/RSS329768960B; fixture
passed unsupported max_replay_examples=0, correctly refused before stream work.
Both paid attempts remain. Simplified fixture to existing production _batch and
accepted native stream constructor, with update/wall limits. No production
admission weakened. One prospective affected-case slot was added, while original
aggregate time/native work/storage ceilings remained. No scientific quota renewed.

Windows3:1PASS,17.23265530000208s/RSS327106560B. Linux1:21PASS,
7.204770418999999s/RSS341381120B. Windows union21passing distinct cases, Linux21;
42passed call outcomes and two preserved fixture failures. The Linux session67571
was polled to terminal exit0 without restart. Linux host61.074238s includes WSL
startup; body time is not substituted for host-inclusive cost.

Windows cumulative:6SGD/backwards14rows,11native constructions/33private draws/
13052initialized scalars,21optimizers,6forwards14rows,194snapshot attempts193returns,
365tensor clones68924scalars. Linux:6SGD/backwards14rows,7constructions/21draws/
7836scalars,17optimizers,6forwards14rows,178snapshot attempts177returns,325clones/
58372scalars. The one refused nonfinite snapshot/platform explains snapshot
attempt/return differences. Remaining attempted counters returned. Named counters
are not all allocations, copies, FLOPs or I/O. All frozen physical caps held.
No teacher/game/physics/probe/final or original TRAIN archive read in fixtures.

Ruff lint/format and production-module mypy default/Linux-target PASS,
5.1647233s statics (initial formatting separately). Tests are linted/formatted;
static type checking covers the production module. Actual Linux statics not run;
no install. Full suite/old real gradient/activation/model-parity cohorts/clean
clone/wheel/native task games and scientific comparisons skipped: no changed
assumption requires repeating them. No scientific or repository readiness claim.

Exact commands use TEMP/TMP D:\Dev\_Temp, physical repository cwd:

```text
.venv\Scripts\python.exe -B artifacts/runs/recurrent-update-norms-20261031/check.py windows-1
.venv\Scripts\python.exe -B artifacts/runs/recurrent-update-norms-20261031/check.py windows-2 tests/test_recurrent_update_norms.py::test_should_observe_existing_delayed_stream_without_changing_source_or_actor
.venv\Scripts\python.exe -B artifacts/runs/recurrent-update-norms-20261031/check.py windows-3 tests/test_recurrent_update_norms.py::test_should_observe_existing_delayed_stream_without_changing_source_or_actor
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp PYTHONDONTWRITEBYTECODE=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/recurrent-update-norms-20261031/check.py linux-1
.venv\Scripts\python.exe -B -m ruff check src/adapters/torch_recurrent_gradient_diagnostics.py tests/test_recurrent_update_norms.py
.venv\Scripts\python.exe -B -m ruff format --check src/adapters/torch_recurrent_gradient_diagnostics.py tests/test_recurrent_update_norms.py
.venv\Scripts\python.exe -B -m mypy --follow-imports=silent --cache-dir artifacts/runs/recurrent-update-norms-20261031/cache/mypy src/adapters/torch_recurrent_gradient_diagnostics.py
.venv\Scripts\python.exe -B -m mypy --follow-imports=silent --platform linux --cache-dir artifacts/runs/recurrent-update-norms-20261031/cache/mypy-linux src/adapters/torch_recurrent_gradient_diagnostics.py
```

Independent SOFTWARE review ACCEPT: four read-only calls/.5787702s; no models,
tests or numerical replication. A later cheap Ruff format --check returned0
(2files already formatted), resolving the review's exact-command note. Host
correctness total93.4754438s includes both failed Windows cohorts and WSLstartup.
The original
132mixed outcomes and143lost trained-health remain separate accepted/FAILED-SPENT
records; no trained weights or lost endpoint was reconstructed. Full acquisition/
retention/recovery, reliability, recurrentRL, heldout/scientific/CPC/release parents
remain open.

## One separately frozen current-TRAIN step trace

After software acceptance, freeze6548048635/PROTOCOL-NATIVE.md and independent
PRE GO (six total review reads/.8816282s) admitted ONE distinct transport
diagnosis, not32health or128acquisition/game retry. Current packaged raw148/K7
TRAIN A-blue/red twice, H8/seed17/lr.03/momentum.9/max16, original meanCE. Source
94705B SHA2565a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508,
existing resource/decoder/role admission. B/C labels were not fit or scored.
Four wakes observe first momentum continuation per head, not a dose search.

All20parameter-block observations have nonzero gradient and actual delta.
Maximum absolute SGD residual across all blocks/steps is1.4007091522251491e-8;
this continuous result was not used to fit a tolerance or tune the model.
The first two original pre-update episode losses per head decrease:

| TRAIN cue | First loss | Second loss | Recurrent gradient L2 first/second | Recurrent actual step L2 first/second |
|---|---:|---:|---:|---:|
| Blue |1.9780595302581787|1.9319700002670288|.0985138287 / .0965538024|.00295541549 / .00554714506|
| Red |1.95184326171875|1.912233591079712|.0968607737 / .0937341135|.00290582560 / .00542052351|

Second-step momentum norm exceeds gradient norm in every block; its direction
produces the observed larger update. Norms alone do not prove causal dominance,
overshoot or harmful interference. All five block metrics and four original
receipts remain in native-result.json; no subset was discarded. These are
pre-update losses, not post-final accuracy. No prediction, activation/SVD,
extra diagnostic gradient, native game or success metric was executed.

This contradicts an absent-gradient/absent-SGD-step explanation for the first
two updates/head on these inputs. It does not rule out later saturation,
objective imbalance, representation limits, insufficient acquisition or other
causes of132's mixed result. It establishes neither useful neural control nor
retention/recovery/reliability/generalization or a CPC advantage. No optimizer
or other learning change is justified solely by this trace.

Execution: one hidden owned process46904, terminal0/no host expiry; body
14.195778399996925s/321056768B, host14.615309s. Actual4SGD+4backwards36rows and
4forwards36rows;6native constructions18private draws7824initialized scalars;
14optimizers116snapshot validations221clones63092scalars. All attempts returned,
all frozen physical caps held. Output10684B under512KiB. Source, real serving
actor and globalTorch/NumPy/Python RNG conservation all true. TRAIN-only stream
has4receipts/36rows and0score forwards; no deployed update or publication.
Original4update60s512MiB budget birth228485.4233051, last clock228485.5383738,
charges4updates/0replay; it was not renewed. Attempt and SPENT markers retained;
no retry or additional source/control cohort. Same execsession19170 observed
to terminal by polling, not restarted after its first output yield.

Prepared native_case.py and native_observe.py have Ruff/format checks0;
unchanged accepted production source was reused. Exact child command:

```text
.venv\Scripts\python.exe -B artifacts/runs/recurrent-update-norms-20261031/native_observe.py
```

PowerShell Start-Process used WindowStyle Hidden, physical D: cwd/TEMP/TMP,
redirected two logs, retained the same process handle and bounded host observation
at100s. Final observer/host/source-birth/limits/counter/error receipts are in stage.
Independent POST ACCEPT of actual source/procedure/results and final report:
eight total read-only calls/1.2104086s for software/PRE/POST, no truncation,
models/tests or numerical replication. Stage63007B/cache28713418B before final
review notes are under8/64MiB. Reviewer-owned REVIEW.md/timings preserve the
optional missing-protocol read and note patch miss; no scientific cap changed.
Future132saved-read authority/source hypothesis was not independently audited.
Finite exception ended; no unused read allowance or new native case remains.

Exact next action: use the existing first-disagreement diagnostic approach to
locate the first TRAIN/live divergence in accepted132's two-head A32blue/red
failure, after checking its saved-read scope. Bound a distinct saved-only read
if permitted, reuse original predictions/teacher transitions, and never restore
weights, repeat fits/games or read lost140/143endpoints. This tests whether the
remaining acquisition gap is already visible on the original teacher trajectory
before choosing a learner change. If required fields/authority are unavailable,
record that dependency and choose another ready learner outcome; do not build
a recovery/source-loader/audit sequel. No next numerical allowance is created.
