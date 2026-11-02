# #398: native attempt failed before simulator creation

Historical attempt1/repair record. A separately amended, cumulative-cost
[attempt2 completed the physical qualification negatively](continuous-affine-native-continuation-20261101.md).
The failure and spent charges below remain unchanged; no native result is
retroactively attributed to attempt1.

The ONE frozen native qualification failed in record conversion. All four
Push17/Pick17/Pick19/Pick20 outcomes are **unobserved**. This is an engineering
failure, not four failed policies and not evidence about native affine control.
No success/retention/recovery/reliability/CPC benefit follows from this attempt.
Full repository goal and398physical criterion5 remain open.

## Actual attempt and costs

From physical D:\Dev\Repositories\Circadian, exactly once:

```powershell
.\artifacts\runs\continuous-affine-native-20261101\run-native.ps1
```

Existing Linux Python3.12.3/NumPy1.26.4 environment was available from
`/mnt/d/Dev/Repositories/Circadian`, matching Mesa ICD, CPU1 and no competing
Python process at metadata-only preflight. Original source fit used NumPy2.4.6;
no solve/residual reproduction was attempted on Linux. TEMP/TMP/TMPDIR pointed
to physical D: in both shells. No install/download/Git publication/global change.

Host PID40884: terminal exit1/child1,5.661515s, no expiry. Producer2.959096991s,
sampled peak57499648B; GNU time3.07s/max55788KiB/exit1. Error:

```text
TypeError: ContinuousAffineFit.__init__() got an unexpected keyword argument 'coefficient_norm'
```

The single cached NEW fitted-map read was charged once. Its2066B length and
accepted hash were checked before JSON reconstruction failed. The existing fit
app writes `dict(asdict(model), coefficient_norm=...)`; the native producer
incorrectly passed that redundant diagnostic to the core constructor. Before
this attempt, the literal port tests had constructed model objects directly and
did not exercise this serialized production report shape.

Reader/fork/snapshot/prediction/native-creation/control/close counters all0.
No fit/SGD/replay/probe/model RNG initialization/physics, no teacher or old raw
source/result reads. Forbidden reads0. Native model-state qualification is null,
not true. Global SDK RNG was not claimed unchanged. No simulator was created,
so no native cleanup owner existed or episode outcome can be inferred.

The failed/spent stage retains protocol, producer, host, markers and receipts:
`artifacts/runs/continuous-affine-native-20261101/`. Actual JSON759B, SHA256
90bb9208d871f63435e178a7ddd91107b4cd9ab28f4486df69aeddd992956175.
Lead and peer each read this new result once; use LEAD-SCALARS.json thereafter.
No result-byte read by producer. Host observed stage27373B/log455B before final
receipt/review additions, within8MiB/512KiB. Combined report759B<4MiB; no Pick
trace exists. Observed RSS/time caps held; budget_ok true does not imply task
completion. The120s/2GiB protocol and native read allowance remain SPENT/no retry.

All12 direct pins matched at PRE, independent POST and root POST before repair.
The producer did not reach its own after-run pin check. Root preserved the old
readonly adapter as `readonly_continuous_affine-before-repair.py` before adding
the helper below. That later addition changes the current adapter's digest;
frozen pins and failed capture must not be rewritten to hide it. This is limited
direct-source evidence, not whole-tree authentication.

## Engineering checks and independent review

New actual-observer AST checks:7PASS/.38s, no producer/model/native import. The
initial7test peak RSS was uninstrumented and remains UNKNOWN/SPENT, not a proved
256MiB gate. Two selected cleanup/lost-return tests then passed with sampler:
.5401833s/46907392B. PRE source review found return metering could throw after a
successful constructor and lose the caller's cleanup owner. Corrected observed
return acknowledgements to increment before any next resource check; a new
elapsed-constructor-boundary literal passed1/.29s. Eight distinct checks,
10executions allPASS/0SKIP; total quoted pytest times.96s plus framework overhead
(only selected2captured end-to-end). No mathematical/numerical allowance renewed.
Producer/test Ruff passed and host syntax errors0; no full producer type claim.

Commands:

```text
python -m pytest -q -o addopts= -p no:cacheprovider --basetemp artifacts/runs/continuous-affine-native-20261101/test-tmp tests/test_continuous_affine_native_observer.py
python -I -B artifacts/runs/continuous-affine-native-20261101/check_resources.py
python -m pytest -q -o addopts= -p no:cacheprovider --basetemp artifacts/runs/continuous-affine-native-20261101/ack-tmp tests/test_continuous_affine_native_observer.py -k acknowledge_returned_constructor
python -m ruff check --no-cache artifacts/runs/continuous-affine-native-20261101/native_case.py tests/test_continuous_affine_native_observer.py
```

Finite independent PRE GO was conditional on return-ACK correction and pinning;
both were applied before launch. Two PRE source calls plus one POST call,
~.78s reported aggregate shell time, read-only/no execution/model/input reads.
POST accepted the **failure record**: exact result hash, counters, caps, source
pins and no native inference. Reviewer ended; this is not native acceptance or
independent numerical replication. Prior22port/4strengthened callback checks and
accepted mixed198fit were reused without baseline or simulator reruns.

## Pure repair, without another experiment

Added `readonly_affine_from_fit_record(record, role="train")` to the existing
readonly adapter. It handles exactly the eight-field serialized model section,
including diagnostic coefficient_norm, bounded13x4 JSON coefficients, <=13
singulars and the four-axis physical spec. Existing core validation checks
finite model/rank/conditioning. The norm diagnostic must agree with robust hypot
within8binary64 epsilons relative/0absolute; this rounding bound applies only to
redundant diagnostic metadata, never fitness, metrics or model selection.
Unknown/missing fields, malformed arrays, corrupt diagnostics and non-TRAIN
roles are refused. Inputs are detached. The helper does not read files or
authenticate release/provenance; those remain the caller's responsibility.

Literal production-shape test uses actual `asdict(model)+_coefficient_norm`
serialization, followed by JSON round trip, without loading any experiment.
Malformed shape/spec/norm/rank/condition and role refusals are covered. No saved
model read, fitting, prediction, random model draw or native work during repair.

```text
python -I -B artifacts/runs/continuous-affine-record-repair-20261101/check.py
python -m ruff check --no-cache src/adapters/readonly_continuous_affine.py tests/test_readonly_affine_record.py
python -m mypy --follow-imports=silent --cache-dir artifacts/runs/continuous-affine-port-20261101/type-cache src/adapters/readonly_continuous_affine.py tests/test_readonly_affine_record.py
```

16PASS/.6432021s/56311808B/budget_ok true; Ruff and types exit0. Separate repair
engineering limits32executions60s256MiB/64record conversions,8MiB artifacts and
reused32MiB type cache. Old core/prediction/class methods/ports unchanged, so
accepted algebra/native-port software evidence remains applicable. This proves
literal record conversion, not conversion of the spent actual cached file.

Independent finite repair SOURCE review ACCEPTED the helper and test scope in
one read-only batch/.2004s shell time, no execution/model/artifact reads. Tests
are lead-executed; this is not independent native/numerical acceptance. The
serialization-review exception ended without granting any retry allowance.

## Open scope and next action

#398 remains open for physical qualification. Original failed producer/protocol
are frozen; no automatic retry, new cached-map read or native allowance follows
from the repair. Original198fit, earlier Push/Pick/MiniGrid negative results,
consent, budgets, clocks, permanent charges and unrelated dirty work are retained.

Next SAME398: determine an explicit prospective continuation protocol for the still-unobserved physical
qualification. It must retain this failed attempt and cumulative costs, use the
same map/all4fixed TRAIN endpoints, and justify any agent-created engineering
limit change before execution. Do not relabel this as an unspent experiment or
rename a retry into a new task. No new fit, baseline/seed/weight/dose selection,
probe/final input or retrospective success threshold. Fullsuite/cleanclone/wheel/
browser/device/GPU/CI/release skipped; no repository-wide readiness claim.
