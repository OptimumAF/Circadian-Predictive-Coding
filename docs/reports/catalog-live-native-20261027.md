# Native learning during live service

[R7.7-CAT.LN / issue251](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/251)
completed its sole frozen native invocation. Source/procedure/result
**self-review**, not independent acceptance. Full repository goal remains active.

The private candidate learned the previously failing legacy task while retaining
both old tasks. The original actor kept serving throughout and still failed
legacy requests. This is useful private adaptation during service, with no
publication or demonstrated controller/circadian advantage.

## Outcomes and costs

The existing `run_case` applied saved245clean→wait/mistakes→wake1 choices on
eight original clean six-row new-context TRAIN batches, delay1, using
`LiveServingPlan(9,(2,6))`. Original BP seed17/D7/H3/K6/lr.03/momentum.9, full
source96updates/576rows and frozen9decision/8wake/24replay envelope were unchanged.
No refit, action/metric/dose/seed selection, extra drain or new runner.

| Native observation | Result |
| --- | --- |
| Private new tasks (legacy/password) | 1/2 before → 2/2 after |
| Private old tasks (invoice/parcel) | 2/2 before → 2/2 after |
| Candidate ordinary task readout | All four complete and successful |
| New optimization | 6 completed wake updates,36rows; native state102/612 |
| Contention | Serving deferrals at ticks2/6; no denied native charge |
| Pending work | All8labels arrived;2fresh packets remain unapplied |
| Live service | 9episodes complete;6successes;29known requests |
| Live publication | Every request generation0/qualified-96/native96 |
| Safety observations | Zero applied permission/revision violations |
| Replay/probe/final/promotion/new controller fit | Zero |

Invoice succeeds at0/4/8, including the stale-once case at4. Password succeeds
at3/7; ordinary parcel succeeds at2. Legacy fails at1/5 on the original actor,
and the deliberately revoked parcel commit at6 is refused. These failures are
retained. Six serving leases began under native admission; that flag does not
prove simultaneous optimizer execution or useful parallel speedup.

Observed instrumented predictor-call latency: p50=16.559495ms,
p95=23.745758ms, maximum23.819525ms. Per-episode elapsed also includes queue,
reservation and workflow work. These are current-process local development
measurements, not SLA or cross-run efficiency estimates.

Outer body11.638976847s/peak338,923,520B, parent15.419287100s/EXIT0/no timeout.
Case-local elapsed8.702078723s/peak336,945,152B is nested, not additional.
The sampler started before Torch and metadata admission; process-tree/hard
allocator bounds are not established. Native resource counters:

| Work | Observed |
| --- | ---: |
| Model components / initializer slots / draw calls | 108 / 4212 / 216 |
| Full snapshots / logical elements | 646 / 88,128 |
| Native forwards / rows | 73 / 143 |
| Backward / SGD | 6 / 6 |
| Typed requests / local workflow dispatches | 59 / 119 |
| Bank certification dispatches | 36 |
| Case owners / selections | 1 / 9 |
| New utility observations / historical consistency checks | 0 / 6 |
| Tensor clones / logical elements | 4825 / 62,944 |
| Tensor deepcopies / logical elements | 9312 / 111,744 |
| Source+fit reads / logical bytes | 104 / 50,128,136 |
| Code-pin reads / bytes | 62 / 1,363,852 |

The forward count reconciles6training+8causal monitoring+47serving/phase
predictions+12candidate decisions. Rows are36+48+47+12=143. Live29requests are
part of47serving/phase predictions, not additional. The original twelve-episode
TRAIN bank was prepared once (36rows, retaining3), even though no replay action
was selected. Snapshot/clone/deepcopy logical counts overlap; initializer counts
are model-construction proxies, not exact FLOPs or allocated bytes.

Source, original actor, caller NumPy/Torch RNG, fitted bytes/choices and code
pins remained unchanged. Original120s512MiB wake/replay budgets, source versions,
causal FIFO receipt prefix, distinct replay charges and9opportunity cap held.
No training/promotion/selection used live outcomes or final/probe data.

## Historical costs and scientific scope

Qualified `_frozen_fit(catalog_fit=True)` validated the original six-trial fit
without restoring/refitting discarded models. The saved result carries48discarded
fits240rows/48replayrows,228prediction calls468rows,216bank-prepared rows; the
whole original study includes56fits288rows,304predictions624rows and288prepared
bank rows, including its applications. Original96fits576rows and source costs
remain separate. These are sunk costs, not new work or additive disjoint groups.
Exact fit overhead is inseparable and all-in utility stays unavailable.

Earlier245negative utilities and replay-worse results remain. The frozen chooser
duplicates a simple drift response on the earlier observed clean/mistake traces;
this single live application does not establish a chooser or CPC advantage.
No controls were retrained or retuned. This is known development grammar, not
independent unseen transfer, reliable sequential loss/relearning or long-duration
deployment. Private success is not deployed benefit: live legacy remained wrong.

#248 remains OPEN FAILED-SPENT with unknown delayed/noisy comparison outcomes.
This one clean6row LIVE case with actual service workers/private task readout is
distinct from that12row/five-policy/deferred question. No retry, continuation,
rescore, allowance rename or scientific tuning occurred. This allowance is now
spent as well; do not rerun the observer.

## Reproducibility and checks

Local ignored evidence: `artifacts/runs/catalog-live-native-20261027/`.
`PROTOCOL.md`, `run.py`, `check.py` and31current code pins were frozen before
models; paths are workspace-relative forward slashes. Existing250128checks per
platform were reused unchanged. The original source and fit hashes were checked
inside the native invocation before models and preserved by original guards:

```text
source ee40be1162f38d22730b3050726687e063c5a54cde6238ade158f9fc1efc6b27
fit    9db55756ddc3a81a4aecf40f3e3fee20c9c6313dcbac1b3d5f1fcd2322212c79
actual dc1e4997619c3b1fb902653727aead5bac9875448aaccff7a28d2e2f89ee0da2
```

Actual result450,578B. Parent stdout1775B/stderr0B; stage492,601B at terminal,
below8MiB. `native-work.json` governs complete protocol qualification; the
result's `case_complete` separately records original case completion. All
declared time/RSS/model/IO/storage limits passed. Exact sole native command:

```text
wsl.exe -d Ubuntu --cd /mnt/c/Users/Avery/Documents/ChatGPT/Circadian -- env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout --kill-after=2s 190s /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/catalog-live-native-20261027/run.py
```

The195s bounded parent recorded its PID/command, polled the same live process
to terminal and preserved raw output. No observation timeout restart. Offline
runtime NumPy1.26.4/Torch2.14.0+cpu matched. Main and future serving worker were
profiled, with synchronized counters; the memory sampler preceded profiling.
This corrected the prior single-thread observer's coverage for this new use.

Software-only commands: Windows `./.venv/Scripts/python.exe -B <stage>/check.py`
and the same WSL interpreter `-B <stage>/check.py`;9stdlib tests/platform PASS,
0failures/errors, .076298600s/.165629765s, no project/NumPy/Torch/model imports.
They cover permanent overruns, missing/returned fits, original dimensions,
forbidden PC, worker entry, shared counters and nonportable pin rejection.
An added local `dict[str,Any]` annotation after initial mypy diagnostics has no
runtime effect, so these behavioral checks were reused.

`check.py --pins` passed on Windows/Linux:31code files681,926B each,
.049668000s/.263490873s, zero scientific payload reads/native imports. Linux
also checked distribution versions through stdlib metadata. One pin-generation
preparation pass read the same31code files681,926B separately; it was not a
second portability preflight. No old-model/source preflight reads or archive
reconstruction. Native prologue/postflight performed their own bounded pin reads.

Final scoped checks use `run.py` and `check.py` in this stage:

```text
./.venv/Scripts/python.exe -m ruff format --check <two files>
./.venv/Scripts/python.exe -m ruff check <two files>
./.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent <two files>
./.venv/Scripts/python.exe -m mypy --platform linux --explicit-package-bases --follow-imports=silent <two files>
```

Final all PASS; exact arrays/outputs/times in `static-final.json`. Initial
format/lint passed; both type modes reported one missing namespace annotation
in the synthetic-thread helper. Explicit `dict[str,Any]` fixed it without an
ignore; initial diagnostics remain in `static.json`. No native launch preceded
the fix, pins, portable preflight or recorded `review-go.json`.
Measured initial+final static time16.841452500s; the first formatting call was
not separately timed, not represented as zero. Mtime-selected cache113,250,304B
is below160MiB (not growth; other inherited caches excluded); final stage526,406B
before review/receipt closeout remains below8MiB. `resource-summary.json` records it.

Saved-only stdlib review loaded the new result once, generated `readback.json`
and passed22checks: terminal status, counts/limits/pins/runtime, complete task/
service evidence, source/actor/RNG, budgets, known native work, FIFO causal
receipts, reserved deferrals and applied permission/revision checks. It imported
no project/NumPy/Torch and did no model reconstruction, prediction or rescoring.
Source/procedure/result self-review inspected these actual outputs, not just
the green checks. No production source changed this session.

Skipped new full suite/clean clone/wheel/install/browser/independent review,
unchanged baseline/math cohorts, comparative/PC/CPC/sequential-recovery/long-duration
studies and release gates. No dependencies/install/download/GPU/remoteCI/agents/
Git mutations/reset/raw publication. Historical plans/log stay frozen per
current AGENTS; issue251/tracker18 carry the current evidence and open parents.

## Critical path and precise next implementation action

Private learning now has a complete native example during live service. The
remaining user-visible gap is publication/use of that learned policy: this API
returns before promotion, and the original actor continues failing legacy.
Source inspection shows `_publish_candidate` already uses the existing
`ServingPromotionController.prepare/commit` and original inner guards. The live
drain path instead requires all8labels consumed; this immutable9tick case leaves
2pending. Extending the horizon or silently treating those packets as consumed
would violate the protocol and is not the next step.

Next implementation action: add pure ownership/version/receipt tests for an
explicit **guarded publication after the live window** opt-in in the existing ticket
driver. Reuse the current promotion controller and guards; preserve all6update
receipts and2pending old-version packet identities as permanently unapplied
when the old chooser/window ends. Do not rebind them, renew budgets or run more
TRAIN decisions. Prove old-owner refusal, exact accepted/rejected/unknown commit
receipt handling, and post-publication predictions before any new native study.
Keep default private/drain/followup paths intact. This targets actual use of a
learned policy, not another capture/audit/CLI sequel. Broad retention/recovery,
fair comparative evidence, long-duration mixed/noisy/poisoning and release
requirements remain open; a later publication qualification requires its own
genuine prospective scope, not a retry of this or248's spent invocation.
