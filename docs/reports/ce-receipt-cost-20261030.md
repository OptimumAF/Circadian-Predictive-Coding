# CE scalar receipt cost diagnosis

R7.1-TW.CE.COST / #345. Previous344 was progress: guarded F16 acquired2fresh
tasks while retaining4old, but full case FAILED/SPENT on original source30s
clock during return2. This fixed model-free probe tests one possible cost; it
does not resume/retry344 or alter any scientific budget/data/metric/seed.

Decision: **no production change justified by this probe**. Neither frozen
18-event triage threshold (meancheck>=0.25ms or meanpublication>=5ms) was reached
on either platform. These are engineering triage thresholds, not scientific
metrics or an equivalence margin. No observed native call frequency or complete
native duty cycle was measured, so this does not prove the component's share
of344runtime or identify the actual dominant cost.

| Prefix events | Windows publication ms | Windows check ms | ProducerWSL publication ms | ProducerWSL check ms |
| --- | ---: | ---: | ---: | ---: |
|0|0.025367|0.001983|0.023129|0.001695|
|1|0.049400|0.005715|0.051300|0.004432|
|8|0.225700|0.031138|0.239284|0.026439|
|16|0.418000|0.060804|0.467780|0.048633|
|18|0.509033|0.068183|0.622952|0.053645|

Actual CEProbeWork._publish/check and paid_report_guard on detached scalar
prefix copies0/1/8/16/18 from344preserved_ce_rehearsal_work. Three publication
timings and100individual check timings perprefix, all stored including outliers.
First publication transitions empty priorview->prefix; second/third priorviews
are equal-sized, so average is not a pure steady-state publication estimate.
Counter wrappers/timers add overhead. JSON/import/copy setup is excluded from
microtimers but included in measured probe scope/RSS. No task or native update
was replayed; fixture counters describe saved metadata, not new exposure.

One additional cProfile sample at18events (1publish+10checks) perplatform:
Windows.004616s, WSL.004557369s. Profiled versus unprofiled timings are distinct
scopes, and profile instrumentation is not a production speed estimate. Top
WSL cumulative functions include publication/deepcopy and scalar check. Full
30function rows retained, no selected favorable sample, warmup or seed search.

Perplatform actual21binds/21publications/526guard closure calls/47top-level
deepcopies/0sourcecallbacks. Twentyone publications include5constructor empty
views,15timed publications and1profiled publication. Recursive deepcopy/leaf
operations add cProfile/time but are not extra top-level calls. Existing loops
and caps bound all work. Torch/torchvision imports and genuine CE entry refused;
forbidden lists empty, genuine models/forwards/gradients/SGD/fits/latent/task/
physics/GPU/final/remote work ZERO. No source actor/learner/clock reconstructed.

Commands from physical D:/Dev/Repositories/Circadian, process TEMP/TMP
D:/Dev/_Temp; WSL TEMP/TMP/TMPDIR=/mnt/d/Dev/_Temp:

```text
.venv/Scripts/python.exe -B artifacts/runs/ce-receipt-cost-20261030/measure.py windows
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/ce-receipt-cost-20261030/measure.py linux
```

WindowsEXIT0/.339517900s/44908544B; WSL EXIT0/.812956487s/48427008B.
Measured probe scope total1.152474387s. External exec_command shell durations
Windows.5276526s/WSL3.841112s total4.3687646s include shell/WSL startup and
are also below10s each/20s aggregate. Internal seconds exclude stdlib startup,
SPENT/prior loading and final serialization/write, so they are not external
launch-to-exit processwall. RSS is sampled, not an OS-enforced hard ceiling;
observed peaks stay below256MiB. All logical work ceilings held. Windows
Python3.14.7/currentNumPy versus producerPython3.12.3/NumPy1.26.4; isolated
measurements are reported separately, not treated as paired identical systems.
Read-only saved JSON never changed, unique SPENT markers and raw500check/15pub
timings/platform retained. Stage below2MiB, scoped cache14201061B below64MiB.
Initial7typing diagnostics retained; normal signatures/types/cProfile.getstats
repair passes scoped Ruff/format/Linux-target mypy. Exact tool argv/timings
statics.json/statics-ready.json. No new dependencies/install/delete/global
settings/commit/push/production edit or other test/model cohort. Existing
343software and341342math evidence reused, no broad/cleanclone/release checks.

Source inspection identifies the next concrete unmeasured path:
run_initialized_probe.boundary invokes ready both before/after its source
guard; native require invokes that boundary and check_initialization=ready
again. ready invokes prepared_guard (the unchanged closure) and paid init
guard; prepared_guard invokes owner_unchanged, which rebuilds native state
and compares native ownership/state/globalRNG. CEProbeWork.boundary separately
wraps the protected boundary with two require calls. This structural nesting
explains why the zero-source-callback microprobe does NOT measure those costs.
It does not prove they dominate or justify deleting checks/caching validity.

Exact next action: freeze a bounded no-gradient timing of native snapshot/RNG
comparison on a fresh engineering owner/state, without rebuilding344models or
replaying its training/tasks, then choose a cost repair only from measured
evidence. Preserve every source/receipt/role/counter/RNG/input/clock check and
all original failed evidence. No automatic native344retry or time extension.

Self-review of actual script/counters/thresholds/result limits complete. Independent actual source/procedure/raw timing/profile review ACCEPT scoped
negative component diagnosis,4readonlycommands/allEXIT0/.7831428s. Physical
REVIEW.md verified and exact receipts retained; no reviewer project imports,
tests/models/reruns/network/children. Finite exception ended; no numerical
replication. Wall/RSS scope qualifications clarified from review feedback;
next native-owner implementation remains a measurement candidate, not an
independently inspected performance repair.
Fullreturn/reliableadaptation/R2/R4/R5/R7/matched science/release/full repository
goal remain open;344failed wholecase and valid F-prefix remain unchanged.

Evidence receipts for R7.1-TW.CE.COST: [canonical closeout](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/345#issuecomment-6537686018); [parent #18](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/18#issuecomment-6537688491), [parent #21](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/21#issuecomment-6537688746), [parent #23](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/23#issuecomment-6537688968), [parent #31](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/31#issuecomment-6537689196).
Machine-readable receipt: artifacts/runs/ce-receipt-cost-20261030/closeout.json.
