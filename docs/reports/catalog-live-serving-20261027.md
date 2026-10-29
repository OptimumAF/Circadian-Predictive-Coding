# Frozen chooser during live service

R7.7-CAT.L / [issue250](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/250).
Single-agent source/procedure/result **self-review**, not independent acceptance.
Full goal and broader R2/R4/R7/scientific/release requirements remain open.

## Working behavior

The existing `run_bp_ticket_serving.run_case` accepts a frozen catalog policy
with the existing `LiveServingPlan` at its original nine-opportunity cap.
The original96-update/576-row actor serves while the private candidate receives
causal TRAIN updates. Source/digest, frozen catalog, separate wake/replay budgets,
leases, admissions, receipts and clocks remain under their existing owners.
Trials, wrong horizons, other controllers, custom/deferred/live TRAIN recipes,
base rebasing, publication and followup refuse before source IO.

No executor, timeline, native model, teacher, learner algorithm, dependency or
CLI was changed. `scripts/ticket_catalog_control.py` is unchanged. Production
edits are confined to `run_case`, `_reference_schedule` and `_case_measures`;
all other module-level AST, including the just-qualified CLI, matches the
session-start copy. Inherited dirty work is preserved. Usage:
[private learning during service](../contextual-controller.md#learn-privately-while-the-original-actor-serves).

Why this: the qualified private chooser and live loop already use the same
sharing gate, but explicit compatibility refusals prevented their composition.
This adds that composition instead of creating another runner or audit layer.

`live_catalog_bp_ticket_private_v1` reports bounded live requests/versions/latency,
private candidate task readouts, wake/replay/native costs and pending packets.
Final completion additionally requires a closed, complete, known service capture.
Worker checks verify the original actor model/publication/configuration/metadata,
policy and source without mutating the main budget clock. Main checks original
plan/owner bindings before more predictions/admissions. Worker failures, changed
source/policy/plan/actor and corrupted native receipts stop the case and retain
paid prefixes. Service outcomes cannot become labels or controller observations.
There is no publication, promotion, final/probe access or new controller fit.
The old comparison utility refuses this distinct report rather than silently
omitting live outcomes/violations; complete native costs remain separately readable.

## Executed checks and commands

Local ignored evidence: `artifacts/runs/catalog-live-serving-20261027/`.
`check.py` reuses the existing finite software meter and optional-native import
trap. Process sampling starts before pytest/project fixture imports. New tests
use actual `TicketServingTimeline` workers, leases, shared runtime and catalog
executor, with scripted native learners, a literal public predictor and bank.
These are correctness fixtures, not trained-policy outcomes.

Final targets on both platforms:

```text
tests/test_catalog_live_serving.py
tests/test_ticket_catalog_control.py
tests/test_catalog_ticket_stream.py
tests/test_ticket_serving_timeline.py
-k 'not compile_only_the_original_complete_train_bank and not native_live_episode_versions_original_causal_receipts_and_priorities and not native_live_task_records_do_not_become_feedback_or_promotion_guards'
```

Windows: `./.venv/Scripts/python.exe -B artifacts/runs/catalog-live-serving-20261027/check.py win-combined <targets above>`.
Linux, followed by the same targets:

```text
wsl.exe -d Ubuntu --cd /mnt/c/Users/Avery/Documents/ChatGPT/Circadian -- env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/catalog-live-serving-20261027/check.py linux-combined
```

Exact argument arrays and pytest arguments are retained in each `*-checks.json`,
with JUnit XML. Both platforms pass **128 cases**:32new,40existing private catalog,
28stream and28pure timeline; zero errors/skips. Three genuine bank/native fixture
cases are intentionally deselected, not counted as executed. Existing mathematical,
native source/bank and prior #245–249 evidence is reused within its original scope.

| Final combined invocation | Body seconds | Sampled peak RSS | Scripted fits/rows | Scripted predictions/rows | Trial callbacks | Pure typed dispatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Windows | 8.366599400 | 87,666,688B | 291 / 2088 | 331 / 2874 | 15 | 330 |
| Linux | 27.228831262 | 76,304,384B | 291 / 2088 | 331 / 2874 | 15 | 330 |

All genuine models, optional native imports, real teacher-bank generation,
simulator interactions, scientific payload reads, final/probe access and native
scientific work are zero. Pure local workflow dispatches are actual fixture
work, not simulator or teacher interactions. Per-run RSS includes this process,
not a process tree/hard allocator bound.

New checks cover all three actions with/without reserved contention, exact
horizon and pre-IO refusals, FIFO/label causality, permanent replay cost after
deferred wake, paid lost returns, concurrent service failure after a successful
update, corrupted preceding replay receipt, worker-side input/actor mutation,
unknown final captures and service-result isolation. Changing public decisions
from successful to terminal wrong actions leaves full candidate, receipts,
training work and all controller observations/actions equal in paired fixtures.

## Failures and observed concurrency

All failed attempts remain in the stage:

1. `win-red`: collection EXIT4; new test recipe omitted required `label_delay`.
   Corrected to1, zero model work,1.598276700s/75,284,480B.
2. `win-red2`: intended RED EXIT1 at the catalog/live compatibility refusal;
   zero model work,1.819321500s/80,769,024B.
3. `win-new`: wrong fixture snapshot API (`VersionedState.generation`) failed
   before a fit; one prediction/six rows paid,2.893804200s/78,725,120B. Fixture
   now reads the original full serving snapshot. Production guard not relaxed.
4. `win-final`:2pass/1fail,5.127421900s/87,879,680B,16scripted fits72rows,
   24predictions144rows. The test incorrectly expected a composite replay and
   wake to both finish while native-started service still held the gate. Actual
   replay committed, subsequent wake deferred. This is legitimate contention,
   not missing work to force through the executor.

The tests now hold the real service lease through the complete action poll with
one ordinary Event, released in `finally` before the existing worker join. This
makes the causal deferral deterministic without changing production scheduling.
With that held lease and no reserved contention, wake1 completes8updates48rows;
replay_wake completes8replays24rows and **zero wakes**, leaving8fresh packets
pending. With reserved ticks2/6, wake1 completes6updates36rows; replay_wake
completes6replays18rows/zero wakes. Wait performs no updates. Actual native timing
can allow a service to finish before a composite wake; returned receipts determine
what happened. No budget renewal, wake retry, forced drain or action substitution.

After correcting that fixture, `win-final2` passed122cases/6.556542900s. Subsequent
worker-side/final-capture fault tests and guards prompted the final128case run.
Original4-invocation engineering cap was amended **prospectively** to6/platform
at issue comment6447767516 after measured11.438824300s and the explained fixture/
concurrency failures.120sbody/512MiB/work/storage limits stayed unchanged; no
scientific allowance was created. One failed patch anchor made zero edits, then
the formatted function signature was inspected and the small patch applied.

Windows cumulative6invocations/26.361966600s/maxRSS87,879,680B:598scripted fits,
4248rows,667predictions5778rows,30trial callbacks. XML records256instances including
one collection error and3failed tests;252passing instances,128distinct final cases.
Linux1invocation/128passing instances, counts as table. No failures discarded.
Typed dispatch metering was added for the final run:330/platform measured there;
earlier Windows typed work is **unmetered, not zero**. Source bounds for earlier
new/timeline cases conservatively stay below1600dispatches (finite9episodes,
six actions, stubbed task ports in other modules), keeping aggregate below4000
without rerunning to manufacture an exact counter. Summaries retain this limitation.

## Static checks and self-review

Final file arguments:

```text
scripts/run_bp_ticket_serving.py
tests/test_catalog_live_serving.py
artifacts/runs/catalog-live-serving-20261027/check.py
```

```text
./.venv/Scripts/python.exe -m ruff format --check <three files>
./.venv/Scripts/python.exe -m ruff check <three files>
./.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent <three files>
./.venv/Scripts/python.exe -m mypy --platform linux --explicit-package-bases --follow-imports=silent <three files>
```

All final commands pass, exact arrays/outputs/durations in `static-final.json`.
Initial format/lint passed; both initial mypy modes found four test-only
attributes on the dynamically replaced public-port factory. Explicit local
`Any` annotations document that monkeypatch boundary, with no ignores or
production change. This annotation-only edit leaves the executed runtime paths
unchanged, so128case behavior evidence is reused rather than rerun. Measured
initial+final static total31.512227400s; earlier formatter calls were not timed
separately and are not represented as zero. Original diagnostics in `static.json`.
Final mtime-selected cache113,233,920B is below160MiB (not growth; inherited other
caches excluded). Stage211,751B before review/receipt closeout is below4MiB;
`resource-summary.json` records the observation and cutoff.

Self-review inspected the combined source diff, original core/timeline procedures,
final tests/receipts and their scope, including the legitimate partial composite
result. Reviewed source is unchanged outside the three intended functions;
catalog executor/timeline bytes and original protocols are preserved. Existing
private/reference and stream suites pass, and unknown live work cannot make a
complete new report or enter the old objective. No independently accepted native
live-catalog behavior, task advantage, matched compute, efficiency, broad recovery,
unseen generalization or release readiness follows.

Skipped full suite/clean clone/wheel/install/browser/independent review, genuine
native live-catalog/application/comparison, unchanged math/baselines and scientific/
release gates. No dependencies/install/download/GPU/CI/agents/Git mutations/reset,
old archive reconstruction or raw publication. #245 negative results and #248
OPEN FAILED-SPENT remain intact; no comparison retry or renewed allowance.
Historical plans/log stay frozen per current AGENTS; issue250/tracker18 carry
the current handoff and open parents.

## Precise next action

Freeze one distinct native **live** qualification of the saved #245 chooser on
the original eight clean six-row TRAIN packets with `LiveServingPlan(9,(2,6))`.
Use its existing frozen clean/wait and mistakes/wake1 choices without refitting,
the original qualified96/576 source and existing native predictor/timeline. Ask
whether the candidate acquires the new task while the original actor continues
serving, and report old/new task endpoints, live failures/latency, pending work
and all model/teacher/guard/readout costs separately. No utility ranking or
controller/CPC benefit inference from one case; a failed acquisition is valid.

First adapt the existing small native observer, declare source/runtime/portable
paths, first-failure/no-retry and all initialization/forward/copy/update/typed/
sourceIO/time/RSS/storage ceilings **before models**, then self-review and run
only after those prerequisites pass. Up to6wake updates36fresh rows follow from
the unchanged9tick plan and reserved2/6; no selected replay/new fit. This would
be a genuinely different live service question, not the failed delayed/noisy
five-case comparison248. This ticket grants **no native allowance** and does not
alter that spent protocol. Broader sequential loss/relearning, mixed/noisy/
poisoning long-duration deployment and scientific/release requirements remain.
