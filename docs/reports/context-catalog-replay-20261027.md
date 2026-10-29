# R7.4-CAT.CTX.R — Contextual TRAIN replay capability

2026-10-27, [issue261](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/261).
Previous goal turn made progress (#260); its source constraint and all prior
negative results are reused. This increment supplies the missing D9 TRAIN bank
and qualifies the existing categorical replay executor on the real conventional
paired-head learner. Full repository/controller/scientific goals remain open.

## Requirement reconciliation

The #260 proposed wait/wake-only catalog attachment would duplicate existing
`ContextualChoice`/`FittedController` support in `context_head_live`. The actual
`ContextualArmCatalog` requires ordered wait, wake1 and replay-then-wake with
six empirical estimates. Its contract was preserved, not reduced to two arms.

Source inspection found the concrete dependency:the existing ticket catalog
composition binds a D7 BP source/history/teacher bank and Torch CE diagnostic.
The context-aware live learner has a different D9 schema, paired-head source
and unit-mistake diagnostic. The shared `CategoricalReplayAction` itself already
supports the necessary owned native/state/exposure ports. A small script-side
authentic bank/composition helper is sufficient; no new orchestration framework
or changes to model math, generic replay, selectors or old runners are needed.
Issue261 freezes this scope/budget before fixtures, preserving full context live
catalog attachment/fitting/application as an unfinished subsequent requirement.

## Working functionality

```text
scripts/context_catalog_replay.py       Exact D9 TRAIN bank + existing executor composition
tests/test_context_catalog_replay.py    Codec/role/alignment and real replay/wake behavior
docs/context-catalog-replay.md          Caller composition, boundaries and limits
```

`compile_context_bank` validates the fixed ordinary old-v1 six-row TRAIN
prototype (invoice/password, original IDs, class/schema/arrays, no prior episode).
It compares every observation/target against the existing `_ticket_trace` public
context sandbox compiler and returns two owned complete TRAIN episodes. Original
teacher/compiler behavior is unchanged. The codec's terminal-zero placeholder
is explicit and is not an observed environment frame. Role declaration and
semantic checks do not independently authenticate physical data provenance.

`compose_context_replay` admits the existing paired-head learner/fixed D9 routing,
then returns the existing `CategoricalReplayAction` with native unit-mistake
diagnostics and `ReplayRetentionBudget(2,4096)`. Caller-owned ordinary TRAIN runtime
and independent finite original budget are reused. No session, restore, fit,
controller decision, file read or model training occurs in either helper.
Two encoded episodes plus dummy targets retain2608numeric bytes; one replay
event trains all6rows in a single native pass. Python/model/temp memory is extra.

The real-learner correctness fixture executes the unchanged full three-action
catalog envelope:wait, replay6-then-wake, ordinary wake with causally arrived
labels. One source fixture TRAIN pass6rows is paid before the runtime is created.
The candidate's new work is one6row replay plus two6row wakes, three passes18rows.
Wake budget charges two updates; separate replay budget charges6rows and zero
wake updates. Original source/actor and inactive v2 head stay unchanged. No
existing source artifact is read or represented as this synthetic fixture.

Replacing equal-valued bank or preceding successful receipt owners blocks
further native work. A deliberately lost replay return keeps6paid exposure rows
and the consolidation-attempt count, commits no model/receipt and stops later
native calls. Original clocks/limits/session owners are preserved. No success
chain or native-state archive is added; unchanged generic guard evidence is reused.

## Executed checks and failures

Final combined Windows `win-4` and Linux `linux-2`:20PASS/0FAIL/0SKIP each. These
are16compiler/admission cases and4real-executor cases, including2bank/receipt
replacement variants. All20 run on the actual current new script/test source.

Exact final commands:

```powershell
./.venv/Scripts/python.exe -B artifacts/runs/context-catalog-replay-20261027/check.py win-4 tests/test_context_catalog_replay.py
wsl.exe -d Ubuntu --cd /mnt/c/Users/Avery/Documents/ChatGPT/Circadian -- env PYTHONDONTWRITEBYTECODE=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-catalog-replay-20261027/check.py linux-2 tests/test_context_catalog_replay.py
```

Both `*-checks.json` and raw JUnit remain local in
[`artifacts/runs/context-catalog-replay-20261027/`](../../artifacts/runs/context-catalog-replay-20261027/).
The prospective per-platform ceilings are issue261:60distinct cases/6calls/
120s512MiB,32conventional TRAIN passes192rows/128constructors256zeroheads/
2000prediction rows/512sandbox transitions, bounded copy/slot visits. No study
or actual-case allowance was created. Software model work is counted below.

| Invocation | Outcome | Seconds | Real passes / rows |
| --- | --- | ---: | ---: |
|win-1|0PASS1FAIL:codec fixture omitted dummy-target argument|2.010733700|0/0|
|win-2|17PASS1FAIL:failure fixture expected no permanent attempt|2.042595500|12/72|
|win-3|16PASS1FAIL:byte assertion omitted two dummy targets|1.916040600|4/24|
|win-4 final|20PASS|1.725883500|12/72|
|linux-1|16PASS1FAIL:same dummy-target byte assertion|8.434478096|4/24|
|linux-2 final|20PASS|7.747920151|12/72|

The codec signature and failed-attempt assertions were corrected in tests;
production codec/accounting stayed unchanged. Retained encoded features2592B
plus two8B zero dummy targets =2608B, still below the original4096B retention
ceiling. No cap was raised or failure removed. One patch failed against a
formatter-split anchor, applied no changes, then succeeded using exact context;
it caused no fixture/model rerun. The superseded downloaded multi-agent prompt
was absent; current AGENTS's single-agent workflow governs without blocking work.

Final each platform:12passes72rows,17learner constructors34zero heads,
39sandbox transitions,252context/384head copy visits42624logical coefficient
slots,0prediction rows/native Torch/history reads. Windows cumulative all four
attempts:28passes168rows,38constructors76heads,105sandbox,585context/892head
visits98976slots,7.695253300s/max sampled73,670,656B. Linux cumulative two
attempts:16passes96rows,22constructors44heads,60sandbox,334context/510head
visits56544slots,16.182398247s/max64,540,672B. Copy accounting conservatively
combines typed deepcopy visits and explicit validation copies, including memo
reuse/double counting; it is not an allocation count. Generic Python objects and
process RSS are additional. All original prospective software ceilings pass.

Scoped three-file Ruffformat/lint/default+Linux-target mypy pass; full exact
commands, outputs and timing are in `statics-*.json`. Final receipt and
`cost-rollup.json` account measured commands and mtime-selected cache payload,
separate from inherited global cache/growth. No installed runtime changed.
Measured static subprocesses22.950137000s plus3.0448725s conservative formatting
tool-wall upper (including the first combined format/check command, which overlaps
its recorded model time). Mtime-selected touched cache113501628B, below160MiB;
it is not global cache size or growth. Finite stage40508B before GitHub receipts.

Single-agent combined source/procedure/result SELF-REVIEW passes, not independent
acceptance. It covers bounded schema/role/data admission, actual compiler reuse,
owned bank/codec bytes, unit-mistake definition, original session/source/state
boundaries, full unchanged catalog contract and all actual partial-work results.
New-file whitespace/local links and finite stage are checked separately. Full
suite, old native math/executor suites, clean clone/install/wheel/browser,
independent review, actual controller fit/application, comparative BP/PC/CPC,
physics, novel tasks and release gates were skipped as outside this scope.

## Scope and next action

This is implemented and executed capability, not learned task usefulness or
scientific benefit. No final/probe/guard labels enter replay. No physical data
source, random model initialization, SGD/PC, promotion, fit/search, seed/metric
selection or historical source reconstruction was executed. Existing #73 D9 BP
failure, #74/#77/#78 conventional results, #248 failed-spent comparison and #259
negative native invoice/retained-old/no-recovery result remain unchanged.
Full #21/#23/#26/#33, reliable adaptation/science/release and full goal stay open.

Next exact implementation action:wire this bank into an explicit **full three-arm**
context catalog private-trial/application option through `context_head_live` and
the existing shared schedule/executor. Preserve default reference modes and
source77 binding, full native origin/clock accounting and separate replay exposure;
do not pass a BP-fitted chooser off as a matched contextual fit. Reuse existing
task readouts and counterfactual fit ports, refuse unsupported combinations before
model work, and freeze a small new protocol before any real trial/application.
Do not build another runner or repeat a spent scientific case. Historical plans
and log remain frozen per AGENTS; issue261/tracker18 carry current status/handoff.
