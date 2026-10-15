# Coupled candidate checkpoint handoff

## Elapsed-retention control

```powershell
python -m scripts.run_coupled_candidate_handoff --describe --elapsed-retention
python -m scripts.run_coupled_candidate_handoff --run --elapsed-retention
```

`make_coupled_elapsed_lifecycle(owner, policy=...)` requires positive elapsed
seconds and supplies the existing coupled measurement ports. Owned-copy limits
are optional. Construct `RetentionExpiryDriver` explicitly and always stop it in
`finally`; the factory starts no thread. The basic and copy factories keep their
explicit elapsed-policy refusals.

The six fixed cases retain the original injected elapsed clock, lifecycle, driver,
copy budget and consent guards across handoff. At the exact two-second deadline,
public prediction, snapshot, restoration and raw ingress refuse before copying or
native work. Both StableActor and PromotableActor direct prediction also recheck
access after trusted copying and prediction callbacks, before publishing a result.
A real local worker then purges retired/current inboxes and the pending checkpoint;
a bounded stop joins it. Terminal driver state keeps public access closed.
The explicit wake requests retry after possible startup contention. Shutdown can
perform more than one empty cleanup; all passes retain the original four-poll,
six-cleanup-attempt allowance and preserve receipts and lifetime accounting.

Cleanup preserves complete learned state, receipts and spent byte/update limits.
The 576-byte lifetime copy charge remains after owned raw bytes fall to zero.
Caller copies, physical memory overwrite, unlearning and process recovery are
outside this guarantee. Injected elapsed time checks deadline semantics; it does
not measure physical deadline latency or establish continuous-service behavior.
See ADR-0296 and `artifacts/runs/coupled-elapsed-retention-20261015/` for the frozen
local protocol, focused checks and evidence. No comparative learning claim.

Focused tests: `python -m pytest -q tests/test_coupled_elapsed_retention.py`.
The retained worker controllers also select the affected old expiry and actor
regressions with actual-native work forbidden. Apply scoped Ruff/check-format
and Windows/Linux mypy to both new adapters, the CLI, both changed actor app
modules and the new test. Exact expanded selections and commands are in the stage.

## Retained raw-copy byte control

```powershell
python -m scripts.run_coupled_candidate_handoff --describe --managed-copies
python -m scripts.run_coupled_candidate_handoff --run --managed-copies
```

The opt-in copy factory supplies five measurements and `ManagedCoupledBuilder`
to the existing lifecycle coordinator. Supported raw graphs are bounded and
typed; aliases count once per copied graph, while cycles, opaque hooks and
unsupported arrays refuse before copying. Complete learned parameters and CPC
chemistry contribute zero raw replay bytes. The basic factory remains separate.

The fixed control uses a 576-byte lifetime copy limit and a separate 192-byte
ingress limit. Three pairs charge144 bytes; capture, restoration and another
capture charge144 each. A second restoration consumes its preparation attempt
but refuses before builder/inbox cloning. Another capture refuses before native
snapshot/inbox copying. Cleanup clears owned raw copies without refunding the
576-byte charge. New ingress still refuses that copy cap despite remaining ingress
capacity. Learned state, original resources and historical receipts remain intact.

These are retained raw array bytes, not a bound on weights, RAM or caller copies.
Native current-state snapshots may occur during restore validation before the
copy reservation. Elapsed retention and process recovery are separate features.
Restore's integrity check borrows validated inbox records under the original
owner lock instead of cloning them. It still serializes data into temporary bytes
for integrity comparison; the retained-array ledger does not bound that temporary
allocation. Public cursor capture continues to return detached payloads.

Focused tests: `python -m pytest -q tests/test_coupled_payload_copies.py
tests/test_coupled_managed_cleanup.py`. Apply scoped Ruff/format/mypy checks below
to the new copy adapter, shared cleanup adapter, CLI, both test modules and
`tests/test_retained_payload_budget.py`, plus the inbox/controller app modules
and their selected cursor/checkpoint test modules. Exact selected controls, commands,
controllers and two-mode native evidence are retained under
`artifacts/runs/coupled-owned-copies-20261015/`; see ADR-0295.

Validation supports119 distinct cases after one fixture repair (initial failure
retained). Scoped checks and independent review accepted. Both six-case native
modes passed with72 total updates/288 energy evaluations; the basic output matches
its previous accepted result apart from transient identity values. Two outer file
wrapper finalization warnings were preserved and fixed with an isolated zero-native
check; native runs were not repeated. Copy charges stayed576 after cleanup/retained0.

## Managed raw cleanup control

```powershell
python -m scripts.run_coupled_candidate_handoff --describe --managed-cleanup
python -m scripts.run_coupled_candidate_handoff --run --managed-cleanup
```

`src/adapters/coupled_data_lifecycle.py` supplies explicit exact-CoupledLearner
footprint/erase callbacks and real numeric NumPy byte measurement to the existing
ManagedDataLifecycle. Coupled native learners retain no raw replay batches, so
both callbacks report typed zero replay payloads without altering learned state.
`make_coupled_data_lifecycle(owner, policy=...)` supports basic ingress, tick,
holder and preparation limits. Optional owned-copy and elapsed-retention policies
refuse explicitly here; use their separate factories described above.

The six-case example retains one retired owner, the current owner and a pending
checkpoint across a native handoff. Opt-out clears all delivered registry-owned
inbox payloads conservatively, including other delivered subjects, and invalidates
the pending checkpoint. Complete learned state/policy, original actor and spent
ingress/holder/update accounting remain unchanged. New data still meets the
original exhausted lifetime ingress bound after cleanup. Receipts and payload-free
erasure metadata remain, as does an explicitly caller-owned inspection copy.

This drops supported owned raw references; it does not overwrite RAM, unlearn
parameters, erase caller snapshots, attest provenance or restore process authority.
Source/tests use the existing app/core APIs and add no dependency or algorithm.

Focused test command: `python -m pytest -q tests/test_coupled_managed_cleanup.py`.
Use the scoped Ruff/format/mypy commands below for the two adapters, CLI, new test
and `tests/test_managed_data_lifecycle.py`. Exact affected-old-test selections,
controllers and results are retained under
`artifacts/runs/coupled-managed-cleanup-20261015/`; see ADR-0294.

Validation:58 focused tests passed with no skips/deselections. Scoped statics and
independent actual review passed. The sole six-case native control passed in2.260s
with36 updates/144 energy evaluations. Original ingress144B/registry4 stayed spent,
both raw inboxes cleared and the pending checkpoint refused. Complete learned
state and historical receipts remained; caller copies were explicitly retained.

## Managed consent control

```powershell
python -m scripts.run_coupled_candidate_handoff --describe --managed-consent
python -m scripts.run_coupled_candidate_handoff --run --managed-consent
```

This fixed six-case control composes the existing managed owner with native
PC, neutral CPC and active CPC under fixed/guarded inference. Three synthetic,
unverified local declarations cover subjects A/B/A. After one update, opting out
A invalidates a pending token before any builder or preparation attempt. A fresh
handoff keeps the original consent owner, catalog, guards and resource quotas.
Only B's later pair trains; A's pending raw pair remains ineligible while capacity
remains. Held-out roles, unmanaged ingress, opted-out ingress, regrant and a fourth
declaration refuse without payload copy or further native work.

Full native states/policies, diagnostics and predictions match a separate direct
reference. PC-neutral parameter frames match and genuine CPC chemistry/wake
history survives. Output retains complete A/B/C envelopes, transfer observations,
final cursor, original/live identities and refusal evidence. It is an engineering
control with local metadata, not attested provenance, human consent or a comparison
of learning benefit. Opt-out blocks future use; raw records and copies remain.
Deletion, unlearning and managed lifecycle/process recovery are separate work.

Focused checks: `python -m pytest -q tests/test_coupled_managed_consent.py
tests/test_managed_experience.py --deselect
tests/test_managed_experience.py::test_should_apply_one_native_wake_then_revoke_second_pair_without_actor_mutation`.
The two deselected old native cases retain their unchanged accepted evidence;
the new native CLI is validated separately. Use the Ruff/format/mypy commands
below with `src/adapters/coupled_managed_consent.py`,
`scripts/run_coupled_candidate_handoff.py`, `tests/test_coupled_managed_consent.py`
and `tests/test_managed_experience.py`. Retained controllers, actual commands and
results: `artifacts/runs/coupled-managed-consent-20261015/`; see ADR-0293.

Acceptance:63 distinct passing focused cases (60 original plus3 repaired fixtures;
initial failures retained, two old native cases deselected). Scoped statics and
independent actual review passed. The sole six-case native run passed in1.846s
with36 updates/144 energy evaluations; raw opted-out records remain deliberately.

`CoupledCheckpointCallbacks` connects the complete learner envelope to the
existing `CandidateCheckpointController`. A checkpoint now checks the learning
rate, inference cap/rate and optional guarded policy as well as native PC/CPC
state before a replacement candidate can become current.

## Run

```powershell
.\.venv\Scripts\python.exe -m scripts.run_coupled_candidate_handoff --describe
.\.venv\Scripts\python.exe -m scripts.run_coupled_candidate_handoff --run
```

The fixed example covers PC, neutral CPC and active CPC under fixed and guarded
inference. It warms twice, queues three train-only samples with label times3/7/9,
and trains the first sample to B. While the original sharing gate is paused, a
wrong-policy preparation is refused without changing B, its cursor or resources.
The second preparation transfers ownership, preserving the original actor,
budget, clock, sharing gate and histories. The old owner and used token refuse
further work. At7 the next sample trains once to C, exactly matching a separate
uninterrupted reference; at9 the third sample remains pending at the original
two-update quota. The serving actor stays at A throughout both preparations.

Seed23, dimensions2→3→2→1, data, rates and neutral controls are fixed. The guarded
policy uses minimum1/gradient tolerance1e6 as a deterministic one-step control.
This example performs36 native updates with CPU threads1, no files or child
processes, and reports full envelopes/cursors/resources as bounded JSON. It is
an engineering equality check and makes no comparative learning-benefit claim.

## Repeated handoffs

```powershell
.\.venv\Scripts\python.exe -m scripts.run_coupled_candidate_handoff --describe --repeat-handoffs
.\.venv\Scripts\python.exe -m scripts.run_coupled_candidate_handoff --run --repeat-handoffs
```

The opt-in control uses one controller and the same actor, budget, clock and
sharing gate for three successive transfers. Four queued train-only samples have
label times3/7/9/11. The first three each train once, then transfer the candidate
under a paused gate. One initial wrong-policy build spends an attempt, so the
three successful transfers leave cumulative preparation counts2/3/4. Every
retired owner and used token refuses work; the actor continues predicting inside
all four builders. Full native state, cursor and resources match before/after
each publication, with exact uninterrupted learning references.

A fourth capture is allowed, but restoration refuses the original exhausted
preparation limit before another builder runs; discarding it frees pending
capacity without refunding attempts. At11 the fourth sample is still pending at
the original three-update sharing/budget quota. Full CPC wake history2/3/4/5 and
PC-neutral parameter equality are checked under both inference modes.

The existing controller validates current snapshot/digest/cursor before its
preparation-limit check. Refusal prevents another builder or native update;
snapshot copying and codec validation still occur. The direct budget probe
raises its original stop, while the shared runtime truthfully remains deferred
with a not-stopped cursor.

Six repeated cases perform48 native updates, at most240 energy evaluations and
no child processes or product datafiles. This small fixed repetition control
does not establish long-run memory bounds or process/lifecycle recovery. The
default command retains the original single-handoff six cases.

## Elapsed-time and RSS refusal controls

```powershell
.\.venv\Scripts\python.exe -m scripts.run_coupled_candidate_handoff --describe --resource-stops
.\.venv\Scripts\python.exe -m scripts.run_coupled_candidate_handoff --run --resource-stops
```

The resource control covers wall and RSS refusal for PC, neutral CPC and active
CPC under both inference modes:12 cases. Two updates and successful handoffs
retain one original clock callable, budget and manually sampled RSS reader.
At simulated time9, the third preparation reaches its original resource limit
before another builder runs. That attempt remains spent. A later update refuses
before native training, leaving sharing admissions3 and completed updates2;
the third sample remains pending and actor A continues serving.

| Observation | Source | Use |
| --- | --- | --- |
| Elapsed time | Injected logical clock0/3/7/9 | Wall refusal at limit8 |
| RSS | Injected reader1024, then4096 bytes | RSS refusal at limit2048 |
| Host RSS | OS readings before/after the saved validation run | Separate boundary telemetry |

The RSS case lowers its reader back to1024 after the first refusal. Its original
peak4096 persists and the subsequent update still refuses. Sampler start/reader,
sample count, peak and budget start survive each handoff; legitimate cumulative
sample/time observations may advance. No sampler thread is needed for these
deterministic controls. The existing before-update refusal leaves the runtime
not stopped, since no native mutation began; the spent sharing quota defers the
following poll.

The example performs72 native updates with fixed prior data/seed/settings and
no files or child processes. `--resource-stops` and `--repeat-handoffs` are
mutually exclusive. Existing default/repeat behavior is retained. These controls
do not measure continuous host peak RSS or establish long-run memory stability.

## Fixed endurance horizon

```powershell
.\.venv\Scripts\python.exe -m scripts.run_coupled_candidate_handoff --describe --endurance
.\.venv\Scripts\python.exe -m scripts.run_coupled_candidate_handoff --run --endurance
```

The endurance mode performs16 handoffs for each of the six model/inference
cases. It retains one controller, actor, budget, event clock, monotonic wall
clock and OS RSS sampler. Seventeen queued train-only samples have label times
3/7/9 through35, with the last sample at37 still pending after the original
16-update quota is spent. An initial wrong-policy preparation consumes one
attempt; successful transfers leave17 spent attempts and17 retained models.

The shared internal loop checks full native/policy/diagnostic/prediction equality
against a separate uninterrupted reference at every cycle. Full cursor and stable
resource state are compared before compacting observations. Output retains each
native envelope, causal receipt, cumulative counts and sampled resource telemetry,
plus the complete final cursor. Source-assertion flags identify in-memory checks;
they do not provide an independent replay of the reference.

The real monotonic budget starts before model construction, with limits45 seconds,
512 MiB sampled RSS and16 runtime updates per case. RSS is sampled manually at
budget/checkpoint boundaries, with its original reader, start, peak and count
preserved. This measures an observed high-water value at those boundaries;
transient peaks between samples may be missed. No sampler thread is started.

The fixed run performs204 native updates including warmups and direct references.
There are no additional seeds, sweeps, downloads or processes. `--endurance` is
mutually exclusive with the other control modes. A finite horizon does not prove
indefinite memory stability or full managed/process/lifecycle recovery.

Endurance validation:54 focused tests passed, including the unchanged repeated
and resource controls, with zero actual native/OS RSS/process calls. Scoped
Ruff/format and Windows/Linux type checks plus independent results review passed.
The single native run completed96 transfers/204 updates in17.85 seconds. Each
case recorded89 OS RSS samples; the largest observed peak was45,903,872 bytes,
within the declared512 MiB sample ceiling. All matched full-state references,
CPC wake histories2..18, neutral frames and original quota refusals passed.

## Integrate

```python
from src.adapters.coupled_checkpoint_callbacks import CoupledCheckpointCallbacks
from src.app.candidate_checkpoint import CandidateCheckpointController

callbacks = CoupledCheckpointCallbacks(
    learner.checkpoint_policy(), binding=binding, limits=limits
)
controller = CandidateCheckpointController(
    shared_runtime,
    build_learner=callbacks.build,
    state_digest=callbacks.state_digest,
    policy_digest=callbacks.policy_digest,
    max_pending=1,
    max_prepare_attempts=2,
)
sharing_gate.pause()  # Must be quiescent; preserve this exact gate and runtime.
token = controller.capture()
replacement = controller.restore(token)
sharing_gate.resume()
```

Policy, binding and limits are detached on construction and revalidated on every
callback. The policy probe requires the exact trusted learner class before
calling its getter and refuses policy mismatch before model work. State digests
and builders reuse the complete bounded envelope and existing restore factory;
neither trains nor predicts. The existing controller charges every preparation
attempt, validates full state/policy, and publishes the replacement atomically.

Why this: native state alone omits separate update settings. Reusing the existing
codec and controller closes that gap without adding an ownership protocol.
Bindings/digests check consistency; they do not authenticate callers.

This requires a surviving trusted same-process owner and its original resource
objects. It does not restore process, lifecycle, erasure or serving authority
from disk. Full managed recovery and long-run/scientific gates remain open.

## Files and checks

```text
src/adapters/coupled_checkpoint_callbacks.py  policy-bound callbacks
src/adapters/coupled_candidate_handoff.py     fixed six-case integration
src/adapters/coupled_repeated_handoff.py      opt-in three-cycle integration
src/adapters/coupled_handoff_resources.py     opt-in elapsed/RSS refusal controls
scripts/run_coupled_candidate_handoff.py     CLI
tests/test_coupled_checkpoint_callbacks.py   synthetic boundary/controller tests
tests/test_coupled_repeated_handoff.py        synthetic repeated/CLI controls
tests/test_coupled_handoff_resources.py       synthetic resource/CLI controls
tests/test_coupled_handoff_endurance.py       synthetic endurance/affected controls
```

Focused command (four old cases explicitly excluded for the zero-native,
zero-serve/guard test budget; unchanged accepted evidence is reused):

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_coupled_checkpoint_callbacks.py tests/test_candidate_checkpoint.py --deselect tests/test_candidate_checkpoint.py::test_should_preserve_exact_both_native_continuation_across_complete_handoff --deselect tests/test_candidate_checkpoint.py::test_should_preserve_promotable_bundle_and_invalidate_old_pending_promotion_ticket --deselect tests/test_candidate_checkpoint.py::test_should_refuse_actor_generation_aba_even_when_its_version_returns_to_base
$handoffPaths = @('src/adapters/coupled_checkpoint_callbacks.py', 'src/adapters/coupled_candidate_handoff.py', 'scripts/run_coupled_candidate_handoff.py', 'tests/test_coupled_checkpoint_callbacks.py')
.\.venv\Scripts\python.exe -m ruff check @handoffPaths
.\.venv\Scripts\python.exe -m ruff format --check @handoffPaths
.\.venv\Scripts\python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @handoffPaths
```

Repeat type checking with `--platform linux`. The saved test/native controllers,
exact commands, counters, results and independent review are retained under
`artifacts/runs/coupled-candidate-handoff-20261015/`. See ADR-0289.

For the repeated control, run `python -m pytest -q
tests/test_coupled_repeated_handoff.py`. Its ten synthetic cases need no native
fixtures. Apply the same Ruff/format/mypy commands to these three paths:
`src/adapters/coupled_repeated_handoff.py`,
`scripts/run_coupled_candidate_handoff.py`, and
`tests/test_coupled_repeated_handoff.py`. Repeated-control evidence is retained
under `artifacts/runs/coupled-repeated-handoff-20261015/`; see ADR-0290.

Repeated-control validation:10 tests passed with no skips/deselections and zero
actual native/model/process work. Three-path Ruff/format and Windows/Linux mypy
passed; independent source/results review accepted. The sole six-case repeated
native run passed in2.272 seconds with48 updates/192 energy evaluations and no
processes. Each case retained original limits over three transfers and refused
spent preparation/update capacity. Full elapsed/RSS, long-run and managed recovery
criteria remain open.

Resource-control checks use `python -m pytest -q
tests/test_coupled_handoff_resources.py`. Apply the same Ruff/format/mypy commands
to `src/adapters/coupled_handoff_resources.py`,
`scripts/run_coupled_candidate_handoff.py` and
`tests/test_coupled_handoff_resources.py`. Exact guarded commands/results and
separate OS RSS boundary observations are retained under
`artifacts/runs/coupled-handoff-resources-20261015/`; see ADR-0291.

Recorded resource validation:14 tests passed with zero actual native/model/process
work; scoped Ruff/format and Windows/Linux mypy plus independent results review
passed. The single12-case native run passed in3.172 seconds with72 updates/
288 energy evaluations,36 guarded one-step diagnostics and no processes. Actual
host RSS boundary readings were37,228,544 and47,001,600 bytes, separate from
injected stop values. No full long-run or recovery acceptance follows.

Endurance checks: `python -m pytest -q tests/test_coupled_handoff_endurance.py
tests/test_coupled_repeated_handoff.py tests/test_coupled_handoff_resources.py`.
Apply the same Ruff/format/mypy commands to those three tests,
`src/adapters/coupled_repeated_handoff.py` and
`scripts/run_coupled_candidate_handoff.py`. Exact guarded commands/results,
before-edit source copies and independent review are retained in
`artifacts/runs/coupled-handoff-endurance-20261015/`; see ADR-0292.

Extend through a new focused integration using the same trusted callbacks and
controller. Preserve original cursor/resources and test refusal before adding
new lifecycle or recovery behavior.

Recorded validation:67 focused tests passed/four explicitly deselected, with
zero actual native/model/process work. Scoped Ruff/format and Windows/Linux
mypy and independent source/results review passed. The single six-case native example passed in0.995 seconds:
36 updates/144 energy evaluations, no processes, exact full-state continuation
and original quotas. Initial launcher failures and annotation repair are retained
in RESULTS.md; broader release, managed recovery and scientific gates stay open.
