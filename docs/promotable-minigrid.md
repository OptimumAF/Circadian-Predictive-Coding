# Play through the published affine actor

`src.adapters.minigrid_serving.run_promotable_minigrid_episode` plays one
caller-named known A3/A4 TRAIN episode through an actual `PromotableActor` with
an exact `ContextLinearLearner` backend. It uses the existing MiniGrid episode
kernel, public initial/current frames, seed17, max16 steps and raw K7 first argmax.
The actor's public `predict` supplies every decision. Promotion and rollback
remain the existing controller's responsibility, between calls.

The existing affine command now offers a fixed first-party workflow:

```console
python -m scripts.run_minigrid_linear_fit_control --example-affine-published --describe
python -m scripts.run_minigrid_linear_fit_control --example-affine-published --output artifacts/published-affine.json
```

Run from the checkout with its optional MiniGrid dependencies. The output must
not already exist. This mode accepts only `--describe` or `--output`; it refuses
source paths, fit overrides and other modes. Description is pure. Execution
checks optional imports/version pins before creating source/model/budget owners.
Missing dependencies produce failed JSON and exit2 with no productive work.

The workflow fits original A blue/red TRAIN batches (2fits/18rows) through the
ordinary delayed stream, uses disjoint original B blue/red examples only as
inner guards, publishes through the existing controller, then plays two A3 TRAIN
games through the actual actor. C examples are unused. Guards retain the fixed
accuracy/gain/drop/latency/resource/action criteria; a refusal stops before games.
No teacher, final/probe/replay, cold/rollback cohort or comparison is launched.

The normal command uses one original2update/90s512MiB TRAIN owner and one
0update/0replay/150s512MiB readout owner, both born before data/models. It checks
source, candidate, actor, original controls and preceding paid reports before
more work. Lost native returns remain unknown even when stopped candidate
readback is refused; accessible receipts and permanent charges remain. Failed
game/readback prefixes stay available, and incomplete final capture cannot
publish a success summary. CLI failure exits2; reports contain no weights or
live authority. Nothing creates durable continuation or refunds failed work.

The actual Windows command achieved2/2 known A successes,9steps/.49375return
each, with2fits18rows and18physicssteps. Complete command/export and costs:
[issue180 report](reports/published-affine-cli-20261025.md). Current previously
built wheels predate this mode; installed delivery has not been qualified here.

```python
from src.adapters.minigrid_serving import run_promotable_minigrid_episode

def play_known_blue(actor, budget, check_boundary, episode_id):
    return run_promotable_minigrid_episode(
        actor,
        cue="blue",
        episode_id=episode_id,
        budget=budget,
        check_boundary=check_boundary,
    )
```

Create the readout `ToyBudgetSession` before source/model work with
`max_training_updates=0`, `max_replay_examples=0`, `max_wall_seconds=150`,
`max_process_rss_bytes=512 * 1024**2`; attach the existing `ProcessRssSampler`.
Reuse that original budget across the declared games. `check_boundary` must
validate original consent, preceding paid evidence and other caller-owned
controls; this adapter does not create or authenticate them. A fitted actor is
supplied through the normal existing stream/controller workflow, never obtained
by treating a result file as live authority. There is no new environment
variable. Optional dependencies remain those in `requirements-minigrid.txt`.

An optional `layout=MiniGridKeyDoorLayout("A", branch_row=4)` selects known A4;
default A3 and `cue="blue"|"red"` are the only admitted choices. All seven native
inputs stay available. No labels, teacher, hidden grid or action repair feed the
policy. Use distinct caller episode IDs; the adapter does not retain an ID bank.

Why this: the ordinary checkpoint evaluator forks and restores a detached
scorer. That is useful for fixed checkpoints, but cannot demonstrate what the
published actor actually serves. This binding adds a narrow branch to the same
kernel and pins the complete serving snapshot, generation, version, native
backend and slot identity. Equal slot/backend replacements, changed metadata,
model, counters, policy, cache clock or configuration stop the binding.

Only exact bounded native records are admitted. Serving cache must be empty;
TTL<=1024 and capacity<=16. Metadata has<=16 flat native scalar entries, keys
1..64 characters and string values<=256; numbers must be finite/bounded.
Version and episode IDs have<=64 characters, generation/cache tick<=2**53-1.
Failed native work and managed lifecycle owners are refused before public
snapshot/prediction callbacks. No arbitrary Python-graph certification or
training provenance is inferred from these checks.

`PromotableActor.serving_snapshot_if_idle()` returns the same full detached
snapshot as `serving_snapshot()` and refuses contention. A physical action
leases the actor's original read gate; ordinary commit/rollback use the same
gate. Trusted native `step` must not synchronously reenter actor/controller
methods while it holds that gate. Public prediction keeps its existing blocking
behavior; this is not hostile-callback isolation or hard preemption.

The returned report includes actor generation/version, native counters and
attempted/returned predictions/actions. `MiniGridEvaluationFailed.report`
retains the failed episode prefix, cleanup/readback errors, unknown work and
stopped binding report. Failed paid work is never refunded or silently retried.
Only one model pin and finite per-episode records are retained.

Focused verification:

```console
python -m pytest -q tests/test_promotable_minigrid_binding.py
```

The51 literal checks pass on Windows and cached Linux; they perform no real
fitting/forward/physics. After a preserved zero-work dependency refusal and a
prospective admission correction, the actual Windows actor achieved0/2 cold,
2/2 published and0/2 rolled-back known-A games. Two fits used18 rows; rollback
reproduced the cold actions/outcomes. This is known-task publication behavior,
not old-task retention, context recovery, generalization or CPC benefit.
See [full evidence and limitations](reports/promotable-minigrid-20261025.md)
and [issue179](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/179).

## Shared prediction requests

Supply the existing live `ResourceSharedRuntime` that owns the same actor:

```python
result = run_promotable_minigrid_episode(
    actor, cue="blue", episode_id="my-shared-blue",
    budget=original_readout_budget,
    check_boundary=check_original_controls_and_paid_work,
    shared_runtime=original_shared_runtime,
)
```

The exact wrapper must own that actor through an ordinary TRAIN runtime and
priority gate. Original wrapper/runtime/gate/resource-probe/native-budget/control
identities are pinned before public observations. Its real public `predict`
leases `sharing.serving` and calls the real actor. Absent the keyword, direct
serving and report shape remain unchanged. No new CLI flag/environment variable.

Original request-capacity/training admission rules apply. A ready update defers
during an active request; a later admission retains its charge. Training pause,
resource and quota denials leave valid serving available. Private weights are
never read, so a partial private update need not block the actor. Request leases
cover prediction only; actions retain the original actor lock. No hard native
preemption, physics-wide request lease, p95 guarantee or training permission.

`result["serving"]["sharing"]` contains start/latest original snapshots, native
budget progress and last readback error. Only two observations are retained;
finite controls and monotonic admission/deferral/progress are checked. Equal
replacements, control edits, counter rollback, foreign/probe owners and malformed
metadata stop the binding. Legitimate pause/private progress stays reported.
This does not authenticate arbitrary concurrent history or authorize restore.
Caller boundary still checks consent and preceding paid update evidence. Failed
requests release their lease and retain attempted/returned work and last valid
observations with uncertainty, without counter/budget resets or automatic retry.

Native evidence: one real serving-active deferral, then a charged private update
after release before first action; both known A games succeed. Actor stays2fits/
18rows, private state becomes3/27. Extra nine A rows were retained re-exposure,
not new feedback or new-task learning. [Evidence and limits](reports/shared-minigrid-serving-20261025.md).

## Fresh A4 private learning with published serving

```console
python -m scripts.run_minigrid_linear_fit_control --example-affine-live-a4 --describe
python -m scripts.run_minigrid_linear_fit_control --example-affine-live-a4 --output live-a4.json
```

This explicit mode publishes the initial two-fit A3 candidate once, then collects
fresh A4 TRAIN teacher feedback and adapts the private learner through the same
original runtime and sharing gate. The existing arrived-cue recipe retains one
earlier episode per cue. Six fits expose90rows:54arrival exposures and36replay-like
exposures from36unique original rows. Eighteen new teacher steps are separate.

The fixed published actor serves A3 blue/red after private updates4 and6. Private
readouts at2/4/6 play A3 and A4 blue/red without training or feeding outcomes to
the controller. These are known TRAIN games, with seed17/max16/rawK7 firstargmax.
One lifetime native6-update120s budget and one readout0-update0-replay150s budget
cover the workflow from before data/models. The finite live JSON limit is8MiB;
existing two-game mode stays4MiB. Output is exclusive; failed paid work must not
automatically retry. No mode mixing, source file, tuning flags or evaluation data.

Actor weights remain at2fits18rows/c0gen1. Private learning retains basea0, so it
does not gain second-publication authority. Original controls, source, receipts,
collection and prior readout records are checked before further work. A failed
fit preserves permanent charges/receipts and marks unavailable state unknown.
This cooperative sequence measures adaptation between requests; it provides no
hard preemption or latency guarantee. Retention is distinct from losing and
regaining a skill. [Implementation and evidence](reports/live-a4-serving-20261025.md).

## Start the next candidate from a published actor

Immediately after a successful publication, an ordinary unmanaged TRAIN owner
can start an independent candidate from the current actor:

```python
original_gate.pause()
previous_owner = shared_runtime._runtime
runtime = controller.start_candidate(
    shared_runtime, original_publication_receipt, candidate_version="c1",
)
original_gate.resume()
# Record new c0-version TRAIN experiences/labels on runtime, then use
# shared_runtime.train_ready(). A later publication still needs fresh guards.
```

The exact latest publication receipt is single-use for this preparation attempt.
The preceding candidate must still equal those published weights, its ledger
must be completely consumed, and history capacity must remain. Advanced private
weights, pending labels/tickets, failed owners, managed/probe/replay/action or
consent-guard configurations are refused. Pause must be explicit; serving stays
available through the existing gate while preparation runs. Failure consumes
the attempt without retiring the old owner or authorizing a retry.

The operation preserves the original native budget, elapsed/event clocks, RSS
sampler, progress and sharing gate/resource/admission counters. It retires the
previous owner and freezes its readable receipt total. New identity capacity is
the previous capacity minus consumed identities; no capacity or budget is renewed.
The new candidate has the current actor as its base and a distinct learner version.
The actor itself does not change. Starting this owner does not authorize training
data reuse or the next publication; normal permissions and guards still apply.

Its `GenerationInboxCursor` format3 explicitly records the prior lifetime-update
offset. New receipts retain cumulative numbering; old receipts are not relabeled
or copied into the new generation. This cursor can use the ordinary checkpoint
handoff while its actor base remains current. Original format1/2 fields stay
unchanged. The offset accounts for prior work, not proof of missing history.
Preserve `previous_owner` and its bounded paid evidence when a workflow requires
that history. Managed lifecycle support requires a separate authority-aware
extension; copying guard callbacks is deliberately refused.

The existing `--example-affine-live-a4` mode remains a private-learning workflow.
Use the explicit deployment command below for a second publication.
[Implementation, checks and native deployment evidence](reports/published-a4-deployment-20261025.md).

## Deploy fresh A4 through the live command

```console
python -m scripts.run_minigrid_linear_fit_control --example-affine-deploy-a4 --describe
python -m scripts.run_minigrid_linear_fit_control --example-affine-deploy-a4 --output deployed-a4.json
```

The command first fits A3 blue/red, checks the existing B sentinel guards and
publishes c0. It records four actual shared-actor A3/A4 games, pauses the original
gate, and starts c1 from the just-published actor with the original receipt. The
old two-update ledger is retired and retained; the new ledger has offset2 and
capacity2. Fresh A4 teacher collection follows that transition. Two more delayed
arrived-cue fits use the earlier A3 packet per cue, then the same B sentinel
guards decide the second publication. Four actual c1 actor games follow.

Four fits expose54rows:36arrival exposures+18explicit repeated A3 rows from36
unique original rows. Eighteen fresh teacher steps are separate. Native4-update/
120s and readout0-update0-replay/150s/512MiB lifetime budgets start before data
and models; the same clock, sharing gate and cumulative four-admission ceiling
cover both generations. Fixed seed17/max16/rawK7 firstargmax games are known TRAIN
readouts, not selection or final evaluation. B sentinel guards do not establish
old-A3 or new-A4 utility; actual game outcomes are reported separately.

`actor_games` records the ordered eight before/after games; `start.old_receipts`
holds the original1/2 prefix and `private.receipts` the new3/4 prefix. `summary`
contains measured old/new rates and acquisition/retention deltas only after a
complete capture. Recovery is unknown because this mode has no loss/regain return
phase. Global RNG witnesses are recorded before serialization. A failed start,
guard, commit, prediction or final capture preserves paid evidence and unknown
work/readback flags; future outcomes are not filled with zero successes. A lost
publication receipt can leave a changed actor, whose observed generation is
reported without fabricating a receipt. Failures must not automatically retry.

The descriptor performs no model, source, environment or budget work. The command
accepts only `--describe` or an optional exclusive `--output`, with a finite8MiB
JSON report. It rejects mode mixing, sources and tuning flags. The private and
two-game published modes retain their defaults. Qualification is for the current
checkout; a newly built installed package needs its own release check.

The native API deployment in issue183 is reused: actual A3 stayed2/2 while A4
improved0/2→2/2 after guarded publication, with the stated54row/18teacher-step cost.
The command's scripted checks verify composition and failure behavior; they do
not constitute another learning experiment. [Command evidence and limitations](reports/deployed-a4-cli-20261025.md).

A [fresh offline wheel and isolated Linux base installation](reports/deployed-a4-installed-package-20261025.md)
now qualify the actual installed `circadian-affine-game --example-affine-deploy-a4 --describe`
entry, current deployment/private/published descriptors and zero-model dispatch.
The packaged resource is available without a checkout. A base-only installation
refuses execution before data/models until the pinned optional runtime is
installed; installed native learning and Windows launcher qualification remain
separate. The earlier command-test time failure is preserved.

A [measured state-comparison repair](reports/deployed-a4-validation-performance-20261026.md)
now qualifies the current checkout's full27-case command cohort on Windows and
Linux within the original120s body limit, with identical work counts. All
preceding evidence is still checked. The original timing failure remains in its
historical record; the installed wheel above retains its original source bytes.
