# Retained joint TRAIN learning during private serving

```powershell
python -B -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --separate-context-heads --paired-serving retained_joint --output NEW.json
```

This explicit owned-source path uses the existing source12-fit/72-row workflow,
ordinary six-packet `context_withheld_blocks_a` recipe (old,new,new,old,old,new;
delay1), serving timeline, `StreamRehearsal` and native consolidation. It leaves
the served actor unchanged, retains only complete arrived TRAIN episodes, and
reads the terminal private candidate on six existing TRAIN sandbox tasks after
all training and serving workers close. There is no promotion, ACK, probe/final
access, controller fitting, seed selection or new comparison framework. Earlier
negative comparisons and #287 FAILED/SPENT remain unchanged.

The optional `ResourceSharedRuntime(wake_rehearsal=..., rehearsal_packets=...)`
path requires a fresh originally bound ordinary dynamic-joint helper, exact
packet metadata, a separate finite replay budget and explicit replay envelope.
Before an actual wake, the existing gate reserves TWO attempts for at most one
wake plus one joint solve. Both reservations remain paid even when replay skips
or fails. `ExitStack` holds that original gate lease until runtime training has
released its own lock and the helper completes. No deferred replay queue or new
capture history is added. Initial no-label polls spend no reservation; paused,
serving, unavailable-resource and quota deferrals perform no fit or retention.
An admitted two-attempt bundle is not preempted between its two calls; serving
can still read its separate original actor. Gate limits count reserved attempts,
not returned fits. Private serving therefore declares12 reservation slots for
six wakes, while physical fit/row charges remain independently reported.

The actual native update origin must match the next original paid packet's
features, targets, sample ID and TRAIN role, with explicit training AND replay
consent recorded at ingress. Default `_record_tick` still records training-only
consent. Candidate data, helper identity, original packet metadata, paid joint
events and the exact sharing-counter increments remain checked; counter changes
are not broadly ignored. Original wake/replay budgets and clocks are not reset.
Default fixed replay/single-wake/ACK paths retain their previous contracts.

## Frozen #300 engineering validation

This is one new serving integration engineering case, not a new scientific
comparison or retry of #287. Pure refusal/deferral gates precede native fixtures.
Per-platform focused fixtures prospectively cost18 numerical LS solves/114rows:
5/33 for a three-packet reserved stream,2/9 for a joint-call refusal before its
numerical solver (9 replay rows still paid),11/72 for ordinary serving on an
explicit literal fixture source. Both platforms total36/228. Source preparation
in the ONE later real workflow costs12/72; child6 wakes/27rows plus up to5 joint
fits/45rows, total workflow23/144. All allowance<=60solves/400rows across tests
and workflow, preserving failures. No automatic workflow retry. At most3 complete
focused cohorts/platform,120seconds and512MiB/process. Native case<=60seconds,
512MiB,8MiB stage/output; original source lifetime remains30seconds from birth.

Native exactly once, after focused correctness/static checks and finite
source/protocol assessment. Fixed ordinary recipe as above, recent policy
every1/maximum12rows with original30-transition/65536-byte retention; no tuning
to outcomes. Native caps:96teacher traces,256episodes,2048dispatches,4096typed
predictions and8192affine/probability/route rows each. No Torch/GPU, optimizer,
latent computation, model random draws, seed changes, download or installation.
Count all source, diagnostics, replay and task readout work; preserve all failures
and inconclusive results. Endpoint tasks are TRAIN known templates, cannot select
training/promotion/probes, and do not establish unseen-task/scientific advantage.
Source/actor unchanged, complete receipts and exact paid work are correctness
criteria; successful task endpoints are an observation, not a condition used to
choose or repeat the run. Read-only reviewer limited to4 bounded source/result
reads, no imports/tests/models/network/writes; lead owns all edits and budgets.

Prospective engineering amendment, issue comment6509971472: initial two51PASS
cohorts consumed36LS228rows, but preceded added complete candidate/receipt pins.
They do not qualify those later guards. Before further numerical work, revise
agent-created TOTAL allowance60/400 to96LS640rows to admit exactly one final
combined cohort/platform (forecast36/228 more) plus the original ONE native
workflow23/144. Forecast total95/600; preserve all original receipts/limits.
Per-process/time/task/storage/native-one-attempt/source30s limits remain unchanged.
This repairs a demonstrated correctness qualification gap, not a scientific retry.
Detached receipt values and raw owner identities now refuse in-place corruption
and replacement; one current candidate and original inbox stay pinned, with no
model history chain. Final full case tests receipt corruption after its already
paid stream, without extra fits.
