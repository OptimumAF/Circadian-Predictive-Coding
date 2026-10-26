# Real BP cue-relative adaptation result

One frozen, independently reviewed local case executed the accepted public
joint TRAIN/F/old-A/A-return route. No production source changed this session.
The original defaults and all prior negative results remain intact.

## Question and protocol

With the original BP H8/seed17/learning-rate0.03/momentum0.9/unweighted CE,
train the original six TRAIN traces for128 joint54-row delayed1 updates.
Score every original trace at0/32/64/96/128. From checkpoints0/128, separately
owned disposable learners perform F16, read old-A fit, then A16 return; all F/A
curves use0/4/8/16. Source, actor, benchmark checkpoints, input packets and
original independent budget/clock bindings remain unchanged. Final D is absent.

The fixed F cohort has separate probe_train/probe_test roles and9-of-9 canonical
TRAIN-unseen inputs per packet. It is a familiar deterministic diagnostic,
not an independent generalization or final-evaluation test. No probe_test label
updates, task/seed/checkpoint selection, replay, promotion or live policy games.

[Frozen protocol](../artifacts/runs/cue-relative-bp-adaptation-20261018/PROTOCOL.md)
and [exact commands/evidence](../artifacts/runs/cue-relative-bp-adaptation-20261018/RESULTS.md).
The existing [CLI and API](cue-relative-probe.md) implement this route; no new
runner, dependency, optimizer, environment setting or schema was introduced.

## Results

TRAIN128 reproduced48/54rows and0/6exact traces, CE0.39177486685667096.
All108 probability rows at0/128 match the earlier constructive result exactly
(maximum difference0.0, frozen tolerance1e-7). Recollected TRAIN and F sources,
layouts, seeds, roles, actions and doses bind the preserved original control.

| Source checkpoint | Encoding | F CE gain after16 | Old-A CE increase during F | A-return CE gain |
| --- | --- | ---: | ---: | ---: |
| 0 | Cue-relative | 0.62135530 | -0.58934784 | 0.26069871 |
| 128 | Cue-relative | 0.54109945 | 0.70592244 | 0.43018053 |
| 128 | Preserved invariant control | 0.72393438 | 0.37882906 | 0.10051323 |

Positive gain means lower CE; positive old-A increase means worse fit. All fixed
points and both cues are preserved in summary.json. Warm cue-relative F CE moves
1.23847275→0.69737331, accuracy12/18→14/18, with zero exact traces. Old-A CE moves
0.39109204→1.09701449 during F, then0.66683395 after return; accuracy16/18→13/18→14/18.
Return improves fit but does not restore its pre-F level. Old A was unacquired
before F at both source checkpoints, so acquired-skill retention/recovery is
inconclusive. F itself is not fully acquired under the exact-trace criterion.

The preserved invariant control ends F at CE0.57678125 with the same14/18accuracy
and zero exact traces. Its old-A CE is0.39374378→0.77257284→0.67205961. The new
encoding has a smaller warm F gain and larger old-A loss; its larger return gain
starts from worse fit. This one case supplies no adaptation advantage or reason
to replace the original default. It establishes no circadian benefit, capacity
limit, robust forgetting, live success or generalization.

## Work and verification

Operational status COMPLETE:128 benchmark updates plus four independent16update
F/return phases total192native updates7488rows,94classifier calls846rows,
10physical teacher episodes90steps. No replay/final/live policy work. Source,
actor, checkpoint and budget bindings pass; unknown-work flags are false.

Native parent27.4448436seconds/actual27.1768056seconds, sampled RSS433999872bytes.
Sole stdlib saved readback0.5158033parent/0.248056actual seconds, RSS23097344bytes.
The separate250second science allowance closes at27.9606469spent; no retry or
additional model. Input costs include16422912benchmark-array bytes,128304template
bytes and470448derived F/monitor/return-array bytes per probe, plus original
cohorts, complete histories, snapshots, transient copies and Python/native state.
Returned projection payloads and all work/timing dictionaries are retained;
wall/RSS includes costs beyond numerical row/call totals.

Independent readback checks270 TRAIN probability rows directly from vectors and
576probe/return decision-loss rows,94scores, all curves/gains, confusion/class
counts, source bindings, copied receipts and return-boundary equality. F/return
raw probability vectors were not persisted by the existing report: their native
argmax and negative-log-probability cannot independently be rederived from these
artifacts. Their saved classes/losses and aggregate arithmetic qualify; accepted
implementation checks supply the runtime isolation/math evidence. These scopes
must remain distinct.

The accepted254scoped checks and static gates were reused because production
source was unchanged. New readback AST, CLI preview, source/protocol and actual
independent reviews passed. Full suite/clean-clone/release, baseline/model reruns,
extra doses/seeds/methods and full final evaluation were skipped. Exact commands,
failures and closure are recorded in RESULTS/CLOSURE. All prior grants remain
closed; all old failed/INCOMPLETE/SPENT results are preserved.

## Reassessed next implementation

Keep the original encoding as default and stop extending this representation
branch on this evidence. Next advance the R2.4 public MiniGrid rehearsal control:
inspect the existing `StreamRehearsal`/`AttributionStream` and Torch rehearsal
adapter's current two-feature/binary fixture, then add explicit raw296/K7 old-A
TRAIN replay composition with complete causal role/schema admission and exact
memory, replay-exposure and computation charges. Verify one bounded isolated
synthetic case before scientific execution. Preserve matched method recipes and
freeze an explicit extra-exposure control before comparing replay outcomes.
No new orchestration framework, fit-dose retry or CPC sweep is needed. Until old
A is acquired, any result describes old-fit preservation rather than retention
of an acquired skill. Full R2.1/R2.4/R4.2, heldout/RL/release and repository goal
remain open.
