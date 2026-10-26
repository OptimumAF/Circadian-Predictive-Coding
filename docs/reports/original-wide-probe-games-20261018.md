# Original-source learning and return games: incomplete study

## What ran

One frozen exploratory case each for BP, PC, neutral CPC and active CPC used the
existing H32 joint128/F16/A16 learnability CLI. Data, seed17, native objectives,
update counts, metrics and comparison directions stayed fixed. The same explicit
60s endpoint/240s whole-case envelope applied to all methods. No retries, tuning,
seed selection, final evaluation, replay or live policy learning occurred.

All four cases stopped at their declared limits. **The comparative study is
unqualified.** The strict saved-results gate fails and publishes no contrasts.
Prior synthetic integration success did not establish original-source completion.

| Method | Body seconds | Paid updates / rows | Complete games | Stop |
| --- | ---: | ---: | ---: | --- |
| BP | 240.0665 | 192 / 7488 | 14 | whole-case limit during source128 A-return games |
| PC | 191.9917 | 176 / 7344 | 11 | source128 F endpoint at60.224s |
| Neutral CPC | 88.0010 | 144 / 7056 | 3 | source0 F endpoint at60.227s |
| Active CPC | 87.1853 | 144 / 7056 | 3 | source0 F endpoint at60.101s |

Shorter durations mean earlier refusal here, not better performance. Unrun phases,
failed games and missing success/return remain unknown rather than zero. The
completed-game counts cannot be used as comparable success-rate denominators.

## What the recorded prefix shows

BP's complete source128 post-F16 endpoint has these actual target-free games:

| Task/cue | Success | Return | Steps |
| --- | --- | ---: | ---: |
| A blue | false | 0 | 16 |
| A red | false | 0 | 16 |
| F blue | false | -1 | 7 |
| F red | true | 0.49375 | 9 |

Thus one new cue works, while the other cue makes a wrong-key failure. F mean
success is0.5 but mean return is−0.253125 at this endpoint; this does not establish
an unconditional winner. Both A games after A16 also finish with zero success and
return; the remaining F readout is incomplete. No full recovery/retention claim
is justified. The accepted earlier known-A red success is a separate descriptive
reference, not a new within-run before-F game baseline.

Every persisted original TRAIN source, all512 benchmark update receipts, all20
TRAIN readouts and their1080 probability rows reproduce accepted controls.
All72 returned F/A monitor checkpoints match the earlier no-game study's recorded
prefixes exactly. Those earlier row-fit findings remain unchanged: BP can improve
new-F fit with old-A loss and incomplete recovery; PC/neutral match and active CPC
showed no advantage. This run adds censored gameplay evidence, not a qualified
new comparative result or a causal explanation of plasticity loss.

Saved-only verification checks521 persisted K7 game distributions, native first
argmax, raw initial148+current148 history and corresponding observation/action/
next-observation chains. There is one additional paid neutral CPC prediction
whose result was not persisted before refusal; it is charged and unknown.

## Isolation and costs

Runtime source/actor checks pass. Reached source-state checks and live endpoint
budget bindings pass. BP's final return-binding flag staysfalse/unproven: the
parent guard expired before its final validation flag could be set. It is not
relabelled preserved. Original fixed budgets and permanent charges remain; fresh
independent cases did not resume failed learners. Full qualification stays refused.
Ambient global-RNG preservation is reused from accepted implementation checks;
this CLI study does not add a separate four-case ambient-RNG assertion.

Actual cumulative work, including incomplete attempts:

| Cost | Observed |
| --- | ---: |
| Native updates / training rows | 656 / 28944 |
| BP SGD steps | 192 |
| Prediction calls / rows | 786 / 2898 |
| Fixed score calls / rows | 264 / 2376 |
| Native game prediction returns | 522 |
| Persisted game predictions / game physics steps | 521 / 521 |
| Teacher resets / steps | 40 / 360 |
| Learner resets | 35 |
| All resets / physics steps | 75 / 881 |
| Maximum sampled external RSS | 468631552bytes |

Copies, optimizer/adaptive/RNG state, projections, evidence validation, Python/
backend/report memory and elapsed time are additional costs. Parameter bytes are
not total memory; equal caps and update counts are not equal computation.
No final-test data, replay or live game updates were used.

[Exact frozen protocol, commands, costs and failures](../../artifacts/runs/original-wide-probe-games-study-20261018/RESULTS.md),
[independent review](../../artifacts/runs/original-wide-probe-games-study-20261018/REVIEW.md),
[full gate: unqualified](../../artifacts/runs/original-wide-probe-games-study-20261018/analysis.json),
[verified partial observations](../../artifacts/runs/original-wide-probe-games-study-20261018/prefix-analysis.json).

## Next useful implementation

Keep original-source full-game qualification and all scientific parent requirements
open. Do not automatically increase clocks or repeat this quartet. The existing
MiniGrid inspector projection admits fixed-A128 reports, so these produced
learnability reports lack its readable phase/cue/playback view. Extend that existing
projection and renderer to show F/A curves, all game outcomes, unrun/unknown states
and censored costs using saved records only. Reuse its public-observation decoder
and playback; add no learner, physics, selection or orchestration framework.
