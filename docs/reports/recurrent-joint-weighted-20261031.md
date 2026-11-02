# Class-balanced joint recurrent TRAIN reference

[#388 / R4.2-RNN.JW](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/388).
Implemented and focused execution passed; native PRE accepted. The ONE native
case completed within caps with a negative comparison; scoped independent POST accepted.
Full repository/scientific/release goal remains open.

## Behavior and rationale

The existing command's explicit `--class-balanced-joint` selects the existing
TorchTrainWeightedLearner and native per-episode class-balanced operation, with
the original six arrived TRAIN sources/joint128 order/1152 rows/raw148/H8/K7/
seed17/lr.03/momentum.9. The analytical current-episode weights are fixed:
n/(present_classes*class_count), absent1. Original TRAIN has five forward actions
and four critical classes occurring once each. No coefficient search, new loss,
optimizer, replay algorithm, runner, evaluator counter relaxation or dependency.
Weighted loss uses sum(target_weight*CE)/sum(target_weight).

Unweighted #384 joint52/54 TRAIN actions and5/6 goals (A-blue7/9/failure), #386's
retention tradeoff, #387's cold failure and stronger simple affine/other controls
are reused. Cold also failing atF48 prevents blaming warm failure alone on lost
plasticity. This TRAIN-only development reference addresses the remaining fit/
control gap; no reserved probe/final labels select weights or tuning. Existing
weighted TRAIN-role refusal remains. Prior sequential weighted #140128updates
FAILED-SPENT/readout expired64/unknown96-128 stay unchanged. A joint upfront
schedule is a distinct frozen question, not refit/recovery/resumption of140.

Only `scripts/recurrent_joint_example.py` changes production. The new small
native-view helper validates the exact existing weighted owner and returns its
original native learner for the existing readonly kernel. Caller pins source,
actor, candidate wrappers and their native identities; source/actor remain cold.
Existing budgets/births/permanent failed work/receipts/RNG/input/protocol and
preceding-evidence guards are preserved. The candidate receives128 weighted
updates,64/576rows per cue. Existing evaluator reads all6 TRAIN teacher scores
at0/128 and6 complete native games at128, with unchanged CE/accuracy/success/
return directions. No probes/replay/promotion/final data are used.

Independent source review found a real callback-ordering gap: a later actor
snapshot could replace an earlier checked native leaf with an equal copy. Final
direct alias reads now check wrapper/native references after all value/receipt/
RNG callbacks, without invoking another snapshot. Three late-callback regression
cases cover source leaf, candidate leaf and complete candidate replacement.
Existing weighted wrapper/native math are unchanged. No architecture/folder
change; existing script→stream/wrapper→native/kernel composition remains.
See [usage](../recurrent-joint-example.md); extend named modes and focused owner/
role checks, not arbitrary counter admission or reserved-role conversions.

## Executed checks, attempts and resource scope

Stage `artifacts/runs/recurrent-joint-weighted-20261031/`, physicalD:
TEMP/TMP/TMPDIR on D:. Existing counted fixture runner blocks genuine native
imports; no Torch/weighted math/model/solver/physics/source payload work.

    ./artifacts/runs/recurrent-joint-weighted-20261031/run-step.ps1 -Step checks -Attempt core -Suite core
    ./artifacts/runs/recurrent-joint-weighted-20261031/run-step.ps1 -Step checks -Attempt final -Suite final
    ./artifacts/runs/recurrent-joint-weighted-20261031/run-step.ps1 -Step checks -Attempt metadata -Suite metadata

- First12cases11PASS/1FAIL,46.01511159999063s129662976B;
  host46.1640847s,PID79108,exit1/no expiry. Full128weighted update/6game composition
  completed before a fixture weight assertion failed. Old literal episodes had
  eight0 actions/one forward, not the actual task's5forward/fourcritical counts.
  Corrected only the NEW literal TRAIN fixture; original real data/objective and
  acceptance requirements unchanged. The other role/failure/owner tests passed.
- Final70PASS/0FAIL/0SKIP,137.94153549999464s145444864B;
  host138.0950633s,PID82408,exit0/no expiry. Includes all three late-callback
  regression cases plus existing ordinary/cold/paired/source2 behavior.
- Post-self-review found the new mode inherited the base native-description
  string saying meanCE. Fixed that one literal to class-weightedCE and added
  a metadata assertion. Final third run2PASS/0SKIP,15.257773200020893s99749888B;
  host15.4038565s,PID78496,exit0, covering that descriptor and the old full F16/A16
  stream/probe/game composition. No model/schedule/budget change after70 gate.

Cumulative84 executions83PASS/1 preserved fixture FAIL,199.21442030000617s,
all three engineering slots spent. Scripted work2480constructors2381port calls/
21429row invocations380teacher3420teacher rows275live. The weighted fake invokes
its ordinary fake method internally; both invocations are conservatively counted,
not claimed to be2381 actual SGD updates. Real new full composition has128
weighted logical updates. All scripted caps held; genuine numerical work0.

```powershell
$stage='artifacts/runs/recurrent-joint-weighted-20261031'
$files=@('scripts/recurrent_joint_example.py','tests/test_recurrent_joint_weighted.py')
./.venv/Scripts/python.exe -B -m ruff format --check @files
./.venv/Scripts/python.exe -B -m ruff check --cache-dir "$stage/ruff-cache" @files
./.venv/Scripts/python.exe -B -m mypy --follow-imports=silent --cache-dir "$stage/mypy-cache" @files
```

Initial statics3.3783466s: format0, two fixture alias lint errors and four fixture
typing errors. Second .7370058s: format/lint0, four new terminal-check protocol
attribute typing errors plus one fixture annotation. Corrected explicit exact
wrapper type guards and fixture annotations; statics0/.7308523s. Final after
descriptor correction format/lint/types0/.8135492s. Observer AST/compile and
PowerShell parse pass with no model execution. Two no-op documentation patches
missed their anchors; simplified direct append/literal replacement. One tool call aborted before dispatch,
then the same test handle was polled to terminal; no process restart/retry.

Full suite, Linux/CI, clean clone, installed wheel, visual/device QA, GPU,
deployment and additional seeds/sweeps are skipped; no readiness claim.

## Independent review and frozen native procedure

One finite peer owns only stage REVIEW.md/reviewer-timings.json, <=4small reads/
20s/4500tokens each, no imports/tests/models/network/numerical replication.
Reads1/2 .3305926s inspected the caller delta, found/reviewed the callback repair,
and reviewed targeted tests. Read3 0ed95a/.1953897s/3594tokens inspected final
terminal gates/statics, positive weights/order and lost-return assertions,
frozen procedure and new weighted observer counters. Scoped PRE GO applies only
to this ONE case. At PRE three reads used .5259823s; one was reserved for direct scalar POST.
Original #387's incomplete
direct16-game review remains explicitly unverified, not silently upgraded.

Final read4 71f0e6/.2143801s/2472tokens directly checked all six native scalar
rows, readout/source/actor/RNG proofs, budgets/clock births, work and host/costs.
Scoped actual POST ACCEPT, four reads/.7403624s, no truncation/raw-array dump,
numerical replication or broader acceptance claim. Teacher51/54 was reviewed as
secondary reported evidence rather than independently recomputed. Finite
exception ENDED; no allowance carries forward.

[NATIVE_PROTOCOL.md](../../artifacts/runs/recurrent-joint-weighted-20261031/NATIVE_PROTOCOL.md)
was frozen after software gates. ONEjointweighted128/1152rows,128 analytical
weight derivations1152rows,12teacher108rows,6sessions/max96learner steps,
154constructors462privatedraws200816values434optimizers. Ordinary native/public
train0 strictly refused. Original body240/TRAIN180/continuousreadout200s,
observer270s768MiB/host285s, output2MiB/stage64MiB/cache32MiB. No retry, warm
baseline rerun, dose/seed/metric selection, future labels or original140renewal.

## Actual result: class balancing did not improve this joint case

ONE native case, PID32116, exit0/no host expiry. Application45.70885869997437s,
observer47.7351829999825s, owned host48.1757964s, peak sampledRSS384819200B.
All attempted operations returned, caps held, all6 distinct environment owners
closed. Source/actor and all three RNG preservation flags true; both readouts
state_unchanged=true, all six game state proofs true. Candidate128 was evaluated
without promotion or further updates. No saved weights or original140 recovery.

| TRAIN task | Weighted teacher correct | Weighted native result | Preserved unweighted native result |
|---|---:|---|---|
| A-blue | 8/9 | fails16steps,0return | fails16steps,0return |
| A-red | 7/9 | fails16steps,0return | succeeds9steps,.49375 |
| B-blue | 9/9 | succeeds9steps,.49375 | succeeds9steps,.49375 |
| B-red | 9/9 | succeeds9steps,.49375 | succeeds9steps,.49375 |
| C-blue | 9/9 | succeeds9steps,.49375 | succeeds9steps,.49375 |
| C-red | 9/9 | succeeds9steps,.49375 | succeeds9steps,.49375 |

Weighted total51/54 teacher decisions,4/6 exact sequences and4/6 native goals;
unweighted #38452/54,5/6 goals. Weighted A-blue fit improves7→8/9 but still fails,
while A-red drops9→7/9 and loses its native success. Four B/C successes remain.
Weighted CE Ablue.398164/Ared.579065/Bblue.292965/Bred.500842/Cblue.330394/
Cred.494598; original complete values/probabilities remain in actual.json.
The fixed analytical objective is not an improvement in this known task/seed
case. Do not tune weights, seeds, dose or metrics to reverse this result.
Unweighted remains the better recurrent reference; simpler affine remains
stronger. No CPC, reliable acquisition, retention/recovery, held-out or release
claim follows from this comparison. No cold native game baseline was run.

Actual costs:128public weighted calls,128analytical derivations/1152TRAIN rows,
128native weighted updates/1152rows/SGD/backward/CE; ordinary public/native train0.
154staged constructors462private draws200816values434optimizers;12teacher108rows,
6sessions68learned predictions,208total forwards1328rows.6environments/resets/
seeds/closes and68physics; live teacher actions0.40538snapshot ports13247clones/
3464562scalars. Global random draws/autograd.grad/solver/LAPACK0. Guard/weight/
copy overhead is paid; equal update/row counts do not match all work/time.

Original TRAIN birth253302.8214203→253317.501432,128updates/0replay;
continuous readout253303.7372957→253346.5506003,42.8133046s of200,0updates/0replay.
No owner, clock, birth, limit, receipt or permanent charge is renewed. Bound
source/protocol hashes match PRE. Actual255873B<2MiB, stage17955391B<64MiB/
cache17633955B<32MiB before final text/POST. outcome.json is a stdlib-only scalar
readback, not another model experiment or resume authority.

## Critical path and precise next action

Stop adding recurrent dose/weight/replay variants without a new concrete defect.
The useful existing joint affine reference already solves F2 plus all oldABC6
in #373's fixed native case. Move toward R4.2's missing held-out evidence on that
stronger frozen control rather than another acquisition-tuning increment.
Targeted source inspection found an existing readonly held-out D kernel, but
its facade accepts only the historical ordinary10/252 endpoint. Current matched
joint affine private endpoint is8/180/two36-row heads, from live source6/108 and
two private fits/72row exposures. Existing first-party private game plan only
reads F/oldABC, not held-out D. These are different paid operations; do not
relax counters, reconstruct JSON weights, relabel final data, or pretend the
historical ordinary result qualifies this private endpoint.

Next [#389 / R4.2-AF.JH](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/389)
is PLANNED, all criteria unchecked. Exact next implementation action: reconcile existing affine workflow's owned
joint_ABC_F endpoint hook and readonly held-out facade, then add the smallest
explicit terminal-only held-out-D readout for actual8/180/two36-row heads. Final
observations/results must be unavailable to fitting, promotion, probe/arm choice,
early stopping or tuning. Preserve normal defaults and historical10/252 route;
no new learner/math/framework. Prefer executing only the fixed joint arm if
the existing factory supports it; do not rerun F-only as a control merely for
logs. Producing a new live endpoint may require fresh prefix fits, all charged
prospectively; saved reports are not model/resume authority. First modelZERO
freeze and boundary tests, then ONE separately frozen local final readout.
No new fixture/model allowance here. Generalization is unproven and a negative
held-out result remains valid; full goal/other deferred requirements stay open.
Existing #126's ordinary10/252 D result (bluefails/redsuccess) was already
accepted and is reused. Its different G/A history cannot qualify this joint
ABC/F endpoint. D is globally seen despite being excluded from this model's
fit; the proposed observation is not a globally untouched statistical test.
The stronger-control choice came from known TRAIN success before consulting
that prior D result. Do not tune or select a later policy from final outcomes.

Final saved-only lead check first hit Windows cp1252 while reading a historical
log; it failed before mutation. Explicit UTF8 and a byte-only local clock-typo
correction preserved unrelated log bytes. Source bindings, doc links, scoped
whitespace and peer END pass; no numerical rerun. Final stage45files17967531B
before FINAL-LEAD receipt, cache17633955B; HEADd9119e8 unchanged/no commit/push.
