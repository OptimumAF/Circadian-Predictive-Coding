# Cold raw-feature learnability control

The existing temporal CLM command can replace its six-update native reserved
probes with a small detached linear diagnostic:

```powershell
python -m scripts.run_clm_temporal_sequence --model-dir C:\prepared\clm --head C:\prepared\head.pt --output fresh-linear-probes.json --catalog-training --reserved-task-probes --probe-linear-fit-control
```

Use prepared offline assets and a separately frozen experiment allowance before
native execution. A new output/provider does not renew an experiment budget.
This software option alone establishes no native fresh-task learning benefit.

The original benchmark still receives twelve catalog updates with its original
delayed TRAIN receipts, clocks and labels. At each existing checkpoint0/4/8/12,
the control fits the two declared `probe_train` markers, tavrix/vemlot, once.
It uses the original fixed minimum-norm one-hot least-squares/intercept/softmax
arithmetic, then reads three groups: those two training markers, two suffix-test
markers and four old-task observations. No test/old-task labels enter the fit.
There are four solves/eight fitting rows/twelve readouts/thirty-two readout rows,
zero probe optimizer/backward/head updates, and no fabricated native receipts.
The original sixteen-text frozen feature union is reused; its encoder work is
still paid. Source snapshots, copies, validation and serialization also cost work.

Why this: a cold feature control helps distinguish raw-feature separability from
difficulties in optimizing a warm native head. It does **not** inherit the source
head's learned weights. An old-task score measures compatibility of a newly
fitted classifier, not forgetting or retention of that native learner. There is
no unfitted-before prediction or before/after plasticity claim. Since the encoder
is fixed, these control fits do not measure changes in native-head plasticity.
It is not a matched-compute comparison, calibration result or CPC advantage.

The option is default off. Existing native probes remain unchanged. It requires
catalog training, reserved probes and the original `none` recipe. Rehearsal,
extra-marker, native probe TRAIN readouts/state-only mode, state-only benchmark,
memory, paired-replay, typed-serving and comparison modes cannot be combined.
Invalid combinations fail before the loader. Existing optional benchmark TRAIN
diagnostics are separate from fitting the probe. Final evaluation remains outside
training, scoring, selection, promotion and control decisions.

`FrozenProbeCatalogBatch` remains `probe_train`; read-only `outer_selection` is
explicitly admitted for the original old-task port. Neither that permission nor
fitting grants deployment: `CLMLinearDecisionPredictor` refuses probe fits before
accessing its encoder/actions. There is no probe export/promotion API here.

The fixed hook retains at most four completed records and one current fit. It
checks original feature values/storage/roles/catalog/targets, native source
owners/weights/work/RNG and completed report owners/values around callbacks.
It validates each paid operation before admitting the next, retains uncertain
fit/readout work and trusted prefixes on failure, and refuses retries after
uncertain paid work. The local fitted model is checked for value and owner changes
between calls. These are trusted local Python guards, not hostile-code containment.

Scoring uses the existing metrics/directions only after the original benchmark
and all four control records pass admission. Every group/endpoint is retained;
no score selects data, thresholds, a fit or a checkpoint. A bad late record is
rejected before any earlier group is scored.

Structure and boundaries:

```text
src/adapters/clm_linear_checkpoint_probes.py  bounded control/hook and cached validation
src/adapters/clm_checkpoint_probes.py         existing plan/source/score dispatch
src/adapters/clm_linear_fit_control.py        existing role-bound fit/readout arithmetic
scripts/run_clm_temporal_sequence.py          existing CLI and stream composition
tests/test_clm_linear_checkpoint_probes.py    literal behavior, isolation and paid failures
```

No dependencies, environment variables or new stream framework. Extend through
these existing ports; preserve explicit initialization/objective/cost semantics
and evaluation isolation. The [software report](reports/clm-linear-probes-20261025.md)
records executed checks, costs, failures, skipped checks and remaining claims.

The subsequent [native baseline diagnostic](reports/clm-linear-marker-native-20261025.md)
completed one cold fit: marker TRAIN2/2, fixed suffix2/2, old-task compatibility2/4.
This is scoped separability evidence; warm retention and context recovery remain
unestablished. Its one-use native and result-reader allowances are spent.

The subsequent [sequential correction port](clm-linear-updates.md) adds an explicit
warm minimum-change update. It preserves the fitting role and counts its residual
solve separately. The original probe hook and cold-fit option remain unchanged.
