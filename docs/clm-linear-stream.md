# Warm linear adaptation in the temporal stream

The existing temporal command supports `--catalog-training --train-linear-update`.
It runs the original twelve delayed-label TRAIN packets through the original
`ExperienceInbox`, clocks, budgets, receipts and post-training scorer, using the
[warm wide categorical correction](clm-linear-updates.md). The native head route
remains the default.

```powershell
python -m scripts.run_clm_temporal_sequence --model-dir <local-encoder> --head <local-head-checkpoint> --output linear-sequence.json --catalog-training --train-linear-update --train-catalog-diagnostics
```

An experiment needs its own prospective allowance before running this command.
The ordinary loader still verifies/loads the original encoder and released heads;
their setup cost is paid even though this learner never forwards or trains those
heads. There are no downloads. A feature-based study can inject already-bound
features through `run_linear_catalog_stream`; retained features need their own
explicit access/protocol allowance. This option does not authorize reusing a
spent experiment, converting probe data into TRAIN or reloading a model.

## Model and class identity

Why this: the native temporal stream changes active tools. Reusing a three-column
classifier by position would incorrectly turn the old `retire` column into a
different tool. The linear model instead owns four fixed columns in the existing
union catalog order: `defer, inspect, retire, route`. The active v1 order is
`inspect, retire, route`; v2 is `defer, inspect, route`. Each TRAIN target maps
by its declared tool ID, with matching original action features, into the union.
Inactive coefficients remain distinct from active ones.

The first batch uses the existing cold one-hot least-squares fit. The next eleven
apply `Wnext = W + lstsq([X,1], Y-[X,1]@W, rcond=1e-12)`. Targets are one-hot over
all four union classes. This is an explicitly different squared-error objective
from the native active-three-class cross entropy, not a matched optimizer or
compute comparison. The intercept participates in coefficient distance. There
is no regularizer, rate tuning, randomness, replay store or optimizer state.

Readout slices the current coefficient columns into the declared active order
**before** stable softmax. An inactive extreme score therefore cannot win or
underflow all active scores during renormalization. No action ID is inferred
from embedding similarity. Up to eight choice batches are explicitly bound at
construction; held-out labels never enter that binding or training.

Before any fit, the model predicts uniform probabilities over active tools.
The reported unchanged control is explicitly `untrained_uniform_union_linear`,
not the released native head. Its kind and exact values are guarded in the
stream report prefix and revalidated before scoring. This is a reference control,
not a trained matched baseline. The original signed metrics/directions and v2
truth on return remain unchanged; the stream does not restore obsolete v1 labels.

## Work, failures and boundaries

An ordinary complete stream has12solves/12commits/36TRAIN-row exposures, five
readouts/20query rows and zero encoder/head forwards or optimizer steps **inside
the learner**. Optional TRAIN diagnostics add three readouts/nine query rows.
The separate frozen feature encoder still costs one forward; loaded head setup,
copying, feature validation and scoring also cost work. Counts are not FLOPs.
Each receipt reports mean last-batch union squared error with its distinct
objective ID. These are actual solve receipts, never simulated SGD steps.

The learner retains one current immutable coefficient control and at most eight
declared choice bindings. No old fitted-model chain or training-row replay
history is retained. Last-batch rank/error/fit_rows are not lifetime counters;
`learner_work` owns cumulative attempts, returns, rows, readouts and commits.
The finite final `linear_last_operation` report retains the most recent solve
prefix and logical array costs, including failed or unknown work.

TRAIN identity, role, ordered active IDs, labels and original action features
are validated before work. Source coefficient/model/identity/policy/action owners
and values and input evidence are guarded around return. Lost/unadmitted solves
keep attempted/returned work and stop the learner without a retry. The app keeps
paid receipts even if a later save or report guard fails. A new object/output
does not renew an external allowance. Guards assume trusted local Python code;
they are not hostile-code containment.

This initial option requires one original `none`-recipe catalog stream. Native
reserved probes, state-only updates, memory/replay candidates, typed serving and
catalog-support comparisons are refused before the loader. Disposable probe
updates remain separate APIs; the linear benchmark route never accepts them.
Final data remain outside training, scoring, replay, promotion and tuning.

## Structure and extension

```text
src/core/clm_linear_catalog.py                policy, objective, reference, work schema
src/adapters/clm_linear_catalog_learner.py     fixed-union TRAIN learner and composition
src/app/clm_temporal_sequence.py              original inbox/clock/receipt runner
scripts/run_clm_temporal_sequence.py          option admission and existing scoring
tests/test_clm_linear_catalog_stream.py       sequence/masking/isolation/failure checks
tests/test_clm_linear_catalog_cli.py          actual literal command and invalid options
```

No new dependencies or environment variables. Extend through these existing
ports and keep budgets, role separation, class identity and solve work explicit.
Any replay extension must count repeated rows and retain original consent;
any probe extension must keep a disposable owner and its separate work allowance.

Focused checks and scoped tooling (observe issue limits before execution):

```powershell
python -m pytest -q tests/test_clm_linear_catalog_stream.py tests/test_clm_linear_catalog_cli.py
python -m ruff check src/core/clm_linear_catalog.py src/adapters/clm_linear_catalog_learner.py src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py
python -m ruff format --check src/core/clm_linear_catalog.py src/adapters/clm_linear_catalog_learner.py src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py
python -m mypy --follow-imports=silent src/core/clm_linear_catalog.py src/adapters/clm_linear_catalog_learner.py src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py
```

The [software report](reports/clm-linear-stream-20261025.md) records actual
commands, failures, passing evidence and resource costs. Literal execution proves
integration; native acquisition, retention, recovery, comparative benefit and
repository/release readiness need separate evidence.

The subsequent [cached-feature case](reports/clm-linear-stream-cached-20261025.md)
fitted TRAIN3/3 at each checkpoint and scored development3/4 after A,4/4 after B
and4/4 on return. Calibration and recovery limitations remain explicit. The
sole attempt is spent; this result grants no rerun or retained-bank allowance.
