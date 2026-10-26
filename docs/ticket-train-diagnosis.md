# Diagnose saved native BP ticket TRAIN fit

Run a read-only diagnosis of the saved chronological BP result:

```powershell
python -B -m scripts.diagnose_ticket_training --input artifacts/runs/native-recurring-tickets-20261023/bp-actual.json --output fresh-bp-diagnosis.json
```

Uses the existing NumPy environment; no Torch/model construction, optimizer,
fitting, probe, replay or final labels. Output must be fresh. The original input
stays sealed. This is a finished-run diagnostic, not a checkpoint restore format
or a new training/promotion/early-stop interface.

## What it checks

The fixed original BP policy/geometry/clocks, TRAIN metadata and complete bounded
episode slots must match. Four saved affine tensor sets must be exact finite
float32 values. The reference uses the existing native tanh affine formula in
NumPy float32. All66previously paid native decisions/330allowed raw scores must
agree within frozen absolute1e-6, and non-ambiguous selected IDs must agree.
It refuses mismatch; no tolerance adjustment from outcomes. This is numerical
readout agreement on the saved inputs, not native optimizer reconstruction or a
universal backend equivalence proof.

Original invoice and parcel teacher traces provide sealed/ready/staged inputs.
TRAIN/public query encodings must match, and identical features cannot carry
conflicting labels. Common sealed and staged inputs intentionally omit concept
and share open/commit targets; ready exposes concept and requires its workflow.

Every checkpoint includes all six TRAIN templates, global6-class and permitted
5-class CE/margins, target/selected action and numerical ambiguity. Historically
pending TRAIN concepts are explicitly exposure0; exposed counts remain separate.
These labels are read after the full source run and never enter an earlier
learner/controller. Stable CE uses float64 reduction; probability margins reuse
the existing core reducer. Original task metrics remain copied unchanged.

## Observed diagnosis

Saved native scores agree to1.1642e-10 maximum absolute raw error. All6teacher/
public inputs align, no contradictory labels, no ambiguous decisions/margins.

| Wake checkpoint | Global mean TRAIN CE | Invoice permitted accuracy | Parcel permitted accuracy |
|---|---:|---:|---:|
|0|1.79106|1/3|1/3 (unexposed)|
|4|1.76131|2/3|1/3 (unexposed)|
|8|1.70189|1/3|1/3|
|12|1.63482|1/3|1/3|

At4invoice opens and chooses inspect, but commit margin remains−0.003881. By12
invoice ready margin is−0.18494 and parcel ready−0.32190; staged commit is still
negative. CE improves while action fit is insufficient for a complete task. This
matches the original wrong-phase loops, and does not prove whether more training,
different optimization, interference or capacity causes the failure.

Next qualify the conventional learner's ability to fit these teacher states in
an isolated, prospectively budgeted TRAIN-only case before changing a benchmark
or adding mechanisms. No CPC/retention/generalization benefit follows.
[Protocol, costs, launch failure and all results](../artifacts/runs/bp-ticket-fit-diagnosis-20261023/RESULTS.md).

## Saved policy-comparison mode

```powershell
python -B -m scripts.diagnose_ticket_training --policy-comparison --input artifacts/runs/phase-matched-controllers-20261023/actual.json --output fresh-policy-fit.json
```

This explicit mode admits the original complete five-controller study (8MiB
input bound and exact original SHA), preserving the default512KiB/three-row clock
guard. It reports common source plus ALL five opportunity4 states, including
unchanged and duplicate states. Qualified native96updates/576rows remain separate
from new updates0/0/2/4/0/4 and their six-row exposures. Prior v2 invoice exposures
and targets are distinguished from revised v1 targets and new exposures; password
had no prior concept-specific teacher rows, while sharing open/commit features.

The actual sandbox-verified v1 TRAIN invoice/password teacher supplies labels.
Saved development traces supply only already-paid numerical score/action parity.
All54decisions/270raw scores must pass original1e-6 tolerance before any fit rows
are interpreted. Source, roles, complete phase captures, native clocks, TRAIN
packet, dtype/geometry/exact finite float32 and original source/fitted hashes are
checked. Failed prefix work is retained. The function consumes saved dictionaries;
the CLI additionally pins exact original input bytes. Neither is restore authority.

| Fixed state | New updates | Global TRAIN CE | Correct rows | Ready invoice route−inspect |
|---|---:|---:|---:|---:|
|Source|0|0.897550|5/6|−3.636213|
|No learning|0|0.897550|5/6|−3.636213|
|Periodic|2|0.883701|5/6|−3.662496|
|Drift|4|0.858183|5/6|−3.635609|
|Budget aware|0|0.897550|5/6|−3.636213|
|Fitted|4|0.858183|5/6|−3.635609|

All states select inspect on the revised invoice ready row. All other teacher
rows fit; password ready margin remains positive. Average loss improvement does
not establish task acquisition. No capacity/plasticity-loss, controller/CPC,
generalization or release conclusion follows. This is self-reviewed, not
independently accepted. Next use the EXISTING isolated fresh-task probe to compare
warm and fresh learning under a separately frozen small protocol; do not tune
the deployed benchmark or rebuild its runner.

The single read-only case used60NumPy projections/90rows/3510nominal affine terms,
1152parameter-array bytes and zero native/TRAIN/probe/final/model-initialization
calls. Maximum saved-native score error2.38419e-7; no ambiguous decisions.
[Frozen protocol, commands, failed fixtures, resource receipts and results](../artifacts/runs/policy-train-fit-20261023/RESULTS.md).

### Original reader checks

```text
scripts/diagnose_ticket_training.py       Saved scalar evidence + existing reducers
tests/test_ticket_training_diagnosis.py   Literal math/admission/alignment/faults
```

```powershell
python -m pytest -q tests/test_ticket_training_diagnosis.py
python -m ruff check scripts/diagnose_ticket_training.py tests/test_ticket_training_diagnosis.py
python -m ruff format --check scripts/diagnose_ticket_training.py tests/test_ticket_training_diagnosis.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/diagnose_ticket_training.py tests/test_ticket_training_diagnosis.py
```

Add `tests/test_policy_train_fit.py` to these scoped commands for the explicit
policy mode. The artifact meter includes both files, counts reference fixture
work and refuses native/perceptron execution. This check scope is not the full suite.

Tests do NumPy projections; respect the recorded bounded correctness scope when
reproducing. Extension should reuse the native head formula and declared TRAIN
contract, preserving all old outcomes and separating new fit budgets. Do not
generalize this reader into a serialization or orchestration framework.
