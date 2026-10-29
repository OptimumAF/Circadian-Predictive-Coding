# Diagnose saved native BP ticket TRAIN fit

## Compare already-paid live clean/noisy decisions

Use the separate saved-score route for a `native_catalog_live_noise_pair_v1`
report:

```powershell
python -B -m scripts.diagnose_ticket_training --live-noise-comparison --input saved-live-pair.json --output fresh-score-comparison.json
```

This route performs no model reconstruction, NumPy reference projection,
teacher generation, prediction or training. It rechecks typed saved decisions,
retains every private task's recorded outcome and scores, and subtracts recorded
scores to show winning margins. It compares only identical saved request/state/
capability prefixes. Once selected actions differ, later different-world states
remain separate. Ties are0; a sole eligible action has no comparative margin.

Input is bounded3MiB; output must be fresh and is bounded256KiB. Malformed,
unknown, inconsistent, nonfinite or oversized evidence fails with retained work
counts. It reads once, hashes the captured bytes and reports
`captured_bytes_only_no_postflight_seal`: this is a snapshot diagnosis, not proof
of later file preservation, execution authenticity or model parity. It cannot
feed training, promotion, tuning, replay or probe selection. Flags are mutually
exclusive; original reference modes below preserve their existing contracts.

[Executed diagnosis](reports/catalog-live-score-diagnosis-20261027.md) found the
first selection difference at legacy's ready step: cleaninspect versusnoisydefer,
with recorded raw gap crossing-0.052939→+0.184587. Password still passed but its
winning raw margin shrank0.253496→0.010614. These are recorded decision facts,
not a causal mechanism, calibrated confidence or evidence that wrong labels
are beneficial. The original native experiment and all negative/spent results
remain unchanged; this mode grants no new model-run budget.

### Inspect the saved comparison offline

Use the existing inspector command on the diagnosis output, with a fresh HTML
destination:

```powershell
python -B -m scripts.build_key_door_inspector --source-artifact fresh-score-comparison.json --output fresh-score-inspector.html
```

The report shows both private candidates' old/new outcomes, first different
action on the common observation prefix, every decision's recorded margins,
all candidate scores, marked TRAIN corruption, exposure and diagnostic costs.
Each observed value links to its exact detached source. Missing values remain
unknown; recorded nulls, zeroes, failures and adverse gaps remain distinct.
The page cannot train or promote and performs no rescoring. Separately supplied
resources have an unverified association. Source snapshots do not authenticate
execution or establish calibrated confidence, causality or a circadian benefit.

New-mode failed CLI summaries now carry an explicit report-type tag and display
their error and paid work without invented outcomes. Historical untagged failed
summaries are not rewritten or guessed into this route. Complete reports require
two known arms/four contexts and bounded decisions/scores; partial reports may
omit unobserved rows.

[Executed export and software checks](reports/catalog-score-inspector-20261027.md)
include all24saved decisions/120scores and940source links without rereading the
original native experiment. Browser appearance and interaction remain unverified;
syntax and source-cell checks are not visual acceptance.

## Original chronological reference diagnosis

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
