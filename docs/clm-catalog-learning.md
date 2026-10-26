# Explicit full-catalog CLM learning

The optional `--catalog-training` mode trains each query against every candidate
in the active catalog. A query's label is an index into an explicitly ordered,
unique candidate ID tuple. Repeated target indices are valid. The default
paired CLM objective and existing commands retain their behavior.

## Objective and boundaries

`clm_full_catalog_cross_entropy_v1` is directional, row-mean cross entropy over
normalized state/action head outputs, scaled by `min(exp(logit_scale), 100)`.
It has no reverse-direction term or group mask. It changes both the candidate
composition and the objective relative to native paired CLM; comparisons cannot
attribute a difference solely to negative coverage.

```text
src/core/clm_catalog.py                 immutable TRAIN catalog/index records
src/adapters/clm_catalog_learner.py      native loss and owned catalog learner
src/adapters/clm_head_learner.py         shared atomic update engine
src/app/clm_temporal_sequence.py        existing causal stream integration
scripts/run_clm_temporal_sequence.py    frozen feature assembly and CLI
tests/test_clm_catalog_learning.py      math, ownership and integration checks
tests/test_clm_catalog_support_learning.py  recipe/receipt/cost/comparison checks
```

Core records validate metadata without importing Torch. The adapter admits
finite CPU float32 NxD query and MxD candidate features before paid updates.
The shared engine stages AdamW, gradient clipping and OneCycle state, then
publishes atomically. Catalog snapshots have a distinct exact public type;
paired snapshots and paired disposable probes cannot install catalog state.
Update attempts, candidate rows and logit values remain charged after restore.
Paid failures poison the owner rather than allowing an uncharged retry.

The app retains delayed labels, TRAIN consent, logical clocks and receipts.
All fixed updates finish before cached development scoring. Final evaluation is
not supplied to training or the CLI. Catalog training can use the existing
[explicit TRAIN support recipes](#explicit-train-support-recipes). Reserved probes
require the none recipe and use an [explicit phase-bound catalog contract](clm-reserved-probes.md#optional-full-catalog-objective).
Ordinary paired prediction and choice readout remain
available through their existing record types.

## Running and checking

Use the existing qualified local CLM assets and environment; no new dependencies
or environment variables are introduced. See the existing native adaptation
instructions for asset configuration. Inspect the CLI before a budgeted run:

```powershell
python -m scripts.run_clm_temporal_sequence --help
python -m pytest -q tests/test_clm_catalog_learning.py
python -m ruff check src/core/clm_catalog.py src/adapters/clm_catalog_learner.py tests/test_clm_catalog_learning.py
python -m ruff format --check src/core/clm_catalog.py src/adapters/clm_catalog_learner.py tests/test_clm_catalog_learning.py
python -m mypy --explicit-package-bases --follow-imports=silent src/core/clm_catalog.py src/adapters/clm_catalog_learner.py tests/test_clm_catalog_learning.py
```

Add `--catalog-training` to the existing temporal CLI command for this explicit
recipe. Its report names the objective and candidate indices; `training_pairs`
retains its legacy meaning of labeled query-target examples. Separate permanent
candidate-row and logit-value counters describe catalog exposure. Do not interpret
equal update counts as matched computation.

The fixed local protocol, exact executed commands, costs, failures and acceptance
are recorded in [the stage artifacts](../artifacts/runs/clm-full-catalog-20261022/).
Scoped software correctness does not establish adaptation benefit or release
readiness. Preserve historical paired controls and all negative results.

The sole frozen native case completes12updates/36query rows within budget. Its
development accuracy is1/4→2/4→3/4→3/4: worse than paired afterA, then tied
accuracy with better NLL/Brier/ECE. The revised invoice rule still fails. This
case changes both candidate composition and directionality and uses historical
paired evidence; it establishes neither general benefit nor recovery.

## Extending safely

Add future catalog/probe integration through the existing stream and explicit
TRAIN-only records. Keep probe owners separate, preserve permanent costs and
freeze a small protocol before comparative runs. Do not silently substitute this
objective for the default native paired objective.

## Explicit TRAIN support recipes

Combine `--catalog-training` with `--catalog-support repeat` or
`--catalog-support route` for a single catalog learner. `--catalog-training --compare-catalog-support`
uses the existing independent repeat/route owners and shared13-template encoder.
The comparison automatically includes TRAIN catalog diagnostics. Both fixed
trajectories finish before any cached metric or TRAIN diagnostic is scored.
Paired defaults, original none recipe and reserved-none probes remain available;
support plus reserved probes refuses before loading.

The exact declared TRAIN packets have3queries in A and4in B/return, all against
3ordered active candidates. Each arm pays12updates44query36candidate132logit
exposures, plus8choice readouts/31query24candidate rows with diagnostics. The
repeat arm duplicates v2 invoice TRAIN; the route arm adds the already-declared
auxiliary route TRAIN query. Class/group/unique-information mix differs despite
equal query counts. No new label, marker, objective or optimization is introduced.

The cached boundary admits both complete objective-specific causal receipt/label/
work prefixes, typed native counters, all caches and exact shared baseline/A/
frozen-A/control parity before either arm can expose metrics. A mixed coherent
paired/catalog report, absent catalog tag or replaced work cannot authenticate a
comparison. App source/feature/clock/budget/consent/paid-prefix guards still bind.

```powershell
python -m scripts.run_clm_temporal_sequence --model-dir LOCAL_ENCODER --head LOCAL_HEAD --output NEW_REPORT.json --catalog-training --compare-catalog-support
python -m pytest -q tests/test_clm_catalog_support_learning.py
python -m ruff check src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_catalog_support_learning.py
python -m ruff format --check src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_catalog_support_learning.py
python -m mypy --explicit-package-bases --follow-imports=silent src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_catalog_support_learning.py
```

Code integration is present; executed qualification and the conditional native
observation are tracked separately in the [frozen protocol and evidence](../artifacts/runs/clm-catalog-support-learning-20261022/).
No dependencies or environment variables added. Preserve all earlier mixed fits,
probe old-task failures and failed budgets. This supplies a conventional fit/control
path, not a retention, recovery, CPC, generalization or release-readiness claim.

### Executed support result and collection limits

224focused checks/platform, scoped statics and independent software review pass.
One frozen native repeat/route comparison completed24updates88query72candidate
264logits,1encoder13seq,90Heads84forwards284projected, inner233.841s; sampledRSS
15.790GB,nottruepeak. No native retry or post-observation dose/metric adjustment.

| catalog recipe | development correct /4 baseline ->A ->B ->return | exact TRAIN correct rows A ;B ;return |
|---|---|---|
| repeat |1 ->2 ->3 ->4|2/3 ;4/4 ;4/4|
| route |1 ->2 ->3 ->3|2/3 ;3/4 ;3/4|

Repeat is preferable in this tiny fixed scope, with lower NLL/Brier/ECE at B and
return. Both have bare TRAIN fit and suffix-transfer gaps at A; invoice is correct
on bare TRAIN at B/return but wrong on suffix B in both and return in route.
Route's auxiliary TRAIN query remains wrong. Repeated rows are not distinct
concepts; no reliable/general/causal/CPC or stable-password recovery claim.

The original outer collector failed after training by shadowing its receipt path
with a loop dict. Original controller1/incomplete parent/qualified source/trace
remain. Separate stdlib saved-output validation and independent QUALIFIED ACTUAL/
interpretation/narrative acceptance do not relabel that producer PASS. Native
child0 is only code-path-inferred; exact outer monotonic time unavailable, original
wait/UTC observational bounds qualified. No model rerun or budget renewal.
[All metric/case/TRAIN/loss endpoints, costs and failed versions](../artifacts/runs/clm-catalog-support-learning-20261022/RESULTS.md).

## Bind a trained catalog learner for typed prediction

`adapters.clm_catalog_decision.CLMCatalogPredictorBinding` connects a current,
successfully trained `CLMCatalogLearner` to the existing `CLMDecisionPredictor`.
It requires the same exact frozen encoder identity, healthy native owners and
unchanged source/encoder work between construction and binding.

```python
from src.adapters.clm_catalog_decision import CLMCatalogPredictorBinding

binding = CLMCatalogPredictorBinding(
    learner, encoder, instructions="Choose the appropriate action.",
    head_revision="local-catalog-update-12",
)
predictor = binding.bind()
result = predictor(decision_request)
receipt = binding.receipt
```

Why this: native snapshot records contain values, not an issuer seal. The binding
takes its own fresh public snapshot and compares the complete current native
state before copying. Arbitrary historical snapshots and probe owners are not
inputs. The existing predictor owns frozen head/scale copies and shares the
existing encoder and its permanent forward allowance. Later source learning
cannot alter the copied predictor heads. The revision string is caller metadata,
not authority or proof that weights came from an earlier reported experiment.

Each binding instance has one permanent attempt, including a failed attempt.
`work` records public snapshot, guard captures and head/scale copy charges;
`snapshot_tensor_elements_copied` includes returned guard and public snapshot
tensors, including private RNG state. It does not measure all native validation
or allocator traffic. Receipts and work are defensive copies. The factory drops
native source references after its attempt and retains no success snapshot chain.
This is inference export, not validation, promotion or dense consolidation.

```powershell
python -m pytest -q tests/test_clm_catalog_decision.py
python -m ruff check src/adapters/clm_catalog_decision.py tests/test_clm_catalog_decision.py
python -m ruff format --check src/adapters/clm_catalog_decision.py tests/test_clm_catalog_decision.py
python -m mypy --explicit-package-bases --follow-imports=silent src/adapters/clm_catalog_decision.py tests/test_clm_catalog_decision.py
```

No dependencies or environment variables added. Qualification status and scoped
costs are recorded in the [binding evidence](../artifacts/runs/clm-catalog-predictor-binding-20261022/).
Native task quality, retention, recovery and release readiness remain separate.

Binding qualification:34new+4existing focused checks pass on Windows and WSL;
scoped Ruff/format/mypy pass and independent software review accepts. The first
static and WSL profiling attempts failed and remain retained. WSL passed under
a separate prospectively reviewed corrective copy allowance; cumulative copies
exceed the original shared ceiling. No pretrained comparison was run for this
binding. [Executed outcomes, costs, failures and next action](../artifacts/runs/clm-catalog-predictor-binding-20261022/RESULTS.md).

## Actual typed choices after a stream

Add `--typed-development-readout` to a single `--catalog-training` run to send
the four existing `after_return` development requests through the trained frozen
predictor. The original twelve updates finish before this inference. The mode
accepts the existing none/repeat/route recipes and refuses comparison and probe
combinations before loading. Defaults remain unchanged.

```powershell
python -m scripts.run_clm_temporal_sequence --model-dir LOCAL_ENCODER --head LOCAL_HEAD --output NEW_REPORT.json --catalog-training --catalog-support repeat --typed-development-readout
python -m pytest -q tests/test_clm_typed_stream_readout.py
python -m ruff check scripts/run_clm_temporal_sequence.py tests/test_clm_typed_stream_readout.py
python -m ruff format --check scripts/run_clm_temporal_sequence.py tests/test_clm_typed_stream_readout.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/run_clm_temporal_sequence.py tests/test_clm_typed_stream_readout.py
```

Why this: cached embedding probabilities did not exercise the live typed request
path. This option keeps the trained source live, uses the admitted public binding,
and reserves five encoder forwards at construction: one feature batch and four
request batches. It never renews a consumed one-forward encoder. Source state and
permanent work are guarded between requests, including the final save callback.

`typed_development` contains canonical requests, actual predictions, original
policy, binding/predictor/source-guard work, encoder before/after ledgers and the
maximum absolute difference from cached return probabilities. The difference
is descriptive; a different batch shape can change native floating-point output.
No labels select further learning, promotion or inference work. A partial record
stays incomplete and cannot authorize cached scoring or conceal its snapshot cost.

No dependencies or environment variables added. This is a development readout;
final evaluation stays unavailable. Literal fixture correctness and any future
pretrained task observations are separate in the [typed stream evidence](../artifacts/runs/clm-typed-stream-readout-20261022/).

Executed qualification:56 focused checks/platform plus one final metadata-only
comparison refusal/platform; scoped Ruff/format/mypy and independent software/
interpretation/narrative review pass. Four tiny trajectories/platform account for
all96TRAIN updates across Windows/WSL. The two initial static failures remain
preserved. No pretrained task validation or native experiment was run here.
[Commands, costs, skipped checks and next protocol](../artifacts/runs/clm-typed-stream-readout-20261022/RESULTS.md).


Native follow-up: one prospectively reviewed single-repeat run now executes all
four typed development requests after the original twelve catalog updates. All
four choose the fixed target; maximum probability difference from cached return
is9.62227554413e-7 (descriptive). Invoice margin is narrow, A/B failures remain,
and the original stable-password recovery flag is false. This is one development
observation, not general retention, recovery or circadian benefit. Total cost:
12updates44TRAINqueries,5encoderforwards29sequences,239.09souter/16.21GBsampledRSS.
[Executed native evidence and limitations](../artifacts/runs/clm-typed-native-return-20261022/RESULTS.md).


## Isolated catalog probe with finite TRAIN rehearsal

Opt in through the existing stream:

```powershell
python -m scripts.run_clm_temporal_sequence --model-dir LOCAL_ENCODER --head LOCAL_HEAD --output NEW_REPORT.json --catalog-training --reserved-task-probes --probe-rehearsal
```

Each non-baseline disposable probe keeps its six original updates and two new
marker rows per update, adding two cyclic rows from its already-observed original
TRAIN triple. Each old row is exposed four times. Baseline has no arrived TRAIN
and uses the original two-marker packets. The main learner still receives its
original twelve updates and 36 query rows. Probe weights remain isolated.

Why this: existing isolated catalog probes showed severe old-task damage. This
provides a conventional rehearsal control using existing learners and the stream.
It adds exposure and computation; it does not establish a retention remedy.
Across all four probes, rehearsal adds 36 old TRAIN query exposures. Total main
plus probe training is 36 updates, 120 query rows, 108 candidate rows and 360
logits, versus 84 query rows and 252 logits without rehearsal. Six updates alone
do not imply matched compute. A matched exposure control must precede comparison.

Replay permission is granted at original TRAIN creation only for this explicit
new mode. Historical training-only experiences are not upgraded. Marker consent
stays training-only, and probe-test, old-task readout, development and final labels
never enter replay. Only the original phase-current bare TRAIN triple is used.
The existing paired, no-rehearsal and typed-request modes retain their defaults;
unsupported combinations refuse before loading.

The probe has separate original six-update and twelve-row replay budgets. The
native inbox receives the update-only budget; two replay rows are charged to the
independent replay session before a native attempt, including failed attempts.
Both original owners, clocks, limits, progress and paid counters are guarded.
Reports keep mixed query work, attempted replay rows, source row indices, retained
packet/history tensor storage bytes and unrefunded failure prefixes. Stored bytes
cover the history and all six materialized packets, deduplicating shared storage;
they do not measure the whole process. Scores remain unavailable until all main
and probe work and their complete cached records are admitted.

```powershell
python -m pytest -q tests/test_clm_probe_rehearsal.py tests/test_clm_catalog_reserved_probe.py
python -m ruff check src/core/clm_catalog_probe.py src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_probe_rehearsal.py
python -m ruff format --check src/core/clm_catalog_probe.py src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_probe_rehearsal.py
python -m mypy --explicit-package-bases --follow-imports=silent src/core/clm_catalog_probe.py src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_probe_rehearsal.py
```

No dependencies or persistent environment variables added. Scoped qualification,
failed attempts and remaining scientific gates are in the
[rehearsal evidence](../artifacts/runs/clm-probe-rehearsal-20261022/).


Functional qualification: 114 focused checks pass on each of Windows and WSL,
with scoped Ruff/format/mypy checks and independent software acceptance. The
initial combined-budget refusal and static failures remain retained. Original
all-attempt literal-input reservation qualification remains FAILED/unproven:
196608 declared elements versus the original131072, with actual input unmetered.
The qualifier's complete flag covers its implemented checks, not that omitted
aggregate gate. No new pretrained retention experiment has run.


## Matched extra-marker exposure control

Use `--probe-marker-control` instead of `--probe-rehearsal` to compare equal
training shapes and dose while changing which labels occupy the extra rows:

```powershell
python -m scripts.run_clm_temporal_sequence --model-dir LOCAL_ENCODER --head LOCAL_HEAD --output NEW_CONTROL_REPORT.json --catalog-training --reserved-task-probes --probe-marker-control
```

At each non-baseline checkpoint, the control repeats the original marker pair
within each of six four-query packets. It uses 24 new-marker query rows, 18
candidate rows and 72 logits, matching rehearsal's 12 new-marker plus 12 arrived
TRAIN query rows. These are repeated new TRAIN exposures, not memory replay;
replay remains zero and consent stays training-only. Baseline uses the original
six two-marker updates in both modes. The main learner remains unchanged and
probe weights remain isolated.

Why this: rehearsal doubles the original probe query exposure. This control
matches updates, query/candidate/logit counts and projection work. Unique labels,
replay consent and retained storage still differ and must be reported. Neither
equal update counts nor this control alone establishes a retention benefit.

`marker_control` records original probe_train provenance, fixed row indices,
unique retained packet storage and extra marker rows attempted. Two extra rows
are charged before each native attempt; failure does not erase that work. Strict
complete cached admission rejects mode, provenance, permission, budget, owner,
feature and counter changes before scores. The two modes are mutually exclusive;
paired, support, comparison and typed readout cross-products refuse before load.

```powershell
python -m pytest -q tests/test_clm_probe_exposure_control.py tests/test_clm_probe_rehearsal.py
python -m ruff check src/core/clm_catalog_probe.py src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py scripts/run_clm_temporal_sequence.py tests/test_clm_probe_exposure_control.py
python -m ruff format --check src/core/clm_catalog_probe.py src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py scripts/run_clm_temporal_sequence.py tests/test_clm_probe_exposure_control.py
python -m mypy --explicit-package-bases --follow-imports=silent src/core/clm_catalog_probe.py src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py scripts/run_clm_temporal_sequence.py tests/test_clm_probe_exposure_control.py
```

No new dependency or persistent environment variable. Functional qualification:
126 focused checks pass on Windows and WSL, with scoped Ruff/format/mypy and
independent software acceptance. This stage checks all attempted cohort input
and copy reservations before declaring qualification complete. The earlier
rehearsal stage's original failed/unproven resource qualification is preserved.
[Commands, costs and native observation status](../artifacts/runs/clm-probe-exposure-control-20261022/RESULTS.md).


## Observe arrived TRAIN retention separately

Add `--probe-train-readouts` to a catalog reserved-probe run (ordinary, rehearsal,
or matched extra-marker control):

```powershell
python -m scripts.run_clm_temporal_sequence --model-dir LOCAL_ENCODER --head LOCAL_HEAD --output NEW_REPORT.json --catalog-training --reserved-task-probes --probe-rehearsal --probe-train-readouts
```

Why this: four suffix old-task heldouts do not reveal whether a probe retained
its three already-arrived bare TRAIN points. The new `arrived_train` score group
reports those original TRAIN examples before and after the six disposable-probe
updates, alongside the unchanged `old_task` suffix group. It is a training-fit
and retention diagnostic, not an independent generalization estimate. Baseline
has no arrived history and adds no group. No label, score or prediction chooses
probe updates, replay rows, stopping or promotion. Final evaluation stays sealed.

The adapter passes the exact original phase TRAIN tuple only at its observed
checkpoint. Owners, features, canonical labels, encoder identity and catalog are
guarded. The private prediction view has role `train`; that role grants no probe
update conversion or replay consent. Rehearsal and diagnostic must share their
original history owner. Source/deployed learners and caller RNG remain isolated.

Per non-baseline probe the two extra reads add six query rows, six candidate rows
and twelve projected rows, with no update or encoder pass. Across all three that
is eighteen query/candidate rows and thirty-six projections. Complete records
have eight choice reads/22 query rows/24 candidate rows; original six readouts keep
their order and schema in a separate field. `arrived_train_readouts` records the
exact source role/phase/row indices/catalog/targets, unique retained tensor storage,
and the two cached endpoints. Scores admit every complete child before exposing
any metric. The total cached probability-row count rises64->82 only in this mode.

Unsupported paired/support/comparison/typed combinations refuse before loading.
No dependencies or persistent environment variables added. Run focused checks:

```powershell
python -m pytest -q tests/test_clm_probe_training_readouts.py tests/test_clm_probe_exposure_control.py tests/test_clm_probe_rehearsal.py
python -m ruff check src/core/clm_probe.py src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py scripts/run_clm_temporal_sequence.py tests/test_clm_probe_training_readouts.py
python -m ruff format --check src/core/clm_probe.py src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py scripts/run_clm_temporal_sequence.py tests/test_clm_probe_training_readouts.py
python -m mypy --explicit-package-bases --follow-imports=silent src/core/clm_probe.py src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py scripts/run_clm_temporal_sequence.py tests/test_clm_probe_training_readouts.py
```

[Executed qualification and costs](../artifacts/runs/clm-probe-training-readouts-20261022/RESULTS.md).


## State-only disposable catalog probe

Use `--probe-state-only` with catalog reserved probes to adapt only the state
projection. The inherited action projection and temperature remain exactly fixed:

```powershell
python -m scripts.run_clm_temporal_sequence --model-dir LOCAL_ENCODER --head LOCAL_HEAD --output NEW_REPORT.json --catalog-training --reserved-task-probes --probe-rehearsal --probe-train-readouts --probe-state-only
```

Why this: the accepted rehearsal diagnostic lost predictions on rehearsed TRAIN
examples. Fixing candidate geometry and temperature is a conventional control for
that interference hypothesis. It is not an established remedy or circadian feature.
All four disposable probes, including baseline, use the explicit scope. The main
learner keeps its original all-parameter updates. Ordinary probes, rehearsal,
extra-marker control and TRAIN readouts compose with this option; support,
comparison, paired and typed readout combinations refuse before model loading.

The private probe engine selects state parameters for AdamW, gradient validation
and clipping. Action projection runs without autograd; temperature is detached
from loss differentiation. Numeric catalog logits and directional cross-entropy
are unchanged. Native `requires_grad` declarations stay intact, and guards check
fixed values, absent gradients, optimizer parameter/state owners and scheduler on
original and staged bundles before publication. A missing optimized gradient is
still a paid failed update. Each probe retains one bounded action-state/scale pin;
it never retains a success-record chain or modifies the source learner.

Reports add `probe_parameter_scope="state_only"` and per-child
`parameter_update_scope` with optimized/fixed/total element counts and validated
fixed-parameter invariance. All child records are admitted before any cached
metric. Default all-parameter runs retain their original report schema. Equal
updates/rows/forwards do not imply equal backward, optimizer or memory computation.
No final data, metrics or predictions select updates or replay rows.

Run focused software checks:

```powershell
python -m pytest -q tests/test_clm_probe_state_only.py tests/test_clm_probe_training_readouts.py tests/test_clm_catalog_probe.py
python -m ruff check src/adapters/clm_head_learner.py src/adapters/clm_catalog_probe.py src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py scripts/run_clm_temporal_sequence.py tests/test_clm_probe_state_only.py
python -m ruff format --check src/adapters/clm_head_learner.py src/adapters/clm_catalog_probe.py src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py scripts/run_clm_temporal_sequence.py tests/test_clm_probe_state_only.py
python -m mypy --explicit-package-bases --follow-imports=silent src/adapters/clm_head_learner.py src/adapters/clm_catalog_probe.py src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py scripts/run_clm_temporal_sequence.py tests/test_clm_probe_state_only.py
```

No new dependencies or persistent environment variables. Scope is a probe
constructor argument rather than a global learner-policy migration; existing
public continuation records remain unchanged.
[Decision](adr/ADR-0546-explicit-state-only-catalog-probes.md) ·
[Executed gates, costs and native evidence](../artifacts/runs/clm-probe-state-only-20261022/RESULTS.md).

## State-only TRAIN catalog learner

`CLMStateOnlyCatalogLearner` accepts ordinary `FrozenCatalogTrainingBatch`
records. It owns its TRAIN updates, optimizer and continuation state independently
of every disposable probe. Select it through the existing temporal runner:

```powershell
python -m scripts.run_clm_temporal_sequence --model-dir LOCAL_ENCODER --head LOCAL_HEAD --output NEW_REPORT.json --catalog-training --train-state-only --train-catalog-diagnostics
```

`--train-state-only` and `--probe-state-only` select different owners. A reserved
probe still defaults to its original all-head scope unless explicitly selected.
No private probe parameters or labels are copied back into TRAIN. All-head runs
remain the default. The new TRAIN learner uses the same numeric catalog CE as
the existing state-only probe, with state-only gradient clipping and AdamW.
Candidate/action parameters and temperature keep their birth values and have
no gradients, while their `requires_grad` declarations remain unchanged.

`CLMStateOnlyCatalogSnapshot` has the existing native state fields and a distinct
exact type. All-head, paired and state-only snapshots cannot restore across
families. Even an exact state-only snapshot cannot change the receiving owner's
fixed birth values. A failed staged restore remains charged and unpublished;
restoration never renews physical training or restore allowances. Snapshots own
their tensors, and the learner retains one bounded fixed birth pin.

Reports add `train_parameter_update_scope` with the scope name, optimized/fixed/
total element counts and validated fixed-parameter invariance. Complete cached
catalog reports validate this declaration before metrics. Existing support
recipes and reserved probes use the same stream; the typed frozen predictor
currently requires the original exact all-head family, so that combination
refuses before loading. Extending its admission remains unfinished.

Focused commands (tiny literal correctness fixtures, no native comparative run):

```powershell
python -m pytest -q tests/test_clm_train_state_only.py tests/test_clm_probe_state_only.py tests/test_clm_catalog_learning.py tests/test_clm_catalog_probe.py
python -m ruff check src/adapters/clm_state_only_catalog_learner.py src/adapters/clm_catalog_probe.py scripts/run_clm_temporal_sequence.py tests/test_clm_train_state_only.py
python -m ruff format --check src/adapters/clm_state_only_catalog_learner.py src/adapters/clm_catalog_probe.py scripts/run_clm_temporal_sequence.py tests/test_clm_train_state_only.py
python -m mypy --explicit-package-bases --follow-imports=silent src/adapters/clm_state_only_catalog_learner.py src/adapters/clm_catalog_probe.py scripts/run_clm_temporal_sequence.py tests/test_clm_train_state_only.py
```

No new dependencies or environment variables. Equal updates do not imply equal
backward, optimizer or memory cost. Accepted private-probe results remain mixed;
they establish no benefit for this continuing TRAIN learner. A native A/B/return
comparison needs a separate frozen protocol after software acceptance.

The first such native continuing-TRAIN observation is now independently accepted:
state-only and saved all-head suffix accuracy both follow 1/4, 2/4, 3/4, 3/4.
State-only has worse B/return NLL and ECE despite lower Brier. Its bare TRAIN
choices reach 3/3 at B/return while the invoice suffix stays wrong; A also has a
bare legacy fit failure. These mixed results establish no retention/recovery or
circadian advantage. The saved all-head run lacks bare TRAIN readouts, so those
fit diagnostics are within-arm only. All metrics, costs and failures are retained
in the evidence below; no native allowance remains open.
[Decision](adr/ADR-0547-state-only-train-catalog-scope.md) ·
[Executed checks and limits](../artifacts/runs/clm-train-state-only-20261022/RESULTS.md).

## Repeat TRAIN to typed requests

The existing all-head repeat recipe can finish its TRAIN stream and then answer
four typed post-return development requests through the existing frozen binding:

```powershell
python -m scripts.run_clm_temporal_sequence --model-dir LOCAL_ENCODER --head LOCAL_HEAD --output NEW_REPORT.json --catalog-training --catalog-support repeat --typed-development-readout
```

It keeps twelve updates but uses 44 TRAIN row exposures: B and return packets
duplicate the invoice row. This is additional exposure, not a distinct task.
The typed stage starts after TRAIN, owns its copied predictor, and cannot update
the source. Reports account for binding snapshots, copy/guard work and four extra
encoder calls. The same encoder is born with five forward calls; that allowance
is not renewed after use.

Literal end-to-end coverage now executes this exact composition: ten new cases
plus three original none-recipe checks pass on Windows and WSL, with scoped
statics and independent executed review. It checks real TRAIN/binding/predictor
operations using a fake encoder, not native task accuracy. Earlier qualified
repeat return 4/4 cached results remain a separate observation. The earlier
accepted R5.3-F.D.X native run also executed these four typed requests correctly,
with a small invoice margin and no broad recovery claim; reuse that observation
instead of executing the mistakenly proposed duplicate D.R.X. No new production
code, dependencies or configuration were needed for this qualification.

```powershell
python -m pytest -q tests/test_clm_repeat_typed_readout.py
python -m ruff check tests/test_clm_repeat_typed_readout.py
python -m ruff format --check tests/test_clm_repeat_typed_readout.py
python -m mypy --explicit-package-bases --follow-imports=silent tests/test_clm_repeat_typed_readout.py
```

[Executed evidence, preserved failure and limits](../artifacts/runs/clm-repeat-typed-readout-20261022/RESULTS.md).

## Phase-delayed TRAIN evidence memory

The existing catalog stream can also maintain the existing causal raw TRAIN
evidence store alongside its unchanged head updates:

```powershell
python -m scripts.run_clm_temporal_sequence --model-dir LOCAL_ENCODER --head LOCAL_HEAD --output NEW_REPORT.json --catalog-training --train-evidence-memory
```

This option currently requires the none recipe, one catalog learner, no reserved
probes or support comparison. It explicitly grants raw TRAIN
reuse consent at stream creation. The fixed corpus's raw mirrors are admitted
only after their original catalog receipts/labels validate. Twelve records per
phase are recorded at checkpoint7/15/23; original observation/label clocks stay
intact. They are not backdated or claimed to have been available to earlier queries.

At baseline/A/B/return, four development queries retrieve top1 current-policy
TRAIN evidence. The observer has no head training, encoder or probe authority.
No eligible evidence yields unavailable. Version isolation means B's password
query cannot silently reuse a v1 label; returned parcel can use already-arrived
v2 B evidence. Scores are produced after all twelve updates, with coverage,
correct counts and accuracy among answered reported separately. Retrieval's
one-hot choices are not calibrated CLM probabilities. This is a fixed-template
memory route, not demonstrated learned generalization or circadian benefit.

The report includes all36events,16retrieval selections, canonical storage bytes,
metadata scans, comparisons, token visits and rendered bytes. Storage is bounded
at36events/64KiB, context768bytes/top1; Python objects and defensive scalar copies
are additional allocations. Failed writes/reads remain paid and stop this owner.
No automatic eviction, retry or native-comparison allowance is supplied.

```powershell
python -m pytest -q tests/test_clm_train_evidence_memory.py
python -m ruff check src/app/clm_train_evidence_memory.py scripts/run_clm_temporal_sequence.py tests/test_clm_train_evidence_memory.py
python -m ruff format --check src/app/clm_train_evidence_memory.py scripts/run_clm_temporal_sequence.py tests/test_clm_train_evidence_memory.py
python -m mypy --explicit-package-bases --follow-imports=silent src/app/clm_train_evidence_memory.py scripts/run_clm_temporal_sequence.py tests/test_clm_train_evidence_memory.py
```

No new dependencies or persistent environment variables. To extend the path,
first add canonical source/receipt and consent tests for the new task; never
convert development/probe/final targets into memory records.

The first frozen native memory observation completed successfully: the head's
losses/probabilities match the saved no-memory producer exactly; memory answers
0/3/3/4 fixed requests correctly where answered and retains B's parcel evidence
at return. B password remains unavailable under strict current-version filtering.
Its36writes16retrievals use22255canonical bytes and add no encoder/head updates.
This is lexical recall with explicit policy metadata, not learned generalization,
weight-only recovery or a circadian advantage. Native allowance is closed.
[All endpoints/costs/limitations](../artifacts/runs/clm-train-evidence-memory-20261022/NATIVE_RESULTS.md).

The generic [evidence-first fallback mode](decision-evidence.md) is separately
available to caller-owned stores and frozen typed predictors. The native CLM
stream observation above did not execute that later policy. The joint typed
serving option below has separate literal correctness evidence and does not
retroactively change that native result.

## Live typed memory with a frozen CLM fallback

Use the existing options together for one all-head none-recipe stream:

```powershell
python -m scripts.run_clm_temporal_sequence --model-dir LOCAL_ENCODER --head LOCAL_HEAD --output NEW_REPORT.json --catalog-training --train-evidence-memory --typed-development-readout
```

After twelve TRAIN updates, the completed memory owner binds one finite serving
view to the ordinary frozen trained-head predictor. It explicitly copies36already-
consented TRAIN events to a separate36-event/64KiB store. The original phase
record, clocks,36writes and16retrievals stay sealed. The copied writes/bytes and
new retrievals are paid separately. Binding is single-attempt, including failures;
there is no automatic chain of stores or renewed head/encoder allowance.

The typed stage submits four canonical after-return requests, answered from
current-policy memory, followed by one fixed unlabeled availability control:
`unknown workflow please`. That request buys one actual frozen CLM callback.
It has no target or accuracy metric and never enters TRAIN, replay or probes.
Reports retain `choice_source`, all five returned decisions, separate memory
binding/view work and the unchanged head binding/source guards. Lost callback
returns preserve the preceding four memory results and all paid work.

The same encoder is born with five forward calls; the joint route uses one
feature pass plus one fallback pass. Four known memory calls add no encoder or
head forwards. The generic fallback returns valid adverse head choices unchanged.
`max_abs_cached_difference` compares only the four known memory predictions
against cached head probabilities: it is an intervention difference, not a claim
of numerical head parity. Existing standalone typed mode keeps four actual head
requests and its original accounting. State-only plus typed serving remains
unsupported; probes/support comparison and non-none memory recipes still refuse.

```text
src/app/clm_train_evidence_memory.py  one paid copied view from completed TRAIN
src/app/clm_memory_serving.py         fixed requests and complete cache admission
scripts/run_clm_temporal_sequence.py  existing binding/encoder/serving composition
tests/test_clm_live_memory_fallback.py actual literal pipeline and paid failures
```

Focused commands use literal heads/fake encoder, not a pretrained comparison:

```powershell
python -m pytest -q tests/test_clm_live_memory_fallback.py
python -m ruff check src/app/clm_train_evidence_memory.py src/app/clm_memory_serving.py scripts/run_clm_temporal_sequence.py tests/test_clm_live_memory_fallback.py tests/test_clm_train_evidence_memory.py
python -m ruff format --check src/app/clm_train_evidence_memory.py src/app/clm_memory_serving.py scripts/run_clm_temporal_sequence.py tests/test_clm_live_memory_fallback.py tests/test_clm_train_evidence_memory.py
python -m mypy --explicit-package-bases --follow-imports=silent src/app/clm_train_evidence_memory.py src/app/clm_memory_serving.py scripts/run_clm_temporal_sequence.py tests/test_clm_live_memory_fallback.py tests/test_clm_train_evidence_memory.py
```

No new dependencies or environment variables. Extend through the same bounded
typed port; declare any new request/unknown control before execution and budget
its actual encoder/copy work. Keep final data and private probes separate.
[Protocol and executed scope](../artifacts/runs/clm-live-memory-fallback-20261022/PROTOCOL.md).
