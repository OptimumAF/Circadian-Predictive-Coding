# Module: `src/core`

## Responsibilities

- Define model behavior (`BackpropMLP`, `PredictiveCodingNetwork`, `CircadianPredictiveCodingNetwork`)
- Define ResNet-50 benchmark variants (`BackpropResNet50Classifier`, `PredictiveCodingResNet50Classifier`, `CircadianPredictiveCodingResNet50Classifier`)
- Implement circadian mechanisms such as chemical gating, reward-modulated wake updates, and adaptive sleep budgeting
- Provide activation utilities
- Define neuron adaptation interfaces and traffic summaries
- Validate NumPy binary training batches before any model state changes
- Validate immutable, JSON-safe sleep-attempt fact records without reading data or writing logs

## Inputs / Outputs

- Inputs: numeric arrays, model hyperparameters
- Outputs: model predictions, train-step metrics, traffic summaries
- Sleep-telemetry inputs: measured model/runner facts; output: a validated version-one value record

## Non-Responsibilities

- CLI handling
- Environment configuration
- Dataset generation and external IO
- Choosing guard roles, scheduling sleep attempts, and persisting event records

The implemented forward, relaxation, update, and diagnostic equations are
specified in [learning mathematics](../learning-mathematics.md). Energy and
update reconciliation is recorded in ADR-0021. One-hidden, deeper circadian,
and gated float64 derivative checks pass under P2.3. The ordinary multilayer
PC lower-latent update does not match the fixed-prior diagnostic gradient;
its matched deeper formulation remains open under P2.6a. Stable relaxation and
nonfinite boundaries for the local contract pass under P2.4 (ADR-0022);
broader numerical input and post-update validation remains P2.8.
The named NumPy and Torch `matched_pc_control()` presets supply a shallow
no-circadian parity control (ADR-0023); deeper NumPy parity is excluded.
The P2.7 float64 binary/two-class fixture compares executed NumPy and Torch
one-hidden PC/circadian forward, hidden update, chemistry, and matched
backprop MLP hidden SGD update, plus separate zero-noise
structural decisions. Its equal-rate output-margin update differs by a
factor of two (ADR-0025); it does not establish full backend parity.

`training_validation.py` takes NumPy feature/target arrays, the expected
input width, and a weight learning rate. It returns no data or raises an
informative `ValueError` for malformed or nonfinite binary training inputs.
Backprop, ordinary PC, and circadian PC call it before state changes. It
does not repair targets or select evaluation roles. See ADR-0026.
The Torch PC/circadian heads validate nonempty feature and class-index
tensors before adaptive state changes. They require matching width,
floating weight dtype/device, `torch.int64` labels, and a valid class
range; ADR-0027 records the error and synchronization contract. NumPy
backprop and ordinary PC stage finite diagnostics, gradients, parameters,
and traffic before committing training updates (ADR-0028). NumPy
circadian and Torch PC heads also guard candidate commits, restoring
provisional adaptive and gradual-prune state on numerical rejection
(ADR-0029). ADR-0030 records finite saturation behavior and validates
post-split/prune tensor widths before training mutation.
ADR-0031 requires finite numeric circadian configuration fields before
either backend can train or sleep.
`dimension_validation.py` enforces positive integer model widths before
NumPy or Torch parameter allocation (ADR-0032).

Circadian `sleep_mode` defaults to `legacy`, which retains the historical
split/prune budget gate. `components` independently switches chemical
reset, homeostasis, splitting, and pruning on both backends and replay on
NumPy. Zero structural budgets still allow enabled consolidation. Torch
has no replay path. `disabled` returns without changing sleep state, even
for a forced event. `SleepEventResult.performed` distinguishes an executed
no-topology event from a skipped one. Warmup and trigger clocks retain
their current behavior; ADR-0033 records the boundary.

`sleep_clocks.py` defines validated completed-runner-epoch progress and a
read-only successful-work snapshot. The four app runners pass typed
`SleepEpochProgress` for warmup and structural progress windows; direct
legacy `current_step`/`total_steps` inputs remain accepted. NumPy and
Torch `get_sleep_clocks()` distinguish wake batches/examples, wake
batches since sleep, replay updates, and performed events. NumPy replay
does not advance its wake clock, and Torch has no replay. Historical
config names have the precise units recorded in ADR-0035; model-owned
state snapshots are described below.

NumPy `configure_replay_retention(ReplayRetentionBudget)` enables an
opt-in, pretraining per-example buffer. It leaves the historical
`replay_memory_size` batch-snapshot policy unchanged for models that do
not opt in. The bounded buffer retains unique labeled rows by the
smallest stable content hashes and enforces both copied-array bytes and
example count. `get_replay_retention()` returns IDs, count, and actual
array bytes for a phase report; `restore_state()` checks the declared
budget and both caps. The app validates role provenance at checkpoint
resume (ADR-0068). This core API does not open validation/test roles or
decide when phase data arrive. The fixed-data
[retention audit](../replay-retention-audit.md) records variable legacy
example/byte counts, cached priorities, duplicate slots, and A→B
content-ID survival before any new policy is introduced.

`replay_retention.py` chooses which distinct content ID to evict when a
bounded replay cap is exceeded. It accepts retained IDs in order and a
typed policy, and returns one ID; it does not select sleep replay,
read data roles, or save artifacts. Direct NumPy core users can opt into
the experimental policies before any wake update:

```python
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.replay_retention import ReplayRetentionPolicy

model.configure_replay_retention(
    ReplayRetentionBudget(max_examples=4, max_bytes=96),
    policy=ReplayRetentionPolicy("recent_fifo"),
)
# Or predeclare policy=ReplayRetentionPolicy("seeded_reservoir", seed=53).
```

Omitting `policy` keeps v4's fixed content-hash retention and unchanged
snapshot fields. The seeded policy is a stable bottom-k rank of distinct
content IDs; it is not Algorithm R over repeated wake occurrences.
ADR-0102 records the choice. Runner exposure and versioned comparison
remain P4.2b.

For opt-in `components` mode, a real adaptive-width change clears the
plateau/budget diagnostic history. This includes split/prune sleep,
NumPy delayed-prune finalization, and external NumPy proposals. An
incomplete component-mode history uses the minimum adaptive budget
scale; legacy history and fallback remain unchanged. ADR-0036 records
the width-normalization reason and the remaining nonstructural limits.

The NumPy circadian model also exposes `snapshot_state()` and
`restore_state(snapshot)` for detached in-memory copies of all model-owned
state, including topology, replay contents, local RNG, and clocks. For
example, save a state before a sleep attempt and restore it after a
rejected proposal. The restore path checks compatible configuration and
tensor alignment before replacing state. Executed sleep now uses that
snapshot as an atomic boundary: mutation errors, nonfinite values, and
invalid topology restore the entry state, including replay and RNG.
Durable checkpointing remains P3.9 (ADR-0037, ADR-0049).

The Torch circadian head's flat `snapshot_state()` dictionary now includes
its split generator, format/config/device identity, adaptive tensors,
diagnostic history, reward state, and counters. `restore_state(saved)`
validates a detached candidate before replacing live head state, so a
rejected noisy split can resume with the same next draw. The ResNet
classifier's current `snapshot_state()` delegates to that head for the
existing sleep guard. Whole-classifier state uses a separate API
(ADR-0038).

`CircadianPredictiveCodingResNet50Classifier.snapshot_full_state()` returns
an in-memory `CircadianClassifierSnapshot` with copied backbone
parameters/buffers, training modes, gradient flags, and the complete head
state. `restore_full_state(snapshot)` checks compatibility and restores
these for deterministic continuation on the same input sequence. The
current circadian training route has no optimizer, scheduler, or scaler;
the input loader and process-global random streams are caller-owned.
The frozen-backbone vision and matched-head guards restore the head on
rejected or failed scoring; their operational rollback counts stay in
the runner. File-backed resume remains P3.9 (ADR-0039, ADR-0050).

NumPy external `NeuronChangeProposal` requests and policy-driven sleep
requests are validated before structural mutation. Invalid indices,
over-budget changes, width violations, and age/cooldown violations raise
an error without changing model state. When an explicit prune request
overlaps a potential split source, the prune takes precedence and the
next eligible split source is used (ADR-0040).
The P4.5 NumPy proposal path parses typed requests, validates active
phase budgets and original-width eligibility, and ranks split sources
with the existing usage score before calling tensor mutation. A policy
still receives copied hidden/chemical `LayerTraffic` and returns
`NeuronChangeProposal`; for example, a one-split/one-prune proposal is
accepted in a phase with both budgets but rejected before mutation when
either phase budget is zero. Torch retains its post-split prune planner,
where a newly created child can be a candidate (ADR-0110).

NumPy's built-in sleep selector also validates its combined original-width
split/prune proposal before drawing split noise. Prune candidates exclude
their original neurons from splitting, and pending gradual prunes reserve
minimum-width capacity. Torch's distinct post-split selector uses the
detached gate described below (ADR-0041).

Torch built-in sleep now plans combined split/prune work on detached head
tensors and a cloned split generator when both structural actions can
run. Its prune candidates are still selected after the noisy split,
including eligible parents or newly appended children. Bad proposals
therefore fail before live mutation; one-action routes avoid the extra
copy. Executed sleep also has the atomic snapshot boundary described
above (ADR-0042, ADR-0049).

NumPy adaptive neurons now have persistent IDs and birth-parent IDs.
`get_neuron_lineage()` returns immutable active IDs and parent references;
children keep their parent reference even when that parent is pruned.
The IDs are model-owned metadata and included in version-2 in-memory
snapshots (ADR-0043). Torch uses the same read-only result and aligned
ID tensors, including within detached post-split planning. Its version-2
head snapshot and trained-state hash include IDs and the next-ID counter;
sleep results still expose positional indices (ADR-0044). Executed sleep
results also contain read-only pre/post lineage snapshots, including
events whose split and prune leave net width unchanged. Skipped results
leave the optional fields empty. A scheduled gradual prune retains the
same active IDs until final removal; explicit proposed/scheduled/removed
telemetry remains P3.6 (ADR-0045).

Isolated NumPy/Torch split tests disable replay, pruning, homeostasis,
chemical reset, and (for the primary contract) split noise. They verify
unchanged predictions across single and repeated splits, duplicated
incoming paths, and parent-plus-child outgoing-row conservation. Seeded
noisy splits are checked separately; a full sleep event may still change
predictions through other enabled components (ADR-0046).

Pruning alignment tests mark every adaptive vector with distinct values
and verify surviving IDs, tensor dimensions, and successful work clocks.
NumPy pending gradual prunes reserve minimum-width capacity and retain
their IDs until wake finalization; Torch prunes immediately (ADR-0047).

`PruneOutcome` reports validated proposed, scheduled, and actually
removed stable IDs separately. Executed sleep events expose it on both
backends; NumPy external proposals return one and NumPy wake results
identify delayed finalization. A pending-ID getter derives current
active marks. Invalid pending mask/TTL combinations are rejected before
training or full-state restore. Existing `pruned_indices` still names
selected tensor positions (ADR-0048).

`sleep_telemetry.py` defines the P3.10a version-one event contract. Frozen
records distinguish stable-ID proposals from applied changes, scheduled
prunes from removals, and core time from the guarded attempt's time. They
validate finite chemistry/guard values, counts, widths, outcomes, and JSON
serialization (ADR-0087). NumPy and Torch `SleepEventResult.telemetry` now
supply model-owned facts for executed and skipped events; NumPy counts exact
replay exposure and Torch records zero replay. Measured core duration is
excluded from deterministic result equality and model snapshots. Torch
captures child identity before its post-split prune (ADR-0088/0089).
The Torch head exposes read-only chemistry summaries for a runner epoch that
does not call core sleep; this supports truthful skipped records without an
app-layer read of private chemical arrays (ADR-0096).
Fixed-feature failed attempts can carry a partial guard-batch exposure count
even when no complete pre-score exists; a complete two-pass decision still
requires both scores and a valid selected-metric delta (ADR-0097).
Unmatched-vision runner logging remains P3.10c. An
accuracy-only guard may leave both cross-entropy scores absent rather than
fabricating measurements; an arrived guard can carry its SHA-256 role hash.
For an aborted attempt, unknown pre/post accuracy and delta remain absent;
the event outcome is `error` and applied work is zero (ADR-0093).

`replay_retention.py` provides pure FIFO and seeded bottom-k eviction and a
typed `ReplayExposureSnapshot`. Only explicitly selected new policies add
observed/duplicate content IDs, applied replay IDs, and update counts to a
NumPy model snapshot. This audit can grow beyond the retained-array byte
cap; it never chooses replay, guard, or final-test data (ADR-0102/0103).

`shared_replay_schedule.py` holds distinct float64 labeled train rows under
the existing example/byte budget and FIFO or seeded bottom-k policy. It
selects newest retained rows without model predictions and returns private
copies for each future trainer. The public retained-ID snapshot is sorted;
the separate retained-order list drives sampling. This module does not
train models, decide sleep timing, or open decision roles (ADR-0106).
The NumPy circadian core exposes read-only retained-order and unprioritized
selected-ID previews for an app preflight; these do not sample afresh or
change replay state (ADR-0107).

The P4.3a isolated fixture and `docs/replay-side-effect-audit.md`
distinguish direct replay mutations from the sleep wrapper. Historical
replay advances chemistry, importance, traffic, reward baseline,
cooldowns, and pending-prune TTL; age, wake clocks, history, and retained
rows stay fixed. `CircadianPredictiveCodingNetwork` also accepts the
pretraining-only `wake_only_adaptive_v1` side-effect policy. Replay then
updates weights and exposed train IDs using the pre-row adaptive gate,
while wake adaptive state and pruning progress stay fixed. The policy is
part of opt-in snapshots and does not change historical snapshot fields.
This core layer does not choose evaluation roles or write ablation results
(ADR-0109).

`difficulty_diagnostics.py` accepts current feedforward binary
probabilities and observed train labels, returning mean absolute error,
the fixed 0.5-clipped error, and BCE. It has no data-source dependency and
does not scale a model update, inspect held-out roles, or replace the
model's relaxed-state supervised-error signal (ADR-0112).

`run_manifest.py` validates required versioned run fields, complete
status, Git-known versus explicitly unavailable source identity,
seed/role hashes, finite resolved config, output paths/protocols, and
timing scope. It produces stable UTF-8/LF manifest bytes. It accepts
facts from app/infra and does not inspect Git, train, or write files
(ADR-0120).
