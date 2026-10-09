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

NumPy `sleep_event(max_replay_examples=...)` optionally preflights the
selected replay snapshot batch lengths before any sleep mutation. The
non-negative argument is the caller's *remaining* allowance; exceeding it
raises `SleepReplayLimitExceeded` with the planned and remaining example
counts. Unbudgeted calls and no-replay/skipped sleeps keep their previous
behavior. The caller owns the cumulative limit and checked cursor
(ADR-0134); core does not write run-state artifacts or inspect evaluation
roles.

NumPy `sleep_event(max_hidden_width=...)` and
`apply_neuron_proposals(..., max_hidden_width=...)` optionally check the
current adaptive hidden width and the width after selected splits before
any mutation. The positive limit is absolute for the circadian layer;
`HiddenWidthLimitExceeded` reports the proposed transient width and cap.
Splits still precede prunes, so a final width within the cap is insufficient.
Core does not clamp split selection to make a run fit (ADR-0135).

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

`local_pilot_budget.py` validates typed planned resources against fixed first-pilot
ceilings and local/simulation/CPU context, returning an immutable request or typed
excess error. It uses only the standard library. Why this: budget policy must be
independent of model objectives and request flags. It performs no measurement,
execution, permission change or promotion; see `docs/local-pilot-budget.md`.


`learner_ports.py` defines a generic native learner port and finite, named training
diagnostic. Inputs, predictions and snapshots are opaque separate type parameters;
no shared loss, array layout, IO, budget clock or promotion belongs here. Outer
adapters implement the port and retain native objectives (ADR-0197).


`experience.py` defines local immutable sample/episode/actor/candidate/action/reward
metadata, source and label arrival ticks, declared train/replay/evaluate flags and
applied native diagnostics. `LogicalClock` supplies explicit monotonic integer
ticks. Inputs/targets remain opaque; validation concerns identifiers/times/roles,
not payload layout, physical provenance, IO or scientific release (ADR-0199).


actor_ports.py defines an owned-fork extension of the existing native port,
versioned prediction/state/candidate records and detached consolidation result/
receipt records. Inputs and state remain native generic payloads; it owns no
threads, clocks, budgets, IO, promotion or scientific release. Fork implementors
prove full model ownership and preserved policy (ADR-0200).


promotion_guard.py defines immutable guard metadata, prospective utility/
retention/numerical/latency/resource/action policy, measured scalar evidence,
deterministic rejection and report bindings. Inputs/predictions remain opaque;
no model operation, IO, serving mutation or scientific release belongs here.
Utility definitions remain separate from native training diagnostics (ADR-0201).


## Serving transaction records

`src/core/serving_ports.py` defines positive TTL/entry configuration, owned cache/snapshot/frame values and local promotion handles/receipts. Inputs are declared configuration and app observations; outputs contain no native learner handle. No IO, model mutation, final-label release or publication belongs in core. See docs/serving-promotion.md.


## Resource sharing records

`src/core/resource_sharing.py` validates declared serving/work/poll bounds and defines detached admission, cumulative priority observations and native update polls. No model/clock/IO/sampler/scheduler/persistence or timing statistic belongs here. These observations do not authorize checkpoint resume or quota renewal.


## Complete inbox cursor

`src/core/inbox_cursor.py` validates exact-format metadata for full source/label/applied histories and capacity/arrival/stopped/work observations. Payloads stay opaque; validation does not copy or score them. Outputs are domain records, not model/time/budget/ownership restore capabilities. No IO, native model or outer app dependency belongs in this module.


### Actual live serving measurement

The generic app native observer/shared-request harness and pure core timing/
nearest-rank/overlap records are documented in docs/live-serving-measurement.md
and ADR-0206. The reserved script boundary executes the finite declared matched
two-native protocol after full correctness/source binding. Retain all raw requests,
native windows and incomplete worker status; no automatic repeat/outcome filtering.
Current measurement acceptance is pending; no scientific advantage is claimed.

R3.5c/original R3.5 current acceptance:396 cases,full651-file platform types/static/source/resource gates and one reserved actual native serving run pass;96/96 shared requests per method fully native-contained. Negative circadian p95 slowdown retained. Evidence:artifacts/runs/r35c-live-serving-20261007/measurement-summary.md. Next R3.6 privacy/replay lifecycle;durable R3.5b2 and broader guards remain open.


## 2026-10-07 — R3.6a permanent data admission (validation pending)

Optional managed local admission: [guide](../managed-experience.md). Pure src/core/data_lifecycle.py metadata and src/app/managed_experience.py original authority install permanent hooks on a fresh ExperienceInbox; legacy uninstalled behavior stays intact. Requires declared training/replay consent and permissions, bounded lifetime grant counts and explicit synthetic/unverified policy. Supported checkpoint handoff retains authority. Opt-out stops future training; native/inbox/checkpoint erasure and unlearning are not claimed. No new environment variables/dependencies. Full R3.6b deletion controls remain the next extension.


R3.6a acceptance:433 passing cases,current654-file Windows/Linux types/static/AST/executed guide/source/resource gates. See artifacts/runs/r36a-data-admission-20261007/admission-summary.md and validation.json. Full R3.6/R3.6b erasure remains unchecked. Exact next action: Prospectively scope R3.6b actual native replay/inbox erasure: inspect replay snapshot identity and full InboxCursor/applied receipt references; design payload-free tombstones preserving consumed IDs, enforce retention lifetimes/quotas, invalidate pending checkpoint payload copies, and test refused resurrection across owned handoff with original clocks/budgets/gates. Start with fake deletion controls, then fixed native controls; do not call erasure parameter unlearning or erase caller-held copies by implication. Preserve original full R3.6 acceptance and durable R3.5b2/R3.7/G3/human-deferred/scientific work.


## 2026-10-07 — R3.6b1 erasure prerequisites (validation pending)

[Payload erasure primitives](../data-erasure.md): core data_erasure.py provides tombstones/counts/optional ReplayPayloadOwner port;InboxCursor format2 and private ExperienceInbox erasure preserve consumed IDs/applied work;NumPy adapters expose native whole-buffer erasure. Only raw replay references are removed;native weights/RNG/policy/counters and exposure hashes remain. Format1 unerased histories remain supported. No new dependencies/environment variables. Full original-authority deletion,quotas/lifetimes/checkpoint/promotion cleanup/non-resurrection is unfinished R3.6b2.


R3.6b1 acceptance:488 passing cases,current656-file win/linux types/static/AST/guide/source/resource gates. Evidence:artifacts/runs/r36b1-erasure-primitives-20261007/validation.json and erasure-summary.md. Full original R3.6/R3.6b/R3.6b2 remains unchecked. Exact next action: R3.6b2: prospectively scope original-authority coordinated deletion. Inspect all owners of payload copies: live/retired candidate and inbox, pending/inspected checkpoints and prepared/failed models, serving promotion/rollback bundles and pending tickets. Define owned-versus-caller copies and bounded supported native/payload measurement ports; implement deletion/expiry/opt-out coordination plus declared record/byte/time policies under original manager/candidate/serving/checkpoint leases. Tests first for partial cleanup failure and refused resurrection; retain budgets/clocks/gates/consumed IDs and weights. Keep transient/audit-only admission disabled until purge semantics pass, and original R3.6/R3.6b unchecked until the complete criteria are proven.


## 2026-10-07 — R3.6b2a retained-copy ownership (validation pending)

[Retained payload ownership](../payload-ownership.md):core metadata/reference ports and app weak registry enumerate supported actor/candidate/checkpoint/promotion owners under nonblocking all-holder quiescence. One opt-in live/lifetime registration allowance survives handoff and GC without renewal. Existing native/budget/serving semantics remain;no new dependencies/environment variables. This is a cleanup prerequisite,not complete deletion or byte/time policy;R3.6b2b retains original-authority all-copy cleanup/non-resurrection acceptance.


R3.6b2a acceptance:512 current tests,full659-file win/linux types/static/AST/guide/source/resource gates;zero NEW native work. Evidence:artifacts/runs/r36b2a-payload-ownership-20261007/validation.json and ownership-summary.md. Full original R3.6/R3.6b/R3.6b2 remains unchecked. Exact next action: R3.6b2b: prospectively scope original-authority coordinated cleanup and retention. Bind current ownership registry and manager plus native replay/inbox tombstone ports; use all-holder references under the existing nonblocking lease, deduplicate by identity, acquire the original sharing/consent authority, and avoid public methods that reacquire leased locks. Define supported native/payload byte measurement and cumulative holder/record/byte/time policies; tests first for pending checkpoint/prepared/failed/retired/promotion/rollback copies, partial cleanup failure, stopped/retired inbox ledger accounting and refused resurrection. Integrate deletion/expiry/opt-out without resetting budgets/clocks/gates/IDs or parameters; keep transient/audit-only admission disabled until actual purge semantics pass. Preserve full original R3.6/R3.6b/R3.6b2 criteria and caller-copy/RAM/unlearning limits.


### Managed cleanup integration — 2026-10-07

data_retention:immutable validated retention policy and payload-free cleanup report;data_erasure replay footprint port observes native array counts without copying. Inputs/outputs and non-responsibilities: [managed lifecycle guide](../managed-data-lifecycle.md). No native learning-equation/dependency/environment-variable change. Aggregate retained bytes,automatic elapsed-time purge,caller-copy deletion and unlearning remain unimplemented;full R3.6b2b is unchecked.


### Owned payload copy byte policy — 2026-10-07

payload_bytes:validated immutable PayloadCopyLimits/PayloadByteSnapshot;data_retention accepts an optional typed owned copy allowance. Inputs/outputs/non-responsibilities and extension guidance:[retained payload guide](../retained-payload-budget.md). No dependency/environment-variable change. Parameters/temporary/caller/Python/RSS memory,unlearning and automatic elapsed purge are outside this increment;full R3.6b2b unchecked.


### Elapsed retention driver — 2026-10-07

retention_driver:typed bounded timing/poll/cleanup observations;data_retention optionally declares elapsed seconds.No IO/threads/payloads. Inputs/outputs/non-responsibilities,tree and commands:[retention expiry guide](../retention-expiry.md). No dependency/environment change. Scalar-only auxiliary expiry remains unfinished;full R3.6 parents unchecked. Clock/OS attestation,caller-copy deletion,RAM/unlearning,durable restart are excluded.


### Owned auxiliary age correction — 2026-10-07

The existing DataRetentionPolicy and immutable driver/copy limits remain unchanged;numeric bytes and content age have distinct meanings. Inputs/outputs/limits/commands:[auxiliary retention guide](../auxiliary-retention.md). No new interface/environment/dependency. Full R3.6 audit passes;durable/R3.7/R3.8/G3 and caller/RAM/unlearning/scientific limits remain separate.


### Recovery admission metadata

`recovery_admission` validates exact bounded saved records against independently trusted fencing/clock/resource facts,without payload IO/copies/native work. Its output is an accounting observation,no restore/training capability. Original limits,spent reservations and downtime remain represented;authentic OS/coordinator adapters and crash integration are unfinished. See [guide](../recovery-admission.md) and ADR-0215.


### Windows recovery observations — 2026-10-07

[Observation guide](../windows-recovery-observation.md) documents core bounded records/port and infra documented API/registered-handle/anchored-clock adapters. Typed observations convey no owner fence or native restore capability. Fake/current-process gates pass;real launcher/payload mismatch retained,corrected worker capture unrun. R3.5b2b/full durable/coordinator-loss recovery remains open. No inherited module changes.


### Direct-worker observation successor — 2026-10-07

R3.5b2b scoped validation now passes strict actual worker identity/time/exit/RSS/cleanup under original coordinator anchor,with unchanged production/test source and280current cases. Prior failed launcher scope/costs retained. No interfaces/dependencies change;transactional independent authority/CAS/live lease/native codec/recovery remain unfinished. See [guide](../windows-recovery-observation.md) and successor evidence.


### recovery_authority

Pure exact AuthorityRecord/AuthorityChange,monotone reservations/conditional handoff,independent high-water validation and RecoveryAuthorityPort. Inputs:original trusted coordinator records and host observations. Outputs:validated metadata changes;no IO,OS handles,live capability or native completion. Original manifest/caps/counters remain bound;uncertain work stops. See [guide](../recovery-authority-journal.md).


### recovery_coordination

Exact bounded RecoveryCosts,closable RecoveryRegistration protocol and host relationship validation. Inputs:independently known authority record,trusted observations and cumulative coordinator clock/RSS floor;outputs:validated sequencing relationships. No IO,process fact authentication,completion receipt or live lease. See [guide](../recovery-coordinator.md).


### recovery_reporting

Exact AuthorityReport,RecoveryReportingPort,FailureWitness and bounded terminal state relationships. Inputs:independent original records/validated attempts and host observations or unavailable facts;outputs:monotone time/RSS/stop reports,never completion/work/cap renewal. Terminal-only RSS overshoot preserves negative evidence. No IO,source authentication,live lease or restart admission. See [guide](../recovery-terminal-authority.md).


## Recovery publication guard port

`recovery_publication.py` defines `RecoveryPublicationPort` and pure original
authority/fresh observation validation. Input:independent exact record and trusted
registered host facts. Output:validated relationship/guard context contract. No IO,
authentication,native completion,callback preemption or side-effect undo. See
../recovery-publication-guard.md and ADR-0220.


## Leased publication reports

RecoveryLeasedPublicationPort extends reporting with a scoped publication lease;
RecoveryPublicationLease exposes read and nonterminal observation reports only.
No work admission/completion/refunds/reopening/IO or process authentication.
Legacy transaction-guard port remains distinct. See ../guarded-recovery-coordinator.md.


`checkpoint_codec.py`: `CodecBinding`,`CodecLimits`,`CheckpointCodec` define
independently supplied source/policy/content comparisons and bounded byte ports.
No filesystem,untrusted object loader,model restoration or ownership authority.
See docs/durable-checkpoint-codecs.md and ADR-0223.


### consolidation_cursor and consolidation_codec_policy

Pure complete immutable ten-field ledger validation and original two-field
attempt/string codec bounds. Inputs:exact typed metadata;outputs:validated records
or ValueError. All failed/consumed IDs,ordered successful receipt/diagnostic fields
and owner flags/counters stay explicit. Native finite diagnostic values/types
are preserved. No copying,clock,model,IO or restore authority.


### managed_lifecycle_state and managed_lifecycle_validation

Complete immutable catalog/lifecycle/driver/registry/copy records and named
original authority references. Validate exact native nested schemas,independent
record/UTF8 bounds,original policies/counters/epochs/aliases and port presence.
No callbacks,clock reads,locks,IO or live restore. Opaque reference equality is
never invoked. Full schemas and explicit absence semantics are documented in
docs/managed-lifecycle-state.md.

managed_lifecycle_validation.require_lifecycle_capture_limits exposes the existing strict independent bound validator for preflight capture. Complete driver original reference contract adds _state_gate;current67source fields/48references. Core still performs no locking,IO,port calls or native work.

lifecycle_codec_policy:complete independent owner/retention/optional driver policies and record/string bounds. Validates original typed native policy rules without wire data,ports or IO;does not authorize restore or budget renewal.

managed_record_state:RuntimeRecordObservation,ManagedRecordMetadata,ManagedRecordCapture and pure joint bounded validation. Complete cursor/lifecycle fields retained;checks matching revisions,flags,enrollment,consumed budget and original authority aliases. Does not attest capture provenance or authorize restore.

managed_record_codec_policy:complete independent lifecycle/consolidation policies;exact native schemas and matching component UTF8 capacities. No wire construction,live reference,port,IO,clock or restore authorization.
