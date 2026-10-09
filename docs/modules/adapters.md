# Module: `src/adapters`

## Responsibilities

- Parse command-line arguments
- Invoke app-level use cases
- Print user-facing experiment summaries
- Expose a dedicated CLI for ResNet-50 speed benchmarking
- In `scripts/run_continual_arrived_confirmation.py`, prepare an immutable
  local v7 study request and invoke ordinary/checkpointed app use cases
  under fixed correctness and work checks
- In `scripts/profile_cifar_representative_feasibility.py`, prepare a
  fixed CUDA development-only cost request and run one bounded child
  feature extraction; `scripts/prepare_cifar_representative_study.py`
  saves the later matched-budget request without training;
  `scripts/run_cifar_representative_selection.py` checks its frozen digests,
  quiet-device gate, and six development-only candidates and writes a
  result, synced attempt journal, manifest, or failure artifact
- `scripts/restore_cifar_representative_selection.py` verifies exact
  result/journal/manifest bytes and source hashes without dataset access;
  it returns a typed frozen manifest plus the audited local evidence
- `scripts/run_cifar_representative_confirmation.py` enforces the restored
  freeze, quiet CUDA gate, separate scope/total time limits, and local
  attempt/seed/result/failure persistence; `scripts/audit_cifar_representative_confirmation.py`
  checks complete typed scope reports and paired identity/work/memory evidence
- `scripts/run_structural_rank_comparison.py` runs the fixed v12 train-only
  gate or globally sealed comparison and writes deterministic JSON to an
  exclusive new local path; it does not implement factor training or
  release final roles
- `scripts/run_sleep_trigger_comparison.py` writes fixed v13 train-only
  or globally sealed outcome JSON to an exclusive new local path; the
  app owns trigger decisions and final-role release
- `scripts/run_continual_trigger_replay_schedule.py` writes fixed v14
  train-only replay opportunities to an exclusive new local JSON path;
  it serializes potential rows/work without model training or scoring
- `scripts/run_continual_trigger_replay_training.py` writes fixed v14
  guarded train-only work and typed decision facts for all six trials to
  an exclusive new local JSON path, omitting durations and final scores
- `scripts/run_continual_trigger_replay_outcomes.py` writes all globally
  sealed v14 scored rows and paired contrasts to an exclusive local
  JSON path without implementing training, role release, or selection
- `scripts/run_versioned_v14_bundle.py` runs the fixed v14 study once,
  serializes train-only bytes before the global final gate, and writes
  or verifies an opt-in local P5.1 bundle. Its opt-in resumable mode
  validates an unscored format-10 prefix before further training or
  final release. It does not choose a model or alter v14 metrics
  (ADRs 0120, 0124). It declares the sole `fixed-v14` preset and rejects
  unknown settings before training (ADR-0126).
- `scripts/project_v14_observations.py` projects or verifies a completed
  P5.1 bundle in an exclusive local directory. It does not train,
  score, or choose a result (ADR-0121).
- `scripts/build_v14_artifact_report.py` publishes or verifies one
  source-checked descriptive JSON/CSV table beside a completed P5.1
  bundle. It does not train, choose a winner, or edit the dashboard
  (ADR-0137).
- `scripts/build_v14_dashboard.py` publishes or verifies one static
  dashboard derived from a checked P5.6a table. It does not train or
  modify the historical dashboard (ADR-0138).
- Opt-in `--capture-wake-diagnostics` on `run_versioned_v14_bundle.py`
  and `--run-measured` / `--verify-measured-run` on
  `project_v14_observations.py` publish genuine train-update metrics
  only after the completed global scored bundle exists (ADR-0122).
- `scripts/run_continual_shift_benchmark.py` parses typed repeatable
  `--override` values for its existing named profiles and optionally
  writes an exclusive fully resolved config record. The app validates
  fields before the adapter invokes training (ADR-0125).
- `scripts/run_multiseed_resnet_benchmark.py` retains its descriptive
  unmatched flags and output names, parses repeatable JSON settings
  after them, checks the app's typed config before training, and embeds
  complete per-seed settings in the JSON result (ADR-0127).
- `predictive_coding_experiment.py` delegates to `src/adapters/cli.py`.
  The adapter keeps the toy baseline/indepth flags, parses typed JSON
  overrides, rejects invalid requests before training, and saves a full
  exclusive resolved-config record or an enriched baseline JSON result
  after completion (ADR-0128).
- `src/adapters/toy_budget_cli.py` handles opt-in toy baseline update,
  time, replay, circadian hidden-width, and process-RSS limits. It claims a local
  lifecycle path, binds resume to the exact resolved config and trusted
  checkpoint hash/cursor, records observed and
  durable wake, replay, and width work plus per-invocation absolute process-RSS
  start/peak/sample facts for completed/incomplete/error attempts,
  and publishes a completed JSON only after final scoring. It does not
  select settings, change training rules, or open final roles (ADRs 0133–0136).
- `resnet50_benchmark.py` delegates to `src/adapters/resnet_benchmark_cli.py`.
  All old defaults come from a typed preset, existing flags remain, and
  repeatable JSON overrides are checked before the Torch runner. The
  adapter writes an exclusive full config or completed result JSON and
  retains the original stdout report (ADR-0129).

## Inputs / Outputs

- Inputs: process arguments and optional environment-backed defaults
- Outputs: console output; the confirmation script also writes an ignored
  request, trusted checkpoint files, and a complete JSON result
- The CIFAR probe and study-preparation scripts write ignored request,
  measured result or failure, and frozen future-study JSON artifacts
- The bounded CIFAR confirmation writes ignored gate, fixed-data journal,
  per-seed wall-time/memory, final result, or failure artifacts

## Non-Responsibilities

- Dataset generation
- Model training internals
- General-purpose persistence or network IO

The confirmation script is a local experiment adapter. Its explicit file
IO is confined to request/result artifacts and the existing trusted
checkpoint store; it does not implement model training or selection rules.

`local_pilot_cli.py` maps resource flags to the pure first-pilot policy and presents
JSON/status codes. `scripts/run_local_pilot_preflight.py` delegates to it. Inputs
are planned resources/context; outputs are stdout and exit status only. There is
no training/download/output-file dispatch or measured enforcement in this adapter.


`numpy_learners.py` wraps owned copies of existing CPC/backprop models and implements
the core native learner port. It preserves diagnostic IDs and feedforward outputs,
reuses CPC model snapshots and adds full ordinary-state graph copy/validated restore.
No dataset/final-role release, disk checkpoint, shared loss or actor promotion.
The app composes the port with existing wake-boundary budgets; see docs/learner-ports.md.


## 2026-10-07 — R3.6b1 erasure prerequisites (validation pending)

[Payload erasure primitives](../data-erasure.md): core data_erasure.py provides tombstones/counts/optional ReplayPayloadOwner port;InboxCursor format2 and private ExperienceInbox erasure preserve consumed IDs/applied work;NumPy adapters expose native whole-buffer erasure. Only raw replay references are removed;native weights/RNG/policy/counters and exposure hashes remain. Format1 unerased histories remain supported. No new dependencies/environment variables. Full original-authority deletion,quotas/lifetimes/checkpoint/promotion cleanup/non-resurrection is unfinished R3.6b2.


R3.6b1 acceptance:488 passing cases,current656-file win/linux types/static/AST/guide/source/resource gates. Evidence:artifacts/runs/r36b1-erasure-primitives-20261007/validation.json and erasure-summary.md. Full original R3.6/R3.6b/R3.6b2 remains unchecked. Exact next action: R3.6b2: prospectively scope original-authority coordinated deletion. Inspect all owners of payload copies: live/retired candidate and inbox, pending/inspected checkpoints and prepared/failed models, serving promotion/rollback bundles and pending tickets. Define owned-versus-caller copies and bounded supported native/payload measurement ports; implement deletion/expiry/opt-out coordination plus declared record/byte/time policies under original manager/candidate/serving/checkpoint leases. Tests first for partial cleanup failure and refused resurrection; retain budgets/clocks/gates/consumed IDs and weights. Keep transient/audit-only admission disabled until purge semantics pass, and original R3.6/R3.6b unchecked until the complete criteria are proven.


### Managed cleanup integration — 2026-10-07

numpy_learners.make_managed_data_lifecycle:outer composition of exact numeric-array byte measurement and exact NumPy learner footprint/erasure ports;the app has no adapter dependency. Inputs/outputs and non-responsibilities: [managed lifecycle guide](../managed-data-lifecycle.md). No native learning-equation/dependency/environment-variable change. Aggregate retained bytes,automatic elapsed-time purge,caller-copy deletion and unlearning remain unimplemented;full R3.6b2b is unchecked.


### Owned payload copy byte policy — 2026-10-07

NumPy exact graph/checkpoint/growth/cache measurement and ManagedNumpyBuilder source-bound preparation ports compose outside app;opaque graphs/builders denied before copying. Inputs/outputs/non-responsibilities and extension guidance:[retained payload guide](../retained-payload-budget.md). No dependency/environment-variable change. Parameters/temporary/caller/Python/RSS memory,unlearning and automatic elapsed purge are outside this increment;full R3.6b2b unchecked.


`backprop_checkpoint_codec.py`: complete exact BackpropSnapshot bytes with
canonical finite float64 frames and the two native legacy aliases. Rejects
unknown fields/additional shared memory before serialization,preflights wire/
array budgets and detaches decoded arrays. It reconstructs neither learner
policy nor live runtime authority. See docs/durable-checkpoint-codecs.md.


`numpy_checkpoint_frames.py` validates explicit finite f8/i8/i4/canonical bool
frames and detached logical array bytes. `circadian_checkpoint_schema.py` freezes
complete native/config/optional-policy fields and validates scalar/topology/
lineage/replay relations. `circadian_checkpoint_codec.py` composes complete
current native bytes through CheckpointCodec with independent bindings and
aggregate preflight. No IO,arbitrary object loader,algorithm update or runtime
restore authority. Structure/example/commands:docs/durable-checkpoint-codecs.md.


### Circadian checkpoint C/F layout qualification

CPC codec wire v2 (`circadian_full_v2`) records exact C/F array order and rejects
old v1,unknown/missing order and unsupported noncontiguous storage. Native
snapshotv2/API/equations remain unchanged. Separate34layout controls compare
fixed native structural continuation across8variants. Frame validation,
native schema and byte orchestration retain their existing inward boundaries;
no dependency/configuration change. See docs/durable-checkpoint-codecs.md and
ADR-0225 for tree,commands,limits,ownership and extension details. Backprop layout
qualification and full composite/native/live recovery remain unfinished.


### Explicit Backprop layout codec qualification

`LayoutBackpropCheckpointCodec` adds separately selected `backprop_layout_v2`
wire frames preserving supported C/F array storage,canonical bytes and native
required aliases. The existing v1 codec remains available for its original
noncontiguous logical-value scope. Exact schema/shared C/F frames depend inward
on existing native contracts and typed codec ports. Every bounded payload is
validated before detached NumPy materialization. Fixed native continuation uses
existing Backprop BCE loss/state/predictions;no model metric/API/equation or
configuration/dependency change. Newmodule tree,examples,commands,limits and
extension path:docs/durable-checkpoint-codecs.md;decision:ADR-0226. Full durable
composite/owner/copy/lifecycle/model/scientific acceptance remains unfinished.


### Complete supported NumPy inbox byte component

NumpyInboxCheckpointCodec wirev2 preserves complete native cursorv1/v2 events,
permissions,unmatched/consumed/applied/tombstone history and actual supported real
numeric C/F payload dtype/byteorder/bytes/ownership. Inner InboxCodecPolicy binds
original independent shape/dtype/record/string/candidate bounds. Role/permission/
metadata checks precede payload inspection;all raw validation precedes detached
NumPy materialization. Existing native/inbox/source/test/API bytes retained;
no dependency/configuration/learning change. Tree/commands/zero-native example/
extension and authority limits:docs/durable-checkpoint-codecs.md andADR0227.
Lifecycle/consolidation/consent/revocation/retention/copy/actor/sharing/Torch/full
composite/singleowner/native/model/coordinatorloss/scientific/human work stays open.


### ConsolidationCheckpointCodec

Implements CheckpointCodec[ConsolidationCursor] with explicit consolidation_full_v1
JSON schema. Original policy/source/content/authority references are independent
inputs. Exact nested fields,metadata counts,UTF8 bytes and canonical aggregate
wire fragments validate;unknown/missing/corrupt/unsupported records refuse.
Shared diagnostic references preserve original first occurrence and exact type/
bits. Outputs are detached observations;no IO,native call or live restore.
Example,commands and extension:docs/consolidation-cursors.md.


### Lifecycle contract before a byte adapter

The complete lifecycle contract lives in core/app. No lifecycle byte adapter
is accepted yet. Future explicit encoding must preserve all current metadata
and independently bound original authority references after actual coherent
capture passes;it must not serialize live callbacks/locks/tokens as renewed
authority. See docs/managed-lifecycle-state.md and original e4/e4b criteria.

lifecycle_checkpoint_schema/codec:explicit complete lifecycle_full_v1 envelope,17record schemas,immutable record/tuple aliases and48slot original reference manifest. Bound bytes,digests,policies,fields,counts,UTF8 and all relationships before typed construction;retain original supplied authority tuple. No clock/measurement/native/IO/restore.

Common-owner paired record capture currently has no byte adapter. Existing lifecycle/consolidation codec bytes and schemas remain unchanged. R3.5b2e4c2 must preserve full paired fields,aliases and independently expected owner/source/policy/content/authority bindings before decoding.

managed_record_checkpoint_schema/codec:complete managed_record_pair_v1 with24explicit native records,13runtime observations,59original reference paths and all immutable metadata aliases. Raw bounded policy/ID/version/stop/epoch/charge/enrollment/alias checks and entire independent original observation comparison precede materialization. lifecycle walkers retain existing defaults;trusted internal descriptors never come from wire input.
