# Module: `src/infra`

## Responsibilities

- Generate deterministic synthetic datasets for experiments
- Generate deterministic synthetic vision datasets for ResNet benchmarks
- Persist trusted local fixed-feature, toy, continual, and seeded vision
  checkpoint files
- Build disjoint torchvision development roles without constructing the
  final CIFAR dataset when explicitly requested by a cost probe

## Inputs / Outputs

- Inputs: sample count, noise, seed
- Outputs: train/test split dataclass
- `vision_datasets.py` keeps final-test construction by default. Its
  `include_final_test=False` CIFAR path creates only train, guard, and
  validation datasets/IDs/hashes and returns a raising final loader. It
  does not choose study seeds, budgets, or candidate settings (ADR-0080).
- `datasets.py` can carry a deferred final-test role when the caller sets
  `hash_test=False` and `defer_test_access=True`. It reads source test
  fields only when the caller later validates, hashes, or scores that role;
  it does not control when the synthetic generator allocates test arrays.
- `continual_roles.py` takes one phase's generated training fields, a
  phase/seed identity, declared inner/outer fractions, and an expected
  final-test count. It returns disjoint train/inner-guard/outer-selection
  arrays with stable row IDs and content hashes. The final role has IDs
  and a declared global-freeze release policy but no values/hash until
  `release_final_test` explicitly reads and validates the source fields.
  An optional unique source-row mapping retains original development IDs
  when the app has already capped Phase B exposure before splitting.
  It neither trains models nor records actual runtime release events
  (ADR-0073).
- `circadian_checkpoint_files.py` takes a runner checkpoint and a
  local path, writes a checksummed temporary file then replaces the target,
  and loads a verified payload. Pickle loading is restricted to trusted
  local files; protocol/config/data validation belongs to the app layer
  (ADR-0053, ADR-0055–0056, ADR-0058). Continual format 5 stores
  unscored trained seed records; the store remains unaware of test roles
  and defers their identity checks to the app layer (ADR-0072). A separate
  v6 store/header accepts only `ArrivedRunnerCheckpoint`; the app validates
  completed and active development roles, replay, and event cursor
  (ADR-0075–0076). A distinct format-8 store accepts only
  `ArrivedSelectionCheckpoint`, atomically replacing the whole ordered
  candidate-manifest cursor with completed unscored trials or an embedded
  active v6 transaction. The app validates candidate identity, trial
  content, independent sleep-history provenance, and frozen choice; the
  file layer knows neither selection metrics nor final-test roles
  (ADR-0078, ADR-0095).
- `toy_result_files.py` takes a completed toy report and a new local path,
  delegates to `local_result_json.py`, and refuses to overwrite a file.
- `toy_run_state_files.py` takes finite JSON run-state records and a local
  path. It exclusively creates the first state, prevents concurrent local
  writers with a sidecar lock, and atomically replaces only the exact bytes
  it read or created. It does not train, score, or validate the scientific
  checkpoint identity (ADR-0133).
- `local_result_json.py` takes a completed dataclass report and a new local
  path, encodes NumPy values and typed sleep records as finite JSON for toy
  and historical continual CLI artifacts. It does not train models, choose
  metrics, or open evaluation roles.
- `difficulty_streams.py` generates fixed balanced A/B synthetic
  development rows and independent final rows on property access. The
  app decides when the existing four-role release function may read final
  fields. This source does not train or choose a condition (ADR-0112).
- `trigger_streams.py` generates paired balanced noisy A/B development
  fields for stationary and axis-shift conditions. Its independent final
  fields are generated only when the app releases their existing four-
  role boundary; it does not schedule sleep or score models (ADR-0115).
- `run_environment.py` captures commit/dirty/workspace content digests,
  required Python/NumPy versions, and CPU facts before a versioned run.
  It reports unavailable Git metadata explicitly and does not read
  decision/final roles or write results (ADR-0120).
- `versioned_run_files.py` writes the two fixed v14 JSON payloads once
  and their completed manifest last, then verifies the saved hashes,
  protocols, source roles, method cells, and contrasts. It does not
  train, choose settings, or resume a stopped run (ADR-0120).
- `observation_projection_files.py` reads only completed verified P5.1
  bundles, writes an exclusive hash-bound derived directory, and checks
  exact regeneration from raw seed records. It does not train, score,
  select, or recover partial writes (ADR-0121).
- `v14_artifact_report_files.py` verifies a completed P5.1 source before
  reading it, publishes an exclusive JSON/CSV table with source and output
  hashes, and re-derives exact bytes on verification. It does not train,
  score, rank methods, or infer failures outside the bundle (ADR-0137).
- `v14_dashboard_files.py` first verifies the P5.6a report and rechecks
  exact table hashes before publishing an exclusive `dashboard-v1` static
  directory. Its verifier re-derives the HTML and four PNGs against the
  current report and rejects stale or edited bytes. It does not train,
  choose a result, or overwrite `docs/index.html` (ADR-0138).
- `measured_observation_files.py` verifies the completed P5.1 source,
  binds canonical wake diagnostics in an exclusive sidecar, and
  re-derives a separate measured projection from that sidecar and raw
  results. It neither trains nor independently proves the numeric
  values or recovers interrupted writes (ADR-0122).
- `atomic_artifact_directory.py` stages validated bytes in a hidden
  sibling directory, records incomplete/failed/canceled publication
  state, and publishes only a complete local directory. It does not
  validate experiment semantics, resume training, or erase failed
  stages (ADR-0123).
- `circadian_checkpoint_files.py` also has a separate immutable
  format-10 v14 store. A store-local mapping-proxy reducer serializes
  sealed role maps without changing global pickle behavior; it does
  not validate trial meaning or read final roles (ADR-0124).
- `v14_resume_files.py` writes a hidden atomic run-state cursor, exact
  immutable checkpoint references, and incomplete/failed/canceled/
  completed status under an OS-released local lock. It does not train,
  score, or validate the stored trial's scientific facts (ADR-0124).

## Non-Responsibilities

`TrustedLocalReplayPolicyCheckpointStore` writes the separate format-9 v8
policy cursor with a distinct magic header and checksum. App validation
owns manifest, role, model, and replay provenance; this adapter only reads
or atomically replaces a trusted local pickle file (ADR-0104).

- Training orchestration
- Model internals
- CLI concerns


### Windows recovery observations — 2026-10-07

[Observation guide](../windows-recovery-observation.md) documents core bounded records/port and infra documented API/registered-handle/anchored-clock adapters. Typed observations convey no owner fence or native restore capability. Fake/current-process gates pass;real launcher/payload mismatch retained,corrected worker capture unrun. R3.5b2b/full durable/coordinator-loss recovery remains open. No inherited module changes.


### Direct-worker observation successor — 2026-10-07

R3.5b2b scoped validation now passes strict actual worker identity/time/exit/RSS/cleanup under original coordinator anchor,with unchanged production/test source and280current cases. Prior failed launcher scope/costs retained. No interfaces/dependencies change;transactional independent authority/CAS/live lease/native codec/recovery remain unfinished. See [guide](../windows-recovery-observation.md) and successor evidence.


### recovery_authority_codec and sqlite_recovery_journal

Infrastructure owns exact bounded canonical JSON and private one-record SQLite CAS through the inner authority port. Inputs:independently known floor,trusted host-derived changes and private path. Outputs:typed metadata/read or committed CAS success;stale false,errors disable instance. Exclusive create/existing-only open,DELETE/FULL/immediate/no retry. No native codec,worker lease/dispatch or coordinator-loss bootstrap. See [guide](../recovery-authority-journal.md).


### SQLite terminal reporting/reconciliation

SqliteRecoveryJournal adds report through the existing bounded full-record CAS and explicit reconcile_terminal through a fresh existing-file connection. Exact independent witness states only;preserve actual committed charges/uncertainty and force stopped. Missing/stale/unwitnessed state refuses;no resumed work,automatic retry,migration or new dependency. Source/host authenticity and power-loss/process/native recovery remain unproven. See [guide](../recovery-terminal-authority.md).


## Private SQLite publication reservation

`SqliteRecoveryJournal.publication_guard` holds BEGIN IMMEDIATE around exact
authority comparison,entry/exit host validation and a trusted callback. All
supported writers contend on that private database with zero timeout. It writes
no records;exceptions disable the adapter and close the connection. It cannot
authenticate callers,undo publication or prove native/coordinator-loss recovery.
The coordinator now uses the shared observation-report lease;see ../guarded-recovery-coordinator.md.


## Shared private journal writer ownership

sqlite_recovery_lock.py owns bounded permanent native lock descriptors;
sqlite_recovery_publication.py exposes restricted durable observation reports
under ownership. Ordinary writes,terminal reconciliation and the legacy guard
share that lock;publication reports use the held capability without reacquisition.
Inactive/cross-thread lease use refuses. Trusted canonical private path only;
no hostile disk/alias/remote-filesystem/native/coordinator-loss claim. See
../guarded-recovery-coordinator.md and ADR-0221.


## Retained original Windows composition

windows_recovery_composition.py accepts independent original AuthorityRecord,
existing canonical private journal and exact retained Windows anchor/worker
registrations. It verifies current physical observer/anchor,PID/creation/shared API,
liveness,terminal flags and exact journal;returns the app coordinator over inner
leased ports. Invalid initial record/types retain caller ownership;later failures
deduplicate closure and preserve primary/all close errors. It does not launch/
repin/terminate workers,bootstrap authority,complete native updates or refund
work. Concrete adapters are tested with private fake API hooks;actual worker/crash
proof is separate. See ../windows-recovery-composition.md and ADR-0222.


## Original RSS floor and actual metadata worker capture

WindowsRecoveryObserver accepts a bounded exact initial peak;composition seeds
it from independent known usage,retaining original caps/regression checks. The
actual native/SQLite composition and writer ownership were exercised with one
strict direct worker and pre-COMMIT process exit. Original charges/uncertainty
persisted;owned native process handles/pipes/reader/Popenhandle cleaned. Capture
markers are absent native components,not codecs;hot replay/power-loss/native/model/
coordinator-loss recovery remains unproven. See ../windows-recovery-process-capture.md.
