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
