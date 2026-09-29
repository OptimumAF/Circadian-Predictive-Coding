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

## Non-Responsibilities

`TrustedLocalReplayPolicyCheckpointStore` writes the separate format-9 v8
policy cursor with a distinct magic header and checksum. App validation
owns manifest, role, model, and replay provenance; this adapter only reads
or atomically replaces a trusted local pickle file (ADR-0104).

- Training orchestration
- Model internals
- CLI concerns
