# Versioned local run manifest (P5.1)

`circadian_run_manifest_v1` is the first complete-run provenance contract.
The first producer is the fixed NumPy v14 full-stack trigger comparison.
Its declared typed preset is `fixed-v14`. Passing `--preset fixed-v14`
explicitly has the same result as omitting it. Unknown preset names and
setting switches fail before training; changing v14 settings requires a
new study identity.
It writes `artifacts/runs/<run-id>/manifest.json` after `training.json` and
`outcomes.json`, which retain the existing v14 protocol bytes. The local
directory is ignored by Git. A run ID is a lowercase letter or digit
followed by at most 63 lowercase letters, digits, or hyphens.

## Required fields

| Field | Meaning and validation |
|---|---|
| `schema_id`, `run_id`, `status` | Exact schema version, safe local slug, and one of completed/incomplete/failed/canceled. The v14 writer publishes only completed runs after both files are present. |
| `protocol_versions`, `algorithm_versions` | Source/training/outcome IDs and the three existing NumPy learning-rule IDs. File protocol IDs must agree. |
| `source` | Git commit, dirty boolean, SHA-256 of porcelain status, SHA-256 of the tracked binary diff, SHA-256 of the commit/diff/untracked-content workspace record, and untracked file count. A source snapshot without Git sets all of these to `null` and states why. |
| `resolved_config`, `config_sha256` | Full fixed v14 manifest and its existing digest. The two result payloads must agree with both. |
| `seed_map` | Explicit per-seed A/B source, role split, B exposure, three model initialization, and circadian-local RNG seeds. |
| `dataset_role_names`, `dataset_split_hashes` | Train, inner guard, outer selection, and final-test content hashes for both arrived phases of every seed. The same hashes must occur in every arm and in the saved payloads. |
| `pretrained_weights` | `not_applicable` with a reason for synthetic seeded NumPy models. `known` requires a SHA-256; `unavailable` requires an explanation. |
| `dependency_versions`, `hardware`, `precision` | Execution Python and NumPy versions; OS, machine, CPU model or explicit unavailable reason, logical CPU count, CPU device; and float64 input/parameter/metric precision. |
| `determinism`, `timing_scope` | Local NumPy RNG policy, observed `PYTHONHASHSEED` or `null`, same-environment scope, and explicit absence of timing measurements. |
| `files` | Relative JSON basenames, SHA-256 values, and protocol IDs. The verifier rejects missing, changed, partial, or mismatched v14 outputs. |

The workspace digest includes the HEAD commit, tracked binary diff, and
content hashes of nonignored untracked files. It does not include ignored
files, installed package contents, or submodule worktrees. The synthetic
v14 source does not read ignored data. This is attribution metadata, not
a signed attestation. A Git-free source is visibly unavailable instead
of being assigned an invented commit.

## Local workflow

From the repository root, using the documented NumPy environment:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --run-id p51-v14-local --preset fixed-v14
.\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --verify-run artifacts/runs/p51-v14-local
```

The writer refuses an existing run ID before training. It captures the
workspace before and after training and refuses to publish if source
contents change. It trains the six fixed trials once, serializes their
unscored trace before final release, then runs the existing global final
gate and serializes every scored cell. It stages files outside the
public run ID, writes `manifest.json` last, and publishes the complete
directory with one same-volume rename (ADR-0123);
the verifier treats an absent or invalid manifest as incomplete. The
same fixed v14 output bytes remain reproducible within the recorded
environment. The run ID and dirty source digest can differ between
manifest files without changing experiment scores.

The separate [P5.2a projector](structured-observation-audit.md) now
derives JSONL and final-row CSV from a verified completed bundle; the
P5.1 producer itself still writes only the raw JSON and manifest.
The separate opt-in [P5.3b checked resume route](v14-checked-resume.md)
continues from complete unscored trial checkpoints while retaining this
same public manifest and payload schema. Neither route changes the
`fixed-v14` configuration or claims
cross-platform bitwise reproducibility (P5.7). New tracks should add a producer and validator
for the same required provenance fields or introduce a new schema ID
when the contract changes; existing v14 JSON and checkpoint formats
remain untouched.

An opt-in [P5.2b wake diagnostic sidecar](measured-wake-observations.md)
can now be requested with `--capture-wake-diagnostics`. It is written
after the complete P5.1 bundle verifies and has its own observation
ID, source hashes, and verifier. It does not extend the frozen P5.1
manifest's two-file schema or alter the v14 raw payload bytes.

The separate [P5.6a descriptive report](../README.md) derives a
seed/arm/method table only after this bundle verifies. Its report
manifest binds the exact source and table bytes; fixed v14 files retain
their original names and hashes (ADR-0137).
The separate [P5.6b dashboard](../README.md) reads only that verified
table, publishes static HTML/four PNGs in an exclusive `dashboard-v1`
directory, and verifies exact regeneration against the current report.
The page identifies the completed-bundle failure scope and the fixed
NumPy track. Historical `docs/index.html` retains its chart data and
displays a separate provenance caveat
(ADR-0138).

The [P5.7 reproducibility scope](reproducibility-scope.md) records two
fresh verified fixed-v14 processes with identical training/outcome bytes;
their manifests differ only by run ID. The fixed v14 model order is
sealed. A small corrected configurable continual run tests order
independence separately, and untested device/version comparisons retain
no approved numeric tolerance (ADR-0139).
