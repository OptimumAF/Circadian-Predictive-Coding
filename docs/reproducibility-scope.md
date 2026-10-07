# Reproducibility scope for the fixed v14 study

The fixed v14 result is a NumPy, binary synthetic continual study on CPU.
Its `fixed-v14` preset seals seeds 47/53, model order, data roles, arms,
metrics, and learning settings. A different model order is a different
protocol request and is rejected before training. The two fresh P5.7 runs
below repeat this preset; they do not retune it or select a favorable seed.

## Verified same-environment repeat

On 2026-09-29, two separate processes wrote ignored local bundles
`artifacts/runs/p57-repro-a/` and `artifacts/runs/p57-repro-b/`. Both pass
the completed-bundle verifier. They recorded commit
`c17a37d792a6f26728a557c5ecc613d528d0f9ad`, the same dirty-workspace
digest `719547e5b317ada43f20019750bcd324999f0afd6f2982f9dd6c4909c14b4a68`,
Python 3.14.7, NumPy 2.4.6, Windows 11, an Intel CPU with 20 logical
cores, and
`compute_device: cpu`. Resolved configuration, seed map, source-role
hashes, protocol/algorithm IDs, precision, and environment facts match.
The only different manifest field is `run_id`, so the manifest file hashes
differ by design. Deterministic payload SHA-256 values match exactly:

| File | Both runs' SHA-256 |
|---|---|
| `training.json` | `174ee7941c0b1e2489783f43b0b481db11f402c4ea55001888c7998cfb28b324` |
| `outcomes.json` | `ea11fc7cc0ac8113eec2fc5512bb28044b99d0885627813c92aad80cf0f2501f` |

To repeat on a stable checkout, choose two unused run IDs and keep tracked
files, nonignored untracked files, Python/NumPy installation, and hardware
unchanged between commands:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --run-id repro-local-a --preset fixed-v14
.\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --verify-run artifacts/runs/repro-local-a
.\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --run-id repro-local-b --preset fixed-v14
.\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --verify-run artifacts/runs/repro-local-b
Get-FileHash artifacts/runs/repro-local-a/training.json,artifacts/runs/repro-local-b/training.json,artifacts/runs/repro-local-a/outcomes.json,artifacts/runs/repro-local-b/outcomes.json -Algorithm SHA256
```

The verifier checks each source's role/cell/protocol/hash facts. The
P5.7 regression test also compares the full manifests excluding `run_id`
and the exact payload bytes. This same-environment check applies **zero
byte tolerance** to deterministic training and outcome files. Measured
sleep durations, wall time, and memory use are outside that byte gate.

## Model-order check

The corrected configurable v6 continual runner accepts a model-order
permutation, so a fixed seed-17, 40-sample, two-epoch-per-phase fixture
trains forward and reversed orders. Its regression compares all three
models' full serialized states after A and B, role hashes, guard decisions,
and sleep/replay facts with measured durations removed. The existing v6
two-seed test also checks final metrics and replay retention. The exact
state and outcome tolerance for this **same-environment v6 test is zero**.
Order-bearing audit event order is expected to change. This test validates
the shared corrected continual execution path; it is not an alternate
fixed-v14 result or a GPU comparison.

## Portability and tolerance policy

| Comparison | Accepted tolerance and status |
|---|---|
| Same recorded source, config, Python/NumPy build, and CPU environment | Exact deterministic payload bytes and seed/role/config facts; only declared run identity may differ. This passed for the two P5.7 runs above. |
| Different CPU, Python/NumPy build, or version | Protocol, algorithm, seed, config, and role identity must be checked first. If role hashes differ, the data are not matched. No cross-build score tolerance has been validated; report every per-seed metric and state delta, and mark equivalence unverified until a tolerance is predeclared and measured on paired controls. |
| CPU versus GPU for fixed v14 | Not applicable: there is no Torch/CUDA v14 replay route. No numeric or bitwise equivalence tolerance is defined. |
| Future Torch CPU/GPU or cross-version track | Use its own versioned protocol and matched data, initialization, capacity, work, and precision. Predeclare metric-specific numeric tolerances and deterministic-operation settings before viewing final scores. Until tested, status is unverified, not within tolerance. |
| Timing and peak memory | No byte or zero-difference tolerance; record device, process scope, warmup, synchronization, sample count, and observed distribution separately. Never treat those measurements as deterministic output fields. |

This policy does not allow widening a threshold after seeing a model
ranking. NumPy limits exact random-stream compatibility to strict
same-build/environment conditions; different CPUs and builds can change
floating-point behavior ([NumPy compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html)).
PyTorch likewise does not guarantee identical results across releases or
CPU/GPU execution ([PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.14/notes/randomness.html)).
The project therefore claims the observed local NumPy repeat only. A
future cross-platform or cross-version claim needs its own checked study.
