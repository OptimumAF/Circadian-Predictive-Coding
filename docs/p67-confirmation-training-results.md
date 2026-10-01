# P6.7b complete unscored confirmation results

Date: 2026-09-30 locally; requests started on 2026-10-01 UTC.
Protocol: `continual_mechanism_confirmation_train_only_v1`.
Base checkout: `86cd5bff71b9c70da94ddcf69d8f62316f2d3382`, with the preserved
development changes and new confirmation components recorded in the log.

## Decision and scope

P6.7b2c3, P6.7b2c, P6.7b2 and P6.7b meet their complete unscored acceptance
criteria. Two separate bounded child processes trained every frozen cell;
both public independent readbacks passed, and the complete deterministic
result files are byte identical. All sixty family-seed instances, fifty
distinct source seeds and 560 cells are retained in original order.

Why this: require a complete source-bound training/copy/validation gate before
opening independent final roles. The run uses the original six protocols,
matched baselines, seeds, settings, metrics and fixed budgets. No seed, arm or
failure was removed, no ranking was inspected, and no cap was increased.
Outer selection and independent final release remain false in both requests
and results. Inner guard evaluation remains the original training behavior.

P6.7, P6.7c, the original matrix and P6.9–P6.12 remain unfinished. These
unscored facts do not establish a benefit or disadvantage for any model.
The next gate is a predeclared seed-level uncertainty/contrast contract under
P6.11, before any independent final value.

## Frozen identities

| Identity | SHA-256 |
|---|---|
| Saved scope record | `622feead54f155521928341151c8496c23c02a54b355b1b3b1e0f68b76c5772f` |
| Full resolved manifest | `8d1ed66b33bbc1bf298cc60604c3741afa22bb4b7e0f6636efe166a52672951b` |
| Execution source map, 78 entries | `b8e6ea624228c7422b61bf48fc736cd187893eedd6ce1fd49558f6b11bbd3735` |
| Separately bound adapter | `e6bafe8bc6c7e2199a578b26ff991a45fbbf4d45085e760fd61d0f9417250445` |
| Complete deterministic result, both runs | `3d85c60627de63769d0f0fc0bf5ec781c77d50468673dab466d8bbe28089e547` |

The 79-file static local dependency closure and twelve historical development
bundles/twenty usage records were independently checked before each request,
by the child and again during readback. No original scientific source pin
changed. The scope/settings/source map are identical in both bundles.

## Complete work, identical in both processes

| Family | Cells | Wake updates | Applied replay | Rejected executed replay | Executed optimizer updates | Guard attempts | Guard evaluations | Guard examples | Peak width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gating | 30 | 720 | 0 | 0 | 720 | 0 | 0 | 0 | 8 |
| Replay | 80 | 1,920 | 360 | 0 | 2,280 | 0 | 0 | 0 | 12 |
| Sleep | 90 | 2,160 | 0 | 0 | 2,160 | 50 | 100 | 2,400 | 12 |
| Schedule | 110 | 2,640 | 342 | 6 | 2,988 | 60 | 120 | 2,160 | 12 |
| Combined | 170 | 4,080 | 1,022 | 40 | 5,142 | 480 | 960 | 17,280 | 14 |
| Parent | 80 | 1,920 | 0 | 0 | 1,920 | 180 | 360 | 6,480 | 13 |
| Total | **560** | **13,440** | **1,724** | **46** | **15,210** | **770** | **1,540** | **28,320** | **14** |

Actual optimizer observations are independent of restored model clocks.
Both runs record 15,210 attempted and successful calls: backprop 3,708,
ordinary PC 3,948 and circadian/parent-control 7,554. Counts match the strict
validator's independently derived work and model-kind partitions. Rejected
replay remains executed work although its model transaction rolls back.
Observed guard/compute differences are preserved; they are not equal-cost
accuracy comparisons.

Each family retains all ten reserved seeds:

- Gating/replay: 101, 103, 107, 109, 113, 127, 131, 137, 139, 149.
- Sleep: 151, 157, 163, 167, 173, 179, 181, 191, 193, 197.
- Schedule: 199, 211, 223, 227, 229, 233, 239, 241, 251, 257.
- Combined: 277, 281, 283, 293, 307, 311, 313, 317, 331, 337.
- Parent: 359, 367, 373, 379, 383, 389, 397, 401, 409, 419.

Retained labeled-array bytes before copies total 46,080 across all instances:
replay 5,760, schedule 7,680, combined 23,040 and parent 9,600; gating/sleep
retain zero. This field measures retained rows, not total process memory.
Both trained boundaries, complete held state, supplied IDs/roles, guard and
selector witnesses and every derived per-seed cost remain in the results.

## Observed resources and lifecycle

Fixed limits: 16,000 executed updates, 600 child seconds and observed
536,870,912-byte (512-MiB) peak RSS, sampled at 0.005 seconds with explicit
boundary samples. No budget override was used.

| Observation | Canonical | Repeat |
|---|---:|---:|
| Request UTC start | `2026-10-01T05:18:28.371043+00:00` | `2026-10-01T05:22:18.089994+00:00` |
| Child PID | 27,868 | 1,596 |
| Observed worker seconds | 32.50460129999556 | 32.650750799999514 |
| Parent elapsed seconds | 44.24597899999935 | 44.65887279999879 |
| Start RSS bytes | 44,978,176 | 44,769,280 |
| Peak RSS bytes | 462,479,360 | 462,348,288 |
| RSS samples, including boundary samples | 32,315 | 32,319 |
| Executed updates | 15,210 | 15,210 |

Same local CPU environment: Python 3.14.7, NumPy 2.4.6,
`Windows-11-10.0.26300-SP0`, Intel Core i7-12700K; request processor string
`Intel64 Family 6 Model 151 Stepping 2, GenuineIntel`. No CUDA experiment.
Resource/time metadata is kept outside deterministic results and naturally
differs between processes. RSS includes held copies, independent validation
and complete result serialization. Sampling can miss brief peaks; the limit
is observed RSS, not an OS allocation ceiling. Stdout framing and parent
publication remain outside that sampled section as declared before data.

Both invocations exit 0 without failure. Each directory contains exactly
request/result/audit files, with no remaining claim or failure marker.
Both independent readers verify current source/request/result/audit bytes,
all complete raw facts and observed counts/resources against the fixed scope.

## Local artifacts

Each bundle is kept locally under ignored `artifacts/runs/`:

| Directory | File | Bytes | SHA-256 |
|---|---|---:|---|
| `p67-confirmation-train` | `confirmation-train.request.json` | 91,195 | `672ec760c83ce7c423058fecd0bf535ff72734bb4bf9cf1fc4a43891c3edbdc3` |
| `p67-confirmation-train` | `confirmation-train.result.json` | 134,554,378 | `3d85c60627de63769d0f0fc0bf5ec781c77d50468673dab466d8bbe28089e547` |
| `p67-confirmation-train` | `confirmation-train.audit.json` | 35,826 | `ff308e7a4b72e7e5ffa4c182223825d4b6e5f3f5768a7c7abf81c26cade9a0e0` |
| `p67-confirmation-train-repeat` | `confirmation-train.request.json` | 91,202 | `2023443417597607c9a16f4cba4f5ad5fb2777ccad54d5af0783059066944456` |
| `p67-confirmation-train-repeat` | `confirmation-train.result.json` | 134,554,378 | `3d85c60627de63769d0f0fc0bf5ec781c77d50468673dab466d8bbe28089e547` |
| `p67-confirmation-train-repeat` | `confirmation-train.audit.json` | 35,826 | `d1f461e1c21e6d0721102ba13349f6156d625cfa761399e09a0409e83047d01a` |

## Commands and validation

These commands were executed locally and all exited 0. Do not rerun training
into occupied directories; the exclusive boundary preserves existing bytes.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p67_confirmation_training --output-dir artifacts/runs/p67-confirmation-train
.\.venv\Scripts\python.exe -m scripts.run_p67_confirmation_training --output-dir artifacts/runs/p67-confirmation-train-repeat
.\.venv\Scripts\python.exe -m scripts.run_p67_confirmation_training --read-only --output-dir artifacts/runs/p67-confirmation-train
.\.venv\Scripts\python.exe -m scripts.run_p67_confirmation_training --read-only --output-dir artifacts/runs/p67-confirmation-train-repeat
```

An additional direct comparison of both result `read_bytes()` values passes;
each has sixty rows/560 cells and false outer-selection/final-release seals.
Canonical/repeat small audits also have exactly equal `work` and source maps;
the complete result equality includes every deterministic checkpoint/fact.

Before the first reserved builder, **444 related tests passed in 76.81 s,
zero skipped**, including **108 new** boundary/resource/artifact tests.
Ruff, seven-file format and mypy on 387 files pass. The execution contract,
ADR-0155 and development log record exact test commands, reproduced/repair
failures and the distinction between metadata IO fixtures and actual runs.
No production code changed between the final correctness gate and these runs.

Full CPU suite, CUDA, large sweeps, scored confirmation, confidence intervals
and final-role release were skipped in this increment. No commit, push,
dependency change or external publication; earlier dirty work is preserved.

## Exact next action

P6.11a's [predeclared analysis contract](p611-confirmation-analysis.md) and pure
complete-scope/numerical/forbidden-read gate now pass. Next freeze and implement
P6.7c's scored reference/source/request/global checkpoint/final-release
boundary, with late corruption and resource/artifact failures before any
actual final value. Preserve every original pair and failure.
Keep P6.11 unchecked until individual confirmation seeds, paired summaries
and intervals are actually reported. P6.7c must bind this contract and both
saved full train gates before separately frozen scoring of all 560 cells,
580 pairs, 1,680 independent final calls and 67,200 examples. Extend via new
modules/protocols; changes to frozen execution need an explicit amendment.
