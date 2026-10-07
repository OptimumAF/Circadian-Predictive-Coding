# P6.7c2: Complete independent confirmation and deterministic repeat

Completed: 2026-10-01. **P6.7c2 and P6.7c are complete.** The subsequent
P6.11b report is also complete; original scientific/resource/reporting parents
retain their own audits. The complete c1 correctness gates and their evidence were
recorded before either actual reserved final execution.

## Fixed scope and outcome

Both real public scored processes and both independent complete readbacks
exited 0. All **560 cells / 580 original pairs / 1,680 predictions / 67,200
independent final examples** are present. All 560 cells and 1,680 endpoints
succeeded; there are zero numerical failures. Each process releases exactly
120 views, reads 120 input and 120 label fields and preserves complete held
training facts/state and every retained content/endpoint/observation link.
All original arms, seeds, duplicate/inactive/negative rows and cost distinctions
remain visible in the full saved results. No favorable early stopping, seed
selection, metric/interval/baseline/setting/cap change or new algorithm.

| Family | Cells per seed | Seeds | Cells | Original pairs per seed |
| --- | ---: | ---: | ---: | ---: |
| Gating | 3 | 10 | 30 | 1 |
| Replay | 8 | 10 | 80 | 3 |
| Sleep | 9 | 10 | 90 | 3 |
| Schedule | 11 | 10 | 110 | 9 |
| Combined | 17 | 10 | 170 | 22 |
| Parent | 8 | 10 | 80 | 20 |
| Total | 56 | 60 family/seed rows | 560 | 58 |

There are **50 distinct source seeds** because gating/replay share ten IDs.
The deterministic repeat does not create additional seed replications. Final
roles contain forty examples per phase. No family pooling or example-level
replication is introduced.

Both entire canonical scientific result files are **byte-for-byte equal**:
**1,116,254 bytes**, SHA-256
`2fae14cc615b2f2bf50716ea3e930514551283b662cded84befab91e39f698a7`.
The fixed pure analysis verifies all 58 contrasts/580 pairs/116 primary
statements and repeats exactly at SHA
`9796965cef1b5e1b3604d6ba69f84d05d6e6cc1e1b7d70ec298c91ba90f47b5c`,
1,885,930 encoded bytes. This additional pairing check does not complete
P6.11b's publication of every seed/interval with joined raw cost facts.

## Observed work and resources

Both actual optimizer observers independently count **15,210** attempted and
executed updates: **13,440 wake + 1,724 applied replay + 46 rejected-executed
replay**. BP/PC/CPC+parent totals are **3,708 / 3,948 / 7,554**. Both match
the complete bound raw training facts, including rolled-back work. Each has
770 guard attempts, 1,540 guard evaluations, 28,320 guard examples, 46,080
retained array bytes before copies and maximum transient width fourteen.

| Observation | Canonical | Repeat |
| --- | ---: | ---: |
| Child elapsed seconds | 68.5221003 | 68.2845622 |
| Claimed parent lifecycle seconds | 71.0737686 | 70.6688824 |
| Start process RSS bytes | 47,513,600 | 47,288,320 |
| Observed peak process RSS bytes | 234,356,736 | 234,295,296 |
| RSS samples at 5 ms | 45,855 | 45,838 |

Original **16,000-update / 600-s / 5-ms sampled absolute 512-MiB** caps pass
unchanged. Sampling covers binding, training, held copies, global proofs,
final evaluation, scientific serialization and exit checks. Stdout framing
and parent publication remain outside the original child RSS scope. Parent
lifecycle time starts with request publication after initial complete-reference
preflight; it is not the entire CLI elapsed time. No per-arm wall/RSS/FLOPs are
invented. P6.10 still owns the broader accuracy/resource presentation.

## Complete local artifacts and provenance

Official ignored directories are `artifacts/runs/p67-confirmation-scored` and
`artifacts/runs/p67-confirmation-scored-repeat`. Neither has failure or claim.

| Bundle/file | SHA-256 | Bytes |
| --- | --- | ---: |
| Canonical request | `a89efeff39467c1d4c55a45707ee5f700242c26dbfd47dc100912202921e1334` | 183,577 |
| Canonical result | `2fae14cc615b2f2bf50716ea3e930514551283b662cded84befab91e39f698a7` | 1,116,254 |
| Canonical audit | `b53bfea96cdcf8c903311f94ac714fe4f2b4d19fa8b0775cfbd0f2d257fcd01f` | 1,039,853 |
| Repeat request | `c34829231ca2d1c85d0211865d920c3747a3d4ccf1121bf9d90f7ad098bf609a` | 183,584 |
| Repeat result | `2fae14cc615b2f2bf50716ea3e930514551283b662cded84befab91e39f698a7` | 1,116,254 |
| Repeat audit | `76688d9eff2018dda2918192f6b7aeb8efc82c806d75d94abdf5d4a53a71c36d` | 1,039,852 |

Each parent freshly reads both complete training bundles and publishes its
exact request before training. Every current source/command/environment,
scope/reference/analysis identity and late artifact link passes in both
workers and public readbacks. Full scored closure stays at **97 sources**,
map `e0cfd897867786271974bcc971b647084f7fcf34b28e8db8a140a2d8c3f7ff8c`,
on authoritative V2 record
`0dc9b7fd3c67d5ba3059324645e7c731f7153795a0ded4eb7c5fb731bbb3bf9c`.
Scientific manifest `76cf873e...3a4223`, analysis `5e33ef28...6b594b1`, all
prior 92 pins and all six original train file bytes remain unchanged.

Additional ignored repetition validation record:
`artifacts/runs/p67-confirmation-scored-repetition-validation.json`, **63,687
bytes**, SHA
`f255d78b10e5315d2b23fec2693a2a446c28be336630877698df82e2d1df9d89`.
It stores exact artifacts, counts, resources, full work vectors, pair scope and
completed c2/c IDs; it explicitly leaves the statistical seed report unfinished.
Its saved validation producer SHA is
`93dda4b7d0188a511bdd9cc7758fa1c84590d8cdfc19959ab58960e6d1ac953e`.

## Commands and next action

The following four commands all exited 0, after the 984-test c1 gate. Existing
official outputs are preserved; their execute commands now correctly refuse
occupied paths. Use the read-only commands to verify them:

```powershell
python -m scripts.run_p67_confirmation_scoring --execute --output-dir artifacts/runs/p67-confirmation-scored
python -m scripts.run_p67_confirmation_scoring --execute --output-dir artifacts/runs/p67-confirmation-scored-repeat
python -m scripts.run_p67_confirmation_scoring --read-only --output-dir artifacts/runs/p67-confirmation-scored
python -m scripts.run_p67_confirmation_scoring --read-only --output-dir artifacts/runs/p67-confirmation-scored-repeat
```

The subsequent [complete original cost join](p611-confirmation-cost-join.md)
passes P6.11b1 with exact repeated cost metadata and four bounded actual
publication/readback commands. All scientific sources and files are preserved.

The subsequent [complete P6.11b2 report](p611-confirmation-report.md) passes
both full publications and independent complete readbacks with byte-identical
JSON/Markdown. Every seed/interval/raw-cost fact, all 626 vectors and all 116
primary statements remain, with no new training/final source access or tuning.
The original scored files and this historical validation record are unchanged.

**Exact next action:** audit original P6.11/P6.9/P6.10/P6.12 acceptance and
implement P6.9a's explicit stage/task matrix over the complete verified report.
Preserve every missing/null/negative/inactive/undefined value and the frozen
three-endpoint contract. Original parents retain their separate acceptance.
