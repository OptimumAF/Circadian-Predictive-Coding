# P6.11b1: Complete original confirmation cost join

Completed: 2026-10-01. **P6.11b1 is complete.** The subsequent full
P6.11b2/P6.11b seed/interval report also passes; original parent audits remain.

## Modules and boundary

```text
src/app/continual_confirmation_report_costs.py
  Pure whole-result binding, raw cost projection and work verification.
src/infra/continual_confirmation_report_cost_references.py
  Sequential complete-reader composition and repeated cost/audit equality.
scripts/inspect_p611_confirmation_costs.py
  Current source closure, exclusive inspection publication and full readback.
tests/test_continual_confirmation_report_costs.py
tests/test_continual_confirmation_report_cost_references.py
tests/test_p611_confirmation_cost_inspection.py
```

The adapter supplies the unchanged `read_completed_bundle` training reader.
Infrastructure depends on app; app has no IO or infrastructure import. Existing
97 scored-source pins are preserved and three consumer modules are added.
See [ADR-0162](adr/ADR-0162-preserve-original-cost-facts-before-attaching-confirmation-scores.md).

## Preserved facts and interpretation

- All 560 original family/seed/arm keys and 60 shared seed contexts.
- Every original raw method field, including nulls, inactive events, rejected
  events, inference/presentation counters, replay and guard facts.
- Initial, after-A and after-B widths/parameter counts, exact model types and
  state/parameter fingerprints. Raw method facts retain peak capacity/history.
- Separate wake, applied-replay and rejected-executed-replay counts. Their sum
  is executed optimizer work; rolled-back work still consumed execution.
- Complete shared context, supplemental guards and original training-role links.
- Independently verified group storage before copies and original whole-run
  wall/RSS observations. No per-arm time, RSS or FLOP estimate is created.

The public projection requires SHA-256
`3d85c60627de63769d0f0fc0bf5ec781c77d50468673dab466d8bbe28089e547`
and 134,554,378 original encoded bytes before and after projection. Streaming
encoding matches the existing ASCII pretty JSON plus LF. It avoids a second
whole encoded string. Mutable nested cost facts are copied.

Both unchanged complete readers independently validate original artifacts,
requests, work, resources and source identities. Their complete reference
report must reproduce `cc1c1deb4c721af5d8250f17783c9daada2501c108be91f825fa37324626b001`.
The projections must equal each other and the independently derived audit work.
Original file/marker checks surround both readbacks; current consumer sources
are checked before, after reading and after publication/readback.

### Observed artifact size

The first two actual publications reproduce 112,635,395 bytes, including a
103,705,403-byte standalone projection. A compact-encoding profile found
1,238,185 bytes in all cell costs and 52,503,109 bytes in shared seed context.
Schedule supplemental guards carry full proof states (3,725,191 compact bytes
in the first row); combined/parent opportunity context also carries original
proofs. Thus the prospective assumption of small shared metadata was incorrect.
The original held-checkpoint maps are omitted, but these raw proof contexts are
preserved literally. Both observed publication commands took approximately
74 seconds, within the declared 180-second local validation timeout.

Why this: preserving the whole original context gives the next report an exact
cost provenance reference now. P6.11b2 can publish all compact per-arm costs
and reference this complete shared-context artifact by byte identity, avoiding
another duplicated proof body while preserving every original cost distinction.
No fields or acceptance criteria were removed in response to this measurement.

## Local usage

```powershell
.\.venv\Scripts\python.exe -m scripts.inspect_p611_confirmation_costs --result-file artifacts/runs/p611-confirmation-costs.json
.\.venv\Scripts\python.exe -m scripts.inspect_p611_confirmation_costs --result-file artifacts/runs/p611-confirmation-costs.json --read-only
```

Publication is exclusive. Occupied outputs fail before entering readers.
Readback reconstructs the entire expected metadata from both complete original
bundles and checks exact canonical output bytes, including source identities.
Failure propagates; a partial or stale inspection cannot pass this readback.
This metadata has `statistical_seed_report_complete=false` and
`scored_bundles_validated=false`. It authorizes no scientific execution.

## Validation and next action

All **55 new/314 related tests pass in 32.45 seconds**, zero skipped. Ruff
across src/tests/scripts, format checks on all six new files, mypy across
428 source files and `git diff --check` pass. Tests cover exact serialization,
all six genuine development families after sealing model/data calls, raw field
and checkpoint preservation, rejected-executed work, detached mutable input,
partial/ambiguous/invalid costs, full source closure and late input/output/source
drift, exclusive collisions, malformed/duplicate/nonfinite JSON and dry CLI.

The actual bounded validation harness runs **two publications and two independent
full readbacks**, sequentially, with a 180-second timeout per child and raising
data/model/train/final sentinels. All four exit 0 with zero forbidden accesses;
parent elapsed times are 73.7242413, 74.3912447, 75.2698046 and 75.3261709
seconds. This is a validation budget, not a replacement for any scientific cap.
Both published complete metadata files are byte-for-byte equal. Each retains
560 unique complete cost keys, 60 contexts and 1,680 checkpoint cost rows.
Work is 13,440 wake + 1,724 applied replay + 46 rejected-executed replay =
15,210 actual updates; shared retained array bytes before copies remain 46,080.

| Local artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `artifacts/runs/p611-confirmation-cost-source.json` | 14,115 | `8400b7c38335199c36f3dd2839c810cb268a4a2bb8fac7a9a3b320fabe83c101` |
| `artifacts/runs/p611-confirmation-costs.json` | 112,635,395 | `214ce7ad00f61ca1aac8f85ad1381f50e9fc4235baeec59d7daf6043cc7d81dd` |
| `artifacts/runs/p611-confirmation-costs-repeat.json` | 112,635,395 | `214ce7ad00f61ca1aac8f85ad1381f50e9fc4235baeec59d7daf6043cc7d81dd` |
| Standalone complete cost projection | 103,705,403 | `42bbe9049ddd863cf8daa7cca0dc4d62bd26ba9f1be5f7ae05be300107e27868` |
| `artifacts/runs/p611-confirmation-cost-validation.json` | 15,682 | `eac94f8e95144d52e389cacf61ad1351ce5b71c07eadb972cfdc8b2cec889e37` |

The 100-source map is
`adc4da3da4a96bd6a9b941137862d34efcccce8751713c916f60fc328b01410f`.
New production identities are app `7881c74a462550f407b16e8f2ccbf761f0faa0ed81a46bee14f3d2d099164298`
(7,819 bytes), infra `7c92d58e1e9c5a168a22afa85e2bfdfdf8ae98ab36f5264c1d121c6bc1a26133`
(3,631 bytes) and CLI `3c3add12dd9de8b5226d4ba7c70dffa4b8b5185a7107d53a860dfd97f8a6b73e`
(4,554 bytes). All original 97 source pins, six training artifact identities
and fixed science/analysis settings are unchanged. Validation metadata binds
the three test files and local harness as well. No new dependency, environment
variable, training, final-source access or scientific resource claim.

```powershell
.\.venv\Scripts\python.exe artifacts/runs/validate-p611-confirmation-costs.py
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests/test_continual_confirmation_report_costs.py tests/test_continual_confirmation_report_cost_references.py tests/test_p611_confirmation_cost_inspection.py tests/test_continual_confirmation_work_validation.py tests/test_continual_confirmation_training_references.py tests/test_p67_scoring_training_reference_inspection.py tests/test_continual_confirmation_validation.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/app/continual_confirmation_report_costs.py src/infra/continual_confirmation_report_cost_references.py scripts/inspect_p611_confirmation_costs.py tests/test_continual_confirmation_report_costs.py tests/test_continual_confirmation_report_cost_references.py tests/test_p611_confirmation_cost_inspection.py
.\.venv\Scripts\python.exe -m mypy
git diff --check
```

The complete harness deliberately refuses occupied publication paths. Reuse
the public `--read-only` command above to verify existing evidence; preserve
all current outputs if a new bounded validation is required.

The subsequent [complete b2 report](p611-confirmation-report.md) joins all
560 raw cost cells, links every complete shared context by exact identity,
and publishes all 626 vectors/116 primary statements. Both actual publications
and both independent complete readbacks pass with whole JSON/Markdown byte
equality and no scientific source access. B2/b are complete; this original
cost evidence and all scientific pins remain unchanged. Next: audit original
parent criteria and implement P6.9a's explicit stage/task matrix presentation.
