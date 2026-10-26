# Bounded paired CLM TRAIN replay

The existing temporal runner accepts `--paired-replay-candidate`. It is off by
default and requires the original paired all-head `none` recipe. Catalog training,
support recipes/comparison, memory, probes, typed serving and extra diagnostics
are refused together with this option before loading assets.

```powershell
.venv/Scripts/python.exe -B scripts/run_clm_temporal_sequence.py --model-dir <prepared-model-dir> --head <prepared-head> --output <new-report.json> --paired-replay-candidate
```

Use prepared pinned local assets and a separately frozen finite native protocol.
The software qualification does not grant a pretrained run or renew old budgets.

## Behavior

The ordinary twelve-update source stream gives TRAIN reuse consent at creation.
Its existing inbox observer retains two actual completed observations, labels and
receipts: the first B packet and the first returned-A packet, both under current
v2 policy. Their original clocks, identities and permissions are preserved.
After source completion, one separate candidate copies warm head/scale values,
clears gradients and creates a fresh native AdamW/OneCycle owner. It alternates
the two fixed records for six updates/eighteen query-row exposures at tick23.

The original paired group-masked symmetric contrastive objective, frozen encoder
identity, projection dimension and native settings remain unchanged. This is a
new candidate with declared finite work, not source optimizer continuation. It
does not install, export or promote weights. The source trajectory and its work
stay sealed; the child report is `paired_replay_candidate`.

One raw four-query/three-candidate development readout before and after replay
recomputes head projections. No development labels select records, dose or
stopping. Raw frozen encoder features can be reused because the encoder remains
fixed; head projections cannot be cached across updates. This path scores no
child labels. A separate frozen study must interpret the saved probabilities
and report acquisition, retention, return and adverse current-rule errors.

## Interfaces and boundaries

```text
src/adapters/clm_replay_candidate.py  separate native TRAIN state and source guard
src/app/clm_replay_candidate.py       causal record admission, inbox and paid replay
src/app/clm_temporal_sequence.py      existing delayed stream; optional native observer
scripts/run_clm_temporal_sequence.py  opt-in composition and fixed replay recipe
tests/test_clm_replay_candidate.py    ownership, loss parity, consent and failure cases
tests/test_clm_replay_stream.py       actual source-to-candidate/default/CLI behavior
```

The app accepts a candidate factory; it imports no backend or adapter. A caller
using `CLMReplayRecord` directly must supply authentic original source records,
receipts and policy-version semantics. Structural checks do not certify an
external caller's provenance. The CLI obtains those records from the actual
inbox observer, without rebuilding them from JSON.

Update and replay budgets are separate existing `ToyBudgetSession` objects.
Admitted replay rows are charged before the native call. An uncertain failure
retains actual work and forbids retry; it cannot renew a source or child budget.
The native adapter retains one bounded source snapshot, not a success chain.
Returned captures and copied RNG elements are reported, with fixture-wide
copy/reserve, RSS, storage and elapsed limits in the qualification protocol.
Those counts are scoped work measures, not total allocator traffic or FLOPs.

Why this: actual runtime receipt reuse supplies a useful end-to-end replay slice
through existing components while keeping the benchmark/deployed learner and
disposable probes separate. Fresh optimizer authority makes the added work
explicit; comparing this child to a source with fewer updates is not matched
computation or a circadian advantage.

## Checks and extension

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_clm_replay_candidate.py tests/test_clm_replay_stream.py
.venv/Scripts/python.exe -m ruff check src/adapters/clm_replay_candidate.py src/app/clm_replay_candidate.py src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_replay_candidate.py tests/test_clm_replay_stream.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/clm_replay_candidate.py src/app/clm_replay_candidate.py src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_replay_candidate.py tests/test_clm_replay_stream.py
.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent src/adapters/clm_replay_candidate.py src/app/clm_replay_candidate.py src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_replay_candidate.py tests/test_clm_replay_stream.py
```

These commands execute model work; the checked-in qualification receipts record
the bounded executed cohorts. Freeze a new scientific protocol before running
prepared pretrained assets. Extend via this same stream and existing native
learner, not a new orchestration framework. Full R5.3 and native comparative,
reliable adaptation and release requirements remain open in GitHub #20/#24.

## Native comparison status

The first finite native equal-dose comparison stopped after its ordinary source
fit and before either candidate. A TorchVersion/JSON-string metadata mismatch
caused refusal despite exact saved source values. A field-specific scalar repair
passes focused checks, but the native case was not retried and its comparison is
unanswered. The aggregate static time gate also failed120s (121.87s observed).
Preserve both failures; see [commands, costs and remaining work](../artifacts/runs/clm-paired-replay-comparison-20261022/RESULTS.md)
and [issue #34](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/34).
