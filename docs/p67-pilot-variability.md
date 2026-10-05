# Complete pilot variability and the P6.7 sample-size gap

Date: 2026-10-05. P6.7d1 is complete for the retrospective scope below. Original P6.7 remains open.

## Original acceptance and finding

P6.7 requires roughly ten confirmation seeds when feasible, **pilot variability
to justify final sample size**, exploratory labeling where uncertainty remains
large, and no favorable stopping. The original scope, ADR-0152, P6.7a planning
log and analysis ADR-0156 reserve ten and declare costs/intervals. They do not
record a paired pilot SD/precision calculation or variance-based count decision.
The complete accepted P6.3 audit proves staging and reservation chronology.
It cannot prove that missing requirement. Original P6.7 remains unchecked.

The original ten-seed confirmation is informative evidence with unchanged
metrics and uncertainty. This new report is **retrospective**. It cannot be
relabeled as a justification recorded before the old final roles were released.

## Complete scope and arithmetic

Preserve all six original development families, 168 cells, 174 seed pairs,
58 named contrasts and two existing primary metrics: final mean task accuracy
and signed forgetting A. This gives **116 vectors / 348 pilot observations**.
Gating's stored `signed_forgetting` name is retained alongside the declared
`signed_forgetting_a` name. Every three-seed vector retains ordered raw values,
source pointers and original difference provenance. No new metric is selected.

For a complete vector, use the original sample SD, with divisor pilot n−1.
At the existing count N=10, conditional SE is `pilot_SD / sqrt(10)`.
Half-width forecasts multiply this SE by unchanged critical values
2.2621571627982053 and 5.403490569214909 from the original df-nine,
116-statement contract. They are conditional precision forecasts and have
no interval endpoints or coverage/power/adequacy claim. They are not empirical
confirmation SDs. Future dispersion is assumed equal to observed pilot
dispersion for this calculation alone.

Why this: pairing preserves within-seed covariance, and changing the sign or
mean cannot improve projected precision. Keep all contrasts and families
separate. Repeated runs add no seeds; gating/replay share sources. Complete
nonconstant pilots are required for forecasts. Missing/insufficient/constant
vectors retain null forecasts and reasons; raw SD remains visible, including
roundoff below the original 1e-12 tolerance. Zero observed variation is not
proof of equivalence or justification for a small sample. Three seeds estimate
variance weakly, and development outer-selection roles differ from independent
final roles. These limitations prevent a prospective count conclusion here.

## Boundaries and usage

```text
src/core/pilot_precision.py                  pure complete-vector SE projection
src/app/continual_pilot_variability.py        complete original pilot consumer
tests/test_pilot_precision.py                known arithmetic/null/type behavior
tests/test_continual_pilot_variability.py     all-scope fabricated input gates
```

The public `build_pilot_variability_report(original_development_inputs)` first
calls the unchanged `build_development_ledger`, validating whole scope/catalog,
all twelve development/eight preflight bundles and every stored metric/role/pair.
It binds the whole inputs and ledger, original manifest and analysis contract.
It owns no filesystem, source/model/train/scoring/final access or experiment
launch. IO authority must come from the complete existing development reader.
The private projection seam accepts fabricated ledgers for clean tests and
grants no original/current scientific authority.

```python
from src.core.pilot_precision import project_pilot_precision
from src.core.seed_statistics import SeedObservation

projection = project_pilot_precision(
    (SeedObservation(41, -0.1), SeedObservation(43, 0.0), SeedObservation(59, 0.1)),
    (41, 43, 59),
    confirmation_seed_count=10,
    zero_deviation_tolerance=1e-12,
)
# Pilot SD .1; conditional SE .1/sqrt(10). No precision target or count decision.
```

## Prospective local correctness and evidence gates

Before tests/derivation, freeze new source/test/contract bytes, original
150-source/79-test/174-input maps, fourteen original whole files, complete prior
current-reader proofs/failures and pre-edit documents. Each metadata/fixture/
saved-input operation has a **hard 180-second** supervisor limit. The unchanged
complete development reader retains its original **120-second** gate and all
twelve development/eight direct preflight/eight canonical-reference dispatches.
No original scientific budget is changed.

Run the new fixtures plus related seed-statistics/development tests; check
Ruff, format and mypy. Then dispatch the complete development reader once,
compare its entire ledger with the accepted whole saved ledger, derive every
vector twice and require exact exclusive JSON output equality. Arm all 24
original source/model/train/predict/final guards. Record commands/outcomes,
zero skips, elapsed limits, full identities and unchanged original pins.
The completed evidence and any failure remain local under `artifacts/runs/`.
Occupied producer output paths must not be rerun or overwritten.

## Exact next action after d1

P6.7d2 must declare a substantive precision objective and rationale independently
of old confirmation outcomes. Account for small-pilot variance uncertainty,
discrete/constant observations and development/final role differences. Justify
a fixed final count, new independent untouched source/seed/final roles, complete
resource envelope, original informative/matched controls and prospective
analysis/stopping/isolation gates before any new confirmation source or score.
Keep d2/d3 and original P6.7 unchecked until their own complete evidence exists.
No seed selection, favorable stopping, narrowed contrasts, baseline tuning or
post-exhaustion cap increase can repair this gap. Later streams/vision remain
unfinished. ADR-0173 records the decision and alternatives.


## Verified d1 evidence

All **43 new / 98 related tests pass, zero skipped**. Ruff, four-file format,
mypy520 and diff checks pass. The complete unchanged development reader
validates twelve development/eight direct preflight/eight canonical-reference
bundles and reproduces the entire accepted ledger in4.0747185s
under its original120-second gate. It is dispatched once; both pure report
derivations retain all116 vectors/348 observations and bind the full43,831,409-byte
ledger at SHA1e58fc0d...72f87. The full inputs also remain bound.

| Family | Primary vectors | Conditional forecasts | Zero-dispersion forecasts suppressed | Conditional SE range |
|---|---:|---:|---:|---:|
|gating|2|1|1|0.007607–0.007607|
|replay|6|5|1|0.003804–0.020127|
|sleep|6|2|4|0.007607–0.015215|
|schedule|18|8|10|0.045644–0.049447|
|combined|44|38|6|0.003804–0.118096|
|parent|40|36|4|0.003804–0.033159|

All SDs/raw values remain, including zeros/roundoff; these conditional ranges
are accuracy fractions and cannot establish prospective sample adequacy.
Both complete317,173-byte reports have SHA-256
`c223c8f245804d71f15f708bca0a138c217ef11047dc314ad4a7584007d1ba03`.
Independent full readback checks every ordered vector and its arithmetic with
standard-library sample SD, including every null forecast; receipt SHA
`213c3bbd4ff37a60bed41269fc89650928608f192531ca58eae1df4a2db95787`.

Actual audit worker69.8029607/parent69.9639461s
is below the remaining179.8718265-second cap. Including the failed0.1281735s
path check, both audit attempts total70.0921196s under shared180 seconds.
Entry metadata55.3847741s also fits its180 gate. All150 original sources,
79 prior tests,174 inputs, fourteen whole originals, prior reader/failure proofs
and document snapshots remain; all24 actual scientific guards are zero.
No scientific confirmation reader or new experiment is dispatched.

### Preserved failures

First fixtures97pass/1fail/0skip because the sentinel named nonexistent
source helpers. First mypy exits2 because a `tests.` fixture import duplicated
module names. The corrected fixture follows existing direct-import conventions,
uses actual `_build_phase_a_roles`/`_build_phase_b_roles`, and additionally seals
final release. Source, old gate, XML, operators/static output and exact copies
remain in `p67-pilot-first-attempt-preservation.json`. Production modules are
unchanged across this repair; accepted98 tests/static gates bind corrected tests.

The first audit exits1 at the producer identity check before helper import:
a separate v2 operator retained the v1 gate's full relative path. Preserve
`p67-pilot-audit-first-attempt-preservation.json` and exact copies. Worker time
and terminal guard counters were not emitted and remain unobserved. A separate
v3 freezes the corrected gate path; it retains passing runtime/test gates and
all original scientific ports/criteria. No original producer is overwritten.

```powershell
.\.venv\Scripts\python.exe artifacts/runs/run-p67-pilot-gate-v2.py tests
.\.venv\Scripts\python.exe artifacts/runs/check-p67-pilot-static-v2.py
.\.venv\Scripts\python.exe artifacts/runs/run-p67-pilot-gate-v3.py audit
```

These successful local producer outputs are occupied. Clean fixture commands
remain runnable without ignored artifacts. Full CPU suite/GPU/clean-clone/
cross-version and new scientific runs were skipped for this pure consumer;
the43 new fixtures and all98 related cases have no runtime skips. Closing
preservation/diff results are recorded in the current development log.
Only d1 closes; d2/d3 and original P6.7 retain their complete acceptance.
