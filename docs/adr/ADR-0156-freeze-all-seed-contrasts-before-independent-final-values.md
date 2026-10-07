# ADR-0156: Freeze all seed contrasts before independent final values

Date: 2026-09-30. Status: accepted; P6.11a correctness verified before final scoring.

## Context

The complete unscored P6.7b gate now repeats every one of 560 reserved cells.
P6.8 froze metric arithmetic and P6.7a retained 58 ordered pairs across six
families, ten seeds per family. P6.7c requires P6.11 predeclaration before
opening any independent final value. No scientific ranking can close this
contract, and shared seeds/repeated runs cannot increase replication.

## Decision

Preserve every original pair as primary for both mean accuracy and signed
forgetting, with a single 116-statement simultaneous family. Pair by seed
and identical final-role fingerprints, keeping all endpoints, per-seed
values, failures and unequal costs. Use complete ten-seed Student-t marginal
95% intervals and Bonferroni simultaneous 95% intervals with explicit normal
seed-outcome/difference assumptions. Freeze independently checked df-nine
critical values without a new dependency. Do not clip interval endpoints.

Suppress intervals/full planned means on missing/failed vectors; retain
observed-only descriptive summaries and null reasons. Suppress intervals
for constant discrete vectors (SD at most 1e−12 accuracy fractions, retaining
raw numbers to distinguish roundoff from count resolution) and all retention ratios before observing
final outcomes. Record zero observed variance without treating it as proof
of equivalence. There is no pooled family mean, adaptive method switch,
composite winner or new cost metric. Document limitations/interpretation in
`docs/p611-confirmation-analysis.md` before synthetic analysis fixtures.

Pure core summaries and app scope/analysis modules accept validated outcome
values and declared identities; they own no data/model/score/IO. P6.7c must
prove actual source/checkpoint/final provenance; P6.11b must report every
actual seed/pair. This increment cannot establish those scientific gates.

## Alternatives

Unpaired intervals discard within-seed covariance. Treating examples or two
identical runs as replications exaggerates sample size. Selecting a smaller
primary contrast family or changing an interval method after outcomes makes
the scientific declaration conditional on the results. A new general
quantile library is unnecessary for this fixed ten-seed design. Bootstrap
or distribution-free procedures would need their own declared scope and
coverage assumptions; they cannot be selected later for narrower intervals.

## Consequences

All original 580 individual pairs/116 primary statements remain visible;
simultaneous intervals can be wide and the model assumptions may be weak
at ten discrete observations. Unresolved and negative outcomes remain valid.
Synthetic correctness evidence precedes separate final release. Original
scientific sources, metrics, settings, caps and incomplete plan criteria
remain intact.

## Evidence

All 102 new/222 related tests pass in 14.20 s, zero skipped. Known paired
counts and NIST's ten-observation mean/SD example verify interval arithmetic;
60-digit independent tail inversion/integration verifies both fixed df-nine
critical values. Exact all-scope output retains all cells/pairs/116 statements
and seed order; late malformed scope/role/contract/accuracy blocks summaries.
Explicit both-side/all-failed, undefined retention, positive transfer,
constant/roundoff, overflow and no source/model/RNG/file/final-release cases
pass. Strengthened metadata/final sentinels additionally pass 2/.39 s; an
unavailable later full-run stream is not completion evidence. Ruff/five-file
format/mypy 392 pass. The contract SHA is
`5e33ef28862bcdf9d92fe14dd6cf6b71672a2336ffd760a1214ef04666b594b1`;
the contract/log record the three new source byte hashes and unchanged
original references. No actual independent score, new experiment/dependency
or setting/metric/baseline/seed change. P6.11/P6.11b/P6.7c remain unfinished.
