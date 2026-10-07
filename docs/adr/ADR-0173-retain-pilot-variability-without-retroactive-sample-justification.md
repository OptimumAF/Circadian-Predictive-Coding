# ADR-0173: Retain pilot variability without retroactive sample justification

Date: 2026-10-05. Status: accepted for P6.7d1 implementation.

## Context

Original P6.7 requires pilot variability to justify final sample size. The
original scope, ADR-0152 and P6.7a log reserve ten independent seeds per
family and scale resource ceilings. ADR-0156 freezes intervals for that
count. None supplies a paired pilot variability calculation or variance-based
count decision. Later confirmation and reporting gates passed, but those
facts do not fill the original prospective justification gap.

## Decision

Leave original P6.7 unchecked and preserve its complete criterion. Add d1 to
retain all 116 original paired primary pilot vectors, three ordered seeds
each, through the unchanged complete development reader. Reuse the pure
seed summary for observed sample SD. Calculate only conditional SE at the
original ten-seed count, and scale it by the two already frozen df-nine
critical values as half-width forecasts. These are retrospective estimates,
not newly measured intervals, power calculations, precision guarantees or
evidence that ten was prospectively justified.

Why this: sign or observed mean does not determine precision. Every original
contrast remains, including constant, negative, null and inactive factors.
Forecasts require complete nonconstant pilots; missing/insufficient/constant
vectors retain raw observations, SD where defined and explicit null reasons.
Use the original 1e-12 roundoff tolerance, preserving the exact raw SD.
Three seeds give a weak variance estimate; different development and final
roles do not establish identical future dispersion. Count example rows,
deterministic repeats or shared families as no additional source replications.

Add separate d2/d3 for a substantive prospective precision objective,
small-pilot uncertainty, fixed sample count, untouched independent roles,
complete resource/analysis/isolation gates and bounded confirmation. A
post-outcome choice cannot repair historical chronology. If the budget is
inadequate, preserve an exploratory/infeasible conclusion and unfinished
acceptance rather than choosing favorable seeds or contrasts.

## Alternatives

Checking P6.7 solely from ten reservations weakens its acceptance. Declaring
new precision targets from observed confirmation results makes justification
conditional on final outcomes. Reopening old final roles or retuning baselines
does not supply prospective evidence. A new generic quantile dependency is
unnecessary for conditional scaling at the existing fixed count.

## Consequences

One small pure core module and one app consumer add no IO or scientific
operations. Public input validation reuses the complete original development
ledger; clean fixtures exercise an explicitly private fabricated seam. Local
evidence checks bind original source/test/input/current-reader/failure history
plus the new modules and retain exclusive output paths. No original scientific
file, protocol, seed, metric, interval, baseline, port or cap is changed.
Original P6.7 and future stream/vision requirements remain unfinished.
