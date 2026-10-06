# ADR-0196: Separate portable boundary fixtures from historical results

Date: 2026-10-06. Status: accepted for R0.3 engineering tests.

## Context

Three Linux combined-factor tests failed before reaching their intended
checkpoint/scoring assertions. They read an ignored historical Windows result
when available and otherwise produced a reference at that original output path.
A complete Linux comparison found1849 differing parameter/state hashes and14
floating fields differing by at most1.11e-16. Manifest, role identity, counts,
decisions and executed work match; cross-environment bitwise equality does not.

## Decision

Generate a test-only current-environment train-fact reference in memory for the
combined-factor boundary tests. Cache its construction within the test process
and return detached copies to each corruption case. Keep exact train-fact and
complete-checkpoint comparisons, all existing corruption cases, phase-arrival
sentinels, evaluation counts and parity assertions unchanged. Historical files,
hashes, protocols and production code remain unchanged. This fixture does not
establish historical reproduction, provenance, source admission or empirical gain.

## Alternatives

Skipping the tests would lose boundary coverage. Tolerant production hash checks
would weaken the scientific contract. Regenerating the retained historical
result would overwrite original evidence. None is justified.

## Consequences

Portable correctness tests can reach their intended boundaries on each platform.
Historical reproduction remains a separate, environment-bound investigation.
Other tests with the same pattern require their own observed failure before this
correction is extended. R0.3 still requires complete supported baseline coverage.

## Observed extension during R0.3c

The complete Linux inventory also found two schedule-factor failures and one
sleep-factor failure with the same cross-environment reference dependency.
Apply the same cached, detached in-memory fixture to those boundary tests.
Neither historical result paths nor production comparisons change.

Two v14 tests confused portable exact before/after byte invariance with recorded
Windows CPython 3.14.7 / NumPy 2.4.6 SHA identity. Preserve both criteria in
separate tests: compare exact training and scored bytes within each environment
and bind actual payload digests in portable manifest tests; keep the original
SHA literals in explicit recorded-environment reproduction tests. Unsupported
environments report those two reproduction tests as skipped. This follows
docs/reproducibility-scope.md; it grants no cross-build scientific equivalence.

Six marshal controls failed because pytest assertion rewriting temporarily
retained a constant inside sys.getrefcount. Bare controls on CPython 3.11,
3.12 and 3.14 showed exact 2/3/2 counts and complete serialization restoration;
3.12 with plain assertions passed all 18 controls. Capture counts into integer
scalars before rewritten assertions, retaining exact counts and full raw-byte,
recursive native-field and mutation checks. The runtime observer is unchanged.

## Complete-inventory extension

The complete corrected Linux inventory exposes additional v14 bundle-chain,
measured-resume, checked-resume, projection and measured-sidecar fingerprint
assumptions. Preserve exact fresh/local/resumed/derived byte checks on every
platform and move each historical hash criterion to a separately marked recorded
environment test. Original hash literals remain unchanged, including wake metrics.

Parent-factor CLI fixtures changed file pins but retained a historical internal
result pin. Bind both identities to the isolated fixture in the parent and real
worker. Sleep/schedule/combined CLI tests likewise now generate isolated local
preflight references, select their exact result pins, and inherit them into real
worker children using a test-only bootstrap. All production physical source pins,
reference readers, complete audit checks, corruption criteria, work/scoring counts,
memory limits and duplicate-output refusal stay unchanged. These tests exercise
portable engineering boundaries and supply no frozen scientific admission. No
fixture construction writes an ignored historical reference directory.
