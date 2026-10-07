# P6.12b3: Complete current findings publication

Status: P6.12b3, b and P6.12 pass their unchanged original acceptance audit.
Prospective freeze, 53 new/304 related tests, static gates, two observer cases
and four complete actual operations pass. ADR-0172 records the IO decision.

## Modules

```text
src/app/continual_findings_readers.py          three complete reader ports
src/config/p612_findings_publication.json      fixed whole IO input declarations
src/infra/continual_findings_current_bindings.py  current sources/inputs/markers
src/infra/continual_findings_current_inputs.py  fresh readers and unchanged synthesis
src/infra/continual_findings_artifacts.py       exclusive publication/fresh readback
scripts/run_p612_complete_findings.py          fixed local CLI composition
tests/findings_publication_fixtures.py          fabricated IO boundary inputs
tests/test_continual_findings_current_*.py      binding/reader/late-drift checks
tests/test_continual_findings_artifacts.py      bytes/ownership/failure/readback checks
tests/test_p612_complete_findings_cli.py        port wiring and override rejection
```

`publish_complete_findings(root, directory, scope, readers)` writes an exclusive
whole request, result, Markdown and audit. `read_completed_findings(...)` requires
a complete bundle and returns its request/result/audit only after fresh complete
reader dispatches, public synthesis/rendering and all final bindings succeed.
Claims/failure markers prevent completion. Occupied output is preserved.

Each operation calls the unchanged complete outcome/cost reader and matrix
reader separately, each with its own complete original report dispatch. The
development port checks all original bundles/preflights. Saved pure validation
is never substituted for these dispatches. Complete current bindings run
before readers, after them and at final artifact verification.

## Budget and local commands

The declared outer cap is 840 seconds per operation, four operations maximum
3,360 seconds. Unchanged inner caps are outcome/cost 240, matrix 180,
development 120 and pure synthesis/presentation 180 seconds. The outer envelope
adds a 120-second binding/artifact allowance to those 720 seconds of ceilings;
each complete binding snapshot also has a 120-second check, while the hard
840-second cap enforces the combined operation limit. Metadata/static/test
gates have separate 180-second ceilings. Actual evidence requires hard-capped
children and flushed complete reader traces. A tool observation timeout is
not terminal process evidence.

Use a new empty output directory for publication. Do not rerun an occupied
producer or published output. The CLI exposes only mode and output directory:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p612_complete_findings --publish --output-dir artifacts/runs/your-new-findings
.\.venv\Scripts\python.exe -m scripts.run_p612_complete_findings --read-only --output-dir artifacts/runs/p612-complete-findings-current-v2
.\.venv\Scripts\python.exe -m pytest -q tests/test_continual_findings_current_bindings.py tests/test_continual_findings_current_inputs.py tests/test_continual_findings_artifacts.py tests/test_p612_complete_findings_cli.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
git diff --check
```

These commands require the complete fixed local artifacts; a clean checkout
alone does not contain ignored evidence. No environment variable/dependency or
scientific override is added. The accepted b2 result (322,781,259 bytes) and
Markdown (1,305,992 bytes) must repeat exactly. H1–H4 remain unresolved within
the complete measured fixed setting. Failed historical claims, unobserved
killed-child counters, all seeds/metrics/contrasts/roles and original measurement
limits remain. The audit's historical validation timing is scoped to its
publication; each readback records its own timing through the operator trace.

## Preserved observer failure

The first CLI publication completed in 563.272 seconds and retained the exact
accepted body/Markdown. Its observer then failed an accounting assertion: each
of the eight original training readers revalidates twelve development bundles
and eight canonical preflight references. Those 96/64 nested validations are
additional to the 12/8 checks owned by the direct development port.

The whole first publication, trace, failed receipt and fifteen snapshots remain.
Terminal child guard counters were not reported and remain unobserved. This
attempt does not count as accepted actual validation. Its 566.761 parent
seconds are charged to the original 3,360-second shared actual budget.

The separately declared observer preserves all 108/72 calls and their exact
ownership: 48/32 under each outcome/cost and matrix port plus 12/8 under the
development port. Two fabricated nesting/error-restoration cases pass before
four corrected actual operations in new `current-v2` directories. Production,
all 150 source/79 test/174 input pins and every original cap remain unchanged.
Remaining-budget exhaustion leaves acceptance open. No cap is increased.

## Complete acceptance evidence

Both exclusive publications and both fresh independent readbacks exit zero in
569.048, 569.692, 571.511 and 572.097 parent seconds, under hard 840-second caps.
Including the first failed observer, actual time is 2,849.109 seconds, below the
original 3,360-second shared limit. Separate final metadata checks take 88.635
seconds under 180. All 150 source, 79 test and 174 input pins remain exact.

Each accepted child records three whole current bindings, two complete reports,
eight original training, four scored and four cost reader returns, all original
development/preflight validation and 24 zero scientific guards. No training,
scoring or new final source is executed. Every JSON and Markdown byte equals b2:

- JSON: 322,781,259 bytes, SHA
  `a090ecf6c16bd20c46af71b6bb625aec7f5cabd5b53fdb114bcb482f6468e743`.
- Markdown: 1,305,992 bytes, SHA
  `5110e724a82452d7d7d53bb8ffd4f4ee654f9bc6e38be6b6a54812d2248dd8af`.

Actual receipt `p612-current-findings-v2-actual-validation.json`: 381,480 bytes /
`f4a8bcec62be03113f0d77f5284f5d9fae55864e4112f9dec52ef1df87749854`.
Original b3/b/P6.12 audit: 413,550 bytes /
`22c2375992456eae4e739189c8a1fa713d5cd209bad35378f27f4e19b3eb2e16`;
98.956 seconds under hard 180. Only those three tasks are newly accepted.
Broader matrix/confirmation tasks remain open. The original b2 body's historical
scope and pending flags remain literal; new IO authority lives in its separate
request/audit, actual operations and original acceptance record.

Extension: compose another complete port in a separately declared increment,
preserving the accepted pure sources and all original reader gates. Freeze its
whole source/test/input scope before fixtures and audit the unchanged original
criteria after actual bounded publication and independent reconstruction.
