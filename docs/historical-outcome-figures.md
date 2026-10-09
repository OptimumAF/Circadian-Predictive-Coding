# Historical outcome and cost figures

## Scope and boundaries

`src/app/historical_outcome_view.py` copies the ordered presentation data and
original uncertainty records from an already accepted P6.10 body. It performs
presentation consistency checks and computes only padded plotting limits.
It does not grant source admission, run scientific readers, change an estimator,
rank models, or execute experiments. Full original state proofs stay in the
unchanged source artifact.

`scripts/export_historical_outcome_figures.py` is the outer filesystem/graphics
boundary. It binds **39,777,631 complete bytes** to SHA256
`73f5892ef7d152a9a1ada89d59f877483e720026ca160a7afb146c53f4a7f4e8`
before projection and checks the source again after exporting. Unknown, partial,
or changed artifacts are refused. Output directories must be unoccupied; failed
partial output is retained and must not be accepted or overwritten.

Why this: the historical publication has accepted complete metadata and original
uncertainty. Rendering those bytes provides useful figures without repeating its
scientific readers or renewing spent experimental budgets. The parent P9.5 still
needs an inventory of remaining historical figures/reports.

## What the figures mean

Each of the six original families has three pages: all six arm metrics, all five
paired contrast metrics, and six work/storage panels. All **560 cells**, **626
original metric vectors** and **116 primary comparison statements** remain in
original manifest order. Gray dots are original seed observations; dark dots are
original means. Vertical displacement separates overlapping seed dots and has
no scientific meaning. No seed or model is selected or reordered by score.

Arm and secondary paired lines use the existing marginal 95% Student-t intervals.
Primary paired lines use the existing Bonferroni 116-statement model-based
intervals, with the original left-minus-right sign. These retain the original
independence/normality assumptions; rendering does not prove those assumptions.
Constant or missing vectors retain absent intervals; retention is descriptive.
Axes include every observation and selected interval endpoint. Negative values,
above-one retention, both null reasons and `[observed/planned]` counts remain.

Work includes executed rollback; latent loops exclude prediction, guard and
selection work. Array storage, shared FIFO, Python overhead, snapshot copies,
recorded parameter history and sampled whole-process RSS have separate scopes.
Stages are not additive live RAM; shared FIFO bytes are not allocated to arms.
Per-arm wall time, RSS, guard duration and continuous history remain unmeasured.
The exact JSON view retains all **12,880** resource fields, **25,400** history
points, original analysis records, three checkpoint byte/status views per cell,
60 shared contexts and four separate historical process segments. Repeats add
no independent replications or new original work.

Every page is labelled historical. H1-H4 remain unresolved; the reopened G0 and
owning-with correctness gates do not grant new source/runtime/scientific access.
These figures do not claim reproduction under the new protocol.

## Run and verify

ReportLab is an **optional export dependency**, outside the required project
environment. This session uses the existing Codex bundled ReportLab 4.4.9 and
Poppler; no package was installed or project constraint changed. The regular
project interpreter can run CLI help, pure tests and source refusals without it.
Use a Python already containing ReportLab to export:

```powershell
python -X utf8 -m scripts.export_historical_outcome_figures --help
python -X utf8 -m scripts.export_historical_outcome_figures --output-dir artifacts/runs/my-historical-figures
pdftoppm -scale-to 1800 -png artifacts/runs/my-historical-figures/historical-outcomes-costs.pdf artifacts/runs/my-historical-figures/page
.\.venv\Scripts\python.exe -X utf8 -m pytest -o addopts='' -q tests/test_historical_outcome_view.py tests/test_historical_figure_export_boundary.py
.\.venv\Scripts\python.exe -X utf8 -m ruff check src tests scripts
.\.venv\Scripts\python.exe -X utf8 -m ruff format --check src/app/historical_outcome_view.py scripts/export_historical_outcome_figures.py tests/test_historical_outcome_view.py tests/test_historical_figure_export_boundary.py
.\.venv\Scripts\python.exe -X utf8 -m mypy --no-incremental --platform win32
.\.venv\Scripts\python.exe -X utf8 -m mypy --no-incremental --platform linux
```

The export contains 18 standalone SVGs, one 18-page vector PDF, `report.md`,
`view.json`, complete plotted-series/extent `figures.json`, and `manifest.json`
with exact source/output identities and rendering environment. It is a derivative
presentation, not an independent official scientific verifier. Inspect rendered
pages and compare every series/interval/cost field to the pinned source before
accepting an export. Numerical data are exact in JSON; visible tick labels round.

Safe extension: add another separately accepted artifact schema and its complete
coverage tests in a new pure app module. Keep source binding and optional graphics
at the outer boundary, preserve unknowns and original metric/seed/interval rules,
and budget rendering prospectively. Do not rerun original readers or loosen this
exporter's fixed accepted-source identity to make another artifact pass.
