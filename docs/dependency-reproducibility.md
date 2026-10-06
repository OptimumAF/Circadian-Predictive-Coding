# Tested dependency environments

Supported ranges remain in `requirements.txt` and `requirements-resnet.txt`.
The dated files in `constraints/` record exact installed versions from a tested
local environment. They constrain installation; they do not change the supported
Python range or install optional packages by themselves.

## Current Windows CPU snapshot

- Constraints: `constraints/windows-cpu-py314-2026-10-06.txt`.
- Interpreter/platform/packages: `constraints/windows-cpu-py314-2026-10-06.environment.json`.
- CPython 3.14.7, Windows AMD64, NumPy 2.4.6, Torch 2.14.0+cpu,
  torchvision 0.29.0+cpu; 25 distributions including development tools.
- Local validation receipts: `artifacts/runs/p93-dependencies/windows-cpu-py314-20261006/`.
  See `docs/development-log.md` for commands, results and skipped checks.

Why this: supported ranges help normal development; exact versions allow a
particular environment to be retained without upgrading a benchmark campaign.
Capture every installed distribution, including transitive and tooling packages,
so this snapshot does not hide a locally present dependency.

## Use the snapshot

In a separate clean checkout with CPython 3.14.7 and no existing `.venv`:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints/windows-cpu-py314-2026-10-06.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-resnet.txt -c constraints/windows-cpu-py314-2026-10-06.txt --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q -ra
```

Omit the optional requirements command for NumPy-only use. The interpreter's
patch version must be checked separately; `py -3.14` selects the available 3.14
interpreter. Constraints for `pip`/`setuptools` record the capture tools; the
commands above do not implicitly install those tools or every captured package.
Compare `python -m pip freeze --all` to the snapshot when reproducing the full
environment, and record any differences. Select the CPU wheel index explicitly.

P9.3a verified the existing environment. P9.3b1 now verifies fresh Windows
NumPy-only and CPU Torch installations at the scoped test gates below. P9.3b2
retains historical publication coverage; P9.6 retains broader installation tests. This is a version snapshot, not a lock
with wheel hashes, a guarantee of future wheel availability or a portability
claim for Linux, other Python versions, CUDA or other hardware. Windows paths
and the active executable are omitted from the shared environment JSON.

## Keep experiment evidence intact

Keep the constraint file immutable once it is associated with an experiment.
For a new environment, add a separately dated file and save its full installed
versions, Python/platform facts, requirements identities, validation commands
and results. Store those facts with the experiment's existing provenance.
Do not overwrite an older snapshot or assign today's versions to an older run.

Existing result schemas and strict source maps remain unchanged. The current
snapshot provides no new seed, source, evaluation or runtime-admission authority.
The deferred mutation-guard regression remains recorded in the development plan.
No new module, dependency, environment variable or model behavior is introduced.

## Files and initial existing-environment validation (P9.3a)

```text
constraints/
  windows-cpu-py314-2026-10-06.txt               # Exact versions
  windows-cpu-py314-2026-10-06.environment.json  # Interpreter/platform/scope
docs/
  dependency-reproducibility.md                # Reproduction workflow
```

The existing-environment verification passed 50 tests with zero failures, errors
or skips, plus dependency consistency, the offline constraints dry-run, Ruff
and mypy over 609 files. The log and local receipts retain complete commands,
outputs, XML case names and version/preservation readback:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip install --dry-run --no-index -r requirements.txt -r requirements-resnet.txt -c constraints/windows-cpu-py314-2026-10-06.txt
.\.venv\Scripts\python.exe -m pytest -q -ra tests/test_run_environment.py tests/test_matched_head_benchmark.py tests/test_matched_head_capacity.py tests/test_isolated_head_memory.py tests/test_process_memory.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src tests scripts
git diff --check
```

No Python module or test changed, so a new source-format check is not applicable.
At P9.3a, the full suite, clean installation, other Python/platform/CUDA
environments and remote CI were not run; the next section records later fresh
Windows installation evidence. To extend this work, add a separately validated dated
snapshot and retain the preceding one; P9.3b/P9.6 track remaining installation
and published-run provenance checks.


## Fresh Windows installation validation (P9.3b1)

Two new independent virtual environments were created on 2026-10-06, using
CPython 3.14.7 and a byte-exact frozen copy of the current 971 repository files.
They live outside the repository; both `pyvenv.cfg` files disable system site
packages, user site access is disabled and actual NumPy/Torch import paths are
inside the intended environment. The original `.venv` and older snapshots remain
unchanged. The frozen source includes current uncommitted work; it is not a clean
Git clone.

| Environment | Installed packages | Required regression outcome |
|---|---:|---|
| NumPy-only | 16 exact constrained runtime/tool packages; Torch/torchvision absent | 184 passed, zero failures/errors/skips |
| CPU Torch | All 25 versions equal the dated snapshot; Torch 2.14.0+cpu, torchvision 0.29.0+cpu, CUDA build absent | 147 passed, zero failures/errors/skips |

The CPU gate includes all eight suites named by the current `torch-cpu` workflow,
plus complete local-gradient, atomic-sleep, Torch component and checkpoint-memory
tests. The NumPy gate covers input/finite-update, sleep components, lineage/full
snapshots, toy/continual resume, model-order/evaluation isolation, bounded replay,
retention, source provenance and declared NumPy gradient cases. Full selection
lists and every XML case are retained in the prospective entry and readback.
Both environments pass `pip check` and the ordinary CLI `--help` command. Fresh
CPU Ruff and mypy over 609 files pass; no source or test edits were necessary.

Installations used `--no-index --find-links <wheelhouse>` with the unchanged
requirements files and dated `-c` constraints. Compatible wheels were copied
from the existing pip HTTP cache, their complete member CRCs and byte identities
checked; Python's pip 26.2.1 bundle supplied the installer. Only the missing
setuptools 84.0.0 binary wheel was downloaded from PyPI under the declared 8 MB
cap. Install pip/setuptools explicitly when reproducing the exact tool versions;
the general commands above do not implicitly pin the installer itself.

Exact creation/installation/help/test/static commands, complete stdout/stderr,
package inventories/import paths, every source/wheel pin and file-size inventory
are in `artifacts/runs/p93-dependencies/clean-install-20261006/`. Readback confirms
all 971 copied source files, 968 nonpermitted original files, current environment
versions and HEAD are unchanged. The retained external workspace is about
1.27 GB, below its declared 3 GiB limit; verification took about 313 seconds of
the 900-second aggregate before final documentation.

This validates the declared Windows install/test scope. The full suite, clean
Git clone, remote CI, Python 3.11/3.12, Linux, CUDA and binary equivalence remain
unverified here. It provides no confirmation-source or runtime-admission authority
and does not establish historical experiment environments. P9.3b2/P9.6 preserve
that unfinished work. Retain this workspace and receipts; use new targets and
a separately declared budget for another installation check.


## Historical environment records (P9.3b2a)

[Saved historical evidence](published-run-environments.md) and
`constraints/published-run-environments-2026-10-06.json` retain exact fields
and source hashes from144 selected saved/configuration/fixture/current records,
with1,339 literal document-reference occurrences. Their origin labels distinguish
old saved declarations from current constraints and nested upstream records.
Two full derivations and independent readback pass, with15 behavior checks.
Missing execution-time tooling/transitive/wheel/build facts remain unknown.
The original complete publication scope stays open in P9.3b2b:420 other artifact
JSON bodies and documented ignored data results remain unreviewed. Preserve this
ledger and all older environments; add a separately scoped successor for new
evidence. Neither this inventory nor today's installs confer scientific admission.


P9.3b2b later reviews all available640 original JSON and68 journals, including
large saved result bodies, in the separately versioned v2 environment ledger
and lossless gzip sidecar. [Full saved-evidence guide](published-run-environments.md)
records all original missing-version/context limits, failed controls and exact
validation scope. CIFAR confirmation results contain no dependency versions;
earlier feasibility-request CUDA versions are not assigned to confirmation.
P9.3b2c retains explicit publication-register completeness; original parents stay
unchecked. Neither these evidence JSON/GZIP files nor old environment directories
are substitutes for the dated installation constraints or original execution facts.


## Original publication register (P9.3b2c)

The [explicit original publication register](published-experiment-register.md)
now reconciles the named published results in all319 frozen repository documents
with all available original structured records and16 original non-JSON evidence
files. Its separate immutable ledger preserves exact original version fields and
missing source/environment facts. Neither current dated constraints nor an earlier
feasibility request supplies versions for a later historical confirmation.

Together with P9.3a and the two fresh P9.3b1 installation checks, this supplies
P9.3's dependency-recording and reconciliation evidence. Full original transitive,
build/wheel/install facts remain unknown where absent; older environments and all
original records are retained. P9.6's broader installation/example/platform gate
and full scientific admission remain unfinished. Final checklist status is
recorded after this session's terminal acceptance check.
