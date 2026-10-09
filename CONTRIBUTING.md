# Contributing

## Project Direction

This project is centered on Circadian Predictive Coding.
Backprop and predictive coding baselines are maintained as comparison anchors.

Contributions should improve one or more of:

- circadian algorithm quality
- benchmark rigor and reproducibility
- engineering reliability and clarity

## Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional benchmark dependencies:

```powershell
pip install -r requirements-resnet.txt
```

## Dependency Reproducibility

Use broad requirements for development and a dated `constraints/` snapshot for
a recorded environment. See [the guide](docs/dependency-reproducibility.md) for
Windows CPU commands and scope. Add a new snapshot when changing environments;
retain the original versions and provenance for published experiments. Record
clean-install checks separately and avoid upgrades during a benchmark campaign.

## Branch And PR Workflow

1. Create a focused branch from `main`/`master` (coding-agent branches use `codex/`).
2. Keep scope narrow and architecture-consistent.
3. Add or update tests for behavior changes.
4. Update docs for user-facing changes.
5. Open a PR using the repository template.

## Required Checks Before PR

```powershell
python -m ruff check .
python -m mypy src tests scripts
python -m pytest -q -ra
```

For changed Python files, also run `python -m ruff format --check` followed by
their actual paths. Keep existing global formatting debt separate from the patch;
do not claim a whole-repository format pass from a changed-file check.

[The current workflow](.github/workflows/ci.yml) declares general Linux checks on
Python 3.11/3.12/3.14, a CPU Torch job with fifteen required suites and a zero-skip
JUnit gate, and a Windows-native job. Native boundary changes require both mypy
platform targets (`--platform win32` and `--platform linux`) and the appropriate
platform tests. Unsupported-platform skips must remain visible; they do not count
as required CPU coverage. Workflow definitions and focused local passes are not
evidence that the complete baseline or current remote jobs passed. Record exact
source/environment, commands, outcomes and skips in the development log.

Keep the checkout and Git identity stable during provenance-sensitive runs: the
v14 source digest includes tracked diffs and nonignored untracked files, including
documentation. For concurrent edits, finish that run first or run against an
independent frozen copy with its own files and Git objects. Do not bypass its
source-change rejection or duplicate another owner's live baseline.

## Coding Standards

- Keep `core` pure and free from dataset/CLI concerns.
- Prefer explicit dataclasses for configuration surfaces.
- Fail early with actionable error messages.
- Avoid hidden coupling between model families.
- Keep benchmark comparisons fair:
  - same dataset split
  - same evaluation protocol
  - clear disclosure of differing hyperparameters
  - predeclared candidate/seed budgets, selection rules and final-test release
  - explicit work, replay/guard/inference costs, capacity, information access,
    backend and process/CUDA memory scope; equal trial counts alone are insufficient
  - retained negative results and all declared seeds/metrics; do not tune anchors,
    choose seeds or change metrics to make the circadian model win

Existing app-to-infra dataset/role imports are recorded boundary debt (P9.4g),
not permission to add more. Define new ports/contracts in inner layers and compose
their outer implementations at the adapter boundary. Preserve current behavior
and frozen identities when reducing the existing coupling.

## Documentation Standards

When behavior changes:

- update `README.md` for usage changes
- update `ARCHITECTURE.md` for boundary or flow changes
- add/update ADRs for major decisions in `docs/adr/`
- add an entry in `CHANGELOG.md`

Use [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) and
[docs/development-log.md](docs/development-log.md) for task acceptance and handoff.
Check a task only after its complete criteria pass and evidence is recorded. Keep
blocked/deferred work unchecked with an exact next action. Record task IDs,
commands/outcomes, skipped tests, artifacts, plan amendments, blockers and the
next action at session end. Preserve original criteria when splitting work.

Portable exact boundary checks and recorded-environment reproduction are separate
claims. See [reproducibility scope](docs/reproducibility-scope.md) and
[ADR-0196](docs/adr/ADR-0196-separate-portable-boundary-fixtures-from-historical-results.md).
Keep original historical hashes/results unchanged; test-local references must stay
isolated and preserve exact assertions. Such fixtures grant no scientific admission.

## Commit Guidance

- Use concise, descriptive commit messages.
- Separate refactors from behavior changes where practical.
- Do not include generated benchmark artifacts unless intentionally publishing results.
- Review the concrete patch and its complete validation before publication. Honor
  the user's existing scope/authorization and any required gates; passing one
  component does not authorize unrelated changes or establish a full baseline.
  Use the PR template to report remaining limits and the rollback action.

## Issue Triage Priorities

1. Reproducible correctness bugs
2. Benchmark regressions
3. Circadian adaptation stability/performance issues
4. Documentation and DX improvements


### Extending managed composite capture

Before adding a supported source,enumerate every current field and classify data,
nested native/source state or original references explicitly. Add complete bounded
preflight and cross-component alias/refusal controls under fresh fixture budgets.
Keep live authority out of payload copies and reserve original raw-payload bytes
before copying. Qualify pending histories,sampler concurrency and current resource
admission before closing R3.5b2e5b;schema presence alone is insufficient.


For managed capture resource changes, preserve original baseline/peak/count,
reader/gate/thread/stop/error authority, cumulative elapsed origin and admitted
copy charges. Test contention, reentry, changed sources and both pre/post-copy
refusals with fresh bounded fixture scopes. Join every actual contender/worker.
Use the same memo for final metadata and verify shared budget/progress segments.
Keep populated histories/native variants/owner/bytes/recovery criteria open until
independently qualified; do not infer them from resource-control success.


Managed capture retained-source bindings (ADR-0237):explicit original enrolled
runtime/controller/pending/token relationships and installed consent/copy guards
now precede holder payload ports and projection. Complete graph bounds precede
checkpoint measurement,integrity and pure promotion checks. Retained checkpoint
payloads require live consent even when the current inbox is empty. Final pending
references and consent are rechecked without invoking controller restore guards.
See docs/managed-composite-capture.md and src/app/managed_composite_bindings.py.
Portable tickets preserve fields/aliases;original ticket and rollback receipt
remain live authority references. Actual promotion issuance/native provenance/
all variants/replay consent/bytes/recovery remain unfinished.


Original managed native update observation (ADR-0238):optional native_observer
ports in inbox/runtime/sharing/managed owner expose original source/label/learner,
actual detached inputs and committed receipt/spent count through synchronous
expiring access. See docs/native-update-origin.md. Original consent/admission,
owner instance fields,default calls and update order remain unchanged. Callback
faults preserve original failure/receipt/resources;returned references remain
caller-owned. No new configuration/dependency/environment variable. Core defines
the reference contract;app manages lifetime;neither imports adapters/infra.
Persistent replay origin and every storage/retention/dedup/eviction/fork/checkpoint/
promotion/restore/erase path remain open under R3.5b2e5b3. Complete compoundcapture,
canonical bytes and recovery gates remain unchecked. Do not infer row lineage
from content hashes or treat these observations as consent/restore permission.
Tests:fixed fake-only origin controls +five selected fake inbox controls +990
current composite/resource/codec controls;both742types/wholeRuff/check format.
For safe extension:add a bounded original replay-write/row port with weak or
owned-accounted payload references;preserve terminal failures and original gates.


Original replay-write observation (ADR0239):core/replay_write_origin defines
a bounded original model/input window;app/replay_write_origin composes it with
the original managed producer. Native copy ranges,new snapshot references and
final retained identities are observed without extra array copies/native fields.
Explicit original ContextVar token/thread/callback lifetime survives refused
close;all default storage/policy/RNG rules preserved. See docs/replay-write-origin.md.
Pure/current gates precede a separately declared tiny native storage-only parity
fixture. No dependency/env/config changes. This is not a persistent row ledger or
consent/restore certificate. Full b3/e5b/e5 retained variants/lineage/bytes/recovery
remain open. Next consume actual copy identity under bounded weak/owned-accounted
retention and original consent/terminal-outcome authority before broadening capture.


### Replay row origin changes

Use bounded fake/pure controls and full current typing/static/preservation gates
before separately budgeted native integration. Enroll the original fresh empty
candidate; never backfill by content hashes. Test revoked/tombstoned/expired/
untracked/foreign origins and committed post-update failures. Document metadata
accounting separately from physical heap/RSS. Qualify original retained holders
before adding capture/fork/actor/promotion/restore/erase support. Exact commands
and receipts: development log and `artifacts/runs/r35b2e5b3c-row-origins-20261008/`.


### Replay capture origins

Require observed original candidate consent/receipts before copying nonempty
replay. Missing or unqualified other-holder lineage refuses before projection.
Keep original lock intervals, failed charges, empty-replay aliases and callback
expiry/reference release controls. Run fake/current/type/static/cache/source gates
before a separately budgeted native capture scope. Commands: development log.


Replay capture lifecycle repair: original runtime open/consent checks use
leased elapsed access during capture; public default reads keep ordinary access.
See docs/replay-capture-origins.md and current development log for qualification.


For original replay copy integration,use the explicit model-copy window at the
actual creation boundary. Bind memo identities rather than content. Implement
original owner/lifecycle/raw-copy/metadata/work/time admission before allocation
and qualify actual retained holder registration before granting capture access.
Run pure controls and both types/static/source/cache gates before separately
budgeted native fixtures;never reuse a spent fixture allowance or failed timeout.


Managed copy admission must consume the original ledger and payload-budget
authorities. Never construct a new allowance to bypass copied live slots or
cumulative charges. Observe actual copier/holder boundaries,keep targets weak
and recheck source consent after resource/measurement callbacks. Record failed
fixture budget scopes separately from test exit codes;exclude them from acceptance
and declare a fresh corrective scope prospectively. Full restoration requires
new observed row/inbox/receipt lineage;constructor witnesses alone are insufficient.


For native state-copy changes, run scalar controls from
`docs/native-state-copy.md`, both configured mypy platform targets and Ruff first.
Run the tiny native integration fixture only within a separately declared local
budget. A copied callback precedes validation/publication and cannot authorize
retained replay by itself. Keep original checkpoint/inbox/receipt lineage gates.


For graph sequence changes,run the scalar controls in docs/native-graph-copies.md
with existing model/state controls and current capture/replay regressions,then
both mypy targets/Ruff/source/default/cache/resource gates. Separately budget
tiny actual checkpoint restore/handoff tests. Raw references kept by a diagnostic
observer need explicit local accounting;the production primitive grants no
authority. Do not replace original ledger/consent/receipt/holder admission gates.

For borrowed state inventory changes, run `tests/test_numpy_replay_graphs.py`
within a prospective bounded fixture scope, then existing copy/origin/capture
controls, both configured mypy platforms and Ruff. Keep original row identities,
exact integer bounds, early count/byte refusal and conservative alias charges.
Do not infer capture permission from schema validation. Coordinate exclusive
paths and exact patch review as recorded in docs/parallel-work.md.

For managed replay publication, run pure content/framework/cleanup controls,
current origin/copy regressions, both full mypy platform targets and Ruff before
any separately budgeted native fixture. Use actual copier memo witnesses and
original per-copy lifetime admission. Preserve original failed preparations and
spent charges. Add late mutation negatives after the LAST opaque callback;
content checks must use fixed inward code, with type/count/byte refusal before
traversal. Never call a probe, copy or fallible validation after retirement.
Release the lexical acquired lock, never reread a callback-mutable gate field.
