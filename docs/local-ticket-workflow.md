# Learned local ticket workflow

The policy chooses every step of an in-memory ticket task: open the sealed
ticket, stage a workflow, then commit it. A bounded dispatcher checks current
capabilities, observation revision, active tools and phase before every effect.
Wrong but permitted workflows can commit and count as task failure.

```powershell
python -m scripts.run_ticket_workflow --output local-ticket-workflow.json
```

The output must be fresh. The fixed driver fits one D7/K6 perceptron through the
existing delayed-label AttributionStream: eight12-row passes/96 TRAIN rows.
There is no seed, dose, tuning, replay, probe, final or promotion option. Before
scoring, it retains states0/8; neither episode results nor goals select updates.

## Task and authority

The known union contains commit/defer/inspect/open/retire/route. Under current v2,
retire is inactive; invoice/password use inspect and legacy/parcel use defer.
All five active capability-permitted tools appear at every phase, so candidates
do not supply the correct next action. Sealed requests hide concept/ticket ID;
ready requests reveal the concept. Request IDs are generic step IDs. The model
gets typed observation data, not the private state or scoring target.

Each control executes four ordinary tickets, four stale-once tickets and one
permanent commit-revocation ticket. Every episode has at most six attempts.
An explicit stale event reseals/increments revision after the first ready choice;
the old command is refused and a new observation can lead to reopening/recovery.
Commit revocation happens after its predicted action and before dispatch. Denial
stops without restoring permission. Scoring separates ordinary success, recovery,
wrong completion, exhausted budgets and desired permission refusal.

Only typed local transitions exist. There are no filesystem/network/subprocess
handlers. Capabilities are an owned grant plus recorded environment changes,
not general host permissions; arbitrary injected Python callbacks are not
contained by this task. Existing ExperiencePermissions remain learning consent.

## Reports, costs and scope

Every paid step retains the observation/capabilities, request, returned/validated
decision, explicit environment event, effect-time authority, transition and final
state. Failed callbacks retain their prefix and unknown unreturned/unvalidated
work. Candidate construction/scanned catalog entries/returned candidates are
separate from typed calls, provider token/comparison assertions and local effects.
TRAIN/demonstration generation, mistakes/affine/update-addition terms, nominal
snapshots and RSS/wall observations are reported separately.

Templates use TRAIN/development suffixes; concepts and states overlap by design.
This is a local functional test, not real-source generalization, modern-model,
CPC, matched-compute, calibrated-quality or whole-repository readiness evidence.
All failed endpoints remain valid results. The frozen protocol permits one small
functional case after independent source/software gates; no native model or SDK.

## Observed local result

After the fixed eight passes, the learned policy completed4/4 ordinary episodes
and4/4 stale-state recoveries. Its commit-revoked episode correctly stopped before
committing. The unchanged policy completed0/9 and retained54 wrong-phase refusals;
it never reached the phases that trigger the perturbations. This is the same
ex-ante task protocol, not identical realized events or matched compute.

One productive case used8updates/96rows/89typed calls, about.89s inner time and
40,583,168B observed RSS maximum (sampled, not true peak). All57 unique cases per
platform are covered by56 full-cohort checks plus6 CLI checks with5 repeats;
scoped lint/format/type checks passed. A zero-work launcher import failure is
preserved separately from the successful corrected launch. The JSON encoder
uses compact encoding if whitespace alone exceeds the unchanged512KiB cap.
[Full evidence, failures and scope](../artifacts/runs/local-ticket-workflow-20261022/RESULTS.md).

## Structure and verification

```text
src/core/ticket_workflow.py       immutable domain transitions and observations
src/app/ticket_workflow.py        bounded typed episode and independent admission
scripts/run_ticket_workflow.py   existing learner/stream plus frozen case matrix
tests/test_ticket_workflow.py    scalar boundary, recovery and paid-failure checks
```

```powershell
python -m pytest -q tests/test_ticket_workflow.py
python -m ruff check src/core/ticket_workflow.py src/app/ticket_workflow.py scripts/run_ticket_workflow.py tests/test_ticket_workflow.py
python -m ruff format --check src/core/ticket_workflow.py src/app/ticket_workflow.py scripts/run_ticket_workflow.py tests/test_ticket_workflow.py
python -m mypy --explicit-package-bases --follow-imports=silent src/core/ticket_workflow.py src/app/ticket_workflow.py scripts/run_ticket_workflow.py tests/test_ticket_workflow.py
```

No new dependencies or environment variables. Extend the typed transition core
with explicit capabilities and terminal correctness before adding model/tool
adapters. Preserve source groups, protected final/probe data and existing native
objectives; freeze costs and protocols before another comparative experiment.

[Protocol](../artifacts/runs/local-ticket-workflow-20261022/PROTOCOL.md) ·
[Review](../artifacts/runs/local-ticket-workflow-20261022/REVIEW.md) ·
[Decision](adr/ADR-0533-enforce-ticket-effects-outside-the-policy.md).

## Optional cached native model

`scripts/run_open_jev_ticket_workflow.py` composes the same twelve demonstrations,
existing native owner and delayed-label inbox with the same dispatcher. It makes
24 one-request TRAIN updates (two passes), then tests unchanged/learned snapshots
on ordinary invoice, stale parcel and commit-revoked invoice, six attempts each.
All training precedes development. These source states and dose differ from the
perceptron study; they are not a matched-compute comparison.

In the existing Linux native environment, with the pinned source and weights
already cached:

```bash
python -B -m scripts.run_open_jev_ticket_workflow --source /path/to/pinned/source --checkpoint /path/to/package/checkpoint --output fresh-native-workflow.json
```

The existing optional native dependencies and offline cache configuration apply;
this command does not install or download them. See the exact frozen invocation
in [the native protocol stage](../artifacts/runs/native-ticket-workflow-20261022/PROTOCOL.md).
The native script is Linux-only for resource observations, as are existing native
drivers. It writes full strict-JSON paid prefixes, including failed steps.

Typed native prediction batches five candidates into one forward. Its permanent
forward counter includes TRAIN and readouts. It provides no native usage/margin;
provider tokens remain unknown while loader sequence-length counts measure
processed tokens separately. The callback checks model/gradient versions, mode,
caller/private RNG and permanent work before returning an action. The dispatcher
still checks effect-time authority independently. No new adapter, checkpoint,
learner objective, dependency, environment variable or privileged tool is added.

Focused verification commands (only run native fixtures within a declared local
budget):

```bash
python -m pytest -q tests/test_native_ticket_workflow.py -k 'not test_should_match_five_candidate_native_port'
python -m pytest -q tests/test_native_ticket_workflow.py -k test_should_match_five_candidate_native_port
python -m ruff check scripts/run_open_jev_ticket_workflow.py tests/test_native_ticket_workflow.py
python -m ruff format --check scripts/run_open_jev_ticket_workflow.py tests/test_native_ticket_workflow.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/run_open_jev_ticket_workflow.py tests/test_native_ticket_workflow.py
```

The single frozen native case finished24 updates and29 typed predictions in
198.17s native time (200.01s wrapper), peak RSS14,258,368,512B. Learned invoice
completes correctly; stale parcel reopens/retries but commits inspect instead
of defer, so it is **not successful recovery**. Revoked invoice commit is denied
without an effect. Raw successes0/3 unchanged versus1/3 learned, with all18
unchanged wrong-phase failures retained. This is mixed functional evidence,
not broad native reliability or a CPC/compute/generalization benefit.
35 scalar checks/platform, one WSL literal port check and scoped statics pass.
[Native outcomes, commands, costs and limitations](../artifacts/runs/native-ticket-workflow-20261022/RESULTS.md).
