# Observable ticket policy context

The ordinary task already has a public policy version, but the original seven
query features omit it. V1 and v2 ready-invoice therefore encode identically
despite requiring route and inspect, both allowed in both versions. An opt-in
observation makes that context visible without changing task rules or permissions.

## API and CLI

```python
from scripts.run_ticket_workflow import CONTEXT_FEATURES, training_cases
from src.core.ticket_workflow import CATALOG, workflow_request
from src.core.decision_features import DecisionFeatureSpec

spec = DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)  # D9
cases = training_cases(policy_version="v1", include_policy_context=True)
# Both the request API and run_workflow_episode accept include_policy_context=True.
```

Pair tagged requests with `CONTEXT_FEATURES`. The two additional vocabulary tokens
are `policy_v1` and `policy_v2`; exactly one is present. It comes from the current
validated task state, at every phase. Target labels, private ticket identity,
future/probe/final labels and staged/committed tool values are not appended.
Request metadata, permitted candidates, revision/effect validation and final task
truth are unchanged. Flags require an exact Python bool before any work.

```powershell
python -B -m scripts.run_ticket_workflow --policy-context --output fresh-context-example.json
```

The existing local example flag selects D9 teacher, learner and episode inputs;
its existing example still trains only v2. This command does execute its original
eight-update perceptron recipe. CLI propagation/model dimensions/cost accounting
are covered by scripted tests here; a new learned context-aware qualification
has **not** run in this increment. No native learning benefit is claimed.

Omitted or False preserves original D7 queries, TRAIN source groups, reports and
frozen recipes. Tagged TRAIN groups end in `-context` and cannot silently enter
the original strict diagnostic. Explicit episode/example markers identify the
observation change. Nominal costs follow selected feature width, including bias:
D7 uses8coefficients/class; D9 uses10. Snapshot coefficients are384/480bytes each.

## Executed verification

80focused checks pass Windows and WSL, plus scoped Ruff/format/fresh default+Linux
mypy. Default teacher matches its accepted saved bytes; False/default episodes
match exactly. Tests cover all versions/concepts/phases, invalid option types,
TRAIN/public feature alignment, typed masks, revocation/stale effects, CLI/fresh
output and changed-dimension cost formulas. Source/procedure/results are
self-reviewed, not independently accepted or repository-wide readiness.

The sole bounded deterministic case builds a table from24TRAIN teacher rows
(12unique contextual vectors), then executes18typed sandbox episodes: both
versions/all four concepts ordinary+stale, plus both invoice commit revocations.
All16ordinary/stale cases succeed and two revocations are correctly refused.
It uses70typed table decisions,94encodes/94dispatches and zero learner/optimizer/
probe/final/random work. This is an exact-template plumbing control, not learned
native task ability, unseen-task generalization or circadian benefit.
[Protocol, failed checks, counts and result](../artifacts/runs/ticket-policy-context-20261023/RESULTS.md).

## Structure and checks

```text
src/core/ticket_workflow.py            Public version token in owned request
src/app/ticket_workflow.py             Same mode through real bounded episodes
scripts/run_ticket_workflow.py         Tagged teacher, D9 spec, existing example CLI
tests/test_ticket_policy_context.py    Alignment/defaults/semantics/CLI/costs
```

```powershell
python -m pytest -q tests/test_ticket_policy_context.py tests/test_ticket_workflow.py
python -m ruff check src/core/ticket_workflow.py src/app/ticket_workflow.py scripts/run_ticket_workflow.py tests/test_ticket_policy_context.py
python -m ruff format --check src/core/ticket_workflow.py src/app/ticket_workflow.py scripts/run_ticket_workflow.py tests/test_ticket_policy_context.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent src/core/ticket_workflow.py src/app/ticket_workflow.py scripts/run_ticket_workflow.py tests/test_ticket_policy_context.py
```

Next extend the existing native BP fit qualification with explicit contextual
joint v1invoice/password and v2invoice/parcel templates, after freezing a small
protocol. D9 adds input parameters; old D7 checkpoints/metrics are not equivalent
controls. Preserve every prior negative; do not infer that input aliasing explains
failed switching, increase probe horizons, or tune CPC from this plumbing result.
