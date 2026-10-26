"""Four-phase replay clock readbacks with fake reports and real receipt records."""

from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters.return_replay_accounting import check_replay_progress
from src.adapters.torch_recurrent_learner import TORCH_RECURRENT_CE_ID
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import AppliedConsolidation
from src.core.learner_ports import TrainingDiagnostic


class Helper:
    def __init__(self, budget, report, receipts):
        self._replay_budget = budget
        self._runtime = SimpleNamespace(
            applied_consolidations=tuple(receipts),
            _base_actor_version="a0",
            _candidate_version="c0",
        )
        self.saved = report
        self.reads = 0

    @property
    def report(self):
        self.reads += 1
        return deepcopy(self.saved)


def fixture(count=256, total=256):
    calls: list[None] = []

    def clock():
        calls.append(None)
        return 0.0

    budget = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=8 * (total // 2)), clock)
    budget.replay_examples_completed = 8 * (count // 2)
    events, receipts, sources = [], [], {}
    for index, updates in enumerate(range(2, total + 1, 2), 1):
        event: dict[str, Any] = dict(
            updates=updates,
            event_id=f"rehearsal-{updates}",
            status="not_run",
            reason="not reached",
            receipt=None,
            charged_exposures=0,
            attempted_updates=0,
            returned_updates=0,
            committed_updates=0,
            selected_ids=[],
            selected_episodes=[],
            diagnostics=[],
        )
        if updates <= count:
            receipt = AppliedConsolidation(
                event["event_id"], "a0", "c0", index, TrainingDiagnostic(TORCH_RECURRENT_CE_ID, 0.5)
            )
            receipts.append(receipt)
            episode = dict(
                episode_id=f"released-{updates}",
                context="C_train-red",
                role="train",
                payload=[updates],
            )
            sources[episode["episode_id"]] = (updates, deepcopy(episode))
            event.update(
                status="complete",
                reason=None,
                receipt=asdict(receipt),
                charged_exposures=8,
                attempted_updates=1,
                returned_updates=1,
                committed_updates=1,
                selected_ids=[f"source-{updates}"],
                selected_episodes=[episode],
                diagnostics=[asdict(receipt.diagnostic)],
            )
        events.append(event)
    committed = count // 2
    report = dict(
        bound=True,
        stopped=False,
        wake_updates_observed=count,
        total_updates=total,
        events=events,
        attempted_updates=committed,
        returned_updates=committed,
        committed_updates=committed,
        charged_exposures=8 * committed,
        replay_budget_charged_exposures=8 * committed,
    )
    state = SimpleNamespace(
        completed_updates=count + committed, training_examples=8 * (count + committed)
    )
    return SimpleNamespace(
        helper=Helper(budget, report, receipts),
        budget=budget,
        state=state,
        sources=sources,
        count=count,
        total=total,
        calls=calls,
    )


def check(data, **options):
    return check_replay_progress(
        data.helper, data.budget, data.count, data.state, training_sources=data.sources, **options
    )


@pytest.mark.parametrize("count", [0, 1, 2, 63, 64, 127, 128, 191, 192, 255, 256])
def test_should_bind_every_four_phase_checkpoint_to_scheduled_work_and_native_clock(count):
    data = fixture(count)
    before = deepcopy(
        (
            data.helper.saved,
            data.helper._runtime.applied_consolidations,
            data.sources,
            vars(data.state),
        )
    )
    result = check(data, total_updates=256)
    committed = count // 2
    assert result == dict(
        wake_updates=count,
        replay_committed_updates=committed,
        replay_charged_examples=8 * committed,
        native_updates=count + committed,
        native_training_examples=8 * (count + committed),
    )
    assert len(data.helper.saved["events"]) == 128
    assert data.budget.budget.max_replay_examples == 1024
    assert data.budget.replay_examples_completed == 8 * committed
    assert data.budget.updates_completed == 0
    assert len(data.calls) == 2 and data.helper.reads == 1
    assert (
        data.helper.saved,
        data.helper._runtime.applied_consolidations,
        data.sources,
        vars(data.state),
    ) == before


def test_should_keep_original192_defaults_exactly_equivalent_to_explicit_total():
    first, second = fixture(192, 192), fixture(192, 192)
    assert (
        check(first)
        == check(second, total_updates=192)
        == dict(
            wake_updates=192,
            replay_committed_updates=96,
            replay_charged_examples=768,
            native_updates=288,
            native_training_examples=2304,
        )
    )
    assert first.helper.saved == second.helper.saved
    assert first.budget.budget.max_replay_examples == 768
    assert len(first.helper.saved["events"]) == 96
    assert len(first.calls) == len(second.calls) == 2


class NoExternalWork:
    def __getattr__(self, name):
        raise AssertionError(f"external field reached: {name}")


@pytest.mark.parametrize(
    "value", [False, True, 192.0, np.int64(256), None, "256", 0, 191, 193, 255, 257, 512]
)
def test_should_admit_total_only_before_any_external_helper_budget_or_state_access(value):
    blocked = NoExternalWork()
    with pytest.raises(ValueError, match="native integer 192 or 256"):
        check_replay_progress(
            blocked, blocked, 0, blocked, training_sources={}, total_updates=value
        )


@pytest.mark.parametrize(
    "field",
    [
        "bound",
        "stopped",
        "wake_updates_observed",
        "total_updates",
        "attempted_updates",
        "returned_updates",
        "committed_updates",
        "charged_exposures",
        "replay_budget_charged_exposures",
    ],
)
def test_should_refuse_report_clock_totals_and_partial_native_work(field):
    data = fixture()
    data.helper.saved[field] = (
        False if field == "bound" else True if field == "stopped" else data.helper.saved[field] - 1
    )
    with pytest.raises(ValueError, match="clocks, charges or consolidation receipts"):
        check(data, total_updates=256)
    assert data.budget.replay_examples_completed == 1024


@pytest.mark.parametrize(
    "problem",
    [
        "original_budget",
        "wake_charge",
        "replay_charge",
        "limit",
        "event_count",
        "event_order",
        "receipt_count",
        "receipt_id",
        "state_count",
        "state_rows",
        "state_bool",
        "state_float",
    ],
)
def test_should_refuse_original_budget_receipt_and_native_progression_corruption(problem):
    data = fixture()
    if problem == "original_budget":
        data.helper._replay_budget = object()
    elif problem == "wake_charge":
        data.budget.updates_completed = 1
    elif problem == "replay_charge":
        data.budget.replay_examples_completed -= 8
    elif problem == "limit":
        data.budget.budget = ToyExecutionBudget(max_replay_examples=768)
    elif problem == "event_count":
        data.helper.saved["events"].pop()
    elif problem == "event_order":
        data.helper.saved["events"][0]["updates"] = 4
    elif problem == "receipt_count":
        data.helper._runtime.applied_consolidations = data.helper._runtime.applied_consolidations[
            :-1
        ]
    elif problem == "receipt_id":
        first, *rest = data.helper._runtime.applied_consolidations
        data.helper._runtime.applied_consolidations = (replace(first, event_id="replaced"), *rest)
    elif problem == "state_count":
        data.state.completed_updates -= 1
    elif problem == "state_rows":
        data.state.training_examples -= 1
    elif problem == "state_bool":
        data.state.completed_updates = True
    else:
        data.state.training_examples = float(data.state.training_examples)
    before = data.budget.replay_examples_completed
    with pytest.raises(ValueError):
        check(data, total_updates=256)
    assert data.budget.replay_examples_completed == before


@pytest.mark.parametrize(
    "problem",
    [
        "status",
        "reason",
        "receipt",
        "charge",
        "attempt",
        "return",
        "commit",
        "selected_count",
        "episode_count",
        "diagnostic_count",
        "diagnostic",
        "attempt_number",
        "actor",
        "candidate",
        "objective",
    ],
)
def test_should_refuse_incomplete_event_or_changed_native_commit(problem):
    data = fixture()
    event = data.helper.saved["events"][0]
    if problem == "status":
        event["status"] = "failed"
    elif problem == "reason":
        event["reason"] = "native commit failed"
    elif problem == "receipt":
        event["receipt"] = None
    elif problem == "charge":
        event["charged_exposures"] = 7
    elif problem in ("attempt", "return", "commit"):
        event[
            {
                "attempt": "attempted_updates",
                "return": "returned_updates",
                "commit": "committed_updates",
            }[problem]
        ] = 0
    elif problem in ("selected_count", "episode_count", "diagnostic_count"):
        event[
            {
                "selected_count": "selected_ids",
                "episode_count": "selected_episodes",
                "diagnostic_count": "diagnostics",
            }[problem]
        ].clear()
    elif problem == "diagnostic":
        event["diagnostics"][0]["value"] = 0.6
    else:
        first, *rest = data.helper._runtime.applied_consolidations
        if problem == "attempt_number":
            changed = replace(first, attempt_number=2)
        elif problem == "actor":
            changed = replace(first, actor_version="a1")
        elif problem == "candidate":
            changed = replace(first, learner_version="c1")
        else:
            changed = replace(first, diagnostic=TrainingDiagnostic("other_objective", 0.5))
        data.helper._runtime.applied_consolidations = (changed, *rest)
        event["receipt"] = asdict(changed)
        event["diagnostics"] = [asdict(changed.diagnostic)]
    with pytest.raises(ValueError, match="complete native commit"):
        check(data, total_updates=256)
    assert data.budget.replay_examples_completed == 1024


@pytest.mark.parametrize("problem", ["missing", "future", "replaced_payload", "replaced_episode"])
def test_should_require_exact_previously_released_training_source(problem):
    data = fixture()
    episode = data.helper.saved["events"][0]["selected_episodes"][0]
    name = episode["episode_id"]
    if problem == "missing":
        data.sources.pop(name)
    elif problem == "future":
        data.sources[name] = (3, data.sources[name][1])
    elif problem == "replaced_payload":
        episode["payload"] = [999]
    else:
        episode["episode_id"] = "another_source"
    with pytest.raises(ValueError, match="released training data"):
        check(data, total_updates=256)
    assert data.budget.replay_examples_completed == 1024


@pytest.mark.parametrize(
    "field",
    [
        "status",
        "receipt",
        "charged_exposures",
        "attempted_updates",
        "returned_updates",
        "committed_updates",
    ],
)
def test_should_refuse_unobserved_future_work_after_partial_success(field):
    data = fixture(128)
    event = data.helper.saved["events"][64]
    event[field] = "running" if field == "status" else {} if field == "receipt" else 1
    with pytest.raises(ValueError, match="unobserved future event work"):
        check(data, total_updates=256)
    assert data.budget.replay_examples_completed == 512


def test_should_allow_absent_runtime_only_at_initial_zero_work_boundary():
    data = fixture(0)
    data.helper._runtime = None
    assert check(data, total_updates=256)["native_updates"] == 0
    data = fixture(2)
    data.helper._runtime = None
    with pytest.raises(ValueError, match="consolidation receipts"):
        check(data, total_updates=256)


def test_should_refuse256_report_under_unchanged192_default():
    data = fixture(192)
    with pytest.raises(ValueError, match="clocks, charges or consolidation receipts"):
        check(data)
    assert data.budget.replay_examples_completed == 768
