"""Deferred H32 ownership and refusal, with no training, predictions or physics.

The sixteen native fixtures declare valid mixed clocks; their training histories
were not executed. Only construction, copying and restoring native states run.
Every game evaluator is replaced with an explicit paid-work fixture.
"""

from copy import deepcopy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from src.adapters import minigrid_deferred_probe_rollout as deferred
from src.adapters import minigrid_probe_rollout as live
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureLearner
from src.adapters.torch_feature_learner import TorchFeatureLearner
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from test_minigrid_deferred_probe_rollout import completed_budget
from test_minigrid_joint_catalogue_route import METHODS
from test_minigrid_wide_probe_games import declared_state, paid_report


def forbidden(*args, **kwargs):
    raise AssertionError("deferred fixtures admit no native training, prediction or physics")


@pytest.fixture(scope="module")
def native_states():
    return {
        method: tuple(
            declared_state(method, origin, relative).snapshot_state()
            for origin in (0, 128)
            for relative in (16, 32)
        )
        for method in METHODS
    }


@pytest.fixture(autouse=True)
def forbid_learning_and_prediction(monkeypatch):
    for owner in (InvariantTorchFeatureLearner, TorchFeatureLearner):
        monkeypatch.setattr(owner, "train_batch", forbidden)
        monkeypatch.setattr(owner, "predict", forbidden)
    monkeypatch.setattr(deferred, "monotonic", lambda: 1.0)
    monkeypatch.setattr(live, "monotonic", lambda: 1.0)
    monkeypatch.setattr(live, "evaluate_memory_probe_checkpoint", forbidden)


@pytest.fixture
def wide_games(monkeypatch):
    work = dict(calls=[], callback=None)

    def evaluate(owner, state, **options):
        work["calls"].append((owner, state, options))
        assert options["method"] in METHODS
        assert options["budget"].updates_completed == 0
        assert options["budget"].replay_examples_completed == 0
        options["check_boundary"]()
        callback = work["callback"]
        return callback(owner, state, options) if callback is not None else paid_report()

    monkeypatch.setattr(live, "evaluate_wide_neural_probe_checkpoint", evaluate)
    return work


def capture_all(native_states, method="bp", *, extended=False):
    manager = deferred.DeferredProbeRollouts(
        wide_method=method, extended_wide_probe_budget=extended
    )
    owners, budgets = [], []
    for index, (origin, phase) in enumerate(deferred.ORDER):
        owner = SimpleNamespace(state=deepcopy(native_states[method][index]))
        owner.snapshot_state = lambda owner=owner: owner.state
        owners.append(owner)
        manager.observer(origin)(owner, phase, qualified)
        if phase == "after_return":
            pair = (completed_budget(), completed_budget())
            manager.bind_completed(origin, *pair)
            budgets.extend(pair)
    return manager, owners, budgets


def qualified(read_ports=None, *, after_clocks=None):
    if read_ports is not None:
        assert read_ports() is None
    if after_clocks is not None:
        assert after_clocks() is None


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("qualified_run", [False, True])
def test_should_defer_all_four_full_native_endpoints_without_renewing_paid_work(
    native_states, wide_games, method, qualified_run
):
    manager, owners, budgets = capture_all(native_states, method)
    before = manager.report()
    assert wide_games["calls"] == []
    assert [row["native_updates"] for row in before["captures"]] == [16, 32, 144, 160]
    assert [row["native_training_examples"] for row in before["captures"]] == [144, 288, 7056, 7200]
    assert 0 < before["capture_work"]["logical_payload_bytes_retained"] <= 620544
    assert before["capture_work"]["payload_bytes_retained"] <= 1024 * 1024
    bindings = [(budget.last_clock, budget.started_at, budget.clock) for budget in budgets]
    # Later source writes cannot change any earlier frozen capture.
    for owner in owners:
        owner.state.native_state.head_state["bias_output"].fill_(0.125)
    operation = manager.run_qualified if qualified_run else manager.run
    result = operation(check_boundary=qualified if qualified_run else lambda: None)
    assert result["status"] == "complete", result["reason"]
    assert result["captures"] == before["captures"]
    assert [(row["source_updates"], row["phase"]) for row in result["records"]] == list(
        deferred.ORDER
    )
    assert len(wide_games["calls"]) == 4
    assert len({id(owner) for owner, _, _ in wide_games["calls"]}) == 2
    for index, (owner, state, options) in enumerate(wide_games["calls"]):
        assert all(owner is not original for original in owners)
        assert _same_state(state, native_states[method][index])
        assert _same_state(pickle.loads(manager._captures[index][2]), state)
        assert options["method"] == method
        assert options["budget"].budget.max_wall_seconds == 40
        assert options["budget"].budget.max_training_updates == 0
        assert options["budget"].budget.max_replay_examples == 0
    work = result["capture_work"]
    for operation_name, expected in (
        ("frozen_reader_factories", 2),
        ("frozen_reader_restores", 4),
    ):
        assert work[f"{operation_name}_attempted"] == expected
        assert work[f"{operation_name}_returned"] == expected
    for operation_name in (
        "snapshot_calls",
        "snapshot_copy_calls",
        "decode_calls",
        "reader_snapshot_calls",
    ):
        assert work[f"{operation_name}_attempted"] == work[f"{operation_name}_returned"]
        assert work[f"{operation_name}_returned"] >= 4
    assert bindings == [(budget.last_clock, budget.started_at, budget.clock) for budget in budgets]
    assert all(budget.updates_completed == 16 for budget in budgets)
    assert all(budget.replay_examples_completed == 0 for budget in budgets)
    result["records"][0]["status"] = "forged"
    assert manager.report()["records"][0]["status"] == "complete"
    manager.check()


@pytest.mark.parametrize("method", METHODS)
def test_should_select_extended_resources_only_for_new_frozen_readers(
    native_states, wide_games, method
):
    manager, _, budgets = capture_all(native_states, method, extended=True)
    originals = [(budget.budget, budget.clock, budget.last_clock) for budget in budgets]
    result = manager.run(check_boundary=lambda: None)
    assert result["status"] == "complete", result["reason"]
    assert all(call[2]["budget"].budget.max_wall_seconds == 60 for call in wide_games["calls"])
    assert originals == [(budget.budget, budget.clock, budget.last_clock) for budget in budgets]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("fault", ["policy", "clock", "tensor", "rogue"])
def test_should_refuse_malformed_full_native_capture_and_preserve_attempted_copy_cost(
    native_states, wide_games, method, fault
):
    state = deepcopy(native_states[method][0])
    if fault == "policy":
        object.__setattr__(state.policy.native_policy, "hidden_dim", 8)
    elif fault == "clock":
        object.__setattr__(state.native_state, "training_examples", 145)
    elif fault == "tensor":
        state.native_state.head_state["bias_output"].fill_(float("nan"))
    else:
        object.__setattr__(state, "rogue", True)
    manager = deferred.DeferredProbeRollouts(wide_method=method)
    with pytest.raises(ValueError):
        manager.observer(0)(SimpleNamespace(snapshot_state=lambda: state), "after_probe", qualified)
    saved = manager.report()
    assert saved["status"] == "failed" and manager._captures == ()
    assert saved["capture_work"]["snapshot_calls_returned"] >= 1
    assert saved["capture_work"]["snapshot_copy_calls_attempted"] >= 1
    assert saved["capture_work"]["snapshot_copy_calls_returned"] == 0
    assert not wide_games["calls"]


@pytest.mark.parametrize(
    "method,fault",
    [("bp", "optimizer"), ("bp", "gradients")]
    + [(method, fault) for method in ("neutral_cpc", "gated_cpc") for fault in ("rng", "adaptive")],
)
def test_should_validate_optimizer_gradients_rng_and_adaptive_state_before_retention(
    native_states, wide_games, method, fault
):
    state = deepcopy(native_states[method][0])
    if fault == "optimizer":
        state.native_state.optimizer_state["state"].clear()
    elif fault == "gradients":
        state.native_state.gradients.clear()
    elif fault == "rng":
        state.native_state.head_state["split_generator_state"] = None
    else:
        state.native_state.head_state["wake_examples"] -= 1
    manager = deferred.DeferredProbeRollouts(wide_method=method)
    with pytest.raises(ValueError):
        manager.observer(0)(SimpleNamespace(snapshot_state=lambda: state), "after_probe", qualified)
    assert manager.report()["capture_work"]["payload_bytes_retained"] == 0
    assert not wide_games["calls"]


@pytest.mark.parametrize("limit", ["MAX_STATE_BYTES", "MAX_PAYLOAD_BYTES", "MAX_LOGICAL_BYTES"])
def test_should_charge_successful_native_copy_before_refusing_a_payload_cap(
    monkeypatch, native_states, wide_games, limit
):
    monkeypatch.setattr(deferred, limit, 1)
    manager = deferred.DeferredProbeRollouts(wide_method="bp")
    with pytest.raises(ValueError):
        manager.observer(0)(
            SimpleNamespace(snapshot_state=lambda: native_states["bp"][0]), "after_probe", qualified
        )
    saved = manager.report()
    assert saved["capture_work"]["snapshot_calls_returned"] >= 1
    assert saved["capture_work"]["snapshot_copy_calls_returned"] >= 1
    assert saved["capture_work"]["payload_bytes_retained"] == 0
    assert saved["captures"][0]["logical_payload_bytes"] > 1
    assert not wide_games["calls"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("wide_method", "pc"),
        ("extended_wide_probe_budget", True),
        ("_captures", ()),
    ],
)
def test_should_refuse_replaced_capture_or_configuration_before_reader_construction(
    native_states, wide_games, field, value
):
    manager, _, _ = capture_all(native_states)
    setattr(manager, field, value)
    result = manager.run(check_boundary=lambda: None)
    assert result["status"] == "failed" and result["records"] == []
    assert result["capture_work"]["frozen_reader_factories_attempted"] == 0
    assert not wide_games["calls"]


@pytest.mark.parametrize("fault", ["updates", "clock", "limits", "replay"])
def test_should_refuse_changes_to_original_completed_budgets_before_any_native_reader(
    native_states, wide_games, fault
):
    manager, _, budgets = capture_all(native_states)
    budget = budgets[0]
    if fault == "updates":
        budget.updates_completed = 15
    elif fault == "clock":
        budget.last_clock += 0.25
    elif fault == "limits":
        object.__setattr__(budget.budget, "max_training_updates", 17)
    else:
        budget.replay_examples_completed = 1
    result = manager.run_qualified(check_boundary=qualified)
    assert result["status"] == "failed" and result["records"] == []
    assert result["capture_work"]["frozen_reader_factories_attempted"] == 0
    assert not wide_games["calls"]


@pytest.mark.parametrize("fault", ["partial", "lost", "closing"])
def test_should_preserve_paid_or_unknown_game_prefix_and_stop_before_second_endpoint(
    native_states, wide_games, fault
):
    manager, _, _ = capture_all(native_states)
    paid = [False]

    def evaluate(owner, state, options):
        paid[0] = True
        if fault == "lost":
            raise OSError("lost paid deferred native readout")
        report = paid_report()
        if fault == "partial":
            report.update(status="failed", reason="paid native game interruption")
            raise live.MiniGridEvaluationFailed(report)
        return report

    def check():
        if fault == "closing" and paid[0]:
            raise ValueError("closing resource refused after paid native work")

    wide_games["callback"] = evaluate
    result = manager.run(check_boundary=check)
    assert result["status"] == "failed" and len(wide_games["calls"]) == 1
    assert len(result["records"]) == 1
    record = result["records"][0]
    assert record["evaluator_admitted"] is True
    if fault == "lost":
        assert record["report"] is None
    else:
        assert record["report"]["work"]["live_prediction_calls_returned"] == 4
    again = manager.run(check_boundary=lambda: None)
    assert again["reason"] == result["reason"]
    assert len(wide_games["calls"]) == 1


def test_should_forward_qualified_final_rereads_exactly_once_after_resource_clocks(
    native_states, wide_games
):
    manager, _, _ = capture_all(native_states)
    events = []

    def outer(read_ports=None, *, after_clocks=None):
        if read_ports is not None:
            read_ports()
        if after_clocks is not None:
            events.append("closing resources")
            after_clocks()

    def evaluate(owner, state, options):
        for _ in range(2):
            options["qualified_boundary"](
                lambda: events.append("native reads"),
                after_clocks=lambda: events.append("final native reread"),
            )
        return paid_report()

    wide_games["callback"] = evaluate
    result = manager.run_qualified(check_boundary=outer)
    assert result["status"] == "complete", result["reason"]
    assert events.count("native reads") == 8
    assert events.count("final native reread") == 8
    for index, event in enumerate(events):
        if event == "final native reread":
            assert events[index - 1] == "closing resources"


def test_should_refuse_native_mutation_by_last_resource_clock_before_next_work(
    native_states, wide_games
):
    manager, _, _ = capture_all(native_states)
    pending = [None]
    reads = []

    def outer(read_ports=None, *, after_clocks=None):
        if read_ports is not None:
            read_ports()
        if after_clocks is not None and pending[0] is not None:
            pending[0]._native._head.bias_output.data.add_(0.125)
        if after_clocks is not None:
            after_clocks()

    def evaluate(owner, state, options):
        pending[0] = owner

        def final_read():
            reads.append("final")
            if not _same_state(owner.snapshot_state(), state):
                raise ValueError("native source changed during closing resource clock")

        options["qualified_boundary"](lambda: reads.append("read"), after_clocks=final_read)
        raise AssertionError("changed closing state admitted paid game work")

    wide_games["callback"] = evaluate
    result = manager.run_qualified(check_boundary=outer)
    assert result["status"] == "failed" and len(wide_games["calls"]) == 1
    assert reads == ["read", "final"]
    assert result["records"][0]["report"] is None


@pytest.mark.parametrize("method", [True, False, 1, "BP", "", "other", np.str_("bp")])
def test_should_refuse_noncanonical_wide_methods_without_any_capture(method):
    with pytest.raises(ValueError):
        deferred.DeferredProbeRollouts(wide_method=method)


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_native_extended_flags_before_capture(flag):
    with pytest.raises(ValueError):
        deferred.DeferredProbeRollouts(wide_method="bp", extended_wide_probe_budget=flag)


def test_should_refuse_extended_count_mode_before_capture():
    with pytest.raises(ValueError):
        deferred.DeferredProbeRollouts(extended_wide_probe_budget=True)


@pytest.mark.parametrize("fault", ["weight", "clock", "rng"])
def test_should_refuse_capture_mutation_by_closing_resources_after_initial_read(
    native_states, wide_games, fault
):
    method = "gated_cpc" if fault == "rng" else "bp"
    state = deepcopy(native_states[method][0])
    manager = deferred.DeferredProbeRollouts(wide_method=method)

    def closing_resources(read_ports=None, *, after_clocks=None):
        if read_ports is not None:
            read_ports()
        if fault == "weight":
            state.native_state.head_state["bias_output"].add_(0.125)
        elif fault == "clock":
            object.__setattr__(state.native_state, "training_examples", 145)
        else:
            state.native_state.head_state["split_generator_state"] = None
        if after_clocks is not None:
            after_clocks()

    with pytest.raises(ValueError):
        manager.observer(0)(
            SimpleNamespace(snapshot_state=lambda: state), "after_probe", closing_resources
        )
    result = manager.report()
    assert result["status"] == "failed" and result["capture_work"]["payload_bytes_retained"] == 0
    assert result["capture_work"]["snapshot_calls_returned"] >= 2
    assert manager._captures == () and not wide_games["calls"]


@pytest.mark.parametrize("fault", ["omit", "double", "swallow"])
def test_should_require_exactly_one_successful_final_capture_read_even_if_boundary_swallows_error(
    native_states, wide_games, fault
):
    state = deepcopy(native_states["bp"][0])
    manager = deferred.DeferredProbeRollouts(wide_method="bp")

    def broken_boundary(read_ports=None, *, after_clocks=None):
        if read_ports is not None:
            read_ports()
        assert after_clocks is not None
        if fault == "omit":
            return
        if fault == "swallow":
            state.native_state.head_state["bias_output"].add_(0.125)
        try:
            after_clocks()
            if fault == "double":
                after_clocks()
        except ValueError:
            pass

    with pytest.raises(ValueError):
        manager.observer(0)(
            SimpleNamespace(snapshot_state=lambda: state), "after_probe", broken_boundary
        )
    result = manager.report()
    assert result["status"] == "failed" and manager._captures == ()
    assert result["capture_work"]["payload_bytes_retained"] == 0
    assert not wide_games["calls"]


def test_should_refuse_same_clock_but_wrong_full_restore_before_admitting_game(
    monkeypatch, native_states, wide_games
):
    manager, _, _ = capture_all(native_states)
    native_factory = deferred.InvariantTorchFeatureLearner

    def wrong_factory(policy):
        reader = native_factory(policy)
        restore = reader.restore_state

        def wrong_restore(state):
            restore(state)
            reader._native._head.bias_output.data.add_(0.125)

        reader.restore_state = wrong_restore
        return reader

    monkeypatch.setattr(deferred, "InvariantTorchFeatureLearner", wrong_factory)
    result = manager.run_qualified(check_boundary=qualified)
    assert result["status"] == "failed" and not wide_games["calls"]
    assert result["capture_work"]["frozen_reader_restores_returned"] == 1
    assert result["records"] == []


@pytest.mark.parametrize("hook", ["_copy", "_same"])
def test_should_refuse_replaced_instance_evidence_interpreter(native_states, wide_games, hook):
    manager, _, _ = capture_all(native_states)
    setattr(manager, hook, lambda *args, **kwargs: True)
    result = manager.run_qualified(check_boundary=qualified)
    assert result["status"] == "failed"
    assert "replaced" in result["reason"]
    assert not wide_games["calls"]


@pytest.mark.parametrize("port", ["read", "final"])
def test_should_refuse_non_none_delegated_native_port_result(
    monkeypatch, native_states, wide_games, port
):
    manager, _, _ = capture_all(native_states)

    def observe(owner, learner, phase, boundary):
        boundary(
            lambda: 1 if port == "read" else None,
            after_clocks=lambda: 1 if port == "final" else None,
        )
        raise AssertionError("invalid port result admitted work")

    monkeypatch.setattr(live.ProbeRollouts, "observe_qualified", observe)
    result = manager.run_qualified(check_boundary=qualified)
    assert result["status"] == "failed"
    assert "must return None" in result["reason"]
    assert not wide_games["calls"]


@pytest.mark.parametrize("secondary", ["omit", "double"])
def test_should_keep_first_capture_port_error_when_boundary_swallows_it(native_states, secondary):
    manager = deferred.DeferredProbeRollouts(wide_method="bp")

    def snapshot():
        raise OSError("first paid snapshot failure")

    def boundary(read_ports=None, *, after_clocks=None):
        for callback in (read_ports, after_clocks if secondary == "double" else None):
            if callback is not None:
                try:
                    callback()
                except Exception:
                    pass

    with pytest.raises(OSError, match="first paid snapshot failure"):
        manager.observer(0)(SimpleNamespace(snapshot_state=snapshot), "after_probe", boundary)
    assert "first paid snapshot failure" in manager.report()["reason"]


def test_should_keep_first_delegated_error_when_later_native_read_also_fails(
    native_states, wide_games
):
    manager, _, _ = capture_all(native_states)

    def swallow(read_ports=None, *, after_clocks=None):
        for callback in (read_ports, after_clocks):
            if callback is not None:
                try:
                    callback()
                except Exception:
                    pass

    def evaluate(owner, state, options):
        def first():
            raise OSError("first delegated failure")

        def second():
            raise ValueError("second delegated failure")

        options["qualified_boundary"](first, after_clocks=second)
        raise AssertionError("swallowed failure admitted work")

    wide_games["callback"] = evaluate
    result = manager.run_qualified(check_boundary=swallow)
    assert result["status"] == "failed"
    assert "first delegated failure" in result["reason"]
    assert "second delegated failure" not in result["reason"]


def test_should_keep_first_native_run_error_while_still_running_closing_callback(
    monkeypatch, native_states, wide_games
):
    manager, _, _ = capture_all(native_states)
    paid = []
    closed = []

    def observe(owner, learner, phase, boundary):
        def read():
            paid.append(True)
            raise OSError("first native run read error")

        boundary(read)

    def closing():
        if paid:
            closed.append(True)
            raise ValueError("secondary closing error")

    monkeypatch.setattr(live.ProbeRollouts, "observe_qualified", observe)
    result = manager.run(check_boundary=closing)
    assert result["status"] == "failed"
    assert "first native run read error" in result["reason"]
    assert closed and not wide_games["calls"]
