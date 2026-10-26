"""Post-training endpoint games with real guards and scripted typed state ports.

The benchmark prefix is the accepted A128 fixture. Probe updates, predictions,
and delegated games are scripted engineering ports, never native count learning
or physics. The actual paid probe/return use cases and full snapshot admission
remain in the composition.
"""

from copy import deepcopy
import pickle
from types import SimpleNamespace
from typing import Any
import pytest

from src.adapters import minigrid_deferred_probe_rollout as deferred
from src.adapters import minigrid_learnability as pilot
from src.adapters.invariant_action_memory_learner import (
    InvariantActionMemoryLearner,
    InvariantMemorySnapshot,
    copy_invariant_memory_snapshot,
)
from src.core.action_memory import ActionMemorySnapshot
from test_minigrid_learning_pilot import evaluator as evaluator
from test_minigrid_memory_control import FLAGS, NATIVE_POLICY, POLICY, memory_environment
from test_minigrid_probe_rollout import ports as ports, report as game_report
from test_torch_feature_rehearsal_pilot import forbidden


def environment(monkeypatch, evaluator):
    """Only owned frozen readers can be constructed for deferred games."""
    monkeypatch.setattr(InvariantActionMemoryLearner, "train_batch", forbidden)
    monkeypatch.setattr(InvariantActionMemoryLearner, "predict", forbidden)
    case = memory_environment(monkeypatch, evaluator)
    readers = []
    restores = []
    snapshots = []
    endpoints = []
    run_probe = pilot.run_fresh_task_probe

    def observe_probe(*args, **options):
        observer = options["endpoint_observer"]

        def observe(learner, phase, boundary):
            endpoints.append((phase, copy_invariant_memory_snapshot(learner.snapshot_state())))
            observer(learner, phase, boundary)

        options["endpoint_observer"] = observe
        return run_probe(*args, **options)

    class Reader:
        def __init__(self):
            self.state = InvariantMemorySnapshot(
                POLICY, ActionMemorySnapshot(NATIVE_POLICY, (), 0, 0)
            )

        def snapshot_state(self):
            snapshots.append(self)
            return copy_invariant_memory_snapshot(self.state)

        def restore_state(self, state):
            self.state = copy_invariant_memory_snapshot(state)
            restores.append((self, deepcopy(state)))

        train_batch = staticmethod(forbidden)
        predict = staticmethod(forbidden)

    def reader_factory():
        reader = Reader()
        readers.append(reader)
        return reader

    monkeypatch.setattr(deferred.memory, "learner", reader_factory)
    monkeypatch.setattr(deferred, "monotonic", lambda: 1.0)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", observe_probe)
    return SimpleNamespace(
        case=case, readers=readers, restores=restores, snapshots=snapshots, endpoints=endpoints
    )


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], object()])
def test_should_reject_nonboolean_deferred_mode_before_resources(monkeypatch, value):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot, "_history_learner", forbidden)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**FLAGS, deferred_probe_rollouts=value)


@pytest.mark.parametrize(
    "changes", [dict(memory_control=False), dict(probe_rollouts=True), dict(return_to_a=False)]
)
def test_should_require_full_count_recipe_and_exclusive_game_mode_before_resources(
    monkeypatch, changes
):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot, "_history_learner", forbidden)
    options = dict(FLAGS, **changes)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**options, deferred_probe_rollouts=True)


def test_should_finish_all_learning_before_restoring_four_genuine_paid_endpoints(
    monkeypatch, evaluator, ports
):
    setup = environment(monkeypatch, evaluator)
    case = setup.case
    originals = pickle.dumps(evaluator.sources)
    observed = []

    def evaluate(reader, state, options):
        assert len(case.work["trains"]) == 64
        assert case.work["arms"][0]["budget"].updates_completed == 128
        assert len(case.work["probes"]) == 2
        assert all(
            call[key].updates_completed == 16
            for call in case.work["probes"]
            for key in ("budget", "return_budget")
        )
        assert all(reader is not owner for owner in case.work["children"])
        assert reader.snapshot_state() == state
        observed.append((state.completed_updates, state.training_examples, options["phase"]))
        options["check_boundary"]()
        return game_report()

    ports["callback"] = evaluate
    result = pilot.run_pilot(**FLAGS, deferred_probe_rollouts=True)
    assert result["status"] == "complete", result["reason"]
    assert observed == [
        (16, 144, "after_probe"),
        (32, 288, "after_return"),
        (144, 1296, "after_probe"),
        (160, 1440, "after_return"),
    ]
    assert 1 <= len(setup.readers) <= 2 and len(setup.restores) == 4
    assert len(setup.snapshots) >= 4
    assert [state.completed_updates for _, state in setup.restores] == [16, 32, 144, 160]
    # A scalar-only reconstruction would lose the actual native count history.
    for (_, restored), (_, captured) in zip(setup.restores, setup.endpoints, strict=True):
        assert restored == captured and restored is not captured
        assert restored.native_state is not captured.native_state
        assert (
            sum(sum(entry.counts) for entry in restored.native_state.entries)
            == restored.training_examples
        )
    phase = result["deferred_probe_rollouts"]
    assert (
        phase["status"] == "complete"
        and phase["entry_validated"] is phase["exit_validated"] is True
    )
    assert len(phase["report"]["captures"]) == len(phase["report"]["records"]) == 4
    assert all(record["status"] == "complete" for record in phase["report"]["records"])
    row = result["methods"][0]
    assert (
        row["charged_updates"] == 128 and row["source_unchanged"] is row["actor_unchanged"] is True
    )
    assert all(
        slot["charged_updates"] == slot["return_charged_updates"] == 16 for slot in row["probes"]
    )
    assert result["work"]["native_train_calls"] == 192
    assert result["work"]["total_score_forwards_completed"] == 94
    assert result["work"]["learner_live_episodes"] == 16
    assert result["work"]["total_prediction_forwards_completed"] == 110
    assert len(case.work["trains"]) == 64 and len(case.work["predicts"]) == 94
    assert pickle.dumps(evaluator.sources) == originals


@pytest.mark.parametrize("fault", ["report", "budget", "source"])
def test_should_refuse_closing_original_cohort_fault_after_retaining_all_paid_games(
    monkeypatch, evaluator, ports, fault
):
    setup = environment(monkeypatch, evaluator)
    case = setup.case

    def corrupt(reader, state, options):
        assert len(case.work["trains"]) == 64
        if len(ports["calls"]) == 1:
            if fault == "report":
                case.work["slots"][0]["report"] = deepcopy(case.work["slots"][0]["report"])
            elif fault == "budget":
                case.work["probes"][0]["budget"].clock = lambda: 1.0
            else:
                object.__setattr__(case.work["arms"][0]["source"].state, "extra", None)
        # Return the paid receipt before the post-port guard refuses corruption.
        return game_report()

    ports["callback"] = corrupt
    result = pilot.run_pilot(**FLAGS, deferred_probe_rollouts=True)
    assert result["status"] == "incomplete" and result["reason"]
    phase = result["deferred_probe_rollouts"]
    assert phase["entry_validated"] is True and phase["exit_validated"] is False
    assert phase["status"] == "failed"
    records = phase["report"]["records"]
    returned = [record for record in records if record.get("report") is not None]
    assert returned
    assert sum(
        record["report"]["work"]["live_prediction_calls_returned"] for record in returned
    ) in (4, 16)
    assert len(case.work["trains"]) == 64
    assert result["work"]["native_train_calls"] == 192
    assert result["methods"][0]["probes"][0]["report"]["completed_updates"] == 16
    assert result["methods"][0]["probes"][0]["report"]["return_phase"]["completed_updates"] == 16


@pytest.mark.parametrize("fault", ["report", "budget"])
def test_should_refuse_unavailable_entry_proof_before_starting_any_game(
    monkeypatch, evaluator, ports, fault
):
    setup = environment(monkeypatch, evaluator)
    phase_runner = pilot._run_deferred_phase

    def enter(*args, **options):
        assert len(setup.case.work["trains"]) == 64
        if fault == "report":
            slot = setup.case.work["slots"][0]
            slot["report"] = deepcopy(slot["report"])
        else:
            setup.case.work["probes"][0]["return_budget"].clock = lambda: 1.0
        return phase_runner(*args, **options)

    monkeypatch.setattr(pilot, "_run_deferred_phase", enter)
    result = pilot.run_pilot(**FLAGS, deferred_probe_rollouts=True)
    assert result["status"] == "incomplete"
    phase = result["deferred_probe_rollouts"]
    assert phase["entry_validated"] is False
    assert phase["status"] == "failed" and phase["reason"]
    assert ports["calls"] == [] and setup.readers == [] and setup.restores == []
    assert len(phase["report"]["captures"]) == 4
    assert len(setup.case.work["trains"]) == 64
    assert result["work"]["native_train_calls"] == 192


def test_should_keep_original_global_deadline_and_paid_game_receipt_on_expiry(
    monkeypatch, evaluator, ports
):
    setup = environment(monkeypatch, evaluator)
    now = [1.0]
    monkeypatch.setattr(pilot, "monotonic", lambda: now[0])

    def expire(reader, state, options):
        assert len(setup.case.work["trains"]) == 64
        now[0] = 182.0  # The existing 180-second origin was 1, not game entry.
        return game_report()

    ports["callback"] = expire
    result = pilot.run_pilot(**FLAGS, deferred_probe_rollouts=True)
    assert result["status"] == "incomplete" and result["reason"]
    phase = result["deferred_probe_rollouts"]
    assert phase["entry_validated"] is True and phase["exit_validated"] is False
    returned = [record for record in phase["report"]["records"] if record["report"] is not None]
    assert len(returned) == 1
    assert returned[0]["report"]["work"]["live_prediction_calls_returned"] == 4
    assert len(ports["calls"]) == 1 and len(setup.case.work["trains"]) == 64
    assert all(
        call[key].started_at == 1.0
        and call[key].budget.max_training_updates == 16
        and call[key].updates_completed == 16
        for call in setup.case.work["probes"]
        for key in ("budget", "return_budget")
    )


def test_should_close_phase_proofs_and_mark_work_unknown_when_the_summary_getter_raises():
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget

    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=128), lambda: 0.0)
    for _ in range(128):
        budget.before_update()
        budget.record_update()
    proofs = []

    def lost():
        raise OSError("lost phase summary")

    manager = SimpleNamespace(check=lambda: None, run_qualified=lambda **options: None, report=lost)
    result = pilot._run_deferred_phase(
        manager,
        [dict(budget=budget)],
        check_phase=lambda: proofs.append("proof"),
        resources=lambda: None,
    )
    assert proofs == ["proof", "proof"]
    assert result["status"] == "failed" and result["report"] is None
    assert result["entry_validated"] is result["exit_validated"] is True
    assert "lost phase summary" in result["reason"]
    work = dict(
        total_score_forwards_completed=94,
        total_score_rows_completed=846,
        source_environment_steps_returned=90,
        failure_work_may_be_unobserved=False,
    )
    deferred.add_work(work, result)
    assert work["learner_live_episodes"] is work["total_prediction_forwards_completed"] is None
    assert work["failure_work_may_be_unobserved"] is True


def test_should_keep_the_paid_training_prefix_when_an_unreached_summary_getter_raises(
    monkeypatch, evaluator, ports
):
    setup = environment(monkeypatch, evaluator)

    def stopped(*args, **options):
        raise RuntimeError("training stopped before capture")

    def lost(self):
        raise OSError("lost unreached summary")

    monkeypatch.setattr(pilot, "_run_probe", stopped)
    monkeypatch.setattr(deferred.DeferredProbeRollouts, "report", lost)
    result = pilot.run_pilot(**FLAGS, deferred_probe_rollouts=True)
    assert result["status"] == "incomplete" and "training stopped" in result["reason"]
    assert result["methods"][0]["charged_updates"] == 128
    assert result["work"]["native_train_calls"] == 128
    assert result["deferred_probe_rollouts"]["report"] is None
    assert result["deferred_probe_rollouts"]["entry_validated"] is False
    assert setup.readers == [] and ports["calls"] == []


def test_should_keep_the_returned_paid_report_without_reading_a_later_failed_getter():
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget

    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=128), lambda: 0.0)
    budget.updates_completed = 128
    returned: dict[str, Any] = dict(
        status="complete",
        reason=None,
        capture_work={},
        records=[
            dict(
                report=game_report(),
                charged_updates=0,
                replay_examples=0,
                evaluator_admitted=True,
                timing_complete=True,
            )
        ],
    )
    proofs = []

    def lost():
        raise AssertionError("later getter must not replace returned paid evidence")

    manager = SimpleNamespace(
        check=lambda: None, run_qualified=lambda **options: returned, report=lost
    )
    result = pilot._run_deferred_phase(
        manager,
        [dict(budget=budget)],
        check_phase=lambda: proofs.append("proof"),
        resources=lambda: None,
    )
    assert result["status"] == "complete" and proofs == ["proof", "proof"]
    assert result["report"] == returned and result["report"] is not returned
    returned["records"].clear()
    work: dict[str, Any] = dict(
        total_score_forwards_completed=94,
        total_score_rows_completed=846,
        source_environment_steps_returned=90,
        failure_work_may_be_unobserved=False,
    )
    deferred.add_work(work, result)
    assert work["probe_rollouts"]["live_prediction_calls_returned"] == 4
    assert work["probe_rollouts"]["failure_work_may_be_unobserved"] is False


def phase_budget():
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.shared.process_memory import ProcessRssSegment

    now = [0.0]
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=128), lambda: now[0])
    budget.updates_completed = 128
    budget.process_rss_segment = ProcessRssSegment(17, 100, 100, 1, 0.01)
    return budget, now


def test_should_bind_the_benchmark_after_its_original_entry_clock_and_rss_observation():
    from dataclasses import replace

    budget, now = phase_budget()
    calls = []

    def proof():
        now[0] += 1.0
        budget.before_final()
        segment = budget.process_rss_segment
        assert segment is not None
        budget.process_rss_segment = replace(segment, sample_count=segment.sample_count + 1)

    def games(**options):
        assert budget.last_clock == 1.0
        assert budget.process_rss_segment.sample_count == 2
        options["check_boundary"]()
        calls.append("games")
        return dict(status="complete", reason=None)

    manager = SimpleNamespace(check=lambda: None, run_qualified=games, report=lambda: None)
    result = pilot._run_deferred_phase(
        manager, [dict(budget=budget)], check_phase=proof, resources=lambda: None
    )
    assert result["status"] == "complete", result["reason"]
    assert result["entry_validated"] is result["exit_validated"] is True
    assert calls == ["games"] and budget.last_clock == 2.0
    assert budget.process_rss_segment.sample_count == 3
    assert budget.updates_completed == 128 and budget.replay_examples_completed == 0


@pytest.mark.parametrize("fault", ["owner", "clock", "limits", "updates", "replay", "progress"])
def test_should_keep_preentry_budget_authority_when_the_phase_proof_changes_it(fault):
    from src.app.toy_execution_budget import ToyExecutionProgress

    budget, now = phase_budget()
    arm = dict(budget=budget)
    proofs = []
    games = []

    def proof():
        proofs.append("proof")
        if len(proofs) != 1:
            return
        now[0] = 1.0
        budget.before_final()
        if fault == "owner":
            arm["budget"] = phase_budget()[0]
        elif fault == "clock":
            budget.clock = lambda: 1.0
        elif fault == "limits":
            object.__setattr__(budget.budget, "max_training_updates", 129)
        elif fault == "updates":
            budget.updates_completed = 127
        elif fault == "replay":
            budget.replay_examples_completed = 1
        else:
            budget.progress = ToyExecutionProgress(updates_completed=128)

    manager = SimpleNamespace(
        check=lambda: None,
        run_qualified=lambda **options: games.append("game"),
        report=lambda: dict(status="not_run", reason=None),
    )
    result = pilot._run_deferred_phase(manager, [arm], check_phase=proof, resources=lambda: None)
    assert result["status"] == "failed" and result["entry_validated"] is False
    assert games == [] and proofs == ["proof", "proof"]


@pytest.mark.parametrize("fault", ["clock", "rss"])
def test_should_refuse_unobserved_benchmark_progress_inside_the_game_phase(fault):
    from dataclasses import replace

    budget, now = phase_budget()
    proofs = []

    def proof():
        proofs.append("proof")
        now[0] += 1.0
        budget.before_final()

    def games(**options):
        if fault == "clock":
            budget.last_clock += 0.5
        else:
            segment = budget.process_rss_segment
            assert segment is not None
            budget.process_rss_segment = replace(segment, sample_count=999)
        options["check_boundary"]()
        raise AssertionError("unobserved progress admitted")

    manager = SimpleNamespace(
        check=lambda: None,
        run_qualified=games,
        report=lambda: dict(status="failed", reason="unobserved progress"),
    )
    result = pilot._run_deferred_phase(
        manager, [dict(budget=budget)], check_phase=proof, resources=lambda: None
    )
    assert result["status"] == "failed" and result["entry_validated"] is True
    assert result["exit_validated"] is False and proofs == ["proof", "proof"]
    assert "unobserved clock or resource progress" in result["reason"]
