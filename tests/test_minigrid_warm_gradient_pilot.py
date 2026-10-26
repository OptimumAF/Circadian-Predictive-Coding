"""A proven warm128 source precedes a separate read-only gradient measurement.

Only orchestration uses fake native ports; accepted gradient mathematics and
the full five-method runner are not reconstructed or requalified here.
"""

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_minigrid_gradient_smoke as cli
from src.adapters import minigrid_learnability as scorer
from src.adapters import minigrid_reward_pilot as reward_pilot
from src.adapters import minigrid_warm_gradient_pilot as pilot
import src.app.toy_execution_budget as budgets
import src.shared.torch_runtime as torch_runtime
from test_minigrid_gradient_route import diagnostic, install_port
from test_minigrid_learning_pilot import same_state, source_batch
from test_minigrid_warm_reward import Native, ports as ports, prepare_fixture
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


@pytest.fixture
def case(monkeypatch, ports):
    work = ports.work
    work.update(raw_predictions=[], readouts=[], budgets=[], pilot_reports=[], guard_calls=0)
    clock = [10.0]
    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "monotonic", lambda: clock[0])
    monkeypatch.setattr(pilot, "_same_state", same_state)
    monkeypatch.setattr(scorer, "_same_state", same_state)
    factory, score_checkpoint = pilot._new_report, pilot._score_checkpoint
    new_budget = budgets.ToyBudgetSession

    def report_factory():
        report = factory()
        work["pilot_reports"].append(report)
        return report

    def prepare(report, memory, deadline, check_boundary):
        assert deadline == 55.0
        work["warm_deadline"] = deadline
        prepare_fixture(monkeypatch, ports, partial=work.get("warm_failure", False))
        prepared = reward_pilot.prepare_warm_start(report, memory, 190.0, check_boundary)
        prepared["sources"] = sources
        source_guard = prepared["guard"]
        warm_budget = new_budget(
            budgets.ToyExecutionBudget(max_training_updates=128, max_wall_seconds=45.0),
            lambda: clock[0],
        )
        warm_budget.updates_completed = 128
        work["warm_budget"] = warm_budget
        limits_owner, clock_owner = warm_budget.budget, warm_budget.clock
        started_at, last_clock = warm_budget.started_at, [warm_budget.last_clock]

        def budget_binding():
            if (
                warm_budget.budget is not limits_owner
                or warm_budget.clock is not clock_owner
                or warm_budget.started_at != started_at
                or warm_budget.updates_completed != 128
                or warm_budget.last_clock != last_clock[0]
            ):
                raise ValueError("original completed warm budget changed")

        def guard():
            work["guard_calls"] += 1
            budget_binding()
            source_guard()
            budget_binding()
            # The accepted warm guard legitimately observes its original clock;
            # the new use case cannot replace it or adopt an outside observation.
            warm_budget.before_final()
            last_clock[0] = warm_budget.last_clock
            if clock[0] >= deadline:
                raise ValueError("original completed warm45 deadline exhausted")
            budget_binding()

        prepared["guard"] = guard
        work["prepared"] = prepared
        mode = work.get("warm_invalid")
        if mode == "status":
            report["status"] = "incomplete"
        elif mode == "parities":
            report["parity_checks"][2] = False
        elif mode == "integer_parities":
            report["parity_checks"] = [1] * 5
        elif mode == "matched":
            report["parity_matched"] = False
        elif mode == "restored":
            report["full_state_restored"] = False
        elif mode == "returned_rows":
            prepared["rows"] = 1153
        elif mode == "returned_updates":
            prepared["updates"] = 127
        elif mode == "returned_sources":
            prepared["sources"] = sources[:2]
        elif mode == "returned_guard":
            prepared["guard"] = None
        return prepared

    def capture_budget(*args, **kwargs):
        budget = new_budget(*args, **kwargs)
        work["budgets"].append(budget)
        return budget

    def predict(owner, features):
        work["raw_predictions"].append((owner, features.copy()))
        return np.tile(np.array([[0.5, 0.5, 0, 0, 0, 0, 0]], dtype=np.float64), (len(features), 1))

    def score(source, state, **options):
        assert work["warm_report"]["status"] == "complete"
        assert work["warm_report"]["parity_checks"] == [True] * 5
        assert (
            work["warm_report"]["parity_matched"]
            is work["warm_report"]["full_state_restored"]
            is True
        )
        assert source is work["prepared"]["source"] and source.completed_updates == 128
        assert state.completed_updates == 128 and state.training_examples == 1152
        assert options["method"] == "recurrent_bc" and options["updates"] == 128
        assert options["gradient_geometry"] is True and options["sources"] is sources
        assert options["budget"] is work["budgets"][0]
        assert options["budget"].budget.max_training_updates == 0
        work["readouts"].append(dict(source=source, state=state, options=options))
        result = score_checkpoint(source, state, **options)
        callback = work.get("after_score")
        if callback:
            callback(result)
        return result

    monkeypatch.setattr(pilot, "_new_report", report_factory)
    monkeypatch.setattr(pilot, "prepare_warm_start", prepare)
    monkeypatch.setattr(pilot, "ToyBudgetSession", capture_budget)
    monkeypatch.setattr(pilot, "_score_checkpoint", score)
    monkeypatch.setattr(Native, "predict", predict)
    return SimpleNamespace(work=work, clock=clock, sources=sources)


def test_should_measure_only_after_all_warm_parity_and_full_state_checks_with_separate_budget(
    monkeypatch, case
):
    calls = install_port(monkeypatch, callback=lambda call, index: case.clock.__setitem__(0, 15.0))
    before = pickle.dumps(case.sources)
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["reason"]
    assert report["warm_start"]["parity_checks"] == [True] * 5
    assert report["gradient_readout"]["updates"] == 128
    assert len(case.work["raw_predictions"]) == 6 and len(calls) == 3
    assert [call["pair"][0].context for call in calls] == ["A", "B", "C_train"]
    assert case.work["budgets"][0] is not case.work["warm_budget"]
    budget = report["diagnostic_budget"]
    assert budget["limits"]["max_training_updates"] == 0
    assert budget["limits"]["max_wall_seconds"] == 20.0
    assert (
        budget["started_at"] == 10.0
        and budget["last_clock"] == 15.0
        and budget["updates_completed"] == 0
    )
    warm_budget = case.work["warm_budget"]
    assert (
        warm_budget.started_at == 10.0
        and warm_budget.last_clock == 15.0
        and warm_budget.updates_completed == 128
    )
    assert case.work["guard_calls"] > 0
    work = report["work"]
    assert work["native_train_calls"] == 128 and work["training_rows"] == 1152
    assert work["warm_start"]["raw_score_rows_completed"] == 90
    assert work["raw_scores"]["score_forward_calls_completed"] == 6
    assert work["raw_scores"]["score_rows_completed"] == 54
    gradient = work["cue_gradient_geometry"]
    assert gradient["pair_calls_completed"] == 3
    assert gradient["forward_rows_completed"] == gradient["backward_rows_completed"] == 54
    assert (
        gradient["geometry_reductions_completed"] == 18
        and gradient["input_array_equivalent_bytes"] == 64368
    )
    assert gradient["optimizer_steps"] == gradient["published_updates"] == 0
    assert work["failure_work_may_be_unobserved"] is False
    assert pickle.dumps(case.sources) == before
    assert not case.work.get("reward_calls") and not case.work["runtimes"]
    assert not case.work["collections"]
    calls[0]["payload"]["joint_geometry"]["cosine"] = 1.0
    assert (
        report["gradient_readout"]["cue_gradient_geometry"][0]["diagnostic"]["joint_geometry"][
            "cosine"
        ]
        == -1.0
    )


@pytest.mark.parametrize(
    "failure",
    [
        "status",
        "parities",
        "integer_parities",
        "matched",
        "restored",
        "returned_rows",
        "returned_updates",
        "returned_sources",
        "returned_guard",
    ],
)
def test_should_refuse_unproven_warm_return_before_diagnostic_budget_or_prediction(
    monkeypatch, case, failure
):
    case.work["warm_invalid"] = failure
    monkeypatch.setattr(pilot, "_score_checkpoint", forbidden)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and report["gradient_readout"]["status"] == "not_run"
    assert case.work["budgets"] == [] and case.work["raw_predictions"] == []
    assert report["work"]["native_train_calls"] == 128 and report["work"]["training_rows"] == 1152
    if failure == "integer_parities":
        assert case.work["guard_calls"] == 0


def test_should_keep_partial_warm_costs_and_skip_unavailable_diagnostics(monkeypatch, case):
    case.work["warm_failure"] = True
    monkeypatch.setattr(pilot, "_score_checkpoint", forbidden)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and "two paid updates" in report["reason"]
    assert (
        report["warm_start"]["training"]["charged_updates"]
        == report["work"]["native_train_calls"]
        == 2
    )
    assert report["work"]["training_rows"] == 18 and case.work["budgets"] == []
    assert report["gradient_readout"]["status"] == "not_run"
    assert report["work"]["cue_gradient_geometry"]["pair_calls_attempted"] == 0


@pytest.mark.parametrize("failure", ["native", "generic"])
def test_should_retain_paid_diagnostic_prefix_and_original_warm_charges(monkeypatch, case, failure):
    def fail(call, index):
        if index == 1:
            if failure == "generic":
                raise RuntimeError("gradient port did not return observed work")
            from src.adapters.torch_recurrent_gradient_diagnostics import GradientDiagnosticFailed

            raise GradientDiagnosticFailed(diagnostic(call["pair"], complete=False))

    calls = install_port(monkeypatch, callback=fail)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and len(calls) == 2
    assert report["gradient_readout"]["status"] == "failed"
    assert report["work"]["native_train_calls"] == 128 and report["work"]["training_rows"] == 1152
    assert report["work"]["raw_scores"]["score_rows_completed"] == 54
    work = report["work"]["cue_gradient_geometry"]
    assert work["pair_calls_attempted"] == 2
    assert work["forward_rows_completed"] == (27 if failure == "native" else 18)
    assert work["failure_work_may_be_unobserved"] is True
    assert case.work["warm_budget"].updates_completed == 128


@pytest.mark.parametrize(
    "fault", ["source", "state", "packet", "warm_report", "diagnostic_budget", "warm_clock"]
)
def test_should_refuse_changed_ownership_after_retaining_first_paid_pair(monkeypatch, case, fault):
    def corrupt(call, index):
        assert index == 0
        if fault == "source":
            case.work["warm_source"]._learner.state.parameters["bias_hidden"][0, 0] += 1
        elif fault == "state":
            case.work["readouts"][0]["state"].parameters["bias_hidden"][0, 0] += 1
        elif fault == "packet":
            case.sources[0].features[0, 0] += 0.1
        elif fault == "warm_report":
            case.work["warm_report"]["parity_checks"][0] = False
        elif fault == "diagnostic_budget":
            case.work["budgets"][0].started_at -= 1
        else:
            case.work["warm_budget"].last_clock += 1

    calls = install_port(monkeypatch, callback=corrupt)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and len(calls) == 1
    assert report["warm_start"]["parity_checks"] == [True] * 5
    assert report["work"]["native_train_calls"] == 128
    assert report["work"]["training_rows"] == 1152
    assert report["work"]["raw_scores"]["score_rows_completed"] == 54
    assert report["work"]["cue_gradient_geometry"]["forward_rows_completed"] == 18
    assert (
        report["gradient_readout"]["cue_gradient_geometry"][0]["diagnostic"]["status"] == "complete"
    )
    assert case.work["warm_budget"].updates_completed == 128


@pytest.mark.parametrize("time", [31.0, 86.0])
def test_should_stop_at_original_diagnostic_or_whole_deadline_without_recharging_warm(
    monkeypatch, case, time
):
    calls = install_port(monkeypatch, callback=lambda call, index: case.clock.__setitem__(0, time))
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and len(calls) == 1
    assert report["work"]["native_train_calls"] == 128
    assert report["work"]["cue_gradient_geometry"]["forward_rows_completed"] == 18
    assert report["diagnostic_budget"]["limits"]["max_wall_seconds"] == 20.0
    assert report["diagnostic_budget"]["updates_completed"] == 0
    assert case.work["warm_budget"].updates_completed == 128
    assert case.work["warm_deadline"] == 55.0


def test_should_refuse_last_source_callback_tamper_and_publish_original_paid_readout(
    monkeypatch, case
):
    calls = install_port(monkeypatch)
    snapshot = Native.snapshot_state
    armed = [False]

    def after_score(result):
        armed[0] = True

    def snapshot_and_corrupt(owner):
        state = snapshot(owner)
        if armed[0] and owner is case.work["warm_source"]._learner:
            armed[0] = False
            case.work["pilot_reports"][0]["gradient_readout"]["scores"][0]["ce"] = 1000.0
        return state

    case.work["after_score"] = after_score
    monkeypatch.setattr(Native, "snapshot_state", snapshot_and_corrupt)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and len(calls) == 3
    assert report["gradient_readout"]["scores"][0]["ce"] != 1000.0
    assert report["work"]["cue_gradient_geometry"]["forward_rows_completed"] == 54
    assert report["work"]["raw_scores"]["score_rows_completed"] == 54
    assert report["work"]["native_train_calls"] == 128


def test_should_describe_fixed_training_and_zero_update_diagnostics_without_native_work(
    monkeypatch,
):
    monkeypatch.setattr(pilot, "prepare_warm_start", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(torch_runtime, "require_torch", forbidden)
    protocol = pilot.describe_protocol()
    assert protocol["protocol_id"] == "minigrid_warm_a128_cue_gradient_geometry_v1"
    assert protocol["planned_work"]["native_train_calls"] == 128
    assert protocol["planned_work"]["diagnostic_optimizer_steps"] == 0
    assert protocol["planned_work"]["gradient_pairs"] == 3
    assert protocol["planned_work"]["diagnostic_raw_rows"] == 54
    assert protocol["planned_work"]["new_teacher_episodes"] == 0
    json.dumps(protocol, allow_nan=False)


def test_should_describe_cli_without_native_setup(monkeypatch, capsys):
    monkeypatch.setattr(torch_runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    assert cli.main(["--describe"]) == 0
    assert json.loads(capsys.readouterr().out) == pilot.describe_protocol()


@pytest.mark.parametrize("fault", ["existing", "missing_parent", "describe_output"])
def test_should_refuse_cli_publication_preflight_before_native_setup(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(torch_runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    path = tmp_path / "result.json"
    if fault == "existing":
        path.write_text("original", encoding="utf8")
    elif fault == "missing_parent":
        path = tmp_path / "absent" / "result.json"
    args = ["--output", str(path)]
    if fault == "describe_output":
        args.append("--describe")
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    if fault == "existing":
        assert path.read_text(encoding="utf8") == "original"
    else:
        assert not path.exists()


@pytest.mark.parametrize("status,expected", [("complete", 0), ("incomplete", 1)])
def test_should_publish_exact_run_result_and_return_honest_status(
    monkeypatch, tmp_path, capsys, status, expected
):
    calls = []
    torch = SimpleNamespace(
        set_num_threads=lambda value: calls.append(("intra", value)),
        set_num_interop_threads=lambda value: calls.append(("interop", value)),
    )
    monkeypatch.setattr(torch_runtime, "require_torch", lambda: torch)
    result = dict(
        status=status,
        reason=None if status == "complete" else "retained failure",
        work={"native_train_calls": 128},
    )
    monkeypatch.setattr(pilot, "run_pilot", lambda: result)
    path = tmp_path / "result.json"
    assert cli.main(["--output", str(path)]) == expected
    assert json.loads(path.read_text(encoding="utf8")) == result
    assert calls == [("intra", 1), ("interop", 1)]
    assert json.loads(capsys.readouterr().out)["bytes"] == path.stat().st_size


@pytest.mark.parametrize("fault", ["nonfinite", "oversized"])
def test_should_not_publish_invalid_or_oversized_cli_json(monkeypatch, tmp_path, fault):
    torch = SimpleNamespace(
        set_num_threads=lambda value: None, set_num_interop_threads=lambda value: None
    )
    monkeypatch.setattr(torch_runtime, "require_torch", lambda: torch)
    result = dict(status="complete", value=float("nan") if fault == "nonfinite" else "paid result")
    monkeypatch.setattr(pilot, "run_pilot", lambda: result)
    if fault == "oversized":
        monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 8)
    path = tmp_path / "result.json"
    with pytest.raises(ValueError):
        cli.main(["--output", str(path)])
    assert not path.exists()
