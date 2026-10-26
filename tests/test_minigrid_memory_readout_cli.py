"""Saved endpoint IO preserves source negatives; no game physics runs here."""

from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any

import pytest

from scripts import run_minigrid_memory_readout as cli
from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters.fifo_action_memory_learner import FifoActionMemoryLearner
from test_minigrid_memory_probe_cli import SOURCE, forbidden, replace
from test_minigrid_memory_return_evaluation import case as case

RETURN_SOURCE = SOURCE.parent.parent / "minigrid-memory-return-20261017/native.json"


@pytest.fixture(scope="module")
def reports():
    return (
        json.loads(SOURCE.read_text(encoding="utf8")),
        json.loads(RETURN_SOURCE.read_text(encoding="utf8")),
    )


def test_should_load_each_report_once_and_restore_all_eight_owned_native_endpoints(monkeypatch):
    calls = []
    original = cli._read_source

    def read(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(cli, "_read_source", read)
    for kind in (ActionMemoryLearner, FifoActionMemoryLearner):
        monkeypatch.setattr(kind, "predict", forbidden)
        monkeypatch.setattr(kind, "train_batch", forbidden)
    before = SOURCE.read_bytes(), RETURN_SOURCE.read_bytes()
    inputs = cli.load_inputs(SOURCE, RETURN_SOURCE)
    assert calls == [SOURCE, RETURN_SOURCE]
    assert [
        (row["source_updates"], row["relative_updates"], row["source_status"]) for row in inputs
    ] == [
        (0, 0, "complete"),
        (0, 16, "complete"),
        (0, 0, "complete"),
        (0, 16, "complete"),
        (128, 0, "complete"),
        (128, 1, "failed"),
        (128, 0, "complete"),
        (128, 16, "complete"),
    ]
    returned = json.loads(before[1])
    for index, row in enumerate(inputs):
        state = row["state"]
        assert state.completed_updates == row["source_updates"] + row["relative_updates"]
        assert state.training_examples == 9 * state.completed_updates
        if index % 2:
            assert (
                json.loads(json.dumps(asdict(state)))
                == returned["cohorts"][index // 2]["end_snapshot"]
            )
    assert inputs[4]["original_cohort_status"] == inputs[5]["original_cohort_status"] == "failed"
    assert inputs[-1]["state"].evictions == 26
    assert inputs[-1]["state"].retired_examples == 704
    assert before == (SOURCE.read_bytes(), RETURN_SOURCE.read_bytes())


@pytest.mark.parametrize(
    "pointer,value",
    [
        ("/status", "complete"),
        ("/protocol/seed", 18),
        ("/source_artifact", "different.json"),
        ("/return_sources/0/role", "final_test"),
        ("/cohorts/2/status", "complete"),
        ("/cohorts/2/charged_updates", True),
        ("/cohorts/2/report/attempted_updates", 1),
        ("/cohorts/2/report/reason", "different refusal"),
        ("/cohorts/2/source_unchanged", False),
        ("/cohorts/2/budget_binding_preserved", False),
        ("/cohorts/3/fifo_retirement_metadata_bytes", 0),
        ("/cohorts/3/end_snapshot/retired_examples", 703),
        ("/cohorts/3/end_snapshot/entries/0/counts/0", True),
    ],
)
def test_should_refuse_changed_source_binding_failed_prefix_or_fifo_evidence(
    reports, monkeypatch, pointer, value
):
    original, returned = deepcopy(reports)
    replace(returned, pointer, value)
    monkeypatch.setattr(cli, "_read_source", lambda path: original if path == SOURCE else returned)
    with pytest.raises(ValueError):
        cli.load_inputs(SOURCE, RETURN_SOURCE)


def test_should_describe_without_loading_sources_or_starting_any_runner(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_inputs", forbidden)
    monkeypatch.setattr(cli, "run_readouts", forbidden)
    assert cli.main(["--describe"]) == 0
    protocol = json.loads(capsys.readouterr().out)
    assert protocol["maximum_predictions"] == 512
    assert protocol["training_updates"] == protocol["score_forwards"] == 0


def test_should_publish_readout_status_without_relabeling_the_original_failed_experiment(
    tmp_path, monkeypatch, capsys
):
    inputs = object()
    monkeypatch.setattr(cli, "load_inputs", lambda *args: inputs)
    report = dict(
        status="complete",
        source_experiment_status="incomplete",
        readouts=[dict(source_status="failed", status="complete")],
    )

    def run(actual):
        assert actual is inputs
        return deepcopy(report)

    monkeypatch.setattr(cli, "run_readouts", run)
    output = tmp_path / "readout.json"
    args = [
        "--source-artifact",
        str(SOURCE),
        "--return-artifact",
        str(RETURN_SOURCE),
        "--output",
        str(output),
    ]
    assert cli.main(args) == 0
    actual = json.loads(capsys.readouterr().out)
    assert actual == json.loads(output.read_text(encoding="utf8"))
    assert actual["source_experiment_status"] == "incomplete"
    assert actual["readouts"] == report["readouts"]
    with pytest.raises(SystemExit):
        cli.main(args)


def test_should_publish_failed_readout_and_return_two(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_inputs", lambda *args: [])
    monkeypatch.setattr(cli, "run_readouts", lambda _: dict(status="failed", reason="paid fault"))
    assert (
        cli.main(["--source-artifact", str(SOURCE), "--return-artifact", str(RETURN_SOURCE)]) == 2
    )
    assert json.loads(capsys.readouterr().out)["reason"] == "paid fault"


def test_should_compute_a_and_e_contrasts_only_from_complete_saved_endpoint_pairs():
    def row(successes, status="complete"):
        return dict(
            status=status,
            source_updates=128,
            method="observation_memory_fifo",
            episodes=[
                dict(context=context, success=success)
                for context, success in zip(("A", "A", "E-blue", "E-red"), successes)
            ],
        )

    contrasts = cli._contrasts([row([True, True, False, False]), row([True] * 4)])
    assert contrasts[0]["A"]["end_minus_initial"] == 0
    assert contrasts[0]["E"]["end_minus_initial"] == 2
    assert cli._contrasts([row([True] * 4), row([], "failed")]) == []


@pytest.mark.parametrize("fail_prediction", [False, True])
def test_should_run_owned_saved_endpoints_without_updates_and_stop_at_first_paid_fault(
    case, fail_prediction
):
    fixture = case()
    fixture.work["prediction_error"] = fail_prediction
    inputs = cli.load_inputs(SOURCE, RETURN_SOURCE)
    before = [asdict(item["state"]) for item in inputs]
    report = cli.run_readouts(inputs)
    assert report["training_updates"] == 0
    assert report["source_experiment_status"] == "incomplete"
    assert [asdict(item["state"]) for item in inputs] == before
    assert report["status"] == ("failed" if fail_prediction else "complete")
    assert len(report["readouts"]) == 8
    if fail_prediction:
        assert report["readouts"][0]["work"]["live_prediction_calls_attempted"] == 1
        assert len(fixture.work["environments"]) == 1
        assert all(row["status"] == "not_run" for row in report["readouts"][1:])
        assert report["contrasts"] == []
    else:
        assert len(fixture.work["environments"]) == 32
        assert len(report["contrasts"]) == 4
        assert report["readouts"][5]["original_cohort_status"] == "failed"
        assert report["readouts"][5]["source_status"] == "failed"


@pytest.mark.parametrize("target", ["prior_report", "future_state", "future_record", "budget"])
def test_should_refuse_changed_parent_evidence_before_the_second_readout_does_work(
    case, monkeypatch, target
):
    from src.adapters import minigrid_policy_evaluation as evaluator

    fixture = case()
    inputs = cli.load_inputs(SOURCE, RETURN_SOURCE)
    original = evaluator.evaluate_memory_return_checkpoint
    returned: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []

    def evaluate(*args, **options):
        if returned:
            if target == "prior_report":
                returned[0]["episodes"][0]["success"] = not returned[0]["episodes"][0]["success"]
            elif target == "future_state":
                entry = inputs[-1]["state"].entries[0]
                object.__setattr__(entry, "counts", entry.counts[1:] + entry.counts[:1])
            elif target == "future_record":
                inputs[-1] = deepcopy(inputs[-1])
            else:
                options["budget"].record_update()
        result = original(*args, **options)
        returned.append(result)
        accepted.append(deepcopy(result))
        accepted[-1].update(
            original_cohort_status=inputs[len(accepted) - 1]["original_cohort_status"],
            original_cohort_reason=inputs[len(accepted) - 1]["original_cohort_reason"],
        )
        return result

    monkeypatch.setattr(evaluator, "evaluate_memory_return_checkpoint", evaluate)
    report = cli.run_readouts(inputs)
    assert report["status"] == "failed" and report["contrasts"] == []
    assert len(fixture.work["environments"]) == 4
    assert len(fixture.work["predictions"]) == 8
    assert report["readouts"][1]["status"] == "failed"
    assert report["readouts"][0] == accepted[0]
    assert report["changed_prior_readout_indices"] == ([0] if target == "prior_report" else [])
    assert all(row["status"] == "not_run" for row in report["readouts"][2:])


def test_should_recheck_parent_after_the_last_scorer_snapshot_before_prediction(case, monkeypatch):
    fixture = case()
    inputs = cli.load_inputs(SOURCE, RETURN_SOURCE)
    original = ActionMemoryLearner.snapshot_state
    snapshots = 0

    def snapshot(owner):
        nonlocal snapshots
        state = original(owner)
        if len(fixture.work.get("environments", [])) == 5 and owner in fixture.work.get(
            "scorers", []
        ):
            snapshots += 1
            if snapshots == 2:
                entry = inputs[-1]["state"].entries[0]
                object.__setattr__(entry, "counts", entry.counts[1:] + entry.counts[:1])
        return state

    monkeypatch.setattr(ActionMemoryLearner, "snapshot_state", snapshot)
    report = cli.run_readouts(inputs)
    assert report["status"] == "failed" and snapshots == 2
    assert len(fixture.work["environments"]) == 5
    assert len(fixture.work["predictions"]) == 8
    assert report["readouts"][1]["work"]["live_prediction_calls_attempted"] == 0
    assert report["contrasts"] == []


@pytest.mark.parametrize("port", ["snapshot", "exit", "exit_budget"])
def test_should_preserve_paid_captures_and_refuse_late_resource_callback_changes(
    case, monkeypatch, port
):
    from src.adapters import minigrid_policy_evaluation as evaluator
    from src.shared.process_memory import ProcessRssSampler

    fixture = case()
    inputs = cli.load_inputs(SOURCE, RETURN_SOURCE)
    original_evaluate = evaluator.evaluate_memory_return_checkpoint
    originals: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    budgets: list[Any] = []

    def evaluate(*args, **options):
        result = original_evaluate(*args, **options)
        budgets.append(options["budget"])
        originals.append(result)
        accepted.append(deepcopy(result))
        accepted[-1].update(
            original_cohort_status=inputs[len(accepted) - 1]["original_cohort_status"],
            original_cohort_reason=inputs[len(accepted) - 1]["original_cohort_reason"],
        )
        return result

    monkeypatch.setattr(evaluator, "evaluate_memory_return_checkpoint", evaluate)
    if port == "snapshot":
        original_snapshot = ProcessRssSampler.snapshot

        def snapshot(sampler):
            result = original_snapshot(sampler)
            if len(originals) == 8:
                originals[0]["episodes"][0]["success"] = not accepted[0]["episodes"][0]["success"]
            return result

        monkeypatch.setattr(ProcessRssSampler, "snapshot", snapshot)
    else:
        original_exit = ProcessRssSampler.__exit__

        def close(sampler, *args):
            original_exit(sampler, *args)
            if port == "exit_budget":
                budgets[-1].record_update()
            else:
                raise RuntimeError("resource close failed after paid games")

        monkeypatch.setattr(ProcessRssSampler, "__exit__", close)
    report = cli.run_readouts(inputs)
    assert report["status"] == "failed" and len(fixture.work["environments"]) == 32
    assert report["readouts"] == accepted
    assert report["training_updates"] == (1 if port == "exit_budget" else 0)
    assert report["contrasts"] == []


def test_should_keep_an_active_unknown_cost_slot_when_a_paid_delegate_report_is_lost(
    case, monkeypatch
):
    from src.adapters import minigrid_policy_evaluation as evaluator

    fixture = case()
    inputs = cli.load_inputs(SOURCE, RETURN_SOURCE)
    original = evaluator.evaluate_memory_return_checkpoint
    calls = 0

    def lose_report(*args, **options):
        nonlocal calls
        calls += 1
        result = original(*args, **options)
        if calls == 2:
            raise RuntimeError("paid readout lost its report")
        return result

    monkeypatch.setattr(evaluator, "evaluate_memory_return_checkpoint", lose_report)
    report = cli.run_readouts(inputs)
    assert report["status"] == "failed" and calls == 2 and report["contrasts"] == []
    assert len(fixture.work["environments"]) == 8
    assert report["readouts"][0]["work"]["environment_steps_returned"] == 8
    lost = report["readouts"][1]
    assert lost["status"] == "failed" and lost["delegated_report_unavailable"] is True
    assert lost["work"]["failure_work_may_be_unobserved"] is True
    assert lost["work"]["environment_steps_attempted"] is None
    assert lost["work"]["live_prediction_calls_attempted"] is None
    assert all(row["status"] == "not_run" for row in report["readouts"][2:])


@pytest.mark.parametrize(
    "args", [[], ["--describe", "--return-artifact", "x"], ["--source-artifact", "x"]]
)
def test_should_reject_incomplete_or_resource_using_preview_arguments_before_loading(
    monkeypatch, args
):
    monkeypatch.setattr(cli, "load_inputs", forbidden)
    with pytest.raises(SystemExit):
        cli.main(args)
