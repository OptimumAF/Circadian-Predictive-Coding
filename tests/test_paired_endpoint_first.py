"""Endpoint-first observation preserves the full five-readout requirement.

Only fake native ports run through the existing stream; no native experiment or
baseline is repeated. Pure/CLI cases reuse the established resource-free ports.
"""

from copy import deepcopy
import json
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.run_minigrid_paired_bc_smoke as cli
from src.adapters import minigrid_paired_bc_pilot as pilot
import src.app.toy_execution_budget as budgets
import src.shared.torch_runtime as runtime
from test_minigrid_paired_pilot import record, route as route
from test_paired_episode_stream import case as case
from test_torch_feature_rehearsal_pilot import Sampler, forbidden

ORDER = [0, 64, 16, 32, 48]


def test_should_observe_endpoint_after_initial_parity_then_finish_all_fixed_midpoints(route):
    report = pilot.run_pilot(endpoint_first=True)
    assert report["status"] == "complete", report["reason"]
    assert route.work["live"] == ORDER
    assert route.work["parity"] == [(0, 0)] and report["initial_parity"] is True
    assert report["training"]["checkpoint_updates"] == list(pilot.POINTS)
    assert len(route.work["trains"]) == report["work"]["charged_updates"] == 64
    assert report["training"]["native_training_examples"] == 1152
    assert [item["updates"] for item in report["raw_readouts"]] == ORDER
    assert [item["updates"] for item in report["evaluations"]] == ORDER
    assert report["work"]["raw_scores"]["score_rows_completed"] == 270
    assert report["work"]["live_evaluation"]["raw_score_rows_completed"] == 90
    assert report["readout_budget"]["updates_completed"] == 0
    assert report["evaluations"][-1]["scores"][0]["exact_sequence_accuracy"] == 0.0
    assert report["outcomes"]["paired"]["joint_acquisition"] is True
    assert report["outcomes"]["paired"]["ce_gain"] == pytest.approx(0.25)
    assert report["outcomes"]["paired"]["live_success_count_change"] == 2
    assert report["outcomes"]["paired"]["mean_return_gain"] == 1.0
    endpoint = report["endpoint_evidence"]
    assert endpoint["status"] == "validated" and endpoint["checkpoint_updates"] == [0, 64]
    assert endpoint["outcomes"] == report["outcomes"]
    assert endpoint["outcomes"] is not report["outcomes"]
    report["evaluations"][1]["scores"][0]["ce"] = 999.0
    assert endpoint["outcomes"]["paired"]["ce_gain"] == pytest.approx(0.25)


def test_should_refuse_endpoint_work_when_original_initial_parity_fails(route):
    route.work["bad_parity"] = True
    report = pilot.run_pilot(endpoint_first=True)
    assert report["status"] == "incomplete" and report["outcomes"] is None
    assert route.work["live"] == [0]
    assert [item["updates"] for item in report["raw_readouts"]] == [0]
    assert report["endpoint_evidence"] is None
    assert report["work"]["charged_updates"] == 64


def test_should_retain_proven_endpoint_when_only_later_original_readout_time_is_exhausted(route):
    def later_timeout(result):
        if result["updates"] == 16:
            route.work["clock"][0] = 31.0

    route.work["after_live"] = later_timeout
    report = pilot.run_pilot(endpoint_first=True)
    assert report["status"] == "incomplete" and report["outcomes"] is None
    assert route.work["live"] == [0, 64, 16]
    assert report["work"]["charged_updates"] == 64
    assert report["training"]["native_training_examples"] == 1152
    endpoint = report["endpoint_evidence"]
    assert endpoint["status"] == "validated" and endpoint["checkpoint_updates"] == [0, 64]
    assert endpoint["outcomes"]["paired"]["joint_acquisition"] is True
    assert [item["updates"] for item in report["evaluations"]][:2] == [0, 64]
    assert report["readout_budget"]["limits"]["max_wall_seconds"] == 20.0
    assert report["readout_budget"]["updates_completed"] == 0


@pytest.mark.parametrize("fault", ["paid_record", "source"])
def test_should_withdraw_endpoint_outcomes_after_later_integrity_refusal(monkeypatch, route, fault):
    sources: list[Any] = []
    construct = pilot.TorchTrainEpisodeBatchLearner

    def factory(*args, **kwargs):
        owned = construct(*args, **kwargs)
        sources.append(owned)
        return owned

    def corrupt(result):
        if result["updates"] == 16:
            if fault == "paid_record":
                route.work["pilots"][0]["raw_readouts"][0]["scores"][0]["ce"] = 999.0
            else:
                sources[0]._learner.state.parameters["weight"][0, 0] += 1

    monkeypatch.setattr(pilot, "TorchTrainEpisodeBatchLearner", factory)
    route.work["after_live"] = corrupt
    report = pilot.run_pilot(endpoint_first=True)
    assert report["status"] == "incomplete" and report["outcomes"] is None
    assert route.work["live"] == [0, 64, 16]
    assert report["endpoint_evidence"]["status"] == "invalid_or_unverified"
    assert report["endpoint_evidence"]["outcomes"] is None
    assert report["endpoint_evidence"]["reason"]
    assert report["work"]["charged_updates"] == 64
    assert report["raw_readouts"][0]["scores"][0]["ce"] != 999.0


def complete_records():
    records = [record(point, final=point == 64) for point in ORDER]
    for item in records:
        item["method"] = pilot.METHOD
    return records


def test_should_require_middle_readouts_to_complete_without_promoting_endpoint_evidence_to_full_success(
    route,
):
    def failed_middle(result):
        if result["updates"] == 16:
            result.update(status="failed", reason="paid midpoint readout incomplete")

    route.work["after_live"] = failed_middle
    report = pilot.run_pilot(endpoint_first=True)
    assert report["status"] == "incomplete" and report["outcomes"] is None
    assert route.work["live"] == [0, 64, 16]
    assert report["evaluations"][-1]["status"] == "failed"
    assert report["work"]["charged_updates"] == 64
    assert report["endpoint_evidence"]["status"] == "invalid_or_unverified"
    assert report["endpoint_evidence"]["outcomes"] is None


def test_should_select_unique_fixed_endpoints_instead_of_last_positional_middle_record():
    records = complete_records()
    assert records[-1]["updates"] == 48 and records[-1]["episodes"][0]["success"] is False
    selected = pilot._metrics(records, final_point=64)
    assert selected["joint_acquisition"] is True and selected["ce_gain"] == pytest.approx(0.25)
    assert selected["live_success_count_change"] == 2 and selected["mean_return_gain"] == 1.0
    assert pilot._metrics(records)["joint_acquisition"] is False
    reference = [record(point, final=point == 128) for point in (0, 32, 64, 96, 128)]
    assert pilot._metrics(reference, final_point=128)["joint_acquisition"] is True


@pytest.mark.parametrize(
    "fault",
    [
        "missing_endpoint",
        "duplicate_endpoint",
        "duplicate_initial",
        "status",
        "role",
        "method",
        "cue",
        "ce",
        "success",
        "exact",
    ],
)
def test_should_refuse_incomplete_or_malformed_fixed_endpoint_metadata(fault):
    records = deepcopy(complete_records())
    endpoint = records[1]
    if fault == "missing_endpoint":
        records.pop(1)
    elif fault == "duplicate_endpoint":
        records.append(deepcopy(endpoint))
    elif fault == "duplicate_initial":
        records.append(deepcopy(records[0]))
    elif fault == "status":
        endpoint["status"] = "failed"
    elif fault == "role":
        endpoint["role"] = "final_test"
    elif fault == "method":
        endpoint["method"] = "recurrent_bc"
    elif fault == "cue":
        endpoint["scores"][1]["cue"] = "blue"
    elif fault == "ce":
        endpoint["scores"][0]["ce"] = float("nan")
    elif fault == "success":
        endpoint["episodes"][0]["success"] = 1
    else:
        endpoint["scores"][0]["exact_sequence_accuracy"] = 0.5
    with pytest.raises(ValueError):
        pilot._metrics(records, final_point=64)


@pytest.mark.parametrize("bad", [None, 0, 1, "true"])
def test_should_refuse_nonbool_mode_before_resource_or_reference_work(monkeypatch, bad):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot, "_load_reference", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(endpoint_first=bad)
    with pytest.raises(ValueError):
        pilot.run_pilot(endpoint_first=bad)


def test_should_preserve_default_protocol_and_training_dose_when_declaring_new_observation_order(
    monkeypatch,
):
    monkeypatch.setattr(pilot, "_load_reference", forbidden)
    monkeypatch.setattr(runtime, "require_torch", forbidden)
    default = pilot.describe_protocol()
    assert default == pilot.describe_protocol(endpoint_first=False)
    endpoint = pilot.describe_protocol(endpoint_first=True)
    assert endpoint["protocol_id"] != default["protocol_id"]
    assert endpoint["checkpoints"] == default["checkpoints"] == list(pilot.POINTS)
    assert endpoint["readout_order"] == ORDER
    assert endpoint["planned_work"] == default["planned_work"]
    assert endpoint["limits"] == default["limits"]


@pytest.mark.parametrize("endpoint_first", [False, True])
def test_should_describe_cli_without_native_setup_and_forward_only_explicit_mode(
    monkeypatch, capsys, endpoint_first
):
    calls: list[dict[str, Any]] = []

    def describe(**options):
        calls.append(options)
        return {"observed_options": options}

    monkeypatch.setattr(pilot, "describe_protocol", describe)
    monkeypatch.setattr(runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    args = ["--describe"] + (["--endpoint-first"] if endpoint_first else [])
    assert cli.main(args) == 0
    expected = {"endpoint_first": True} if endpoint_first else {}
    assert calls == [expected]
    assert json.loads(capsys.readouterr().out) == {"observed_options": expected}


@pytest.mark.parametrize(
    "endpoint_first,status,code",
    [(False, "complete", 0), (True, "complete", 0), (True, "incomplete", 1)],
)
def test_should_preserve_cli_exit_and_optional_keyword_without_running_a_model(
    monkeypatch, capsys, endpoint_first, status, code
):
    calls: list[dict[str, Any]] = []
    torch = SimpleNamespace(
        set_num_threads=lambda value: None, set_num_interop_threads=lambda value: None
    )
    monkeypatch.setattr(runtime, "require_torch", lambda: torch)

    def run(**options):
        calls.append(options)
        return dict(status=status, reason=None if code == 0 else "retained later failure")

    monkeypatch.setattr(pilot, "run_pilot", run)
    assert cli.main(["--endpoint-first"] if endpoint_first else []) == code
    assert calls == [{"endpoint_first": True} if endpoint_first else {}]
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_should_reprove_source_after_last_resource_callback_on_later_wall_refusal(
    monkeypatch, route
):
    sources: list[Any] = []
    changed = [False]
    construct = pilot.TorchTrainEpisodeBatchLearner

    def factory(*args, **kwargs):
        source = construct(*args, **kwargs)
        sources.append(source)
        return source

    class LateSampler(Sampler):
        exited = False

        def __exit__(self, *args):
            self.exited = True
            return False

        def sample(self):
            report = route.work["pilots"][0] if route.work["pilots"] else None
            if (
                self.exited
                and report is not None
                and report["total_seconds"] is not None
                and not changed[0]
            ):
                changed[0] = True
                sources[0]._learner.state.parameters["weight"][0, 0] += 1
            return 100

    def later_timeout(result):
        if result["updates"] == 16:
            route.work["clock"][0] = 31.0

    monkeypatch.setattr(pilot, "TorchTrainEpisodeBatchLearner", factory)
    monkeypatch.setattr(pilot, "ProcessRssSampler", LateSampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", LateSampler)
    route.work["after_live"] = later_timeout
    report = pilot.run_pilot(endpoint_first=True)
    assert changed[0] is True
    assert sources[0]._learner.state.parameters["weight"][0, 0] == 1.0
    assert report["status"] == "incomplete" and report["outcomes"] is None
    assert report["endpoint_evidence"]["status"] == "invalid_or_unverified"
    assert report["endpoint_evidence"]["outcomes"] is None
    assert report["work"]["charged_updates"] == 64
    assert report["training"]["native_training_examples"] == 1152
    assert [item["updates"] for item in report["evaluations"]][:2] == [0, 64]


@pytest.mark.parametrize("fault", ["forged", "deleted"])
def test_should_refuse_pending_endpoint_field_tamper_even_when_readout_time_also_expires(
    route, fault
):
    def corrupt(result):
        if result["updates"] == 16:
            public = route.work["pilots"][0]
            if fault == "forged":
                public["endpoint_evidence"] = dict(
                    status="validated", outcomes={"manufactured": True}
                )
            else:
                del public["endpoint_evidence"]
            route.work["clock"][0] = 31.0

    route.work["after_live"] = corrupt
    report = pilot.run_pilot(endpoint_first=True)
    assert route.work["live"] == [0, 64, 16]
    assert report["status"] == "incomplete" and report["outcomes"] is None
    assert report["endpoint_evidence"]["status"] == "invalid_or_unverified"
    assert report["endpoint_evidence"]["outcomes"] is None
    assert report["work"]["charged_updates"] == 64
    assert [item["updates"] for item in report["evaluations"]][:2] == [0, 64]


def test_should_refuse_wrong_raw_endpoint_header_before_live_endpoint_and_capture(
    monkeypatch, route
):
    score = pilot._score_checkpoint

    def wrong_header(*args, **kwargs):
        paid = score(*args, **kwargs)
        if kwargs["updates"] == 64:
            paid["updates"] = 16
        return paid

    monkeypatch.setattr(pilot, "_score_checkpoint", wrong_header)
    report = pilot.run_pilot(endpoint_first=True)
    assert report["status"] == "incomplete" and report["outcomes"] is None
    assert route.work["live"] == [0]
    assert report["endpoint_evidence"] is None
    assert report["raw_readouts"][-1]["updates"] == 16
    assert report["work"]["raw_scores"]["score_rows_completed"] == 108
    assert report["work"]["charged_updates"] == 64


@pytest.mark.parametrize("secondary", [None, "close_error", "episode_readback", "report_readback"])
def test_should_classify_only_wall_exhaustion_without_secondary_cleanup_or_readback_failure(
    secondary,
):
    report: dict[str, Any] = dict(
        reason="original readout time exhausted",
        readback_error=None,
        episodes=[dict(close_error=None, readback_error=None)],
    )
    if secondary == "close_error":
        report["episodes"][0]["close_error"] = "native environment close failed"
    elif secondary == "episode_readback":
        report["episodes"][0]["readback_error"] = "native episode prefix unknown"
    elif secondary == "report_readback":
        report["readback_error"] = "final learned-state proof failed"
    cause = budgets.ToyExecutionStopped(
        budgets.ToyExecutionStop(
            reason="max_wall_seconds",
            updates_completed=0,
            elapsed_seconds=21.0,
            checkpoint_position=None,
        )
    )
    failure = pilot.MiniGridEvaluationFailed(report)
    failure.__cause__ = cause
    assert pilot._wall_stop(cause) is True
    assert pilot._wall_stop(failure) is (secondary is None)
    assert failure.__cause__ is cause
