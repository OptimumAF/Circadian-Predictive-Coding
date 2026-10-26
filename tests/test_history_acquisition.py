"""Endpoint-only public-history acquisition composes accepted ports and references.

Native math, cold expansion and evaluator behavior have separate qualification.
The bounded training stub here verifies admission/order/paid evidence without a
second 64-update fake runner or any baseline/native/environment execution.
"""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.run_minigrid_paired_bc_smoke as cli
from src.adapters import minigrid_paired_bc_pilot as pilot
from src.adapters import torch_train_episode_batch as paired
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
import src.app.toy_execution_budget as budgets
import src.shared.torch_runtime as runtime
from src.core.minigrid_key_door import predefined_layouts
from test_minigrid_learning_pilot import source_batch
from test_paired_episode_stream import case as case
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


def endpoint(point, *, history=False):
    ce = 2.0 if point == 0 else (2.2 if history else 1.8)
    correct_probability = float(np.exp(-ce))
    probabilities = np.full((9, 7), (1 - correct_probability) / 6, dtype=np.float64)
    probabilities[:, 1] = correct_probability
    exact = float(np.argmax(probabilities[0]) == 1)
    returned = -1.0 if history and point == 64 else 0.0
    return dict(
        method=pilot.METHOD,
        updates=point,
        role="train",
        status="complete",
        reason=None,
        scores=[
            dict(
                context="A",
                cue=cue,
                ce=ce,
                accuracy=exact,
                exact_sequence_accuracy=exact,
                probabilities=probabilities.tolist(),
            )
            for cue in ("blue", "red")
        ],
        episodes=[
            dict(
                context="A",
                cue=cue,
                status="complete",
                success=False,
                predictions=[],
                environment=dict(transitions=[]),
                **{"return": returned},
            )
            for cue in ("blue", "red")
        ],
        work=dict(
            raw_score_forwards_attempted=2,
            raw_score_forwards_completed=2,
            raw_score_rows_attempted=18,
            raw_score_rows_completed=18,
            live_prediction_calls_attempted=4,
            live_prediction_calls_returned=4,
            live_sessions_started=2,
            environment_resets_completed=2,
            environment_steps_returned=4,
        ),
    )


@pytest.fixture
def history_route(monkeypatch, case):
    work = case.work
    work.update(events=[], evaluations=[], reports=[], clock=[10.0])
    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    saved = [endpoint(point) for point in (0, 64)]
    old = deepcopy(saved[0])
    old["method"] = "recurrent_bc"
    admitted = dict(
        sources=sources, provenance={"original_train_reference": True}, evaluations=[old]
    )
    paired_reference = dict(
        evaluations=saved,
        metrics=pilot._metrics(saved, final_point=64),
        provenance=dict(full_status="incomplete", endpoint_status="validated"),
    )
    monkeypatch.setattr(pilot, "monotonic", lambda: work["clock"][0])
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "_load_reference", lambda: {})
    monkeypatch.setattr(pilot, "_admit_reference", lambda raw: admitted)
    monkeypatch.setattr(pilot, "_load_paired_reference", lambda: paired_reference)
    monkeypatch.setattr(pilot, "_score_checkpoint", forbidden)
    builtin_dict = dict

    def capture(*args, **kwargs):
        value = builtin_dict(*args, **kwargs)
        if "protocol" in kwargs and "training" in kwargs:
            work["reports"].append(value)
            monkeypatch.delattr(pilot, "dict")
        return value

    monkeypatch.setattr(pilot, "dict", capture, raising=False)

    def initialize(schedule, original_sources, *, diagnostics=None):
        assert original_sources is sources and len(schedule) == 128
        assert all(
            episode.role == "train"
            and episode.context == "A"
            and len(episode.transitions[0].observation) == 296
            for episode in schedule
        )
        learner = paired.TorchTrainEpisodeBatchLearner(
            TorchRecurrentPolicy(296, 8, 7, 17), schedule
        )
        details: dict[str, Any] = diagnostics if diagnostics is not None else {}
        details.update(
            status="complete",
            before_training=True,
            initial_law_matches=True,
            source_unchanged=True,
            raw_source_unchanged=True,
            forward_calls=4,
            forward_rows=36,
            forward_calls_attempted=4,
            forward_rows_attempted=36,
            curves=[
                dict(
                    context="A",
                    cue=score["cue"],
                    raw_probabilities=deepcopy(score["probabilities"]),
                    history_probabilities=deepcopy(score["probabilities"]),
                    maximum_absolute_difference=0.0,
                )
                for score in saved[0]["scores"]
            ],
        )
        work["initialization"] = details
        work["events"].append("initialized")
        fault = work.get("initialization_fault")
        if fault == "paid_failure":
            details.update(
                status="failed",
                reason="second initial-law prediction failed",
                forward_calls=1,
                forward_rows=9,
                forward_calls_attempted=2,
                forward_rows_attempted=18,
            )
            raise ValueError("second initial-law prediction failed")
        if fault == "retained_law":
            details["curves"][0]["raw_probabilities"][0][0] += 1e-5
        elif fault is not None:
            key, value = fault
            details[key] = value
        return learner, details

    def train(row, packets, memory, deadline, boundary, **options):
        assert options["public_history"] is True and options["checkpoint_updates"] == pilot.POINTS
        assert deadline == 55.0 and len(packets) == 64
        assert all(
            packet.features.shape == (18, 296)
            and packet.episode is None
            and packet.role == "train"
            and packet.context == "A"
            for packet in packets
        )
        row["charged_updates"] = 0
        source = options["learner_factory"](pilot.METHOD)
        assert work["events"] == ["initialized"]
        boundary()
        initial = source.snapshot_state()
        states = [
            (
                point,
                replace(
                    initial,
                    completed_updates=point,
                    training_episodes=2 * point,
                    training_examples=18 * point,
                ),
            )
            for point in pilot.POINTS
        ]
        row.update(
            status="trained",
            charged_updates=64,
            native_training_episodes=128,
            native_training_examples=1152,
            source_unchanged=True,
            actor_unchanged=True,
            checkpoint_updates=list(pilot.POINTS),
            stream_result=dict(
                completed_updates=64,
                training_examples=1152,
                applied_updates=[dict(update_number=i + 1) for i in range(64)],
            ),
        )
        work["events"].append("trained64")
        if work.get("corrupt_initialization_during_training"):
            work["reports"][0]["initialization"]["forward_rows"] = 0
            boundary()
        seal, source_seal, states_seal = (
            pickle.dumps(row),
            pickle.dumps(initial),
            pickle.dumps(states),
        )
        budget = SimpleNamespace(updates_completed=64, started_at=10.0, last_clock=10.0)

        def guard():
            if (
                pickle.dumps(row) != seal
                or pickle.dumps(source.snapshot_state()) != source_seal
                or pickle.dumps(states) != states_seal
                or budget.updates_completed != 64
            ):
                raise ValueError("preceding history TRAIN proof changed")

        arm = dict(
            source=source,
            states=states,
            budget=budget,
            stream=SimpleNamespace(finished=True),
            guard=guard,
        )
        work["arm"] = arm
        return arm

    def evaluate(source, state, **options):
        assert work["events"][:2] == ["initialized", "trained64"]
        assert options["public_history"] is True and options["sources"] == sources[:2]
        assert options["layouts"] == predefined_layouts()[:1] and options["role"] == "train"
        assert state.policy.feature_dim == 296
        point = options["updates"]
        options["check_boundary"]()
        work["evaluations"].append(point)
        value = endpoint(point, history=True)
        if work.get("failed_readout") == point:
            value.update(status="failed", reason="paid history readout failed")
            value["work"]["live_prediction_calls_returned"] = 1
            raise pilot.MiniGridEvaluationFailed(value)
        callback = work.get("after_readout")
        if callback:
            callback(value)
        options["check_boundary"]()
        return value

    monkeypatch.setattr(pilot, "_history_learner", initialize)
    monkeypatch.setattr(pilot.shared, "_train_arm", train)
    monkeypatch.setattr(pilot, "evaluate_checkpoint", evaluate)
    return SimpleNamespace(work=work, sources=sources, saved=paired_reference)


def test_should_describe_distinct_endpoint_protocol_and_preserve_default_wire(monkeypatch):
    for name in (
        "_load_reference",
        "_load_paired_reference",
        "_history_learner",
        "evaluate_checkpoint",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    original = pilot.describe_protocol()
    assert pilot.describe_protocol(public_history=False) == original
    protocol = pilot.describe_protocol(public_history=True)
    assert protocol["protocol_id"] != original["protocol_id"]
    assert protocol["feature_dim"] == 296 and protocol["checkpoints"] == [0, 64]
    assert protocol["training_checkpoints"] == [0, 16, 32, 48, 64]
    assert protocol["parameter_counts"] == dict(raw=1319, history=2503, added=1184)
    work = protocol["planned_work"]
    assert (work["native_training_updates"], work["episode_exposures"], work["training_rows"]) == (
        64,
        128,
        1152,
    )
    assert (work["initialization_forward_calls"], work["initialization_forward_rows"]) == (4, 36)
    assert work["raw_scorer_calls"] == work["raw_scorer_rows"] == 0
    assert (
        work["live_evaluator_raw_calls"],
        work["live_evaluator_raw_rows"],
        work["live_episodes"],
    ) == (4, 36, 4)


@pytest.mark.parametrize(
    "options",
    [
        dict(public_history=1),
        dict(public_history=None),
        dict(public_history=np.bool_(True)),
        dict(public_history=True, endpoint_first=True),
    ],
)
def test_should_refuse_invalid_or_combined_modes_before_resource_work(monkeypatch, options):
    monkeypatch.setattr(pilot, "monotonic", forbidden)
    monkeypatch.setattr(pilot, "_load_reference", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(**options)
    with pytest.raises(ValueError):
        pilot.run_pilot(**options)


def test_should_initialize_then_finish_training_then_read_only_fixed_endpoints(history_route):
    route = history_route
    saved_before = pickle.dumps(route.saved)
    report = pilot.run_pilot(public_history=True)
    assert report["status"] == "complete", report["reason"]
    assert route.work["events"] == ["initialized", "trained64"]
    assert route.work["evaluations"] == [0, 64] and report["raw_readouts"] == []
    assert report["initialization"]["before_training"] is True
    assert report["initialization"]["initial_law_matches"] is True
    assert report["initial_parity"] is True
    assert report["training"]["checkpoint_updates"] == list(pilot.POINTS)
    assert (
        report["training"]["charged_updates"],
        report["training"]["native_training_examples"],
    ) == (64, 1152)
    outcomes = report["outcomes"]
    assert outcomes["history"]["joint_acquisition"] is False
    assert outcomes["raw_paired"]["joint_acquisition"] is False
    assert outcomes["history_minus_raw_paired"]["ce_gain"] == pytest.approx(-0.4)
    assert outcomes["history_minus_raw_paired"]["mean_return_gain"] == -1.0
    assert report["work"]["initialization"] == dict(forward_calls=4, forward_rows=36)
    assert report["readout_budget"]["updates_completed"] == 0
    assert report["work"]["charged_updates"] == 64
    assert pickle.dumps(route.saved) == saved_before


@pytest.mark.parametrize(
    "fault",
    [
        ("status", "failed"),
        ("before_training", False),
        ("initial_law_matches", False),
        ("initial_law_matches", 1),
        ("source_unchanged", False),
        ("raw_source_unchanged", False),
        ("forward_calls", 3),
        ("forward_rows", 35),
        "retained_law",
    ],
)
def test_should_refuse_unverified_initialization_before_any_training_or_readout(
    history_route, fault
):
    history_route.work["initialization_fault"] = fault
    report = pilot.run_pilot(public_history=True)
    assert report["status"] == "incomplete" and report["outcomes"] is None
    assert history_route.work["events"] == ["initialized"]
    assert history_route.work["evaluations"] == []
    assert report["training"].get("charged_updates", 0) == 0


def test_should_preserve_paid_initialization_failure_without_any_training_or_readout(history_route):
    history_route.work["initialization_fault"] = "paid_failure"
    report = pilot.run_pilot(public_history=True)
    assert report["status"] == "incomplete" and report["outcomes"] is None
    assert history_route.work["events"] == ["initialized"]
    assert history_route.work["evaluations"] == []
    assert report["training"].get("charged_updates", 0) == 0
    assert report["initialization"]["status"] == "failed"
    assert (
        report["initialization"]["forward_calls"],
        report["initialization"]["forward_rows"],
    ) == (1, 9)
    assert (
        report["initialization"]["forward_calls_attempted"],
        report["initialization"]["forward_rows_attempted"],
    ) == (2, 18)
    assert report["work"]["initialization"]["forward_rows"] == 9


def test_should_keep_initialization_sealed_through_completed_training_before_readouts(
    history_route,
):
    history_route.work["corrupt_initialization_during_training"] = True
    report = pilot.run_pilot(public_history=True)
    assert report["status"] == "incomplete" and report["outcomes"] is None
    assert history_route.work["events"] == ["initialized", "trained64"]
    assert history_route.work["evaluations"] == []
    assert report["training"]["charged_updates"] == 64
    assert report["initialization"]["forward_rows"] == 36


@pytest.mark.parametrize("point", [0, 64])
def test_should_keep_failed_paid_readout_prefix_without_claiming_acquisition(history_route, point):
    history_route.work["failed_readout"] = point
    report = pilot.run_pilot(public_history=True)
    assert report["status"] == "incomplete" and report["outcomes"] is None
    assert report["training"]["charged_updates"] == 64
    assert report["evaluations"][-1]["status"] == "failed"
    assert report["evaluations"][-1]["work"]["live_prediction_calls_returned"] == 1
    assert history_route.work["evaluations"] == ([0] if point == 0 else [0, 64])


@pytest.mark.parametrize("fault", ["initialization", "source", "paid_training", "reference"])
def test_should_refuse_changed_initialization_or_completed_evidence_before_more_work(
    history_route, fault
):
    route = history_route

    def corrupt(record):
        if record["updates"] != 0:
            return
        if fault == "initialization":
            route.work["reports"][0]["initialization"]["forward_rows"] = 0
        elif fault == "source":
            route.work["arm"]["source"]._learner.state.parameters["weight"][0, 0] = 99
        elif fault == "paid_training":
            route.work["reports"][0]["training"]["charged_updates"] = 0
        else:
            route.saved["metrics"]["ce_gain"] = 99.0

    route.work["after_readout"] = corrupt
    report = pilot.run_pilot(public_history=True)
    assert report["status"] == "incomplete" and report["outcomes"] is None
    assert report["training"]["charged_updates"] == 64
    assert route.work["evaluations"] == [0]
    assert report["initialization"]["forward_rows"] == 36


def test_should_refuse_final_resource_callback_changes_after_all_paid_readouts(
    history_route, monkeypatch
):
    route = history_route

    class LateSampler(Sampler):
        exited = False

        def __exit__(self, *args):
            self.exited = True
            return False

        def snapshot(self):
            if self.exited:
                route.work["reports"][0]["initialization"]["forward_rows"] = 0
            return super().snapshot()

    monkeypatch.setattr(pilot, "ProcessRssSampler", LateSampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", LateSampler)
    report = pilot.run_pilot(public_history=True)
    assert route.work["evaluations"] == [0, 64]
    assert report["status"] == "incomplete" and report["outcomes"] is None
    assert report["training"]["charged_updates"] == 64
    assert report["initialization"]["forward_rows"] == 36
    assert len(report["evaluations"]) == 2


def test_should_describe_history_cli_without_native_setup(monkeypatch, capsys):
    monkeypatch.setattr(runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    assert cli.main(["--public-history", "--describe"]) == 0
    assert json.loads(capsys.readouterr().out) == pilot.describe_protocol(public_history=True)


def test_should_refuse_combined_cli_flags_before_native_setup(monkeypatch):
    monkeypatch.setattr(runtime, "require_torch", forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(["--public-history", "--endpoint-first"])
    assert raised.value.code == 2


@pytest.mark.parametrize("status,exit_code", [("complete", 0), ("incomplete", 1)])
def test_should_forward_only_explicit_history_and_keep_honest_cli_exit(
    monkeypatch, capsys, status, exit_code
):
    calls: list[dict[str, Any]] = []
    torch = SimpleNamespace(
        set_num_threads=lambda value: None, set_num_interop_threads=lambda value: None
    )
    monkeypatch.setattr(runtime, "require_torch", lambda: torch)

    def run(**options):
        calls.append(options)
        return dict(status=status)

    monkeypatch.setattr(pilot, "run_pilot", run)
    assert cli.main(["--public-history"]) == exit_code
    assert calls == [dict(public_history=True)]
    assert json.loads(capsys.readouterr().out)["status"] == status


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "protocol",
        "full_status",
        "endpoint_status",
        "points",
        "counter",
        "bool_counter",
        "source",
        "metrics",
        "cue",
        "duplicate_point",
        "parity",
        "readback",
    ],
)
def test_should_admit_only_the_retained_validated_paired_endpoints(tmp_path, monkeypatch, fault):
    records = [endpoint(0), endpoint(64)]
    saved: dict[str, Any] = dict(
        protocol=pilot.describe_protocol(endpoint_first=True),
        status="incomplete",
        initial_parity=True,
        readback_error=None,
        training=dict(
            charged_updates=64,
            native_training_episodes=128,
            native_training_examples=1152,
            source_unchanged=True,
            actor_unchanged=True,
        ),
        endpoint_evidence=dict(
            status="validated",
            checkpoint_updates=[0, 64],
            outcomes=dict(paired=pilot._metrics(records, final_point=64)),
        ),
        evaluations=records,
    )
    if fault == "protocol":
        saved["protocol"]["seed"] = 18
    elif fault == "full_status":
        saved["status"] = "complete"
    elif fault == "endpoint_status":
        saved["endpoint_evidence"]["status"] = "invalid_or_unverified"
    elif fault == "points":
        saved["endpoint_evidence"]["checkpoint_updates"] = [0, 48]
    elif fault == "counter":
        saved["training"]["native_training_examples"] = 1151
    elif fault == "bool_counter":
        saved["training"]["charged_updates"] = True
    elif fault == "source":
        saved["training"]["source_unchanged"] = False
    elif fault == "metrics":
        saved["endpoint_evidence"]["outcomes"]["paired"]["ce_gain"] += 0.1
    elif fault == "cue":
        saved["evaluations"][1]["scores"][0]["cue"] = "red"
    elif fault == "duplicate_point":
        saved["evaluations"][1]["updates"] = 0
    elif fault == "parity":
        saved["initial_parity"] = False
    elif fault == "readback":
        saved["readback_error"] = "changed prior state"
    path = tmp_path / "retained-endpoints.json"
    path.write_text(json.dumps(saved), encoding="utf8")
    monkeypatch.setattr(pilot, "PAIRED_REFERENCE_PATH", str(path))
    monkeypatch.setattr(runtime, "require_torch", forbidden)
    if fault is not None:
        with pytest.raises(ValueError):
            pilot._load_paired_reference()
    else:
        admitted = pilot._load_paired_reference()
        assert admitted["evaluations"] == records
        assert admitted["provenance"]["full_status"] == "incomplete"
        assert admitted["provenance"]["endpoint_status"] == "validated"
        assert admitted["metrics"] == pilot._metrics(records, final_point=64)
