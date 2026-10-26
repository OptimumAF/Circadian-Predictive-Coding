"""Paired recurrent objectives retain identical data and unweighted readouts."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.run_minigrid_weighted_bc_smoke as cli
from src.adapters import minigrid_weighted_bc_pilot as pilot
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import torch_train_weighted_learner as weighted
from src.adapters.torch_recurrent_learner import TORCH_RECURRENT_CE_ID, TorchRecurrentPolicy
from src.adapters.torch_recurrent_weighted import TORCH_RECURRENT_WEIGHTED_CE_ID
from src.app.coupled_attribution import AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import episode_arrays
from src.core.learner_ports import TrainingDiagnostic
from src.core.minigrid_key_door import predefined_layouts
import src.shared.torch_runtime as torch_runtime
from scripts.run_minigrid_history_evaluation_smoke import _fixed_a_packets
from test_minigrid_history_evaluation import history_case as history_case
from test_minigrid_fixed_a_acquisition import record
from test_minigrid_learning_pilot import Learner, evaluator as evaluator
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


@pytest.fixture(scope="module")
def raw_reference():
    return pilot._load_reference()


@pytest.fixture(scope="module")
def admitted_reference(raw_reference):
    return pilot._admit_reference(deepcopy(raw_reference))


def test_should_admit_only_fixed_saved_control_and_its_original_six_sources(
    raw_reference, admitted_reference
):
    selected = next(row for row in raw_reference["methods"] if row["method"] == "recurrent_bc")
    assert admitted_reference["provenance"]["historical_control"] == selected
    assert admitted_reference["row"]["method"] == "recurrent_bc"
    assert admitted_reference["row"]["objective_definition"] == TORCH_RECURRENT_CE_ID
    assert admitted_reference["row"]["charged_updates"] == 128
    assert admitted_reference["row"]["native_training_examples"] == 1152
    assert [record["updates"] for record in admitted_reference["evaluations"]] == [
        0,
        32,
        64,
        96,
        128,
    ]
    assert all(record["method"] == "recurrent_bc" for record in admitted_reference["evaluations"])
    assert [
        (packet.context, packet.episode.episode_id.rsplit("-", 1)[-1])
        for packet in admitted_reference["sources"]
    ] == [(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")]
    assert all(
        packet.role == packet.episode.role == "train" and packet.features.shape == (9, 148)
        for packet in admitted_reference["sources"]
    )


@pytest.mark.parametrize("fault", ["method", "receipt", "source", "frame"])
def test_should_refuse_corrupted_reference_before_any_new_training_without_touching_artifact(
    monkeypatch, raw_reference, fault
):
    artifact = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/r21-fixed-a-acquisition-20261016/native-result.json"
    )
    before = artifact.read_bytes()
    report = deepcopy(raw_reference)
    row = next(row for row in report["methods"] if row["method"] == "recurrent_bc")
    if fault == "method":
        row["method"] = "recurrent_weighted_bc"
    elif fault == "receipt":
        row["stream_result"]["applied_updates"][0]["diagnostic"]["definition"] = (
            TORCH_RECURRENT_WEIGHTED_CE_ID
        )
    elif fault == "source":
        step = report["training_sources"][0]["transitions"][0]
        step["action"] = (step["action"] + 1) % 7
    else:
        endpoint = next(
            value
            for value in report["evaluations"]
            if value["method"] == "recurrent_bc" and value["updates"] == 0
        )
        endpoint["episodes"][0]["predictions"][1]["observation"][0] = 0.2
    monkeypatch.setattr(pilot, "_load_reference", lambda: report)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: None)
    monkeypatch.setattr(pilot, "TorchRecurrentLearner", forbidden)
    monkeypatch.setattr(pilot, "TorchTrainWeightedLearner", forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)
    result = pilot.run_pilot()
    assert result["status"] == "incomplete" and result["reason"]
    assert result["methods"][1]["charged_updates"] in (None, 0)
    assert all(record["status"] == "not_run" for record in result["evaluations"][5:])
    assert artifact.read_bytes() == before


def test_should_refuse_foreign_schedule_seal_before_calling_its_serializer(monkeypatch, evaluator):
    native_fixture(monkeypatch, evaluator)
    source = pilot._learner_factory(tagged_packets(evaluator))("recurrent_weighted_bc")

    class ForeignSeal:
        def __reduce__(self):
            pytest.fail("foreign schedule seal serializer must never run")

    source._schedule_seal = ForeignSeal()
    with pytest.raises(ValueError, match="schedule"):
        source.snapshot_state()


def labels_weight(episode):
    counts = [sum(step.action == action for step in episode.transitions) for action in range(7)]
    present = sum(count > 0 for count in counts)
    return tuple(len(episode.transitions) / (present * count) if count else 1.0 for count in counts)


def tagged_packets(evaluator):
    sources = list(deepcopy(evaluator.sources))
    patterns = ((2, 2, 2, 2, 2, 1, 1, 3, 5), (2, 2, 2, 1, 1, 1, 3, 5, 5))
    for index, actions in enumerate(patterns):
        source = sources[index]
        episode = replace(
            source.episode,
            transitions=tuple(
                replace(step, action=action)
                for step, action in zip(source.episode.transitions, actions, strict=True)
            ),
        )
        features, targets = episode_arrays(episode)
        sources[index] = replace(source, episode=episode, features=features, targets=targets)
    return _fixed_a_packets(tuple(sources))


def native_fixture(monkeypatch, evaluator):
    calls = []
    ordinary = Learner.train_batch

    def mean(owner, features, targets):
        result = ordinary(owner, features, targets)
        return TrainingDiagnostic(TORCH_RECURRENT_CE_ID, result.value)

    def train_weighted(owner, episode, weights):
        assert episode.role == "train" and episode.num_classes == 7
        assert len(episode.transitions) == 9 and len(episode.transitions[0].observation) == 148
        assert tuple(weights) == labels_weight(episode)
        calls.append((deepcopy(episode), tuple(weights), owner.state.completed_updates))
        features, targets = episode_arrays(episode)
        result = ordinary(owner, features, targets)
        return TrainingDiagnostic(TORCH_RECURRENT_WEIGHTED_CE_ID, result.value)

    monkeypatch.setattr(Learner, "train_batch", mean)
    monkeypatch.setattr(Learner, "train_weighted_episode", train_weighted, raising=False)
    monkeypatch.setattr(pilot, "TorchRecurrentLearner", Learner)
    monkeypatch.setattr(weighted, "TorchRecurrentLearner", Learner)
    monkeypatch.setattr(
        weighted, "_same_state", lambda left, right: pickle.dumps(left) == pickle.dumps(right)
    )
    return calls


@pytest.mark.parametrize("method", ["recurrent_bc", "recurrent_weighted_bc"])
def test_should_route_two_released_current_148_k7_episodes_through_real_stream_and_matching_objective(
    monkeypatch, evaluator, method
):
    calls = native_fixture(monkeypatch, evaluator)
    packets = tagged_packets(evaluator)
    assert len(packets) == 128 and len({packet.episode.episode_id for packet in packets}) == 128
    source = pilot._learner_factory(packets)(method)
    before = pickle.dumps(source.snapshot_state())
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    stream = AttributionStream(
        source,
        packets,
        (),
        checkpoint_updates=(0, 2, 128),
        label_delay=1,
        budget=budget,
        training_only=True,
    )
    stream.run_until(2)
    assert stream.completed_updates == budget.updates_completed == 2
    assert pickle.dumps(source.snapshot_state()) == before
    assert stream._runtime.actor.snapshot_state().state.completed_updates == 0
    state = stream._runtime.candidate_snapshot().state
    assert state.completed_updates == 2 and state.training_examples == 18
    assert state.optimizer_state["momentum"] == 2
    receipts = stream._runtime.applied_updates
    objective = (
        TORCH_RECURRENT_WEIGHTED_CE_ID
        if method == "recurrent_weighted_bc"
        else TORCH_RECURRENT_CE_ID
    )
    assert [receipt.update_number for receipt in receipts] == [1, 2]
    assert all(receipt.diagnostic.definition == objective for receipt in receipts)
    assert len(calls) == (2 if method == "recurrent_weighted_bc" else 0)
    if calls:
        assert [episode for episode, _, _ in calls] == [packet.episode for packet in packets[:2]]
        assert [point for _, _, point in calls] == [0, 1]
        assert calls[0][1] != calls[1][1] and any(value != 1 for value in calls[0][1])
    assert all(values.shape == (9, 148) for _, values, _ in evaluator.work["trains"])
    assert not evaluator.work.get("predicts") and not evaluator.work.get("environments")


def test_should_evaluate_weighted_marker_as_same_fresh_148_recurrent_sessions(history_case):
    case = history_case
    policy = TorchRecurrentPolicy(148, 8, 7, 17)
    case.source.policy = case.source.state.policy = policy
    state = case.source.snapshot_state()
    state.completed_updates, state.training_examples = 128, 1152
    report = evaluation.evaluate_checkpoint(
        case.source,
        state,
        method="recurrent_weighted_bc",
        updates=128,
        layouts=predefined_layouts()[:1],
        role="train",
        sources=case.sources[:2],
        budget=case.budget,
        check_boundary=case.guard,
    )
    assert report["status"] == "complete" and report["work"]["live_sessions_started"] == 2
    assert len(case.work["sessions"]) == 2 and all(
        session.completed_steps == 2 for session in case.work["sessions"]
    )
    assert report["work"]["raw_score_forwards_completed"] == 2
    assert all(values.shape[1] == 148 for _, values in case.work["predicts"])
    assert all(
        "features" not in decision
        for slot in report["episodes"]
        for decision in slot["predictions"]
    )
    assert all(score["ce"] == pytest.approx(float(np.log(2))) for score in report["scores"])
    assert case.budget.updates_completed == 0 and not case.work.get("trains")


@pytest.mark.parametrize("role", ["train", "final_test"])
def test_should_restrict_weighted_marker_to_fixed_train_a_pair_before_fork(history_case, role):
    case = history_case
    policy = TorchRecurrentPolicy(148, 8, 7, 17)
    case.source.policy = case.source.state.policy = policy
    state = case.source.snapshot_state()
    state.completed_updates, state.training_examples = 128, 1152
    with pytest.raises(ValueError):
        evaluation.evaluate_checkpoint(
            case.source,
            state,
            method="recurrent_weighted_bc",
            updates=128,
            layouts=predefined_layouts()[:3] if role == "train" else predefined_layouts()[-1:],
            role=role,
            sources=case.sources if role == "train" else (),
            budget=case.budget,
            check_boundary=case.guard,
        )
    assert not case.work.get("forks") and not case.work.get("environments")


@pytest.mark.parametrize("target", ["episodes", "seal", "both", "extra_fields"])
def test_should_refuse_replaced_148_schedule_evidence_before_any_weighted_update(
    monkeypatch, evaluator, target
):
    native_fixture(monkeypatch, evaluator)
    packets = tagged_packets(evaluator)
    source = pilot._learner_factory(packets)("recurrent_weighted_bc")
    if target in ("episodes", "both"):
        object.__setattr__(source._episodes[0], "context", "B")
    if target in ("seal", "both"):
        object.__setattr__(source._schedule_seal[0], "context", "B")
    if target == "extra_fields":
        object.__setattr__(source._episodes[0], "unobserved", True)
    with pytest.raises(ValueError):
        source.snapshot_state()
    assert not evaluator.work.get("trains")


def test_should_describe_pair_resource_free_and_leave_original_five_control_recipe_unchanged(
    monkeypatch,
):
    original = pilot.shared.describe_protocol()
    for name in (
        "TorchRecurrentLearner",
        "TorchTrainWeightedLearner",
        "ProcessRssSampler",
        "require_torch",
        "evaluate_checkpoint",
        "_load_reference",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    protocol = pilot.describe_protocol()
    assert json.loads(json.dumps(protocol, allow_nan=False)) == protocol
    assert protocol["methods"] == ["recurrent_bc", "recurrent_weighted_bc"]
    assert protocol["objectives"] == {
        "recurrent_bc": TORCH_RECURRENT_CE_ID,
        "recurrent_weighted_bc": TORCH_RECURRENT_WEIGHTED_CE_ID,
    }
    assert protocol["feature_dim"] == 148 and protocol["hidden_dim"] == 8
    assert protocol["seed"] == 17 and protocol["checkpoints"] == [0, 32, 64, 96, 128]
    assert protocol["protocol_id"] == "minigrid_reference_weighted_a128_acquisition_v1"
    assert protocol["planned_work"]["native_training_updates"] == 128
    assert protocol["planned_work"]["training_rows"] == 1152
    assert protocol["planned_work"]["raw_score_forwards"] == 10
    assert protocol["planned_work"]["raw_score_rows"] == 90
    assert protocol["planned_work"]["live_episodes"] == 4
    assert protocol["planned_work"]["teacher_episodes"] == 0
    assert protocol["retained_historical_exposure"]["unweighted_updates"] == 128
    assert protocol["retained_historical_exposure"]["teacher_physics"] == 54
    assert pilot.shared.describe_protocol() == original
    assert original["methods"] == ["bp", "pc", "neutral_cpc", "gated_cpc", "recurrent_bc"]


def paired_case(monkeypatch, evaluator, admitted_reference, *, fault=None):
    work: dict[str, Any] = dict(arms=[], calls=[], outputs=[])
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: None)
    monkeypatch.setattr(pilot, "monotonic", lambda: 10.0)

    raw = {"fixed_reference_fixture": True}
    reference = deepcopy(admitted_reference)
    report_factory = pilot._new_report

    def new_report():
        value = report_factory()
        work["report"] = value
        return value

    def admit(value):
        assert value is raw
        work["reference"] = reference
        return reference

    def train(row, packets, sampler, deadline, check_boundary, **kwargs):
        assert row["method"] == "recurrent_weighted_bc"
        if fault == "peer_train":
            work["report"]["methods"][0]["stream_result"]["applied_updates"][0]["update_number"] = (
                999
            )
        check_boundary()
        assert kwargs["checkpoint_updates"] == pilot.POINTS
        assert len(packets) == 128 and all(
            packet.context == "A" and packet.features.shape == (9, 148) for packet in packets
        )
        source = Learner(TorchRecurrentPolicy(148, 8, 7, 17))
        states = []
        for point in pilot.POINTS:
            state = source.snapshot_state()
            state.completed_updates, state.training_examples = point, point * 9
            states.append((point, state))
        budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=128), lambda: 0.0)
        budget.updates_completed = 128
        result = dict(
            completed_updates=128,
            training_examples=1152,
            applied_updates=[
                dict(
                    update_number=i + 1,
                    diagnostic=dict(definition=pilot.OBJECTIVES[row["method"]], value=0.5),
                )
                for i in range(128)
            ],
        )
        row.update(
            status="trained",
            charged_updates=128,
            native_training_examples=1152,
            source_unchanged=True,
            actor_unchanged=True,
            stream_result=result,
            checkpoint_updates=list(pilot.POINTS),
            wake_calls_attempted=128,
            wake_calls_returned=128,
        )
        sealed = pickle.dumps(
            (
                source.snapshot_state(),
                states,
                budget.budget,
                budget.started_at,
                budget.updates_completed,
            )
        )

        def guard():
            if (
                pickle.dumps(
                    (
                        source.snapshot_state(),
                        states,
                        budget.budget,
                        budget.started_at,
                        budget.updates_completed,
                    )
                )
                != sealed
            ):
                raise ValueError("previous peer native state changed")

        arm = dict(row=row, source=source, states=states, budget=budget, guard=guard)
        work["arms"].append(arm)
        return arm

    def read(source, state, **kwargs):
        assert len(work["arms"]) == 1
        assert kwargs["method"] == "recurrent_weighted_bc"
        if work["outputs"] and fault == "readout":
            work["outputs"][0]["scores"][0]["ce"] = 999.0
        elif work["outputs"] and fault == "receipt":
            work["report"]["methods"][0]["stream_result"]["applied_updates"][0]["diagnostic"][
                "definition"
            ] = "corrupted"
        elif work["outputs"] and fault == "peer_state":
            work["arms"][0]["source"].state.parameters["weight"] += 1
        kwargs["check_boundary"]()
        method, point = kwargs["method"], kwargs["updates"]
        arm = next(arm for arm in work["arms"] if arm["row"]["method"] == method)
        assert source is arm["source"] and state is next(
            value for update, value in arm["states"] if update == point
        )
        assert kwargs["budget"] is arm["budget"] and arm["budget"].updates_completed == 128
        assert len(kwargs["sources"]) == 2 and all(
            packet.features.shape == (9, 148) for packet in kwargs["sources"]
        )
        live = "layouts" in kwargs
        assert live == (point in (0, 128))
        if live:
            assert kwargs["layouts"] == predefined_layouts()[:1] and kwargs["role"] == "train"
            assert "public_history" not in kwargs
        value = record(
            method, point, gain=1.0 if method == "recurrent_bc" else 0.25, raw=False, goals=False
        )
        for index, score in enumerate(value["scores"]):
            score["ce"] = reference["evaluations"][0]["scores"][index]["ce"] - 0.25 * point / 128
        if point == 0:
            value = deepcopy(reference["evaluations"][0])
            value["method"] = "recurrent_weighted_bc"
            if fault == "initial_mismatch":
                value["scores"][0]["probabilities"][0][0] += 0.0001
                value["scores"][0]["probabilities"][0][1] -= 0.0001
        else:
            value["work"] = dict(
                raw_score_forwards_attempted=2,
                raw_score_forwards_completed=2,
                raw_score_rows_attempted=18,
                raw_score_rows_completed=18,
                live_prediction_calls_attempted=4 if live else 0,
                live_prediction_calls_returned=4 if live else 0,
                live_sessions_started=2 if live else 0,
                environment_resets_completed=2 if live else 0,
                environment_steps_returned=4 if live else 0,
                failure_work_may_be_unobserved=False,
            )
        if fault == "raw_failure" and point == 64:
            value.update(status="failed", reason="paid raw failure")
            value["scores"] = value["scores"][:1]
            value["work"].update(
                raw_score_forwards_attempted=1,
                raw_score_forwards_completed=1,
                raw_score_rows_attempted=9,
                raw_score_rows_completed=9,
            )
        work["calls"].append((method, point, live))
        work["outputs"].append(value)
        return value

    monkeypatch.setattr(pilot, "_new_report", new_report)
    monkeypatch.setattr(pilot, "_load_reference", lambda: raw)
    monkeypatch.setattr(pilot, "_admit_reference", admit)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", train)
    monkeypatch.setattr(pilot, "evaluate_checkpoint", read)
    monkeypatch.setattr(pilot, "_score_checkpoint", read)
    return SimpleNamespace(work=work)


def test_should_reuse_fixed_mean_reference_and_finish_only_weighted_stream_before_new_readouts(
    monkeypatch, evaluator, admitted_reference
):
    reference_before = pickle.dumps(admitted_reference)
    case = paired_case(monkeypatch, evaluator, admitted_reference)
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["reason"]
    assert case.work["calls"] == [
        ("recurrent_weighted_bc", point, point in (0, 128)) for point in pilot.POINTS
    ]
    assert len(report["evaluations"]) == 10
    assert sum(len(value["scores"]) for value in report["evaluations"]) == 20
    assert sum(len(value.get("episodes", [])) for value in report["evaluations"]) == 8
    assert report["work"]["charged_updates"] == 256
    assert report["work"]["observed_native_training_rows"] == 2304
    assert report["work"]["raw_score_forwards_completed"] == 20
    assert report["work"]["raw_score_rows_completed"] == 180
    assert report["work"]["live_sessions_started"] == 8
    assert report["initial_condition_match"] is True
    assert report["new_work"]["charged_updates"] == 128
    assert report["new_work"]["observed_native_training_rows"] == 1152
    assert report["new_work"]["raw_score_forwards_completed"] == 10
    assert report["new_work"]["raw_score_rows_completed"] == 90
    assert report["new_work"]["live_sessions_started"] == 4
    assert report["new_work"]["teacher_reports"] == 0
    assert report["historical_work"]["charged_updates"] == 128
    assert report["historical_work"]["teacher_environment_steps_observed"] == 54
    assert pickle.dumps(admitted_reference) == reference_before
    assert report["evaluations"][:5] == admitted_reference["evaluations"]
    assert report["methods"][0] == admitted_reference["row"]
    assert report["scientific_advantage_verified"] is False
    assert all(row["joint_acquisition"] is False for row in report["acquisition"]["methods"])
    comparison = report["acquisition"]["signed_comparisons"][
        "recurrent_weighted_bc-minus-recurrent_bc"
    ]
    reference_gain = (
        sum(score["ce"] for score in admitted_reference["evaluations"][0]["scores"]) / 2
        - sum(score["ce"] for score in admitted_reference["evaluations"][-1]["scores"]) / 2
    )
    assert comparison["ce_gain"] == pytest.approx(0.25 - reference_gain)
    assert comparison["ce_gain"] < 0
    assert all(row["final_evaluation"]["status"] == "not_run" for row in report["methods"])
    assert (
        not evaluator.work.get("trains")
        and not evaluator.work.get("predicts")
        and not evaluator.work.get("environments")
    )
    case.work["outputs"][0]["scores"][0]["ce"] = 999
    assert (
        report["evaluations"][5]["scores"][0]["ce"]
        == admitted_reference["evaluations"][0]["scores"][0]["ce"]
    )


@pytest.mark.parametrize(
    "fault", ["readout", "receipt", "peer_state", "raw_failure", "peer_train", "initial_mismatch"]
)
def test_should_preserve_paid_charges_and_null_all_acquisition_after_late_refusal(
    monkeypatch, evaluator, admitted_reference, fault
):
    case = paired_case(monkeypatch, evaluator, admitted_reference, fault=fault)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and report["reason"]
    assert report["work"]["charged_updates"] == (128 if fault == "peer_train" else 256)
    assert report["new_work"]["charged_updates"] == (0 if fault == "peer_train" else 128)
    assert all(
        row["status"] == "unavailable" and row["joint_acquisition"] is None
        for row in report["acquisition"]["methods"]
    )
    assert all(
        value is None
        for value in report["acquisition"]["signed_comparisons"][
            "recurrent_weighted_bc-minus-recurrent_bc"
        ].values()
    )
    assert report["methods"][0]["stream_result"]["applied_updates"][0]["update_number"] == 1
    assert (
        report["methods"][0]["stream_result"]["applied_updates"][0]["diagnostic"]["definition"]
        == TORCH_RECURRENT_CE_ID
    )
    if fault == "peer_train":
        assert case.work["calls"] == [] and all(
            slot["status"] == "not_run" for slot in report["evaluations"][5:]
        )
    elif fault == "raw_failure":
        assert len(case.work["calls"]) == 3
        assert (
            report["evaluations"][7]["status"] == "failed"
            and len(report["evaluations"][7]["scores"]) == 1
        )
        assert all(slot["status"] == "not_run" for slot in report["evaluations"][8:])
    else:
        assert len(case.work["calls"]) == 1
        assert (
            report["evaluations"][5]["scores"][0]["ce"]
            == admitted_reference["evaluations"][0]["scores"][0]["ce"]
        )
        if fault == "initial_mismatch":
            assert report["initial_condition_match"] is False


@pytest.mark.parametrize("raw,goals", [(False, True), (True, False), (True, True)])
def test_should_require_both_fixed_raw_sequences_and_both_native_goals_in_new_pair(raw, goals):
    evaluations = [record(method, point) for method in pilot.METHODS for point in pilot.POINTS]
    endpoint = evaluations[-1]
    endpoint["scores"][1]["exact_sequence_accuracy"] = float(raw)
    endpoint["scores"][1]["accuracy"] = float(raw)
    endpoint["episodes"][1]["success"] = goals
    endpoint["episodes"][1]["return"] = float(goals)
    row = pilot._outcomes(evaluations, True)["methods"][1]
    assert row["raw_both_cues_exact"] is raw and row["live_both_cues_goals"] is goals
    assert row["joint_acquisition"] is (raw and goals)


def test_should_null_pair_claims_on_whole_case_refusal_even_with_all_readouts_present():
    evaluations = [record(method, point) for method in pilot.METHODS for point in pilot.POINTS]
    result = pilot._outcomes(evaluations, False)
    assert all(row["ce_gain"] is row["joint_acquisition"] is None for row in result["methods"])
    assert all(
        value is None
        for value in result["signed_comparisons"][
            "recurrent_weighted_bc-minus-recurrent_bc"
        ].values()
    )


def test_should_route_cli_describe_without_optional_native_runtime(monkeypatch, capsys):
    monkeypatch.setattr(torch_runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    monkeypatch.setattr(pilot, "describe_protocol", lambda: {"protocol_id": "fixed-pair"})
    assert cli.main(["--describe"]) == 0
    assert json.loads(capsys.readouterr().out) == {"protocol_id": "fixed-pair"}


@pytest.mark.parametrize("status", ["complete", "incomplete"])
def test_should_publish_exact_cli_result_with_one_thread_and_operational_exit_status(
    monkeypatch, tmp_path, capsys, status
):
    calls = []
    torch = SimpleNamespace(
        set_num_threads=lambda count: calls.append(("threads", count)),
        get_num_interop_threads=lambda: 2,
        set_num_interop_threads=lambda count: calls.append(("interop", count)),
    )
    monkeypatch.setattr(torch_runtime, "require_torch", lambda: torch)
    report = {"status": status, "methods": [{"charged_updates": 1}], "acquisition": None}
    monkeypatch.setattr(pilot, "run_pilot", lambda: report)
    path = tmp_path / "paired.json"
    assert cli.main(["--output", str(path)]) == (0 if status == "complete" else 1)
    assert calls == [("threads", 1), ("interop", 1)]
    assert json.loads(path.read_text(encoding="utf8")) == report
    assert json.loads(capsys.readouterr().out)["bytes"] == path.stat().st_size


@pytest.mark.parametrize("fault", ["existing", "parent", "describe_output"])
def test_should_refuse_unpublishable_cli_request_before_native_setup(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(torch_runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    path = tmp_path / "paired.json"
    arguments = ["--output", str(path)]
    if fault == "existing":
        path.write_text("original attempt", encoding="utf8")
    elif fault == "parent":
        arguments = ["--output", str(tmp_path / "missing" / "paired.json")]
    else:
        arguments.append("--describe")
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2
    if fault == "existing":
        assert path.read_text(encoding="utf8") == "original attempt"


@pytest.mark.parametrize("fault", ["nonfinite", "size"])
def test_should_refuse_invalid_cli_json_before_creating_result(monkeypatch, tmp_path, fault):
    torch = SimpleNamespace(set_num_threads=lambda count: None, get_num_interop_threads=lambda: 1)
    monkeypatch.setattr(torch_runtime, "require_torch", lambda: torch)
    report = {"status": "complete", "value": float("inf") if fault == "nonfinite" else "x" * 100}
    monkeypatch.setattr(pilot, "run_pilot", lambda: report)
    if fault == "size":
        monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 32)
    path = tmp_path / "unpublished.json"
    with pytest.raises(ValueError):
        cli.main(["--output", str(path)])
    assert not path.exists()
