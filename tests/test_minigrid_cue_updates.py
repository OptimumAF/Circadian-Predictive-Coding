"""One original cue update on each disposable full-state warm fork.

The native port is deterministic and records optimizer continuation. Accepted
native CE and scoring mathematics are reused rather than reproduced here.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_cue_updates as cue_updates
from src.adapters import torch_demonstration_reward_learner as typed
from src.adapters.torch_recurrent_learner import TORCH_RECURRENT_CE_ID, TorchRecurrentPolicy
import src.app.toy_execution_budget as budgets
from src.core.learner_ports import TrainingDiagnostic
from test_minigrid_learning_pilot import same_state, source_batch
from test_minigrid_warm_reward import Native, snapshot
from test_torch_feature_rehearsal_pilot import Sampler


def raw_readout(sources, *, updates=128, cue=None):
    scores = []
    for packet in sources:
        episode = packet.episode
        episode_cue = episode.episode_id.rsplit("-", 1)[-1]
        change = 0.0 if cue is None else (-0.2 if episode_cue == cue else 0.3)
        correct_probability = float(np.exp(-(1.0 + change)))
        scores.append(
            dict(
                context=packet.context,
                cue=episode_cue,
                episode_id=episode.episode_id,
                row_ids=list(packet.row_ids),
                ce=1.0 + change,
                accuracy=0.0,
                exact_sequence_accuracy=0.0,
                probabilities=np.tile(
                    np.array(
                        [[1.0 - correct_probability, correct_probability, 0, 0, 0, 0, 0]],
                        dtype=np.float64,
                    ),
                    (9, 1),
                ).tolist(),
            )
        )
    return dict(
        method="recurrent_bc",
        updates=updates,
        role="train",
        status="complete",
        reason=None,
        scores=scores,
        source_unchanged=True,
        state_unchanged=True,
        work=dict(
            score_forward_calls_attempted=6,
            score_forward_calls_completed=6,
            score_rows_attempted=54,
            score_rows_completed=54,
            failure_work_may_be_unobserved=False,
        ),
    )


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = dict(owners=[], trains=[], scores=[], budgets=[], boundaries=0)
    monkeypatch.setattr(Native, "work", work)
    monkeypatch.setattr(typed, "TorchRecurrentLearner", Native)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(cue_updates, "_same_state", same_state)
    clock = [10.0]
    monkeypatch.setattr(cue_updates, "monotonic", lambda: clock[0])
    policy = TorchRecurrentPolicy(148, 8, 7, 17)
    source = typed.TorchDemonstrationRewardLearner(policy)
    source.restore_state(snapshot(policy, 128, 1152))
    state = source.snapshot_state()
    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    baseline = raw_readout(sources)
    memory: Any = Sampler()
    caller_budget = budgets.ToyBudgetSession(
        budgets.ToyExecutionBudget(
            max_training_updates=0, max_wall_seconds=60.0, max_process_rss_bytes=768 * 1024**2
        ),
        lambda: clock[0],
    )
    caller_budget.attach_memory(memory)
    new_budget = budgets.ToyBudgetSession

    def capture_budget(*args, **kwargs):
        budget = new_budget(*args, **kwargs)
        work["budgets"].append(budget)
        return budget

    def boundary():
        work["boundaries"] += 1
        callback = work.get("boundary_callback")
        if callback:
            callback()

    def train(owner, episode):
        record = dict(owner=owner, episode=deepcopy(episode), before=owner.snapshot_state())
        work["trains"].append(record)
        if work.get("fail_train") == len(work["trains"]):
            raise RuntimeError("atomic cue update refused")
        for index, (key, parameter) in enumerate(owner.state.parameters.items()):
            momentum = owner.state.optimizer_state["state"][index]["momentum_buffer"]
            carried = 0.9 * momentum + np.full_like(parameter, 0.1)
            owner.state.optimizer_state["state"][index]["momentum_buffer"] = carried.copy()
            owner.state.parameters[key] = parameter - 0.03 * carried
            owner.state.gradients[key] = np.full_like(parameter, 0.1)
        owner.state.completed_updates += 1
        owner.state.training_examples += len(episode.transitions)
        owner.trained_cue = episode.episode_id.rsplit("-", 1)[-1]
        record["after"] = owner.snapshot_state()
        callback = work.get("after_train")
        if callback:
            callback(record)
        return TrainingDiagnostic(TORCH_RECURRENT_CE_ID, 0.5)

    def score(owner, checkpoint, **options):
        assert options["method"] == "recurrent_bc" and options["updates"] == 129
        assert options.get("gradient_geometry", False) is False
        assert options["sources"] is sources and options["budget"] is caller_budget
        assert checkpoint.completed_updates == 129 and checkpoint.training_examples == 1161
        options["check_boundary"]()
        result = raw_readout(sources, updates=129, cue=owner._learner.trained_cue)
        work["scores"].append(dict(owner=owner, state=deepcopy(checkpoint), result=result))
        callback = work.get("after_score")
        if callback:
            callback(result)
        return result

    monkeypatch.setattr(cue_updates, "ToyBudgetSession", capture_budget)
    monkeypatch.setattr(cue_updates, "_score_checkpoint", score)
    monkeypatch.setattr(Native, "train_episode", train, raising=False)
    return SimpleNamespace(
        work=work,
        clock=clock,
        source=source,
        state=state,
        sources=sources,
        baseline=baseline,
        caller_budget=caller_budget,
        memory=memory,
        boundary=boundary,
    )


def run(case, report):
    return cue_updates.run_cue_updates(
        report,
        case.source,
        case.state,
        case.sources,
        case.baseline,
        case.caller_budget,
        check_boundary=case.boundary,
    )


def test_should_update_two_independent_full_warm_states_and_score_all_original_train_packets(case):
    original_state = pickle.dumps(case.source.snapshot_state())
    original_packets = pickle.dumps(case.sources)
    original_baseline = pickle.dumps(case.baseline)
    report: dict[str, Any] = {}
    run(case, report)
    assert report["status"] == "complete", report["reason"]
    assert len(case.work["trains"]) == len(case.work["scores"]) == 2
    left, right = case.work["trains"]
    assert left["owner"] is not right["owner"]
    assert all(item["owner"] is not case.source._learner for item in (left, right))
    for index, record in enumerate((left, right)):
        assert pickle.dumps(record["before"]) == original_state
        assert record["episode"] == case.sources[index].episode
        assert record["episode"] is not case.sources[index].episode
        assert len(record["episode"].transitions) == 9
        assert record["episode"].role == "train"
        assert record["after"].completed_updates == 129
        assert record["after"].training_examples == 1161
        for parameter_index, key in enumerate(record["before"].parameters):
            previous = record["before"].optimizer_state["state"][parameter_index]["momentum_buffer"]
            expected = 0.9 * previous + np.full_like(previous, 0.1)
            np.testing.assert_array_equal(
                record["after"].optimizer_state["state"][parameter_index]["momentum_buffer"],
                expected,
            )
            assert record["before"].gradients[key] is not None
        arm = report["arms"][index]
        assert arm["update_committed"] is arm["commit_validated"] is True
        assert arm["diagnostic"]["definition"] == TORCH_RECURRENT_CE_ID
        assert len(arm["readout"]["scores"]) == 6
    assert pickle.dumps(case.source.snapshot_state()) == original_state
    assert (
        pickle.dumps(case.sources) == original_packets
        and pickle.dumps(case.baseline) == original_baseline
    )
    work = report["work"]
    assert work["native_updates_attempted"] == work["native_updates_committed"] == 2
    assert work["committed_training_rows"] == work["attempted_training_rows"] == 18
    assert work["score_forward_calls_completed"] == 12 and work["score_rows_completed"] == 108
    assert work["observed_recurrent_unroll_steps"] == 126
    assert work["failure_work_may_be_unobserved"] is False
    assert report["budget"]["limits"]["max_training_updates"] == 2
    assert report["budget"]["limits"]["max_wall_seconds"] == 20.0
    assert report["budget"]["updates_completed"] == 2
    assert case.caller_budget.updates_completed == 0
    for blue, red, comparison in zip(
        report["deltas"]["blue"],
        report["deltas"]["red"],
        report["deltas"]["blue_minus_red"],
        strict=True,
    ):
        assert blue["ce_delta"] == pytest.approx(blue["after_ce"] - 1.0)
        assert red["ce_delta"] == pytest.approx(red["after_ce"] - 1.0)
        assert comparison["ce_difference"] == pytest.approx(blue["after_ce"] - red["after_ce"])
    assert report["deltas"]["blue"][0]["ce_delta"] < 0 < report["deltas"]["blue"][1]["ce_delta"]
    case.work["scores"][0]["result"]["scores"][0]["ce"] = 999.0
    assert report["arms"][0]["readout"]["scores"][0]["ce"] != 999.0


@pytest.mark.parametrize(
    "fault", ["status", "updates", "method", "role", "missing", "order", "ce", "row_ids"]
)
def test_should_refuse_invalid_baseline_before_fork_or_update(case, fault):
    if fault == "status":
        case.baseline["status"] = "incomplete"
    elif fault == "updates":
        case.baseline["updates"] = 129
    elif fault == "method":
        case.baseline["method"] = "bp"
    elif fault == "role":
        case.baseline["role"] = "final_test"
    elif fault == "missing":
        case.baseline["scores"].pop()
    elif fault == "order":
        case.baseline["scores"].reverse()
    elif fault == "ce":
        case.baseline["scores"][0]["ce"] = float("nan")
    else:
        case.baseline["scores"][0]["row_ids"][0] = "replaced-source-row"
    report: dict[str, Any] = {}
    with pytest.raises(ValueError):
        run(case, report)
    assert case.work["trains"] == case.work["scores"] == []
    assert len(case.work["owners"]) == 1
    assert report == {}


@pytest.mark.parametrize("fault", ["role", "width", "context", "counter"])
def test_should_refuse_nonoriginal_train_inputs_before_more_native_work(case, fault):
    if fault == "role":
        packet = case.sources[0]
        case.sources = (
            replace(
                packet, role="probe_train", episode=replace(packet.episode, role="probe_train")
            ),
            *case.sources[1:],
        )
    elif fault == "width":
        packet = case.sources[0]
        case.sources = (replace(packet, features=packet.features[:, :-1].copy()), *case.sources[1:])
    elif fault == "context":
        case.sources = (*case.sources[2:4], *case.sources[:2], *case.sources[4:])
    else:
        case.state.completed_updates = 127
    report: dict[str, Any] = {}
    with pytest.raises(ValueError):
        run(case, report)
    assert case.work["trains"] == case.work["scores"] == []


@pytest.mark.parametrize("failed_update", [1, 2])
def test_should_preserve_only_committed_prefix_after_atomic_native_failure(case, failed_update):
    case.work["fail_train"] = failed_update
    report: dict[str, Any] = {}
    run(case, report)
    assert report["status"] == "incomplete" and "atomic cue update refused" in report["reason"]
    assert len(case.work["trains"]) == failed_update
    assert report["work"]["native_updates_committed"] == failed_update - 1
    assert report["work"]["committed_training_rows"] == 9 * (failed_update - 1)
    assert report["budget"]["updates_completed"] == failed_update - 1
    assert report["deltas"] is None
    if failed_update == 2:
        assert report["arms"][0]["readout"]["status"] == "complete"
        assert report["work"]["score_rows_completed"] == 54


@pytest.mark.parametrize("fault", ["source", "state", "packet", "baseline", "budget"])
def test_should_retain_paid_first_update_when_postcommit_evidence_changes(case, fault):
    def corrupt(record):
        if fault == "source":
            case.source._learner.state.parameters["bias_hidden"][0, 0] += 1
        elif fault == "state":
            case.state.parameters["bias_hidden"][0, 0] += 1
        elif fault == "packet":
            case.sources[0].features[0, 0] += 0.1
        elif fault == "baseline":
            case.baseline["scores"][0]["ce"] = 999.0
        else:
            case.work["budgets"][0].started_at -= 1

    case.work["after_train"] = corrupt
    report: dict[str, Any] = {}
    run(case, report)
    assert report["status"] == "incomplete" and len(case.work["trains"]) == 1
    assert report["work"]["native_updates_committed"] == 1
    assert report["budget"]["updates_completed"] == (0 if fault == "budget" else 1)
    assert report["work"]["committed_training_rows"] == 9
    assert report["arms"][0]["update_committed"] is True
    assert report["arms"][0]["status"] != "complete"
    assert report["deltas"] is None and case.work["scores"] == []


def test_should_refuse_changed_preceding_readout_before_starting_red_update(case):
    report: dict[str, Any] = {}
    armed = [False]

    def after_score(result):
        armed[0] = True

    def boundary():
        if armed[0] and report.get("arms") and report["arms"][0].get("readout"):
            armed[0] = False
            report["arms"][0]["readout"]["scores"][0]["ce"] = 999.0

    case.work["after_score"] = after_score
    case.work["boundary_callback"] = boundary
    run(case, report)
    assert report["status"] == "incomplete" and len(case.work["trains"]) == 1
    assert report["work"]["native_updates_committed"] == 1
    assert report["work"]["score_rows_completed"] == 54
    assert report["arms"][0]["readout"]["scores"][0]["ce"] != 999.0
    assert report["deltas"] is None


@pytest.mark.parametrize("fault", ["replace", "clear", "append"])
def test_should_publish_private_paid_prefix_when_public_arm_container_is_corrupted(case, fault):
    report: dict[str, Any] = {}
    armed = [False]
    case.work["after_score"] = lambda result: armed.__setitem__(0, True)

    def boundary():
        if armed[0] and report.get("arms") and report["arms"][0].get("readout"):
            armed[0] = False
            if fault == "replace":
                report["arms"] = []
            elif fault == "clear":
                report["arms"].clear()
            else:
                report["arms"].append({"ce": 999.0})

    case.work["boundary_callback"] = boundary
    run(case, report)
    assert report["status"] == "incomplete" and len(case.work["trains"]) == 1
    assert len(report["arms"]) == 2
    assert report["arms"][0]["readout"]["status"] == "complete"
    assert report["work"]["native_updates_committed"] == 1
    assert report["work"]["score_rows_completed"] == 54
    assert report["deltas"] is None


def test_should_keep_returned_native_commit_and_diagnostic_when_budget_recording_raises(
    monkeypatch, case
):
    def fail_record(owner):
        raise RuntimeError("budget receipt failed after native return")

    monkeypatch.setattr(budgets.ToyBudgetSession, "record_update", fail_record)
    report: dict[str, Any] = {}
    run(case, report)
    assert report["status"] == "incomplete" and len(case.work["trains"]) == 1
    assert report["arms"][0]["update_committed"] is True
    assert report["arms"][0]["diagnostic"]["definition"] == TORCH_RECURRENT_CE_ID
    assert report["work"]["native_updates_committed"] == 1
    assert report["work"]["committed_training_rows"] == 9
    assert report["work"]["failure_work_may_be_unobserved"] is True
    assert report["budget"]["updates_completed"] == 0


def test_should_stop_at_original_twenty_second_update_budget_with_first_charge_intact(case):
    case.work["after_train"] = lambda record: case.clock.__setitem__(0, 31.0)
    report: dict[str, Any] = {}
    run(case, report)
    assert report["status"] == "incomplete"
    assert len(case.work["trains"]) == 1 and case.work["scores"] == []
    assert report["budget"]["started_at"] == 10.0
    assert report["budget"]["limits"]["max_wall_seconds"] == 20.0
    assert report["budget"]["updates_completed"] == 1
    assert report["work"]["native_updates_committed"] == 1
    assert report["work"]["committed_training_rows"] == 9
    assert case.caller_budget.updates_completed == 0
    assert report["deltas"] is None


@pytest.mark.parametrize("failed_arm", [0, 1])
def test_should_preserve_partial_raw_score_work_without_deriving_deltas(case, failed_arm):
    def incomplete(result):
        if len(case.work["scores"]) - 1 == failed_arm:
            result.update(status="failed", reason="second raw prediction failed")
            result["scores"] = result["scores"][:1]
            result["work"].update(
                score_forward_calls_attempted=2,
                score_forward_calls_completed=1,
                score_rows_attempted=18,
                score_rows_completed=9,
                failure_work_may_be_unobserved=True,
            )

    case.work["after_score"] = incomplete
    report: dict[str, Any] = {}
    run(case, report)
    assert report["status"] == "incomplete"
    assert len(case.work["trains"]) == failed_arm + 1
    assert report["work"]["native_updates_committed"] == failed_arm + 1
    assert report["work"]["score_rows_completed"] == 54 * failed_arm + 9
    assert report["arms"][failed_arm]["readout"]["scores"]
    assert report["work"]["failure_work_may_be_unobserved"] is True
    assert report["deltas"] is None
