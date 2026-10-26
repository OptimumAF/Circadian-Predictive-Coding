"""Scripted TRAIN prior-gradient composition, without models or environments."""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import pytest

from src.adapters import minigrid_gradient_readouts as gradients
from src.adapters import minigrid_learning_pilot as shared
from src.adapters.torch_recurrent_gradient_diagnostics import GradientDiagnosticFailed
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition


def sources():
    observation = (-1.0,) * 296
    packets = []
    for context in ("A", "B", "C_train"):
        for cue in ("blue", "red"):
            episode = DemonstrationEpisode(
                f"minigrid-source-{context}-{cue}",
                context,
                "train",
                tuple(
                    DemonstrationTransition(
                        step, observation, step % 7, 0.0, observation, step == 8, 7
                    )
                    for step in range(9)
                ),
                7,
            )
            packets.append(shared._batch(episode))
    return tuple(packets)


def report():
    return dict(
        scores=[dict(episode=index, ce=0.5) for index in range(6)],
        prior_gradient_geometry=[],
        work=dict(
            score_forward_calls_completed=6,
            score_rows_completed=54,
            prior_gradient_geometry=gradients.empty_prior_work(),
        ),
    )


def payload(*, complete=True):
    work = dict.fromkeys(gradients.PRIOR_NATIVE_WORK_KEYS, 0)
    work.update(
        snapshot_validations_attempted=1,
        snapshot_validations_completed=1,
        validation_temporary_optimizers=0,
        projection_calls_attempted=2,
        projection_calls_completed=2 if complete else 1,
        projection_rows_attempted=18,
        projection_rows_completed=18 if complete else 9,
        projection_array_bytes_returned=(18 if complete else 9) * 296 * 8,
        forward_calls_attempted=2,
        forward_calls_completed=2 if complete else 1,
        forward_rows_attempted=18,
        forward_rows_completed=18 if complete else 9,
        backward_calls_attempted=2 if complete else 1,
        backward_calls_completed=2 if complete else 1,
        backward_rows_attempted=18 if complete else 9,
        backward_rows_completed=18 if complete else 9,
        failure_work_may_be_unobserved=not complete,
    )
    return dict(
        status="complete" if complete else "incomplete",
        reason=None if complete else "second prior forward refused",
        definition="torch_invariant_feature_prior_mean_ce_gradient_geometry_v1",
        joint_geometry=dict(cosine=-0.5),
        work=work,
    )


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(
        gradients, "_same_state", lambda first, second: pickle.dumps(first) == pickle.dumps(second)
    )
    return dict(
        report=report(),
        state={"weights": [1.0], "optimizer": {"momentum": 2.0}},
        sources=sources(),
    )


def install_reader(monkeypatch, *, callback=None):
    calls: list[dict[str, Any]] = []

    def read(state, pair):
        assert type(pair) is tuple and len(pair) == 2
        assert pair[0].context == pair[1].context in ("A", "B", "C_train")
        assert all(
            episode.role == "train"
            and episode.num_classes == 7
            and len(episode.transitions) == 9
            and len(episode.transitions[0].observation) == 296
            for episode in pair
        )
        call = dict(state=state, pair=pair, payload=payload())
        calls.append(call)
        if callback is not None:
            result = callback(call, len(calls) - 1)
            if result is not None:
                return result
        return call["payload"]

    monkeypatch.setattr(gradients, "diagnose_prior_gradients", read)
    return calls


def run(case, *, check_boundary=lambda: None):
    gradients.run_prior_gradient_readouts(
        case["report"], case["state"], case["sources"], check_boundary=check_boundary
    )


def test_should_compose_three_owned_prior_pairs_and_separate_projection_costs(monkeypatch, case):
    calls = install_reader(monkeypatch)
    before = pickle.dumps(case["state"])
    run(case)
    result = case["report"]
    assert [row["context"] for row in result["prior_gradient_geometry"]] == ["A", "B", "C_train"]
    assert len(calls) == 3 and pickle.dumps(case["state"]) == before
    assert all(call["state"] is not case["state"] for call in calls)
    work = result["work"]["prior_gradient_geometry"]
    assert work["pair_calls_attempted"] == work["pair_calls_completed"] == 3
    assert work["geometry_reductions_completed"] == 15
    assert work["train_rows_admitted"] == 54 and work["input_array_equivalent_bytes"] == 128304
    assert work["forward_rows_completed"] == work["backward_rows_completed"] == 54
    assert work["projection_calls_attempted"] == work["projection_calls_completed"] == 6
    assert work["projection_rows_attempted"] == work["projection_rows_completed"] == 54
    assert work["projection_array_bytes_returned"] == 127872
    assert work["optimizer_steps"] == work["published_updates"] == 0
    assert work["failure_work_may_be_unobserved"] is False
    assert result["work"]["score_forward_calls_completed"] == 6
    assert "cue_gradient_geometry" not in result and len(result["scores"]) == 6
    calls[0]["payload"]["joint_geometry"]["cosine"] = 999
    assert result["prior_gradient_geometry"][0]["diagnostic"]["joint_geometry"]["cosine"] == -0.5


@pytest.mark.parametrize("failure", ["native", "incomplete", "generic"])
def test_should_preserve_paid_pair_and_partial_costs_on_second_diagnostic_failure(
    monkeypatch, case, failure
):
    def fail(call, index):
        if index == 1:
            if failure == "generic":
                raise RuntimeError("reader return unobserved")
            partial = payload(complete=False)
            if failure == "native":
                raise GradientDiagnosticFailed(partial)
            return partial
        return None

    calls = install_reader(monkeypatch, callback=fail)
    with pytest.raises((GradientDiagnosticFailed, RuntimeError, ValueError)):
        run(case)
    result = case["report"]
    records = result["prior_gradient_geometry"]
    assert len(calls) == len(records) == 2 and len(result["scores"]) == 6
    assert records[0]["diagnostic"]["status"] == "complete"
    if failure == "generic":
        assert records[1]["diagnostic"] is None
    else:
        assert records[1]["diagnostic"]["status"] == "incomplete"
    work = result["work"]["prior_gradient_geometry"]
    assert work["pair_calls_attempted"] == 2 and work["geometry_reductions_completed"] == 5
    rows = 18 if failure == "generic" else 27
    assert work["forward_rows_completed"] == work["projection_rows_completed"] == rows
    assert work["failure_work_may_be_unobserved"] is True
    assert work["optimizer_steps"] == work["published_updates"] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "state",
        "diagnostic_state",
        "sources",
        "scores",
        "score_owner",
        "work",
        "work_owner",
        "record",
        "record_owner",
        "boundary",
    ],
)
def test_should_keep_returned_payload_before_refusing_changed_prior_evidence(
    monkeypatch, case, fault
):
    result = case["report"]
    paid_scores = deepcopy(result["scores"])

    def corrupt(call, index):
        if fault == "state":
            case["state"]["weights"][0] = 9.0
        elif fault == "diagnostic_state":
            call["state"]["weights"][0] = 9.0
        elif fault == "sources":
            case["sources"][0].features[0, 0] = -0.8
        elif fault == "scores":
            result["scores"][0]["ce"] = 9.0
        elif fault == "score_owner":
            result["scores"] = deepcopy(result["scores"])
        elif fault == "work":
            result["work"]["prior_gradient_geometry"]["forward_rows_completed"] = 999
        elif fault == "work_owner":
            result["work"] = deepcopy(result["work"])
        elif fault == "record":
            result["prior_gradient_geometry"][0]["context"] = "replaced"
        elif fault == "record_owner":
            result["prior_gradient_geometry"][0] = deepcopy(result["prior_gradient_geometry"][0])

    calls = install_reader(monkeypatch, callback=corrupt)

    def boundary():
        if calls and fault == "boundary":
            raise ValueError("earlier completed stream evidence changed")

    with pytest.raises(ValueError, match="evidence changed"):
        run(case, check_boundary=boundary)
    assert len(calls) == 1 and result["scores"] == paid_scores
    assert len(result["prior_gradient_geometry"]) == 1
    assert result["prior_gradient_geometry"][0]["context"] == "A"
    assert result["prior_gradient_geometry"][0]["diagnostic"]["status"] == "complete"
    work = result["work"]["prior_gradient_geometry"]
    assert work["pair_calls_completed"] == 1 and work["geometry_reductions_completed"] == 5
    assert work["forward_rows_completed"] == work["projection_rows_completed"] == 18
    assert work["failure_work_may_be_unobserved"] is True
    assert result["work"]["score_forward_calls_completed"] == 6


def test_should_refuse_replaced_preceding_record_after_preserving_second_return(monkeypatch, case):
    def corrupt(call, index):
        if index == 1:
            case["report"]["prior_gradient_geometry"][0] = deepcopy(
                case["report"]["prior_gradient_geometry"][0]
            )

    calls = install_reader(monkeypatch, callback=corrupt)
    with pytest.raises(ValueError, match="evidence changed"):
        run(case)
    result = case["report"]
    assert len(calls) == len(result["prior_gradient_geometry"]) == 2
    assert all(
        row["diagnostic"]["status"] == "complete" for row in result["prior_gradient_geometry"]
    )
    assert result["work"]["prior_gradient_geometry"]["geometry_reductions_completed"] == 10


@pytest.mark.parametrize(
    "fault", ["order", "missing", "role", "width", "rows", "binding", "discrete", "row_ids"]
)
def test_should_reject_invalid_train_source_before_calling_reader(monkeypatch, case, fault):
    packets = case["sources"]
    first = packets[0]
    if fault == "order":
        packets = (packets[1], packets[0], *packets[2:])
    elif fault == "missing":
        packets = packets[:-1]
    elif fault == "role":
        episode = first.episode
        assert episode is not None
        first = replace(first, role="probe_train", episode=replace(episode, role="probe_train"))
    elif fault == "width":
        first = replace(first, features=first.features[:, :148].copy())
    elif fault == "rows":
        first = replace(first, features=first.features[:8].copy())
    elif fault == "binding":
        first.features[0, 0] = -0.8
    elif fault == "discrete":
        episode = first.episode
        assert episode is not None
        observation = (-0.9,) + episode.transitions[0].observation[1:]
        transitions = tuple(
            replace(row, observation=observation, next_observation=observation)
            for row in episode.transitions
        )
        first = shared._batch(replace(episode, transitions=transitions))
    else:
        first = replace(first, row_ids=packets[1].row_ids)
    if fault not in ("order", "missing"):
        packets = (first, *packets[1:])
    case["sources"] = packets
    calls = install_reader(monkeypatch)
    with pytest.raises(ValueError):
        run(case)
    assert not calls and not case["report"]["prior_gradient_geometry"]
    assert case["report"]["work"]["prior_gradient_geometry"] == gradients.empty_prior_work()


def test_should_sum_prior_costs_without_changing_recurrent_work_schema():
    first = gradients.empty_prior_work()
    first.update(projection_rows_completed=18, forward_rows_completed=18, seconds=0.25)
    second = gradients.empty_prior_work()
    second.update(projection_rows_completed=9, seconds=0.5, failure_work_may_be_unobserved=True)
    result = gradients.sum_prior_work([first, second])
    assert result["projection_rows_completed"] == 27 and result["seconds"] == 0.75
    assert (
        result["forward_rows_completed"] == 18 and result["failure_work_may_be_unobserved"] is True
    )
    assert "projection_rows_completed" not in gradients.empty_work()
    assert "projection_rows_completed" not in gradients.sum_work([first])
