"""Matched recurring row exposure through real admission; no numerical solver."""

from copy import deepcopy
from dataclasses import replace
import pickle
import random
from typing import Any

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_context_linear_learner as port
from src.adapters import minigrid_train_catalogue as catalogue
from src.app.coupled_attribution import AttributionStream
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.demonstration_episode import episode_arrays
from src.core.reference_controller import ReferencePolicy
from test_arrived_cue_catalogue import episode

MULTIPLICITIES = (1, 1, 2, 2, 3, 3, 4, 4, 4, 4)
WORK = dict(solver_stubs=0, solver_rows=0, scores=0, score_rows=0)


def sources():
    return tuple(
        episode(f"minigrid-source-{context}-{name}", cue, mark=mark, action=action)
        for context, mark, action in (
            ("A", 1, 2),
            ("B", 3, 5),
            ("C_train", 5, 1),
            ("G_train", 7, 4),
        )
        for name, cue in (("blue", 2), ("red", 0))
    )


def scheduled():
    original = sources()
    return tuple(
        replace(original[index], episode_id=f"arrival-{i}")
        for i, index in enumerate((*range(8), 0, 1))
    )


def run(packets, **options):
    return cli.run_context_linear_stream(
        packets, **{"recurring_catalogue": True, "recurring_current_only": True, **options}
    )


def bind_ports(monkeypatch):
    box: dict[str, Any] = dict(fits=[], scores=[], fail_at=None, handoff=[])

    def solve(x, y, classes):
        assert classes == 7
        WORK["solver_stubs"] += 1
        WORK["solver_rows"] += len(x)
        assert WORK["solver_stubs"] <= 1600 and WORK["solver_rows"] <= 57600
        box["fits"].append((x.copy(), y.copy()))
        result = CategoricalLinearFit(296, 7, len(x), ((0.0,) * 7,) * 297, 0, (0.0,) * len(x), 0.0)
        if len(box["fits"]) == box["fail_at"]:
            raise RuntimeError("scripted lost solve return")
        return result

    def read(owner, x):
        WORK["scores"] += 1
        WORK["score_rows"] += len(x)
        assert WORK["scores"] <= 7680 and WORK["score_rows"] <= 69120
        box["scores"].append(len(box["fits"]))
        return np.full((len(x), 7), 1 / 7, dtype=np.float64)

    def forbidden(*args, **kwargs):
        pytest.fail("genuine solve/runtime/source read is forbidden")

    monkeypatch.setattr(port, "fit_categorical_linear", solve)
    monkeypatch.setattr(port.ContextLinearLearner, "predict", read)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    return box


@pytest.fixture
def ports(monkeypatch):
    return bind_ports(monkeypatch)


@pytest.fixture(scope="module")
def complete():
    with pytest.MonkeyPatch.context() as patch:
        box = bind_ports(patch)
        packets = sources()
        original = pickle.dumps(packets)
        rng = pickle.dumps((np.random.get_state(), random.getstate()))
        result = run(packets, endpoint_observer=box["handoff"].append)
        assert pickle.dumps(packets) == original
        assert pickle.dumps((np.random.get_state(), random.getstate())) == rng
        return result, box, packets


def test_should_match_joint_shapes_but_repeat_only_current_train_with_owned_arrays():
    packets = scheduled()
    before = pickle.dumps(packets)
    current = catalogue.compose_repeated_current_training(packets, recurring=True)
    joint = catalogue.compose_arrived_cue_catalogue(packets, retained_per_cue=3, max_arrivals=10)
    assert [len(b.features) for b in current.training] == [len(b.features) for b in joint.training]
    assert (current.arrival_row_exposures, current.repeated_current_row_exposures) == (90, 162)
    assert current.fit_row_exposures == joint.fit_row_exposures == 252
    assert current.input_array_bytes == joint.input_array_bytes == 598752
    for i, (batch, wake) in enumerate(zip(current.training, current.wakes, strict=True)):
        x, y = episode_arrays(packets[i])
        assert np.array_equal(batch.features, np.tile(x, (MULTIPLICITIES[i], 1)))
        assert np.array_equal(batch.targets, np.tile(y, (MULTIPLICITIES[i], 1)))
        assert batch.episode is None and not batch.packet_episodes and batch.role == "train"
        assert {s.episode_id for s in wake.source_spans} == {packets[i].episode_id}
        assert (wake.observed_at, wake.arrived_at, wake.repetitions) == (
            i,
            i + 1,
            MULTIPLICITIES[i],
        )
        assert all(
            not np.shares_memory(batch.features, other.features)
            for other in current.training
            if other is not batch
        )
    current.training[-1].targets[0, 0] = 6
    assert current.training[-1].targets[9, 0] != 6
    assert pickle.dumps(packets) == before


def test_should_preserve_default_six_and_explicit_false_arrays_ids_and_costs():
    packets = scheduled()[:6]
    left = catalogue.compose_repeated_current_training(packets)
    right = catalogue.compose_repeated_current_training(packets, recurring=False)
    assert [w.repetitions for w in left.wakes] == [1, 1, 2, 2, 2, 2]
    assert pickle.dumps(left) == pickle.dumps(right)


def test_should_never_use_future_or_earlier_targets_in_a_current_only_wake():
    packets = scheduled()
    original = catalogue.compose_repeated_current_training(packets, recurring=True)
    changed = tuple(
        replace(p, transitions=tuple(replace(t, action=6) for t in p.transitions))
        if i in (0, 9)
        else p
        for i, p in enumerate(packets)
    )
    newer = catalogue.compose_repeated_current_training(changed, recurring=True)
    for i in range(1, 9):
        assert np.array_equal(original.training[i].features, newer.training[i].features)
        assert np.array_equal(original.training[i].targets, newer.training[i].targets)


@pytest.mark.parametrize("flag", [1, None, "true"])
def test_should_reject_ambiguous_recurring_flag_before_allocation(monkeypatch, flag):
    monkeypatch.setattr(np, "tile", lambda *a: pytest.fail("allocation reached"))
    with pytest.raises(ValueError, match="exact bool"):
        catalogue.compose_repeated_current_training(scheduled(), recurring=flag)


@pytest.mark.parametrize(
    "fault",
    [
        "count",
        "list",
        "probe_train",
        "probe_test",
        "final_test",
        "rows",
        "cue",
        "pixels",
        "duplicate",
        "label",
    ],
)
def test_should_refuse_invalid_original_train_before_repetition(monkeypatch, fault):
    packets: Any = scheduled()
    if fault == "count":
        packets = packets[:-1]
    elif fault == "list":
        packets = list(packets)
    elif fault in ("probe_train", "probe_test", "final_test"):
        packets = (replace(packets[0], role=fault), *packets[1:])
    elif fault == "rows":
        packets = (episode("short", rows=8), *packets[1:])
    elif fault == "cue":
        packets = (packets[1], packets[0], *packets[2:])
    elif fault == "pixels":
        object.__setattr__(packets[-1].transitions[0], "observation", (0.0,) * 296)
    elif fault == "duplicate":
        packets = (packets[0], packets[0], *packets[2:])
    else:
        object.__setattr__(packets[-1].transitions[0], "action", 7)
    monkeypatch.setattr(np, "tile", lambda *a: pytest.fail("allocation reached"))
    with pytest.raises(ValueError):
        catalogue.compose_repeated_current_training(packets, recurring=True)


def test_should_refuse_storage_cap_before_allocating_next_repetition(monkeypatch):
    monkeypatch.setattr(catalogue, "MAX_INPUT_ARRAY_BYTES", 1)
    monkeypatch.setattr(np, "tile", lambda *a: pytest.fail("allocation reached"))
    with pytest.raises(ValueError, match="arrays exceed"):
        catalogue.compose_repeated_current_training(scheduled(), recurring=True)


@pytest.mark.parametrize("fault", ["label", "equal_observation", "record_dictionary"])
def test_should_refuse_changed_or_replaced_source_during_preparation(monkeypatch, fault):
    packets = scheduled()
    tile, calls = np.tile, []

    def corrupt(*args):
        result = tile(*args)
        calls.append(1)
        if len(calls) == 1:
            row = packets[-1].transitions[0]
            if fault == "label":
                object.__setattr__(row, "action", 6)
            elif fault == "equal_observation":
                object.__setattr__(row, "observation", tuple(list(row.observation)))
            else:
                object.__setattr__(row, "__dict__", dict(vars(row)))
        return result

    monkeypatch.setattr(np, "tile", corrupt)
    with pytest.raises(ValueError, match="changed"):
        catalogue.compose_repeated_current_training(packets, recurring=True)


@pytest.mark.parametrize(
    "options",
    [
        dict(recurring_current_only=1),
        dict(recurring_current_only=None),
        dict(recurring_catalogue=False),
        dict(retain_arrived=True),
        dict(abc_catalogue=True),
        dict(a4_training=True),
        dict(a4_current_only=True),
        dict(a4_current_repeated=True),
        dict(reference_policy=ReferencePolicy(mode="no_learning")),
        dict(train_update="minimum_change"),
    ],
)
def test_should_refuse_mixed_intents_before_sources_or_owner(monkeypatch, options):
    monkeypatch.setattr(port, "ContextLinearLearner", lambda: pytest.fail("owner allocated"))
    with pytest.raises(ValueError):
        run(object(), **options)


def test_should_execute_only_ten_delayed_fits_with_original_source_actor_and_budgets(complete):
    result, box, packets = complete
    assert result["status"] == "complete", result["reason"]
    assert result["source_unchanged"] is result["actor_unchanged"] is True
    assert [len(x) for x, _ in box["fits"]] == [9 * n for n in MULTIPLICITIES]
    assert result["observed_checkpoints"] == [0, 2, 4, 6, 8, 10]
    assert [r["state"]["training_examples"] for r in result["endpoints"]] == [
        0,
        18,
        54,
        108,
        180,
        252,
    ]
    assert len(box["handoff"]) == 1 and box["handoff"][0].training_examples == 252
    budget = result["budget"]
    assert budget["initial_limits"] == budget["limits"]
    assert budget["updates_completed"] == budget["limits"]["max_training_updates"] == 10
    assert budget["replay_examples_completed"] == 0
    assert result["stream"]["evaluation_scope"] == "training_only"
    for i, ((x, y), receipt) in enumerate(
        zip(box["fits"], result["committed_receipts"], strict=True)
    ):
        packet = packets[(*range(8), 0, 1)[i]]
        assert y[:, 0].tolist() == [t.action for t in packet.transitions] * MULTIPLICITIES[i]
        assert (
            receipt["observed_at"],
            receipt["arrived_at"],
            receipt["applied_at"],
            receipt["update_number"],
        ) == (i, i + 1, i + 1, i + 1)


def test_should_separate_current_exposure_from_retained_replay_and_defer_all_scores(complete):
    result, box, _ = complete
    costs = result["current_repetition"]
    assert "catalogue" not in result
    assert (
        costs["committed_rows"]
        == costs["verified_rows"]
        == dict(arrivals=90, repeated_current=162, native=252)
    )
    assert costs["prior_packet_replay_rows"] == costs["retained_bank_rows"] == 0
    assert len(costs["verified_receipts"]) == 10
    assert costs["input_array_bytes"] == 598752
    assert not costs["failure_work_may_be_unobserved"]
    assert box["scores"] == [10] * 48
    assert result["score_rows_completed"] == 432


def test_should_keep_eight_paid_receipts_and_unknown_ninth_fit_work(ports):
    ports["fail_at"] = 9
    result = run(sources(), endpoint_observer=ports["handoff"].append)
    assert result["status"] == "failed"
    assert len(ports["fits"]) == 9 and not ports["scores"] and not ports["handoff"]
    assert result["budget"]["updates_completed"] == len(result["committed_receipts"]) == 8
    assert result["current_repetition"]["committed_rows"] == dict(
        arrivals=72, repeated_current=108, native=180
    )
    assert result["current_repetition"]["failed_attempt_repeated_rows_upper_bound"] == 27
    assert result["current_repetition"]["failure_work_may_be_unobserved"]
    assert result["failure_work"]["work"]["fit_rows_attempted"] == 36
    assert result["budget"]["initial_limits"] == result["budget"]["limits"]


@pytest.mark.parametrize(
    "fault",
    [
        "source_role",
        "equal_observation",
        "candidate",
        "receipt",
        "budget_birth",
        "consent",
        "charges",
    ],
)
def test_should_stop_after_paid_prefix_when_authority_or_evidence_changes(
    ports, monkeypatch, fault
):
    packets = sources()
    advance = AttributionStream.run_until

    def corrupt(owner, count):
        advance(owner, count)
        if count != 2:
            return
        if fault == "source_role":
            object.__setattr__(packets[-1], "role", "final_test")
        elif fault == "equal_observation":
            row = packets[-1].transitions[0]
            object.__setattr__(row, "observation", tuple(list(row.observation)))
        elif fault == "candidate":
            learner = owner._runtime._candidate
            learner._state = replace(learner._state, training_examples=19)
        elif fault == "receipt":
            receipt = next(iter(owner._runtime._inbox._applied.values()))
            object.__setattr__(receipt, "sample_id", "corrupted")
        elif fault == "budget_birth":
            owner._budget.started_at += 1
        elif fault == "consent":
            owner._replay_consent = True
        else:
            owner._budget.updates_completed = 0

    monkeypatch.setattr(AttributionStream, "run_until", corrupt)
    result = run(packets)
    assert result["status"] == "failed" and len(ports["fits"]) == 2
    assert not ports["scores"]
    assert len(result["current_repetition"]["verified_receipts"]) <= 2
    if fault == "receipt":
        assert result["committed_receipts"] is None and result["receipt_capture_error"]
    elif fault not in ("budget_birth", "consent", "charges"):
        assert result["budget"]["updates_completed"] == 2


def test_should_preserve_original_recurring_joint_and_default_false_outputs(ports):
    packets = sources()
    implicit = cli.run_context_linear_stream(packets, retain_arrived=True, recurring_catalogue=True)
    explicit = cli.run_context_linear_stream(
        packets, retain_arrived=True, recurring_catalogue=True, recurring_current_only=False
    )
    assert implicit["status"] == explicit["status"] == "complete"
    assert "current_repetition" not in implicit and "current_repetition" not in explicit
    assert implicit["catalogue"] == explicit["catalogue"]
    assert implicit["protocol"] == explicit["protocol"]
    assert implicit["endpoints"] == explicit["endpoints"]
    assert implicit["catalogue"]["committed_rows"] == dict(arrivals=90, replay=162, native=252)


def test_should_describe_without_any_owner_or_solver(ports):
    description = cli.describe_context_linear_recurring_current_only()
    assert description["fit_calls"] == 10 and description["fit_rows"] == 252
    assert description["current_packet_multiplicities"] == MULTIPLICITIES
    assert description["catalogue_replay_row_exposures"] == 0
    assert description["repeated_current_row_exposures"] == 162
    assert deepcopy(description) == description
    assert not ports["fits"] and not ports["scores"]
