"""Fixed repeated-current control on real stream ownership and literal fit ports."""

from copy import deepcopy
from dataclasses import replace
import pickle
import random
from typing import Any

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_train_catalogue as catalogue
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner, DIAGNOSTIC
from src.app.coupled_attribution import AttributionStream
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from src.core.minigrid_context_encoding import context_public_history
from src.core.reference_controller import ReferencePolicy
from test_a4_adaptation import sources
from test_arrived_cue_catalogue import episode
from test_affine_reference_readout import literal_state

WORK = dict(
    owners=0,
    snapshots=0,
    trains=0,
    train_rows=0,
    predicts=0,
    predict_rows=0,
    validations=0,
    validation_rows=0,
    forbidden=0,
)


def scheduled():
    original = sources()
    return tuple(
        replace(original[index], episode_id=f"wake-{i}")
        for i, index in enumerate((0, 1, 2, 3, 0, 1))
    )


def bind_ports(monkeypatch):
    box: dict[str, Any] = dict(fits=[], predictions=[], fail_at=None)
    original_init = ContextLinearLearner.__init__
    original_snapshot = ContextLinearLearner.snapshot_state
    validate = catalogue._validate_sources

    def empty(owner):
        WORK["owners"] += 1
        assert WORK["owners"] <= 200
        original_init(owner)

    def snapshot(owner):
        WORK["snapshots"] += 1
        assert WORK["snapshots"] <= 10000
        return original_snapshot(owner)

    def train(owner, x, y):
        WORK["trains"] += 1
        WORK["train_rows"] += len(x)
        assert WORK["trains"] <= 48 and WORK["train_rows"] <= 720
        assert x.shape == (len(x), 296) and y.shape == (len(x), 1)
        box["fits"].append((x.copy(), y.copy()))
        if len(box["fits"]) == box["fail_at"]:
            owner._state = replace(owner._state, failed_work=(1, len(x), 0, 0, True))
            raise TrainingUpdateFailed(
                "literal fit return lost",
                DIAGNOSTIC,
                dict(
                    fit_calls_attempted=1,
                    fit_rows_attempted=len(x),
                    fit_calls_completed=0,
                    fit_rows_completed=0,
                    failure_work_may_be_unobserved=True,
                ),
            )
        head = replace(
            literal_state(1).heads[0], fit_rows=len(x), rank=0, singular_values=(0.0,) * len(x)
        )
        cue = int(context_public_history(tuple(float(v) for v in x[0]))[297])
        heads = list(owner._state.heads)
        heads[cue] = head
        owner._state = replace(
            owner._state,
            heads=tuple(heads),
            completed_updates=owner._state.completed_updates + 1,
            training_examples=owner._state.training_examples + len(x),
        )
        return TrainingDiagnostic("literal_no_solver", 0.0)

    def predict(owner, x):
        WORK["predicts"] += 1
        WORK["predict_rows"] += len(x)
        assert WORK["predicts"] <= 120 and WORK["predict_rows"] <= 1080
        box["predictions"].append(len(box["fits"]))
        return np.full((len(x), 7), 1 / 7, dtype=np.float64)

    def forbidden(*args, **kwargs):
        WORK["forbidden"] += 1
        raise AssertionError("genuine solver/forward/physics/data work forbidden")

    def validation(*args, **kwargs):
        WORK["validations"] += 1
        assert WORK["validations"] <= 400
        return validate(*args, **kwargs)

    for name in ("canonicalize_public_history", "context_public_history"):
        project = getattr(catalogue, name)

        def counted(value, operation=project):
            WORK["validation_rows"] += 1
            assert WORK["validation_rows"] <= 4000
            return operation(value)

        monkeypatch.setattr(catalogue, name, counted)
    monkeypatch.setattr(catalogue, "_validate_sources", validation)
    monkeypatch.setattr(ContextLinearLearner, "__init__", empty)
    monkeypatch.setattr(ContextLinearLearner, "snapshot_state", snapshot)
    monkeypatch.setattr(ContextLinearLearner, "train_batch", train)
    monkeypatch.setattr(ContextLinearLearner, "predict", predict)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    return box


@pytest.fixture(autouse=True)
def ports(monkeypatch):
    return bind_ports(monkeypatch)


@pytest.fixture(scope="module")
def complete():
    # Obtain the normal function fixture through a finite patch, not hidden fits.
    with pytest.MonkeyPatch.context() as patch:
        box = bind_ports(patch)
        packets = sources()
        before = pickle.dumps(packets)
        rng = pickle.dumps((random.getstate(), np.random.get_state()))
        handoff: list[Any] = []
        result = cli.run_context_linear_stream(
            packets, a4_current_repeated=True, endpoint_observer=handoff.append
        )
        box["handoff"] = handoff
        assert pickle.dumps(packets) == before
        assert pickle.dumps((random.getstate(), np.random.get_state())) == rng
        return result, box, packets


def test_should_run_six_delayed_updates_and_score_only_after_all_fits(complete):
    result, box, packets = complete
    assert result["status"] == "complete", result["reason"]
    assert result["source_unchanged"] is result["actor_unchanged"] is True
    assert [len(x) for x, _ in box["fits"]] == [9, 9, 18, 18, 18, 18]
    assert box["predictions"] == [6] * 16
    assert result["stream"]["evaluation_scope"] == "training_only"
    assert result["budget"]["updates_completed"] == 6
    assert result["budget"]["replay_examples_completed"] == 0
    assert result["budget"]["limits"] == result["budget"]["initial_limits"]
    assert [r["state"]["training_examples"] for r in result["endpoints"]] == [0, 18, 54, 90]
    assert len(box["handoff"]) == 1 and box["handoff"][0].training_examples == 90
    for tick, (index, copies, (x, y), receipt) in enumerate(
        zip(
            (0, 1, 2, 3, 0, 1),
            catalogue.CURRENT_REPETITIONS,
            box["fits"],
            result["committed_receipts"],
            strict=True,
        )
    ):
        packet = packets[index]
        assert x.tolist() == [list(t.observation) for t in packet.transitions] * copies
        assert y[:, 0].tolist() == [t.action for t in packet.transitions] * copies
        assert (
            receipt["observed_at"],
            receipt["arrived_at"],
            receipt["applied_at"],
            receipt["update_number"],
        ) == (tick, tick + 1, tick + 1, tick + 1)


def test_should_report_repeated_current_rows_separately_from_prior_replay(complete):
    result = complete[0]
    costs = result["current_repetition"]
    assert "catalogue" not in result
    assert (
        costs["arrival_row_exposures"],
        costs["repeated_current_row_exposures"],
        costs["fit_row_exposures"],
        costs["prior_packet_replay_rows"],
        costs["retained_bank_rows"],
    ) == (54, 36, 90, 0, 0)
    assert costs["input_array_bytes"] == 90 * 297 * 8
    assert costs["committed_rows"] == dict(arrivals=54, repeated_current=36, native=90)
    assert costs["verified_rows"] == costs["committed_rows"]
    assert len(costs["verified_receipts"]) == 6
    assert costs["failure_work_may_be_unobserved"] is False
    assert costs["failed_attempt_repeated_rows_upper_bound"] is None
    assert [w["original_episode_id"] for w in costs["wakes"]] == [
        sources()[i].episode_id for i in (0, 1, 2, 3, 0, 1)
    ]


def test_should_detach_arrays_and_prove_each_copy_without_fabricated_episode():
    packets = scheduled()
    before = pickle.dumps(packets)
    plan = catalogue.compose_repeated_current_training(packets)
    assert [w.repetitions for w in plan.wakes] == [1, 1, 2, 2, 2, 2]
    all_ids = [row for batch in plan.training for row in batch.row_ids]
    assert len(set(all_ids)) == len(all_ids) == 90
    for i, (batch, wake) in enumerate(zip(plan.training, plan.wakes, strict=True)):
        assert batch.episode is None and batch.role == "train" and batch.num_classes == 7
        assert [(s.episode_id, s.start, s.stop) for s in wake.source_spans] == [
            (packets[i].episode_id, copy * 9, (copy + 1) * 9) for copy in range(wake.repetitions)
        ]
        assert all(
            not np.shares_memory(batch.features, other.features)
            for other in plan.training
            if other is not batch
        )
    plan.training[2].features[0, 0] = 0.75
    plan.training[2].targets[0, 0] = 6
    assert plan.training[2].features[9, 0] != 0.75
    assert plan.training[2].targets[9, 0] != 6
    assert pickle.dumps(packets) == before


@pytest.mark.parametrize("fault", ["count", "list", "role", "rows", "cue", "history", "duplicate"])
def test_should_refuse_bad_composer_sources_before_allocating(monkeypatch, fault):
    packets = scheduled()
    if fault == "count":
        packets = packets[:-1]
    elif fault == "list":
        packets = list(packets)
    elif fault == "role":
        packets = (replace(packets[0], role="probe_train"), *packets[1:])
    elif fault == "rows":
        packets = (episode(packets[0].episode_id, rows=8), *packets[1:])
    elif fault == "cue":
        packets = (packets[1], packets[0], *packets[2:])
    elif fault == "history":
        object.__setattr__(packets[0].transitions[0], "observation", (0.0,) * 296)
    else:
        packets = (packets[0], packets[0], *packets[2:])
    monkeypatch.setattr(np, "tile", lambda *a: pytest.fail("allocation reached"))
    with pytest.raises(ValueError):
        catalogue.compose_repeated_current_training(packets)


def test_should_refuse_array_cap_before_repetition_allocation(monkeypatch):
    monkeypatch.setattr(catalogue, "MAX_INPUT_ARRAY_BYTES", 1)
    monkeypatch.setattr(np, "tile", lambda *a: pytest.fail("allocation reached"))
    with pytest.raises(ValueError, match="arrays exceed"):
        catalogue.compose_repeated_current_training(scheduled())


@pytest.mark.parametrize(
    "options",
    [
        dict(a4_current_repeated=1),
        dict(a4_current_repeated=None),
        dict(a4_current_only=True),
        dict(a4_training=True),
        dict(retain_arrived=True),
        dict(abc_catalogue=True),
        dict(recurring_catalogue=True),
        dict(reference_policy=ReferencePolicy(mode="no_learning")),
    ],
)
def test_should_refuse_unfrozen_or_conflicting_modes_before_owner_work(ports, options):
    before = WORK["owners"]
    with pytest.raises(ValueError):
        cli.run_context_linear_stream(sources(), **{"a4_current_repeated": True, **options})
    assert WORK["owners"] == before and not ports["fits"]


def test_should_keep_two_receipts_and_explicit_lost_fit_costs(ports):
    ports["fail_at"] = 3
    result = cli.run_context_linear_stream(sources(), a4_current_repeated=True)
    assert result["status"] == "failed" and len(ports["fits"]) == 3
    assert result["budget"]["updates_completed"] == 2
    assert len(result["committed_receipts"]) == 2
    costs = result["current_repetition"]
    assert costs["committed_rows"] == dict(arrivals=18, repeated_current=0, native=18)
    assert costs["failed_attempt_repeated_rows_upper_bound"] == 9
    assert costs["failure_work_may_be_unobserved"] is True
    assert result["failure_work"]["work"]["fit_rows_attempted"] == 18
    assert not ports["predictions"]


def test_should_refuse_corrupted_prior_receipt_before_another_fit(ports, monkeypatch):
    original = AttributionStream.run_until

    def run(owner, updates):
        original(owner, updates)
        if updates == 2:
            receipt = next(iter(owner._runtime._inbox._applied.values()))
            object.__setattr__(receipt, "sample_id", "corrupted")

    monkeypatch.setattr(AttributionStream, "run_until", run)
    result = cli.run_context_linear_stream(
        sources(), a4_current_repeated=True, check_boundary=lambda: None
    )
    assert result["status"] == "failed" and len(ports["fits"]) == 2
    assert result["budget"]["updates_completed"] == 2
    assert result["committed_receipts"] is None and result["receipt_capture_error"]
    assert result["current_repetition"]["committed_rows"] is None
    assert not ports["predictions"]


@pytest.mark.parametrize(
    "mode,rows",
    [(dict(a4_current_only=True), 54), (dict(a4_training=True, retain_arrived=True), 90)],
)
def test_should_preserve_old_positive_modes_and_report_shapes(ports, mode, rows):
    result = cli.run_context_linear_stream(sources(), **mode)
    assert result["status"] == "complete", result["reason"]
    assert sum(len(x) for x, _ in ports["fits"]) == rows
    assert "current_repetition" not in result
    assert ("catalogue" in result) is (rows == 90)
    assert result["source_unchanged"] is result["actor_unchanged"] is True


@pytest.mark.parametrize("fault", ["original", "candidate"])
def test_should_refuse_changed_original_or_changed_candidate_before_more_work(
    ports, monkeypatch, fault
):
    packets = sources()
    original = AttributionStream.run_until

    def run(owner, updates):
        original(owner, updates)
        if updates == 2:
            if fault == "original":
                object.__setattr__(packets[0], "role", "final_test")
            else:
                learner = owner._runtime._candidate
                learner._state = replace(learner._state, training_examples=19)

    monkeypatch.setattr(AttributionStream, "run_until", run)
    result = cli.run_context_linear_stream(packets, a4_current_repeated=True)
    assert result["status"] == "failed" and len(ports["fits"]) == 2
    assert result["budget"]["updates_completed"] == 2 and not ports["predictions"]
    assert len(result["current_repetition"]["verified_receipts"]) <= 2


def test_should_describe_fixed_control_without_work(ports):
    description = cli.describe_context_linear_a4_current_repeated()
    assert description["fit_calls"] == 6 and description["fit_rows"] == 90
    assert description["checkpoint_updates"] == (0, 2, 4, 6)
    assert description["current_packet_multiplicities"] == (1, 1, 2, 2, 2, 2)
    assert description["catalogue_replay_row_exposures"] == 0
    assert not ports["fits"] and not ports["predictions"]
    assert deepcopy(description) == description
