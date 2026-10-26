"""Cumulative released TRAIN packets through one bounded existing runtime.

Prospective per-platform numerical fixtures:9 initializers/9 frozen A fits
(18 rows/36 feature values),18 prototype appends/36 rows,0 RNG/SGD/native work.
Only the shared success predicts:7 callback calls/20 rows/100 distance candidates
and200 feature values. Append+prediction normalization is25 calls/56 rows/112
coordinates, plus8 validation normalizations/16 rows/32 coordinates. Other
callbacks return metadata or inject bounded faults.
Admission cases do no numerical work; the capacity refusal uses257 metadata rows.
The root runner owns all executable checks; this file was authored source-only.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from typing import Any

import numpy as np
import pytest

from src.adapters import continuous_stream_demo as demo
from src.adapters import continuous_stream_sequence as sequence
from src.adapters.numpy_nearest_continuous_bc import OBJECTIVE, NumpyNearestContinuousBCLearner
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)


def episode(name: str, *, rows: int = 2, role: str = "train", spec=None) -> ContinuousEpisode:
    spec = spec or ContinuousActionSpec(2, ("m",), (-1.0,), (1.0,))
    start = 0 if name in ("warm", "return") else 10 if name == "new" else 20
    actions = (
        (-1.0, -0.5)
        if name in ("warm", "return")
        else (1.0, 0.5)
        if name == "new"
        else (0.25, 0.75)
    )
    return ContinuousEpisode(
        name,
        "warm-context" if name == "return" else name + "-context",
        role,
        spec,
        tuple(
            ContinuousTransition(
                index,
                (float(start + index), 0.0),
                (actions[index % 2],),
                float(index == rows - 1),
                (float(start + index + 1), 0.0),
                index == rows - 1,
                False,
            )
            for index in range(rows)
        ),
    )


def inputs():
    return episode("warm"), episode("new"), (episode("third"), episode("return"))


def assert_state(first, second) -> None:
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    np.testing.assert_array_equal(first.prototype_features, second.prototype_features)
    np.testing.assert_array_equal(first.prototype_actions, second.prototype_actions)


@pytest.fixture(scope="module")
def completed():
    warm, new, additional = inputs()
    original_inputs = tuple(asdict(item) for item in (warm, new, *additional))
    captured: dict[str, Any] = {"runtime": [], "checkpoints": [], "trains": []}
    factory, train = sequence.ActorShadowRuntime, NumpyNearestContinuousBCLearner.train_batch

    def observer(readout, metadata):
        owned_metadata = deepcopy(metadata)
        state = readout.snapshot_state()
        captured["checkpoints"].append((readout, owned_metadata, state))
        raw = [(0.0, 0.0), (1.0, 0.0)]
        if metadata["phase"] == "after_release":
            current = {"new": new, "third": additional[0], "return": additional[1]}[
                metadata["episode_id"]
            ]
            raw.extend(row.observation for row in current.transitions)
        prediction = readout.predict_with_work(np.asarray(raw, dtype=np.float64))
        # Snapshot copies and callback metadata are caller-owned, not live state.
        state.prototype_actions[0, 0] = 0.99
        metadata["tick"] = -999
        metadata["known_episode_ids"] = ["callback-only"]
        return {
            "actions": prediction.actions.tolist(),
            "work": {
                "prediction_calls": 1,
                "prediction_rows": len(raw),
                "distance_candidates": prediction.distance_candidates,
                "distance_feature_values": prediction.distance_feature_values,
            },
        }

    with pytest.MonkeyPatch.context() as patch:

        def runtime(source, **options):
            created = factory(source, **options)
            captured["runtime"].append((source, created, options, source.snapshot_state()))
            return created

        def append(model, features, targets):
            captured["trains"].append((model, features.copy(), targets.copy()))
            return train(model, features, targets)

        patch.setattr(sequence, "ActorShadowRuntime", runtime)
        patch.setattr(NumpyNearestContinuousBCLearner, "train_batch", append)
        report = demo.run_continuous_demo(
            warm, new, additional_episodes=additional, checkpoint_observer=observer
        )
    assert report["complete"] is True
    assert tuple(asdict(item) for item in (warm, new, *additional)) == original_inputs
    return report, captured


def test_should_hold_each_packet_for_two_ticks_and_charge_one_shared_budget(completed):
    report, captured = completed
    assert report["protocol_id"] == "continuous_nearest_sequence_v1"
    assert [row["tick"] for row in report["events"]] == list(range(9))
    assert [row["packet_index"] for row in report["events"]] == [0] * 3 + [1] * 3 + [2] * 3
    assert [len(row["applied"]) for row in report["events"]] == [0, 0, 1] * 3
    assert [row["candidate_prototype_rows"] for row in report["events"]] == [
        2,
        2,
        4,
        4,
        4,
        6,
        6,
        6,
        8,
    ]
    assert [row["budget_updates"] for row in report["events"]] == [0, 0, 1, 1, 1, 2, 2, 2, 3]
    assert len(report["applied"]) == 3
    for index, receipt in enumerate(report["applied"]):
        assert receipt["observed_at"] == 3 * index
        assert receipt["arrived_at"] == receipt["applied_at"] == 3 * index + 2
        assert receipt["update_number"] == index + 1
        assert receipt["episode_id"] == ("new", "third", "return")[index]
        assert receipt["diagnostic"] == {"definition": OBJECTIVE, "value": 2}
    _, _, options, _ = captured["runtime"][0]
    assert options["budget"].budget.max_training_updates == 3
    assert options["budget"].updates_completed == 3
    assert options["budget"].replay_examples_completed == 0


def test_should_keep_source_actor_old_prefix_and_warm_statistics_exact(completed):
    _, captured = completed
    owner, runtime, _, initial = captured["runtime"][0]
    assert len(captured["runtime"]) == 1 and len(captured["trains"]) == 4
    assert_state(initial, owner.snapshot_state())
    assert_state(initial, runtime.actor.snapshot_state().state)
    final = runtime.candidate_snapshot().state
    assert final.completed_updates == 4 and final.training_examples == 8
    assert final.policy == initial.policy
    assert final.policy.observation_standardization.means == (0.5, 0.0)
    assert final.policy.observation_standardization.scales == (0.5, 1.0)
    np.testing.assert_array_equal(final.prototype_features[:2], initial.prototype_features)
    np.testing.assert_array_equal(
        final.prototype_actions[:, 0], [-1.0, -0.5, 1.0, 0.5, 0.25, 0.75, -1.0, -0.5]
    )


def test_should_show_acquisition_retention_and_return_through_fresh_detached_observers(completed):
    report, captured = completed
    checkpoints = report["checkpoints"]
    assert [row["phase"] for row in checkpoints] == [
        "warm",
        "before_release",
        "after_release",
        "before_release",
        "after_release",
        "before_release",
        "after_release",
    ]
    assert [row["tick"] for row in checkpoints] == [0, 0, 2, 3, 5, 6, 8]
    assert [row["prototype_rows"] for row in checkpoints] == [2, 2, 4, 4, 6, 6, 8]
    assert [row["completed_updates"] for row in checkpoints] == [1, 1, 2, 2, 3, 3, 4]
    assert [row["known_episode_ids"] for row in checkpoints] == [
        ["warm"],
        ["warm"],
        ["warm", "new"],
        ["warm", "new"],
        ["warm", "new", "third"],
        ["warm", "new", "third"],
        ["warm", "new", "third", "return"],
    ]
    assert all(row["result"]["actions"][:2] == [[-1.0], [-0.5]] for row in checkpoints)
    assert [checkpoints[index]["result"]["actions"][2:] for index in (2, 4, 6)] == [
        [[1.0], [0.5]],
        [[0.25], [0.75]],
        [[-1.0], [-0.5]],
    ]
    assert all(
        row["tick"] >= 0 and "callback-only" not in row["known_episode_ids"] for row in checkpoints
    )
    owners = [item[0] for item in captured["checkpoints"]]
    assert len({id(owner) for owner in owners}) == 7
    source, runtime, _, _ = captured["runtime"][0]
    assert all(
        owner is not source
        and owner is not runtime._candidate
        and owner is not runtime.actor._learner
        for owner in owners
    )
    assert sum(row["result"]["work"]["prediction_rows"] for row in checkpoints) == 20
    assert sum(row["result"]["work"]["distance_candidates"] for row in checkpoints) == 100
    assert sum(row["result"]["work"]["distance_feature_values"] for row in checkpoints) == 200
    json.dumps(report, allow_nan=False)


def test_should_account_for_appends_observers_validation_and_bounded_memory(completed):
    report, _ = completed
    assert report["work"] == {
        "fit_attempts": 1,
        "fit_returns": 1,
        "fit_rows": 2,
        "model_initializations": 1,
        "model_initialization_attempts": 1,
        "model_random_draws": 0,
        "warm_append_attempts": 1,
        "warm_append_returns": 1,
        "warm_append_rows": 2,
        "runtime_append_attempts": 3,
        "runtime_append_returns": 3,
        "runtime_append_rows": 6,
        "validated_updates": 3,
        "observer_attempts": 7,
        "observer_returns": 7,
        "observer_validated": 7,
        "optimizer_updates": 0,
        "replayed_training_rows": 0,
        "validation_normalization_rows": 6,
    }
    assert report["memory"]["source_prototype_bytes"] == 48
    assert report["memory"]["actor_prototype_bytes"] == 48
    assert report["memory"]["candidate_prototype_bytes"] == 192
    assert report["memory"]["readout_prototype_bytes_max"] == 192
    assert report["memory"]["retained_episode_rows"] == 8
    assert report["memory"]["checkpoint_records"] == 7
    assert report["experiment_state"] == {"completed_updates": 3, "next_tick": 9}
    assert report["model_state"] == {
        "source_publications": 1,
        "candidate_publications": 4,
        "candidate_prototype_rows": 8,
        "durable_checkpoint": False,
    }


@pytest.mark.parametrize(
    "bad",
    ["list", "too_many", "foreign", "probe", "final", "duplicate", "spec", "capacity", "observer"],
)
def test_should_refuse_all_future_packet_admission_before_fit(monkeypatch, bad):
    warm, new, additional = inputs()
    observer: Any = None
    if bad == "list":
        additional = list(additional)
    elif bad == "too_many":
        additional = additional + (episode("fourth"),)
    elif bad == "foreign":
        additional = (object(),)
    elif bad == "probe":
        additional = (episode("third", role="probe_train"),)
    elif bad == "final":
        additional = (episode("third", role="final_test"),)
    elif bad == "duplicate":
        additional = (new,)
    elif bad == "spec":
        additional = (episode("third", spec=ContinuousActionSpec(2, ("N",), (-1.0,), (1.0,))),)
    elif bad == "capacity":
        warm, new, additional = (
            episode("warm", rows=128),
            episode("new", rows=128),
            (episode("third", rows=1),),
        )
    else:
        observer = object()

    def forbidden(*args, **kwargs):
        pytest.fail("unadmitted future packet reached fit/model/runtime")

    monkeypatch.setattr(sequence, "fit_training_standardization", forbidden)
    monkeypatch.setattr(sequence, "NumpyNearestContinuousBCLearner", forbidden)
    monkeypatch.setattr(sequence, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_demo(
            warm, new, additional_episodes=additional, checkpoint_observer=observer
        )


@pytest.mark.parametrize("kind", ["not_dict", "nonfinite", "oversized"])
def test_should_refuse_invalid_observer_result_before_new_labels(kind):
    warm, new, additional = inputs()

    def observer(readout, metadata):
        if kind == "not_dict":
            return []
        return {"bad": float("nan")} if kind == "nonfinite" else {"bad": "x" * (256 * 1024)}

    report = demo.run_continuous_demo(
        warm, new, additional_episodes=additional, checkpoint_observer=observer
    )
    assert report["complete"] is False
    assert report["applied"] == [] and report["events"] == []
    assert "error" in report
    assert report["work"]["observer_attempts"] == report["work"]["observer_returns"] == 1
    assert report["work"]["observer_validated"] == 0


def test_should_retain_paid_callback_failure_and_stop_before_later_release():
    warm, new, additional = inputs()
    seen: list[dict[str, Any]] = []

    def observer(readout, metadata):
        seen.append(deepcopy(metadata))
        if metadata["phase"] == "after_release":
            raise RuntimeError("synthetic paid observer failure")
        return {"observed": metadata["phase"]}

    report = demo.run_continuous_demo(
        warm, new, additional_episodes=additional, checkpoint_observer=observer
    )
    assert (
        report["complete"] is False
        and "synthetic paid observer failure" in report["error"]["message"]
    )
    assert len(report["applied"]) == 1 and len(report["events"]) == 3
    assert [row["phase"] for row in seen] == ["warm", "before_release", "after_release"]
    assert report["applied"][0]["arrived_at"] == 2


def test_should_retain_returned_observer_report_but_refuse_changed_detached_learner():
    warm, new, additional = inputs()

    def observer(readout, metadata):
        readout._state.prototype_actions[0, 0] = 0.75
        return {"paid_observation": "returned before state validation"}

    report = demo.run_continuous_demo(
        warm, new, additional_episodes=additional, checkpoint_observer=observer
    )
    assert report["complete"] is False and report["applied"] == []
    assert report["checkpoints"][0]["result"] == {
        "paid_observation": "returned before state validation"
    }
    assert report["events"] == []


def test_should_refuse_changed_preceding_receipt_before_the_next_packet(monkeypatch):
    warm, new, additional = inputs()
    captured = []
    factory = sequence.ActorShadowRuntime

    def runtime(source, **options):
        owner = factory(source, **options)
        captured.append((owner, options))
        return owner

    def observer(readout, metadata):
        if metadata["phase"] == "after_release":
            owner = captured[0][0]
            key, receipt = next(iter(owner._inbox._applied.items()))
            owner._inbox._applied[key] = replace(receipt, arrived_at=3)
        return {"returned": True}

    monkeypatch.setattr(sequence, "ActorShadowRuntime", runtime)
    report = demo.run_continuous_demo(
        warm, new, additional_episodes=additional, checkpoint_observer=observer
    )
    assert report["complete"] is False
    assert captured[0][1]["budget"].updates_completed == 1
    assert len(report["events"]) == 3 and len(report["applied"]) == 1
    assert report["checkpoints"][-1]["result"] == {"returned": True}


def test_should_refuse_empty_final_poll_mutation_after_three_paid_appends(monkeypatch):
    warm, new, additional = inputs()
    captured = []
    factory = sequence.ActorShadowRuntime

    def runtime(source, **options):
        owner = factory(source, **options)
        captured.append((owner, options))
        ready = owner.train_ready

        def poll(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if not applied and options["budget"].updates_completed == 3:
                candidate = owner._candidate
                assert type(candidate) is NumpyNearestContinuousBCLearner
                candidate._state.prototype_actions[-1, 0] = 0.99
            return applied

        monkeypatch.setattr(owner, "train_ready", poll)
        return owner

    monkeypatch.setattr(sequence, "ActorShadowRuntime", runtime)
    report = demo.run_continuous_demo(warm, new, additional_episodes=additional)
    assert report["complete"] is False
    assert len(report["applied"]) == 3
    assert captured[0][1]["budget"].updates_completed == 3


def test_should_retain_native_paid_receipt_when_update_throws_after_commit(monkeypatch):
    warm, new, additional = inputs()
    captured = []
    factory = sequence.ActorShadowRuntime

    def runtime(source, **options):
        owner = factory(source, **options)
        captured.append((owner, options))
        ready = owner.train_ready

        def update(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if applied:
                raise RuntimeError("synthetic native postcommit failure")
            return applied

        monkeypatch.setattr(owner, "train_ready", update)
        return owner

    monkeypatch.setattr(sequence, "ActorShadowRuntime", runtime)
    report = demo.run_continuous_demo(warm, new, additional_episodes=additional)
    assert report["complete"] is False
    assert "synthetic native postcommit failure" in report["error"]["message"]
    assert report["experiment_state"]["completed_updates"] == 1
    assert captured[0][1]["budget"].updates_completed == 1
    assert report["applied"] == []
    observed = report["retained_applied_on_failure"]
    assert len(observed) == 1
    assert observed[0]["episode_id"] == "new"
    assert observed[0]["arrived_at"] == 2
    assert observed[0]["diagnostic"] == {"definition": OBJECTIVE, "value": 2}
    assert report["work"]["runtime_append_attempts"] == 1
    assert report["work"]["runtime_append_returns"] == 0
    assert len(report["events"]) == 2


def test_should_retain_failed_setup_fit_attempt_without_constructing_model(monkeypatch):
    warm, new, additional = inputs()

    def failed_fit(record):
        raise RuntimeError("synthetic frozen fit failure")

    monkeypatch.setattr(sequence, "fit_training_standardization", failed_fit)
    monkeypatch.setattr(
        sequence,
        "NumpyNearestContinuousBCLearner",
        lambda *args: pytest.fail("failed fit constructed learner"),
    )
    report = demo.run_continuous_demo(warm, new, additional_episodes=additional)
    assert report["complete"] is False
    assert report["error"]["message"] == "synthetic frozen fit failure"
    assert report["work"]["fit_attempts"] == 1 and report["work"]["fit_returns"] == 0
    assert report["work"]["model_initializations"] == 0
    assert report["events"] == report["checkpoints"] == report["applied"] == []


@pytest.mark.parametrize("explicit", [False, True])
def test_should_keep_default_empty_none_on_legacy_route_without_new_work(monkeypatch, explicit):
    def legacy(*args):
        raise RuntimeError("legacy admission reached")

    monkeypatch.setattr(demo, "_training_episodes", legacy)
    options: dict[str, Any] = (
        {"additional_episodes": (), "checkpoint_observer": None} if explicit else {}
    )
    invalid: Any = object()
    with pytest.raises(RuntimeError, match="legacy admission reached"):
        demo.run_continuous_demo(invalid, invalid, **options)
