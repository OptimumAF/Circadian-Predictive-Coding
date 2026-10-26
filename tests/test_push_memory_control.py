"""Five delayed fixed-input releases and scalar memory-control routing.

One full198 sequence plus one three-row paid callback failure. Stream callbacks
make no predictions. Driver tests use controlled owners and stage doubles only.
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.adapters import continuous_stream_sequence as sequence
from src.adapters import numpy_nearest_continuous_bc as nearest
from src.adapters import numpy_continuous_bc as bp
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition


def episode(name, count):
    observations = tuple((float(index),) + (0.0,) * 11 for index in range(count))
    return ContinuousEpisode(
        name,
        "TRAIN-" + name,
        "train",
        cube_taskspace_spec(),
        tuple(
            ContinuousTransition(
                index,
                observation,
                (0.0, 0.0, 0.0, -1.0),
                0.0,
                observations[min(index + 1, count - 1)],
                False,
                index == count - 1,
            )
            for index, observation in enumerate(observations)
        ),
    )


def records(*, tiny=False):
    return tuple(
        episode(name, 1 if tiny else count)
        for name, count in zip(
            ("a", "b", "train20", "push", "corrections", "current"),
            (15, 50, 15, 18, 50, 50),
            strict=True,
        )
    )


def forbidden(*args, **options):
    pytest.fail("memory control fixture admitted neural arithmetic, fitting or prediction")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts: defaultdict[str, int] = defaultdict(int)
    initialize, append = (
        nearest.NumpyNearestContinuousBCLearner.__init__,
        nearest.NumpyNearestContinuousBCLearner.train_batch,
    )

    def initialized(owner, policy):
        counts["nearest_initializations"] += 1
        return initialize(owner, policy)

    def appended(owner, features, targets):
        counts["append_attempts"] += 1
        counts["append_rows"] += len(features)
        result = append(owner, features, targets)
        counts["append_returns"] += 1
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "__init__", initialized)
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "train_batch", appended)
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "predict_with_work", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", forbidden)
        patch.setattr(sequence, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("PUSH_MEMORY_CONTROL_LEDGER"):
        with Path(os.environ["PUSH_MEMORY_CONTROL_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(dict(counts), output, sort_keys=True)
            output.write("\n")
    assert counts["nearest_initializations"] == 2
    assert counts["append_attempts"] == counts["append_returns"] == 9
    assert counts["append_rows"] == 201
    assert counts["driver_sequence_calls"] == 22
    assert counts["driver_push_calls"] == 26 and counts["driver_pick_calls"] == 42


@pytest.fixture(scope="module")
def full_sequence(ledger):
    data = records()
    before = tuple(asdict(item) for item in data)
    captured: dict[str, Any] = dict(readouts=[])
    factory = sequence.ActorShadowRuntime

    def runtime(source, **options):
        value = factory(source, **options)
        captured.update(source=source, runtime=value, budget=options["budget"])
        return value

    def observe(readout, metadata):
        captured["readouts"].append((readout, deepcopy(metadata), readout.snapshot_state()))
        metadata["tick"] = -99
        return dict(scope="metadata only; no predictions")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sequence, "ActorShadowRuntime", runtime)
        result = sequence.run_sequence(
            data[0], data[1:], observe, fixed_input=True, maximum_releases=5
        )
    assert tuple(asdict(item) for item in data) == before
    return result, captured, data


def test_should_append_all198_in_original_order_with_five_permanent_arrival_charges(full_sequence):
    report, captured, data = full_sequence
    assert report["complete"] is True, report
    assert report["release_scope"] == dict(
        maximum_releases=5,
        default_maximum_releases=3,
        scope="explicit fixed-input five-release sequence; no prior budget renewal",
    )
    assert [row["arrived_at"] for row in report["applied"]] == [2, 5, 8, 11, 14]
    assert [row["observed_at"] for row in report["applied"]] == [0, 3, 6, 9, 12]
    assert [row["episode_id"] for row in report["applied"]] == [
        item.episode_id for item in data[1:]
    ]
    assert len(report["events"]) == 15 and len(report["checkpoints"]) == 11
    assert report["model_state"]["source_publications"] == 1
    assert report["model_state"]["candidate_publications"] == 6
    assert report["model_state"]["candidate_prototype_rows"] == 198
    assert report["experiment_state"] == dict(completed_updates=5, next_tick=15)
    assert (
        captured["budget"].updates_completed == 5
        and captured["budget"].budget.max_training_updates == 5
    )
    assert report["work"]["validated_updates"] == 5 and report["work"]["runtime_append_rows"] == 183
    assert report["work"]["fit_attempts"] == report["work"]["model_random_draws"] == 0
    candidate = captured["runtime"].candidate_snapshot().state
    assert candidate.prototype_features.tolist() == [
        list(row.observation) for item in data for row in item.transitions
    ]
    assert candidate.prototype_actions.tolist() == [
        list(row.action) for item in data for row in item.transitions
    ]


def test_should_keep_warm_source_actor_and_all11_detached_checkpoint_snapshots(full_sequence):
    report, captured, _ = full_sequence
    warm = captured["source"].snapshot_state()
    actor = captured["runtime"].actor.snapshot_state().state
    assert warm.training_examples == actor.training_examples == 15
    np.testing.assert_array_equal(warm.prototype_features, actor.prototype_features)
    observations = captured["readouts"]
    assert len({id(item[0]) for item in observations}) == 11
    assert [item[1]["prototype_rows"] for item in observations] == [
        15,
        15,
        65,
        65,
        80,
        80,
        98,
        98,
        148,
        148,
        198,
    ]
    assert all(row["tick"] >= 0 for row in report["checkpoints"])
    for owner, metadata, initial in observations:
        np.testing.assert_array_equal(owner._state.prototype_features, initial.prototype_features)
        assert owner._state.training_examples == metadata["prototype_rows"]
        assert not np.shares_memory(
            owner._state.prototype_features, captured["source"]._state.prototype_features
        )
    assert (
        report["memory"]["source_prototype_bytes"]
        == report["memory"]["actor_prototype_bytes"]
        == 1920
    )
    assert report["memory"]["candidate_prototype_bytes"] == 25344
    assert "candidate" not in report and "learner" not in report


def test_should_retain_two_paid_updates_and_callback_result_then_stop_on_readout_corruption():
    data = records(tiny=True)

    def observe(readout, metadata):
        if metadata["phase"] == "after_release" and metadata["packet_index"] == 1:
            readout._state.prototype_actions[0, 0] = 0.125
        return dict(phase=metadata["phase"], index=metadata["packet_index"])

    report = sequence.run_sequence(data[0], data[1:], observe, fixed_input=True, maximum_releases=5)
    assert report["complete"] is False and len(report["applied"]) == 2
    assert report["experiment_state"]["completed_updates"] == 2
    assert report["work"]["runtime_append_returns"] == 2
    assert len(report["checkpoints"]) == 5
    assert report["checkpoints"][-1]["result"] == dict(phase="after_release", index=1)
    assert report["observed_native_on_failure"]["candidate"]["training_examples"] == 3


# Six is an explicit fixed-input scope; seven remains outside the bounded API.
@pytest.mark.parametrize(
    "maximum,fixed", [(True, True), (None, True), (3.0, True), (4, True), (7, True), (5, False)]
)
def test_should_refuse_invalid_or_nonfixed_release_scope_before_work(monkeypatch, maximum, fixed):
    monkeypatch.setattr(sequence, "NumpyNearestContinuousBCLearner", forbidden)
    data = records()
    with pytest.raises(ValueError):
        sequence.run_sequence(data[0], data[1:], fixed_input=fixed, maximum_releases=maximum)


@pytest.mark.parametrize("fault", ["default_three", "role", "duplicate", "capacity"])
def test_should_preserve_original_scope_limits_roles_ids_and_capacity_before_constructor(
    monkeypatch, fault
):
    data = records()
    options: dict[str, Any] = dict(fixed_input=True, maximum_releases=5)
    if fault == "default_three":
        options.pop("maximum_releases")
    elif fault == "role":
        from dataclasses import replace

        data = data[:-1] + (replace(data[-1], role="probe_test"),)
    elif fault == "duplicate":
        from dataclasses import replace

        data = data[:-1] + (replace(data[-1], episode_id=data[0].episode_id),)
    else:
        data = tuple(episode(str(index), 50) for index in range(6))
    monkeypatch.setattr(sequence, "NumpyNearestContinuousBCLearner", forbidden)
    with pytest.raises(ValueError):
        sequence.run_sequence(data[0], data[1:], **options)


@pytest.mark.parametrize("explicit", [False, True])
def test_should_preserve_default_three_release_admission_callshape(monkeypatch, explicit):
    calls: list[Any] = []

    def admit(*args, **options):
        calls.append((args, options))
        raise ValueError("preadmission sentinel")

    data = records(tiny=True)
    monkeypatch.setattr(sequence, "_episodes", admit)
    options: dict[str, Any] = {"maximum_releases": 3} if explicit else {}
    with pytest.raises(ValueError, match="preadmission sentinel"):
        sequence.run_sequence(data[0], data[1:2], fixed_input=True, **options)
    assert calls == [((data[0], data[1:2]), {})]


def controlled_owner(updates, rows):
    # Exact native type with literal bank/counters; no constructor, append or query.
    owner = object.__new__(nearest.NumpyNearestContinuousBCLearner)
    owner._policy = nearest.FixedInputNearestContinuousBCPolicy(cube_taskspace_spec())
    owner._state = nearest.NearestContinuousBCSnapshot(
        owner._policy,
        np.zeros((rows, 12), np.float64),
        np.tile(np.asarray([(0.0, 0.0, 0.0, -1.0)], np.float64), (rows, 1)),
        updates,
        rows,
    )
    return owner


def native_receipt(*, push, updates, rows, seed=17):
    value = dict(
        complete=True,
        source_updates=updates,
        source_training_rows=rows,
        final_success=False,
        any_success=False,
        sparse_return=0.0,
        prediction_attempts=1,
        prediction_returns=1,
        nearest_prediction_attempts=1,
        nearest_prediction_returns=1,
        nearest_distance_candidate_upper_bound=rows,
        nearest_distance_candidates_completed=rows,
        nearest_distance_feature_upper_bound=12 * rows,
        nearest_distance_feature_values_completed=12 * rows,
        nearest_selected_indices=[0],
        work=dict(completed_control_physics_steps=5, possible_failed_physics_steps=0),
    )
    if push:
        value.update(
            transitions=[dict(observation=[0.0] * 12)],
            source_unchanged=True,
            unreturned_prediction_possible=False,
            unreturned_native_work_unknown=False,
            unvalidated_nearest_prediction_work_unknown=False,
            control_attempts=1,
            control_returns=1,
            metadata=dict(
                task_id="PushCube-v1",
                seed=17,
                role="train",
                context="push-cube-train-17",
                recipe={},
                controller={},
                horizon=50,
            ),
        )
    else:
        value.update(
            episode=dict(transitions=[dict(observation=[0.0] * 42)]),
            seed=seed,
            role="train",
            task="PickCube-v1",
            context=f"pick-cube-train-{seed}",
            controller="pd_ee_delta_pos",
            observation_recipe="state42",
            action_history=dict(recipe={"mode": "executed"}),
            frozen_state_exact=True,
            original_owner_exact=True,
            action_history_exact=True,
            optimizer_updates=0,
        )
    return value


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "stream_failure",
        "stream_receipt",
        "checkpoint",
        "push_incomplete",
        "push_unknown",
        "work",
        "lost_push",
        "reset",
        "source_owner",
        "prior_raw",
        "saved",
        "corrections",
        "source_bytes",
        "original_bytes",
        "after_reset",
        "pick_counter",
        "holder",
        "clock",
        "prior_nan",
        "saved_tail",
        "stream_clock",
    ],
)
def test_should_route_two_fixed_memory_checkpoints_and_keep_paid_refusals_without_more_scenes(
    tmp_path, monkeypatch, ledger, fault
):
    from types import SimpleNamespace
    from scripts import push_cube_memory_control as driver

    original = records()
    packets = dict(
        zip(
            ("a", "b", "train20", "push", "corrections", "current_corrections"),
            original,
            strict=True,
        )
    )
    groups = tuple(
        (name, packets[name])
        for name in ("current_corrections", "corrections", "push", "a", "b", "train20")
    )
    baseline = dict(
        native_push=native_receipt(push=True, updates=128, rows=198),
        pick=[
            native_receipt(push=False, updates=128, rows=198, seed=seed) for seed in (17, 19, 20)
        ],
    )
    receipt = dict(
        original_correction=dict(path="synthetic-original", bytes=7), accepted_study=dict(bytes=11)
    )
    calls: list[tuple[str, str]] = []
    raw_returns: list[dict[str, Any]] = []
    holders: list[Any] = []
    mode: dict[str, Any] = {}
    stream_capture: dict[str, Any] = {}

    def bound(path, *, work):
        work["attempts"] += 1
        if mode.get("source_bad"):
            raise ValueError("accepted source bytes changed")
        raw = json.dumps(baseline).encode()
        work["returns"] += 1
        work["bytes"] += len(raw)
        return raw

    def original_unchanged(value, work):
        work["attempts"] += 1
        if mode.get("original_bad"):
            raise ValueError("original correction bytes changed")
        work["returns"] += 1
        work["bytes"] += 7

    def push(owner, *, label):
        ledger["driver_push_calls"] += 1
        calls.append((label, "push"))
        if fault == "lost_push":
            raise RuntimeError("native return unavailable")
        raw = native_receipt(
            push=True, updates=owner._state.completed_updates, rows=owner._state.training_examples
        )
        if fault == "push_incomplete":
            raw["complete"] = False
        elif fault == "push_unknown":
            raw["unvalidated_nearest_prediction_work_unknown"] = True
        elif fault == "work":
            raw["nearest_distance_candidates_completed"] -= 1
        elif fault == "reset" or fault == "after_reset" and label == "memory-after":
            raw["transitions"][0]["observation"][0] = 1e-15
        elif fault == "source_owner":
            object.__setattr__(
                owner._state, "prototype_features", owner._state.prototype_features.copy()
            )
        raw_returns.append(raw)
        return raw

    def pick(path, *, learner, train_seed, action_history_recipe):
        ledger["driver_pick_calls"] += 1
        calls.append(
            ("before" if learner._state.training_examples == 80 else "after", str(train_seed))
        )
        assert action_history_recipe.mode == "executed"
        raw = native_receipt(
            push=False,
            updates=learner._state.completed_updates,
            rows=learner._state.training_examples,
            seed=train_seed,
        )
        if fault == "pick_counter":
            raw["source_updates"] -= 1
        elif fault in ("prior_raw", "prior_nan"):
            raw_returns[0]["sparse_return"] = float("nan") if fault == "prior_nan" else 9.0
        elif fault == "source_bytes":
            mode["source_bad"] = True
        elif fault == "original_bytes":
            mode["original_bad"] = True
        elif fault == "holder" and len(holders) == 1:
            holders[0].report = deepcopy(holders[0].report)
        saved = dict(raw, complete=False) if fault == "saved" else raw
        with path.open("x", encoding="utf8") as handle:
            json.dump(saved, handle)
        raw_returns.append(raw)
        holder = SimpleNamespace(
            report=raw, corrections=object() if fault == "corrections" else None
        )
        holders.append(holder)
        return holder

    def run(warm, releases, *, observer, fixed_input, maximum_releases):
        ledger["driver_sequence_calls"] += 1
        assert warm is packets["a"]
        assert releases == tuple(
            packets[name] for name in ("b", "train20", "push", "corrections", "current_corrections")
        )
        assert fixed_input is True and maximum_releases == 5
        value: dict[str, Any] = dict(
            complete=fault != "stream_failure",
            fixed_input=True,
            release_scope=dict(maximum_releases=5),
            experiment_state=dict(completed_updates=5, next_tick=15),
            model_state=dict(
                source_publications=1,
                candidate_publications=6,
                candidate_prototype_rows=198,
                durable_checkpoint=False,
            ),
            checkpoints=[],
            applied=[
                dict(
                    observed_at=3 * i,
                    arrived_at=3 * i + 2,
                    applied_at=3 * i + 2,
                    update_number=i + 1,
                )
                for i in range(5)
            ],
            work=dict(validated_updates=5, fit_attempts=0),
        )
        stream_capture["report"] = value
        if fault == "stream_failure":
            return value
        try:
            population = 15
            for position in range(11):
                phase = (
                    "warm"
                    if position == 0
                    else "before_release"
                    if position % 2
                    else "after_release"
                )
                packet_index = None if position == 0 else (position - 1) // 2
                if phase == "after_release":
                    population += len(releases[packet_index].transitions)
                updates = 1
                if packet_index is not None:
                    updates = packet_index + (2 if phase == "after_release" else 1)
                metadata = dict(
                    phase=phase,
                    packet_index=packet_index,
                    prototype_rows=population,
                    completed_updates=updates,
                )
                if fault == "checkpoint" and phase == "after_release" and packet_index == 1:
                    metadata["prototype_rows"] = 79
                result = observer(controlled_owner(updates, population), metadata)
                assert len(json.dumps(result).encode()) < 256 * 1024 and "scenes" not in result
                value["checkpoints"].append(dict(metadata, result=result))
            if fault == "stream_receipt":
                value["applied"][3]["arrived_at"] += 1
        except Exception as error:
            value["complete"] = False
            value["error"] = str(error)
        return value

    monkeypatch.setattr(driver, "read_aggregated_inputs", lambda path: (groups, receipt))
    monkeypatch.setattr(driver, "_bound_bytes", bound)
    monkeypatch.setattr(driver, "_require_original_unchanged", original_unchanged)
    monkeypatch.setattr(driver, "run_sequence", run)
    monkeypatch.setattr(driver, "roll_push", push)
    monkeypatch.setattr(driver, "run_frozen_rollout", pick)
    times = iter((10.0, 11.0))
    read_saved = driver._read_saved

    def saved_read(path, work):
        value = read_saved(path, work)
        if fault == "saved_tail" and mode.get("final_clock"):
            raw_returns[0]["sparse_return"] = 9.0
        return value

    monkeypatch.setattr(driver, "_read_saved", saved_read)

    def clock():
        value = next(times)
        if value == 11.0:
            mode["final_clock"] = True
        if fault == "clock" and value == 11.0:
            raw_returns[0]["sparse_return"] = 9.0
        if fault == "stream_clock" and value == 11.0:
            stream_capture["report"]["applied"][0]["arrived_at"] = 999
        return value

    monkeypatch.setattr(driver, "monotonic", clock)
    output = tmp_path / "run.json"
    result = driver.run_control(output, source=tmp_path / "accepted.json")
    assert json.loads(output.read_text()) == json.loads(json.dumps(result))
    assert result["complete"] is (fault is None), result.get("error")
    assert result["stream"] is stream_capture["report"]
    if fault is None:
        assert len(calls) == 8 and len(result["readouts"]) == 2
        assert [item["label"] for item in result["readouts"]] == ["before", "after"]
        assert [item["metadata"]["prototype_rows"] for item in result["readouts"]] == [80, 198]
        assert tuple(result["comparisons"]["acquisition_retention"]) == driver.SCENES
        for value in result["comparisons"]["acquisition_retention"].values():
            assert value["final_success"] == value["any_success"] == value["sparse_return"] == 0
        assert result["source_bytes_unchanged"] is True
    else:
        assert result["error"]
        expected_calls = (
            0
            if fault in ("stream_failure", "checkpoint")
            else 8
            if fault in ("stream_receipt", "clock", "saved_tail", "stream_clock")
            else 5
            if fault == "after_reset"
            else 3
            if fault == "holder"
            else 2
            if fault
            in (
                "prior_raw",
                "prior_nan",
                "saved",
                "corrections",
                "source_bytes",
                "original_bytes",
                "pick_counter",
            )
            else 1
        )
        assert len(calls) == expected_calls
        if fault == "lost_push":
            assert result["unreturned_stage_work_unknown"] is True
            assert result["readouts"][0]["scenes"] == {}
        elif fault == "source_owner":
            assert result["pending_paid_return"]["stage"] == "before-push17"
            assert result["pending_paid_return"]["report"]["source_training_rows"] == 80
            assert result["readouts"][0]["scenes"] == {}
        elif calls:
            assert result["readouts"][0]["scenes"]["push17"]["source_training_rows"] == 80
        if fault in ("prior_raw", "prior_nan"):
            assert result["readouts"][0]["scenes"]["push17"]["sparse_return"] == 0.0
            assert result["pending_paid_return"]["stage"] == "before-pick17"
            assert result["rejected_prior_evidence"]
        elif fault == "holder":
            assert result["pending_paid_return"]["stage"] == "before-pick19"
        elif fault in ("clock", "saved_tail"):
            assert result["readouts"][0]["scenes"]["push17"]["sparse_return"] == 0.0
        elif fault == "stream_clock":
            assert result["stream"]["applied"][0]["arrived_at"] == 2
            assert result["rejected_stream_evidence"]["applied"][0]["arrived_at"] == 999


@pytest.mark.parametrize("location", ["main", "before", "after"])
def test_should_refuse_any_existing_scene_path_before_input_or_stream_work(
    tmp_path, monkeypatch, location
):
    from scripts import push_cube_memory_control as driver

    output = tmp_path / "run.json"
    occupied = output if location == "main" else tmp_path / f"run-{location}-pick-19.json"
    occupied.write_text("preserve")
    monkeypatch.setattr(driver, "read_aggregated_inputs", forbidden)
    monkeypatch.setattr(driver, "run_sequence", forbidden)
    with pytest.raises(ValueError):
        driver.run_control(output)
    assert occupied.read_text() == "preserve"


@pytest.mark.parametrize("complete", [False, True])
def test_should_route_memory_cli_without_implicit_new_study_work(
    tmp_path, monkeypatch, capsys, complete
):
    import sys
    from scripts import push_cube_memory_control as driver

    calls: list[Path] = []
    output = tmp_path / "run.json"

    def run(path):
        calls.append(path)
        return dict(complete=complete, error=None if complete else "partial", elapsed_seconds=1.0)

    monkeypatch.setattr(driver, "run_control", run)
    monkeypatch.setattr(sys, "argv", ["memory", "--output", str(output)])
    assert driver.main() == (0 if complete else 1)
    assert calls == [output] and json.loads(capsys.readouterr().out)["complete"] is complete
