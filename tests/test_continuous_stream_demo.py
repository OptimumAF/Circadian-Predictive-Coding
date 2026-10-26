"""Released continuous labels through the existing runtime, without native simulation."""

from dataclasses import asdict, replace
import json
import sys

import numpy as np
import pytest

from scripts import run_native_stream_demo as cli
from src.adapters import continuous_stream_demo as demo
from src.adapters.numpy_nearest_continuous_bc import (
    NearestContinuousBCPolicy,
    NumpyNearestContinuousBCLearner,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.observation_standardization import ObservationStandardization
from src.infra.maniskill_continuous import ACTION_UNITS


def episode(name="warm", *, role="train", rows=2, declared=None):
    declared = ContinuousActionSpec(2, ("m",), (-1.0,), (1.0,)) if declared is None else declared
    start = 0 if name == "warm" else 10
    observations = [
        ((float(start + index),) + (0.0,) * (declared.observation_dim - 1))
        for index in range(rows + 1)
    ]
    return ContinuousEpisode(
        name,
        name + "-context",
        role,
        declared,
        tuple(
            ContinuousTransition(
                index,
                observations[index],
                (
                    (-1.0 + index * 0.5)
                    if name == "warm" and index < 2
                    else 1.0
                    if index % 2 == 0
                    else 0.5,
                )
                * len(declared.action_units),
                float(index == rows - 1),
                observations[index + 1],
                index == rows - 1,
                False,
            )
            for index in range(rows)
        ),
    )


def assert_same_state(first, second):
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    np.testing.assert_array_equal(first.prototype_features, second.prototype_features)
    np.testing.assert_array_equal(first.prototype_actions, second.prototype_actions)


def test_should_apply_one_complete_packet_only_at_tick_two_and_keep_actor_warm():
    warm, new = episode(), episode("new")
    before_warm, before_new = repr(warm), repr(new)
    report = demo.run_continuous_demo(warm, new)
    assert report["complete"] is True and report["schema_version"] == 1
    assert report["protocol_id"] == "continuous_nearest_delayed_stream_v1"
    assert report["scope"] == "saved_train_integration"
    assert repr(warm) == before_warm and repr(new) == before_new
    assert all(report["checks"].values())
    assert report["experiment_state"] == dict(format_version=1, completed_updates=1, next_tick=3)
    assert report["model_state"]["snapshot_type"] == "NearestContinuousBCSnapshot"
    assert report["model_state"]["durable_checkpoint"] is False
    assert "snapshot_format_version" not in report["model_state"]
    assert report["model_state"]["source_publications"] == 1
    assert report["model_state"]["candidate_publications"] == 2
    events = report["events"]
    assert [event["tick"] for event in events] == [0, 1, 2]
    assert [event["label_available"] for event in events] == [False, False, True]
    assert [len(event["applied"]) for event in events] == [0, 0, 1]
    assert [event["candidate_prototype_rows"] for event in events] == [2, 2, 4]
    assert [event["candidate_publications"] for event in events] == [1, 1, 2]
    assert [event["budget_updates"] for event in events] == [0, 0, 1]
    assert all(event["actor_predictions"] == [[-0.5], [-0.5]] for event in events)
    assert report["readouts"]["actor_after_new"] == [[-0.5], [-0.5]]
    assert report["readouts"]["candidate_warm"] == [[-1.0], [-0.5]]
    assert report["readouts"]["candidate_new"] == [[1.0], [0.5]]
    assert report["readouts"]["candidate_warm_indices"] == [0, 1]
    assert report["readouts"]["candidate_new_indices"] == [2, 3]
    (applied,) = report["applied"]
    assert applied["sample_id"] == "new-episode-pairs" and applied["episode_id"] == "new"
    assert applied["observed_at"] == 0 and applied["arrived_at"] == applied["applied_at"] == 2
    assert applied["diagnostic"] == dict(definition="continuous_nearest_bc_added_pairs_v1", value=2)
    decoded = json.loads(json.dumps(report, allow_nan=False))
    assert decoded["events"] == report["events"]
    assert decoded["policy"]["spec"]["action_units"] == ["m"]


def test_should_account_fit_append_serving_readout_and_retained_memory_separately():
    report = demo.run_continuous_demo(episode(), episode("new"))
    work, memory = report["work"], report["memory"]
    for name, expected in dict(
        fit_attempts=1,
        fit_returns=1,
        fit_rows=2,
        fit_feature_values=4,
        fit_numeric_bytes=32,
        warm_append_attempts=1,
        warm_append_returns=1,
        warm_append_rows=2,
        runtime_append_returns=1,
        runtime_append_rows=2,
        source_rows_ingested=4,
        replayed_training_rows=0,
        optimizer_updates=0,
        actor_prediction_calls=4,
        actor_prediction_rows=8,
        actor_prediction_bytes=64,
        actor_distance_candidates=16,
        actor_distance_feature_values=32,
        candidate_prediction_calls=2,
        candidate_prediction_rows=4,
        candidate_prediction_bytes=32,
        candidate_distance_candidates=16,
        candidate_distance_feature_values=32,
        normalization_calls=8,
        normalization_rows=16,
    ).items():
        assert work[name] == expected
    assert (
        work["actor_distance_count_source"]
        == "derived_from_exact_unchanged_typed_actor_and_query_shapes"
    )
    for name, expected in dict(
        source_prototype_bytes=48,
        actor_prototype_bytes=48,
        candidate_prototype_bytes=96,
        readout_prototype_bytes=96,
        prototype_growth_bytes=48,
        statistics_bytes_per_owner=32,
        inbox_feature_target_bytes=48,
    ).items():
        assert memory[name] == expected
    assert all(value >= 0 for value in report["timings"].values())
    assert report["settings"]["max_experiences"] == report["settings"]["max_training_updates"] == 1
    assert report["settings"]["simulator_calls"] == report["settings"]["promotion_calls"] == 0


@pytest.mark.parametrize(
    "bad",
    ["warm_probe", "new_probe", "final", "spec", "identity", "capacity", "corrupt", "foreign"],
)
def test_should_refuse_invalid_source_episodes_before_fit_or_runtime(monkeypatch, bad):
    warm, new = episode(), episode("new")
    if bad == "warm_probe":
        warm = episode(role="probe_train")
    elif bad == "new_probe":
        new = episode("new", role="probe_test")
    elif bad == "final":
        new = episode("new", role="final_test")
    elif bad == "spec":
        new = episode("new", declared=replace(new.spec, action_units=("N",)))
    elif bad == "identity":
        new = episode()
    elif bad == "capacity":
        warm, new = episode(rows=129), episode("new", rows=128)
    elif bad == "corrupt":
        object.__setattr__(new.transitions[0], "next_observation", (999.0, 0.0))
    else:
        new = object()
    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_: pytest.fail("invalid source reached fitting"),
    )
    monkeypatch.setattr(
        demo,
        "ActorShadowRuntime",
        lambda *_args, **_kwargs: pytest.fail("invalid source reached runtime creation"),
    )
    with pytest.raises(ValueError):
        demo.run_continuous_demo(warm, new)


def setup_runtime(*, capacity=256, identities=1, updates=1):
    declared = episode().spec
    source = NumpyNearestContinuousBCLearner(
        NearestContinuousBCPolicy(
            declared,
            ObservationStandardization((0.0, 0.0), (1.0, 1.0), 2),
            capacity,
        )
    )
    source.train_episode(episode())
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=updates), lambda: 0.0)
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=identities,
        max_consolidations=0,
    )
    return source, runtime, clock, budget


def packet(sample="batch", *, arrived_at=2):
    features = np.array([[10.0, 0.0], [11.0, 0.0]])
    targets = np.array([[1.0], [0.5]])
    return (
        Experience(
            sample, "new", 0, "actor-0", features, "train", ExperiencePermissions(training=True)
        ),
        LabelArrival("label-" + sample, sample, "new", arrived_at, "actor-0", targets),
    )


@pytest.mark.parametrize("future_label", [False, True])
def test_should_not_append_or_charge_withheld_or_future_labels(future_label):
    source, runtime, clock, budget = setup_runtime()
    initial = source.snapshot_state()
    experience, label = packet()
    runtime.record_experience(experience)
    if future_label:
        runtime.record_label(label)
    for tick in (0, 1):
        clock.advance_to(tick)
        assert runtime.train_ready() == () and budget.updates_completed == 0
        assert_same_state(initial, runtime.candidate_snapshot().state)
    clock.advance_to(2)
    if not future_label:
        assert runtime.train_ready() == () and budget.updates_completed == 0
        runtime.record_label(label)
    assert len(runtime.train_ready()) == 1 and budget.updates_completed == 1
    assert runtime.train_ready() == () and budget.updates_completed == 1
    assert runtime.candidate_snapshot().state.training_examples == 4
    np.testing.assert_array_equal(
        runtime.actor.predict(experience.features).prediction, [[-0.5], [-0.5]]
    )
    assert_same_state(initial, source.snapshot_state())


def test_should_preserve_lifetime_identity_and_duplicate_limits_without_extra_charges():
    _, runtime, clock, budget = setup_runtime()
    experience, label = packet()
    runtime.record_experience(experience)
    runtime.record_label(label)
    with pytest.raises(ValueError, match="duplicate"):
        runtime.record_experience(experience)
    with pytest.raises(ValueError, match="duplicate"):
        runtime.record_label(label)
    with pytest.raises(ValueError, match="capacity"):
        runtime.record_experience(packet("other")[0])
    assert budget.updates_completed == 0
    clock.advance_to(2)
    assert len(runtime.train_ready()) == 1
    with pytest.raises(ValueError, match="capacity"):
        runtime.record_experience(packet("other")[0])
    assert budget.updates_completed == 1


def test_should_refuse_another_append_at_budget_before_calling_learner(monkeypatch):
    _, runtime, clock, budget = setup_runtime(identities=2)
    first, label = packet()
    runtime.record_experience(first)
    runtime.record_label(label)
    clock.advance_to(2)
    assert len(runtime.train_ready()) == 1
    before = runtime.candidate_snapshot().state
    second, second_label = packet("second")
    runtime.record_experience(second)
    runtime.record_label(second_label)
    monkeypatch.setattr(
        NumpyNearestContinuousBCLearner,
        "train_batch",
        lambda *_: pytest.fail("spent budget admitted another append"),
    )
    with pytest.raises(ToyExecutionStopped) as caught:
        runtime.train_ready()
    assert caught.value.stop.reason == "max_training_updates"
    assert budget.updates_completed == 1 and len(runtime.applied_updates) == 1
    assert_same_state(before, runtime.candidate_snapshot().state)


@pytest.mark.parametrize("failure", ["capacity", "uncertain"])
def test_should_stop_failed_candidate_but_keep_warm_actor_available(monkeypatch, failure):
    source, runtime, clock, budget = setup_runtime(capacity=2 if failure == "capacity" else 256)
    before = source.snapshot_state()
    experience, label = packet()
    runtime.record_experience(experience)
    runtime.record_label(label)
    if failure == "uncertain":

        def broken(model, *_):
            model._state.prototype_actions[0, 0] = 0.25
            raise RuntimeError("uncertain append failure")

        monkeypatch.setattr(NumpyNearestContinuousBCLearner, "train_batch", broken)
    clock.advance_to(2)
    with pytest.raises((ValueError, RuntimeError)):
        runtime.train_ready()
    assert budget.updates_completed == 0 and runtime.applied_updates == ()
    with pytest.raises(ValueError, match="stopped"):
        runtime.train_ready()
    with pytest.raises(ValueError, match="stopped"):
        runtime.candidate_snapshot()
    np.testing.assert_array_equal(
        runtime.actor.predict(experience.features).prediction, [[-0.5], [-0.5]]
    )
    assert_same_state(before, source.snapshot_state())


def test_should_own_source_actor_candidate_snapshots_and_registered_array_payloads():
    source, runtime, clock, _ = setup_runtime()
    initial = source.snapshot_state()
    experience, label = packet()
    runtime.record_experience(experience)
    runtime.record_label(label)
    experience.features.fill(999.0)
    label.targets.fill(999.0)
    actor_state, candidate = runtime.actor.snapshot_state(), runtime.candidate_snapshot()
    actor_state.state.prototype_actions.fill(0.25)
    candidate.state.prototype_features.fill(999.0)
    source.train_batch(np.array([[5.0, 0.0]]), np.array([[0.0]]))
    assert_same_state(initial, runtime.actor.snapshot_state().state)
    assert_same_state(initial, runtime.candidate_snapshot().state)
    clock.advance_to(2)
    assert len(runtime.train_ready()) == 1
    final = runtime.candidate_snapshot().state
    np.testing.assert_array_equal(final.prototype_features[-2:], [[10.0, 0.0], [11.0, 0.0]])
    np.testing.assert_array_equal(final.prototype_actions[-2:], [[1.0], [0.5]])
    assert_same_state(initial, runtime.actor.snapshot_state().state)


def test_should_detect_candidate_mutation_during_post_drain_empty_poll(monkeypatch):
    original = ActorShadowRuntime.train_ready
    calls = []

    def corrupt_empty_poll(runtime, **kwargs):
        result = original(runtime, **kwargs)
        calls.append(result)
        if len(calls) == 4:
            assert result == ()
            runtime._candidate._state.prototype_actions[0, 0] += 0.01
        return result

    monkeypatch.setattr(ActorShadowRuntime, "train_ready", corrupt_empty_poll)
    with pytest.raises(RuntimeError, match="invariant"):
        demo.run_continuous_demo(episode(), episode("new"))
    assert len(calls) == 4


def teacher_receipt(seed=17, *, rows=2):
    context = "pick-cube" if seed == 17 else "pick-cube-train-19"
    declared = ContinuousActionSpec(42, ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    data = episode("warm" if seed == 17 else "new", rows=rows, declared=declared)
    data = replace(data, episode_id=f"pick-cube-task-teacher-{seed}", context=context)
    return dict(
        qualified=True,
        final_success=True,
        observed_success=True,
        error=None,
        task="PickCube-v1",
        controller="pd_ee_delta_pos",
        role="train",
        seed=seed,
        context=context,
        candidate_episode=asdict(data),
    )


@pytest.mark.parametrize("seed", [17, 19])
def test_should_read_bound_qualified_train_receipts_and_original_a_without_top_context(
    tmp_path, seed
):
    raw = teacher_receipt(seed)
    if seed == 17:
        del raw["context"]
    path = tmp_path / "teacher.json"
    path.write_text(json.dumps(raw), encoding="utf8")
    before = path.read_bytes()
    data = cli._read_teacher(path, seed)
    assert type(data) is ContinuousEpisode and data.role == "train"
    assert data.episode_id == f"pick-cube-task-teacher-{seed}"
    assert data.context == ("pick-cube" if seed == 17 else "pick-cube-train-19")
    assert data.spec.observation_dim == 42 and data.spec.action_units == ACTION_UNITS
    assert len(data.transitions) == 2 and data.transitions[-1].reward == 1.0
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "bad",
    [
        "unqualified",
        "not_final_success",
        "error",
        "role",
        "seed",
        "seed_type",
        "task",
        "controller",
        "top_context",
        "missing_b_context",
        "episode_role",
        "episode_context",
        "episode_id",
        "dimension",
        "units",
        "bounds",
        "horizon",
        "final_reward",
        "fractional_reward",
        "large_reward",
        "nonfinite",
        "seam",
        "terminal",
    ],
)
def test_should_refuse_unqualified_misbound_or_corrupt_teacher_receipts(tmp_path, bad):
    raw = teacher_receipt(19, rows=51 if bad == "horizon" else 2)
    data = raw["candidate_episode"]
    if bad in ("unqualified", "not_final_success"):
        raw["qualified" if bad == "unqualified" else "final_success"] = False
    elif bad == "error":
        raw["error"] = "native failure"
    elif bad == "role":
        raw["role"] = "final_test"
    elif bad == "seed":
        raw["seed"] = 23
    elif bad == "seed_type":
        raw["seed"] = 19.0
    elif bad == "task":
        raw["task"] = "Other-v1"
    elif bad == "controller":
        raw["controller"] = "pd_joint_pos"
    elif bad == "top_context":
        raw["context"] = "pick-cube"
    elif bad == "missing_b_context":
        del raw["context"]
    elif bad.startswith("episode_"):
        field, value = {
            "episode_role": ("role", "probe_train"),
            "episode_context": ("context", "pick-cube"),
            "episode_id": ("episode_id", "pick-cube-task-teacher-17"),
        }[bad]
        data[field] = value
    elif bad in ("dimension", "units", "bounds"):
        field, value = {
            "dimension": ("observation_dim", 43),
            "units": ("action_units", ["m"] * 4),
            "bounds": ("lower_bounds", [-2.0] * 4),
        }[bad]
        data["spec"][field] = value
    elif bad == "final_reward":
        data["transitions"][-1]["reward"] = 0.0
    elif bad in ("fractional_reward", "large_reward"):
        data["transitions"][0]["reward"] = 0.5 if bad == "fractional_reward" else 2.0
    elif bad == "nonfinite":
        data["transitions"][0]["observation"] = [float("nan")] * 42
    elif bad == "seam":
        data["transitions"][0]["next_observation"] = [999.0] * 42
    elif bad == "terminal":
        data["transitions"][-1]["terminated"] = False
    path = tmp_path / "bad-teacher.json"
    path.write_text(json.dumps(raw), encoding="utf8")
    with pytest.raises(ValueError):
        cli._read_teacher(path, 19)


def test_should_bound_teacher_file_read_and_reject_reserved_seed_before_open(tmp_path):
    oversized = tmp_path / "large.json"
    oversized.write_bytes(b" " * (1024**2 + 1))
    with pytest.raises(ValueError, match="1MiB"):
        cli._read_teacher(oversized, 17)
    with pytest.raises(ValueError, match="seed"):
        cli._read_teacher(tmp_path / "absent.json", 23)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--scenario", "continuous-nearest"],
        ["--scenario", "continuous-nearest", "--warm-teacher", "warm.json"],
        ["--scenario", "continuous-nearest", "--new-teacher", "new.json"],
        [
            "--scenario",
            "continuous-nearest",
            "--warm-teacher",
            "warm.json",
            "--new-teacher",
            "new.json",
            "--seed",
            "23",
        ],
        [
            "--scenario",
            "continuous-nearest",
            "--warm-teacher",
            "warm.json",
            "--new-teacher",
            "new.json",
            "--diagnose-units",
        ],
        ["--scenario", "stream", "--warm-teacher", "warm.json"],
        ["--scenario", "promotion-rejection", "--new-teacher", "new.json"],
        ["--scenario", "coupled-neutral", "--warm-teacher", "warm.json"],
        ["--scenario", "matched-baselines", "--new-teacher", "new.json"],
    ],
)
def test_should_refuse_incompatible_cli_options_before_reading_or_running(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["stream", *arguments])
    monkeypatch.setattr(
        cli, "_read_teacher", lambda *_: pytest.fail("invalid options reached receipt read")
    )
    monkeypatch.setattr(
        cli,
        "run_continuous_demo",
        lambda *_: pytest.fail("invalid options reached continuous demo"),
    )
    monkeypatch.setattr(
        cli, "run_demo", lambda *_: pytest.fail("invalid options reached legacy demo")
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_bind_cli_files_to_fixed_train_seeds_and_emit_only_json(
    monkeypatch, tmp_path, capsys
):
    warm, new = episode(), episode("new")
    paths = [tmp_path / "warm.json", tmp_path / "new.json"]
    calls = []

    def read(path, seed):
        calls.append((path, seed))
        return warm if seed == 17 else new

    def run(first, second):
        assert first is warm and second is new
        return dict(complete=True, boundary="synthetic-cli")

    monkeypatch.setattr(cli, "_read_teacher", read)
    monkeypatch.setattr(cli, "run_continuous_demo", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            "continuous-nearest",
            "--warm-teacher",
            str(paths[0]),
            "--new-teacher",
            str(paths[1]),
        ],
    )
    cli.main()
    captured = capsys.readouterr()
    assert captured.err == "" and json.loads(captured.out) == dict(
        complete=True, boundary="synthetic-cli"
    )
    assert calls == [(paths[0], 17), (paths[1], 19)]


def test_should_preserve_legacy_default_seed_without_requiring_teacher_files(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "run_demo", lambda seed: calls.append(seed) or dict(legacy=True))
    monkeypatch.setattr(
        cli, "_read_teacher", lambda *_: pytest.fail("legacy default read teacher files")
    )
    monkeypatch.setattr(sys, "argv", ["stream"])
    cli.main()
    assert calls == [23] and json.loads(capsys.readouterr().out) == dict(legacy=True)
