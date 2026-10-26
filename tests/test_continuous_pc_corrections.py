"""Arrived PC corrections and fake robot IO; no simulator or gain claims.

Prospective per-platform numerical budget: six warm + 24 shared correction +
two paid-return failures + eight late-readout + eight recipe + eight projected
+ one sink observer + eight metadata observer updates = 65 published native
updates / 130 rows. One failed native proposal
adds one attempt / two rows. Four PC initializers delegate to four BP initializers
and draw 92 normal values. No fit, direct-reference retraining or real native IO.
The 31 cases bound native energy entries at330 and latent proposals at264;
the fake three-scene recipe adds six external prediction rows / six controls.
"""

from dataclasses import asdict, replace
import json
from typing import Any, cast

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import continuous_correction_demo as demo
from src.adapters import numpy_continuous_pc as pc
from src.adapters.continuous_learnability_demo import ContinuousLearnabilityResult
from src.adapters.continuous_learnability_modes import RecordedContinuousPC
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.continuous_pc import CONTINUOUS_PC_ENERGY_ID
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_corrections import PickCubeCorrectionBatch, PickCubeCorrectionLabel
from src.core.pick_cube_public_vectors import public_vector_projection
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries
from test_maniskill_bc_smoke import FakeEnvironment


FAMILIES = ("pc", "neutral-cpc", "active-cpc")


def assert_state(first, second):
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)
    if first.wake_state is None:
        assert second.wake_state is None
    else:
        assert first.wake_state.wake_updates == second.wake_state.wake_updates
        np.testing.assert_array_equal(
            first.wake_state.chemical_state, second.wake_state.chemical_state
        )
        np.testing.assert_array_equal(
            first.wake_state.plasticity_state, second.wake_state.plasticity_state
        )


def native_model(port):
    return port._learner if type(port) is RecordedContinuousPC else port


def coherent_wake(model):
    state = model._state
    config, wake = state.policy.circadian, state.wake_state
    chemicals = wake.chemical_state + 0.1
    gates = np.clip(np.exp(-config.plasticity_sensitivity * chemicals), config.min_plasticity, 1.0)
    model._state = replace(
        state, wake_state=replace(wake, chemical_state=chemicals, plasticity_state=gates)
    )


def make_batch(dimension=3):
    spec = ContinuousActionSpec(dimension, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)

    def observation(step):
        return (float(step),) + (0.0,) * (min(dimension, 21) - 1) + (0.25,) * max(0, dimension - 21)

    transitions = tuple(
        ContinuousTransition(
            i,
            observation(i),
            (0.125, -0.25, 0.5, -0.5),
            float(i == 1),
            observation(i + 1),
            i == 1,
            False,
        )
        for i in range(2)
    )
    episode = ContinuousEpisode(
        "pick-cube-smoke-19", "pick-cube-train-19", "train", spec, transitions
    )
    labels = tuple(
        PickCubeCorrectionLabel(i, row.observation, (0.5, 0.0, 1.0, 1.0), "align")
        for i, row in enumerate(transitions)
    )
    return PickCubeCorrectionBatch(19, episode, labels)


def new_owner(family, dimension=3):
    spec = ContinuousActionSpec(dimension, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    stats = ObservationStandardization(
        (0.5,) + (0.0,) * (dimension - 1), (0.5,) + (1.0,) * (dimension - 1), 2
    )
    config = (
        None
        if family == "pc"
        else CoupledCircadianConfig.matched_pc_control()
        if family == "neutral-cpc"
        else CoupledCircadianConfig()
    )
    return pc.NumpyContinuousPCLearner(
        pc.ContinuousPCPolicy(
            pc.bc.ContinuousBCPolicy(
                spec, hidden_dim=2, seed=17, observation_standardization=stats
            ),
            circadian=config,
        )
    )


@pytest.fixture(scope="module")
def sources():
    result = {}
    features = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    targets = np.array([[0.2, -0.2, 0.4, -1.0], [-0.3, 0.5, -0.2, 1.0]], dtype=np.float64)
    for family in FAMILIES:
        owner = new_owner(family)
        for _ in range(2):
            owner.train_batch(features, targets)
        assert any(np.any(value != 0) for value in owner.snapshot_state().velocity)
        result[family] = owner
    return result


@pytest.fixture(scope="module")
def completed(sources):
    records: dict[str, Any] = {}
    captured: dict[str, list[Any]] = {"runtime": [], "native": []}
    original_runtime = demo.ActorShadowRuntime
    original_train = pc.NumpyContinuousPCLearner.train_batch_with_work
    with pytest.MonkeyPatch.context() as patch:

        def runtime(source, **options):
            created = original_runtime(source, **options)
            captured["runtime"].append((created, options))
            return created

        def train(owner, features, targets):
            before = owner.snapshot_state()
            returned = original_train(owner, features, targets)
            captured["native"].append(
                (
                    owner,
                    before,
                    owner.snapshot_state(),
                    features.copy(),
                    targets.copy(),
                    asdict(returned),
                )
            )
            return returned

        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(pc.NumpyContinuousPCLearner, "train_batch_with_work", train)
        for family in FAMILIES:
            source = sources[family]
            before = source.snapshot_state()
            batch = make_batch()
            original_batch = asdict(batch)
            result = demo.run_continuous_correction_update(source, batch, now=3)
            assert result.report["complete"] is True
            assert asdict(batch) == original_batch
            records[family] = (source, before, result)
    return records, captured


@pytest.mark.parametrize("family", FAMILIES)
def test_should_continue_full_native_state_through_eight_private_updates(completed, family):
    records, captured = completed
    source, initial, result = records[family]
    index = FAMILIES.index(family)
    runtime, options = captured["runtime"][index]
    native = captured["native"][8 * index : 8 * (index + 1)]
    assert (
        type(result) is ContinuousLearnabilityResult
        and type(result.candidate) is pc.NumpyContinuousPCLearner
    )
    assert len(native) == 8 and all(row[0] is native_model(runtime._candidate) for row in native)
    assert result.candidate is not source and result.candidate is not native_model(
        runtime._candidate
    )
    assert_state(initial, source.snapshot_state())
    assert_state(initial, runtime.actor.snapshot_state().state)
    assert_state(initial, native[0][1])
    state = result.candidate.snapshot_state()
    assert_state(state, native[-1][2])
    assert_state(state, runtime.candidate_snapshot().state)
    assert state.completed_updates == 10 and state.training_examples == 20
    assert state.policy == initial.policy
    assert options["budget"].updates_completed == options["budget"].budget.max_training_updates == 8
    assert options["budget"].replay_examples_completed == 0
    if family != "pc":
        assert initial.wake_state is not None and state.wake_state is not None
        assert initial.wake_state.wake_updates == 2 and state.wake_state.wake_updates == 10
        assert np.any(state.wake_state.chemical_state != initial.wake_state.chemical_state)
    for _, before, after, features, targets, _ in native:
        assert after.completed_updates == before.completed_updates + 1
        np.testing.assert_array_equal(features, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        np.testing.assert_array_equal(targets, np.array([[0.5, 0.0, 1.0, 1.0]] * 2))
        assert not np.array_equal(targets, np.array([[0.125, -0.25, 0.5, -0.5]] * 2))


@pytest.mark.parametrize("family", FAMILIES)
def test_should_preserve_arrival_receipts_and_actual_native_work_without_bp_formulas(
    completed, family
):
    records, _ = completed
    report = records[family][2].report
    assert report["protocol_id"] == "continuous_pc_correction_8_v1"
    assert report["source_updates"] == 2 and report["source_training_rows"] == 4
    assert (
        len(report["events"])
        == len(report["applied_updates"])
        == len(report["native_updates"])
        == 8
    )
    for index, (event, receipt, returned) in enumerate(
        zip(report["events"], report["applied_updates"], report["native_updates"], strict=True)
    ):
        assert event["success_validated"] is event["update_attempted"] is True
        assert event["applied"] == [receipt]
        assert receipt["sample_id"] == f"corrective-batch-{index}"
        assert receipt["event_id"] == f"corrective-label-{index}"
        assert receipt["episode_id"] == "pick-cube-smoke-19"
        assert receipt["actor_version"] == "actor-0" and receipt["learner_version"] == "candidate-0"
        assert receipt["observed_at"] == 2 and receipt["arrived_at"] == receipt["applied_at"] == 3
        assert receipt["update_number"] == index + 1
        assert receipt["diagnostic"] == returned["diagnostic"]
        assert returned["diagnostic"]["definition"] == CONTINUOUS_PC_ENERGY_ID
        assert returned["training_rows"] == 2
    work = report["work"]
    assert work["optimizer_updates"] == 8 and work["training_rows"] == 16
    assert work["unique_correction_rows"] == 2 and work["repeated_training_rows"] == 14
    assert work["fit_calls"] == work["reference_replay_rows"] == 0
    assert (
        work["normalization_calls"],
        work["normalization_rows"],
        work["normalization_values"],
    ) == (8, 16, 48)
    assert work["model_numeric_bytes_per_owner"] == 320
    assert work["statistics_numeric_bytes_per_owner"] == 48
    assert work["wake_numeric_bytes_per_owner"] == (0 if family == "pc" else 32)
    assert work["named_model_owners"] == 4
    assert work["inbox_feature_target_numeric_bytes"] == 896
    assert report["settings"]["max_training_updates"] == 8
    assert all(
        value == 0 for key, value in report["settings"].items() if key != "max_training_updates"
    )
    assert not ({"forward_passes", "backward_passes", "dense_multiply_accumulates"} & work.keys())
    for key in report["native_updates"][0]["work"]:
        expected = sum(row["work"][key] for row in report["native_updates"])
        assert work["native_update_counts"][key] == expected
        assert work["native_event_row_weights"][key] == expected * 2
    assert report["progress"]["updates_completed"] == 8
    assert report["progress"]["replay_examples_completed"] == 0
    json.dumps(report, allow_nan=False)


def test_should_keep_neutral_and_plain_pc_parameters_momentum_losses_and_common_native_work_exact(
    completed,
):
    records, _ = completed
    plain = records["pc"][2]
    neutral = records["neutral-cpc"][2]
    a, b = plain.candidate.snapshot_state(), neutral.candidate.snapshot_state()
    assert replace(b.policy, circadian=None) == a.policy
    for left, right in zip(a.parameters + a.velocity, b.parameters + b.velocity, strict=True):
        np.testing.assert_array_equal(left, right)
    for first, second in zip(
        plain.report["native_updates"], neutral.report["native_updates"], strict=True
    ):
        assert first["diagnostic"] == second["diagnostic"]
        assert (
            first["task_loss"] == second["task_loss"] and first["inference"] == second["inference"]
        )
        assert first["work"] == {key: second["work"][key] for key in first["work"]}
    assert b.wake_state.wake_updates == 10
    np.testing.assert_array_equal(b.wake_state.plasticity_state, np.ones(2))


def test_should_own_returned_candidate_and_report_without_mutating_source_or_runtime(completed):
    records, captured = completed
    source, initial, result = records["active-cpc"]
    state = result.candidate.snapshot_state()
    state.parameters[0].flat[0] += 1
    state.velocity[0].flat[0] -= 1
    state.wake_state.chemical_state.flat[0] += 1
    assert_state(initial, source.snapshot_state())
    assert_state(
        result.candidate.snapshot_state(), captured["runtime"][2][0].candidate_snapshot().state
    )


@pytest.mark.parametrize(
    "fault",
    [
        "source_role",
        "statistics",
        "source_type",
        "future",
        "batch_role",
        "spec",
        "pairing",
        "projection",
    ],
)
def test_should_reject_unreleased_or_incompatible_inputs_before_any_fork(
    sources, monkeypatch, fault
):
    owner: Any = sources["active-cpc"].fork()
    batch = make_batch()
    options: dict[str, Any] = {"now": 3}
    if fault in ("source_role", "statistics"):
        state = owner.snapshot_state()
        base = replace(
            state.policy.base,
            **(
                {"training_role": "probe_train"}
                if fault == "source_role"
                else {"observation_standardization": None}
            ),
        )
        policy = replace(state.policy, base=base)
        owner._policy, owner._state = policy, replace(state, policy=policy)
    elif fault == "source_type":
        owner = object()
    elif fault == "future":
        options["now"] = 2
    elif fault == "batch_role":
        object.__setattr__(batch.episode, "role", "final_test")
    elif fault == "spec":
        object.__setattr__(batch.episode.spec, "action_units", ("other",) * 4)
    elif fault == "pairing":
        object.__setattr__(batch.labels[0], "observation", (99.0, 0.0, 0.0))
    else:
        options["observation_projection"] = public_vector_projection()
    monkeypatch.setattr(
        pc.NumpyContinuousPCLearner, "fork", lambda *_: pytest.fail("unreleased PC forked")
    )
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(owner, batch, **options)


@pytest.mark.parametrize("failure", ["bad_receipt", "postcommit", "native_proposal"])
def test_should_preserve_paid_failure_native_return_or_failed_work_without_retry(
    sources, monkeypatch, failure
):
    source = sources["active-cpc"].fork()
    initial = source.snapshot_state()
    factory = demo.ActorShadowRuntime
    captured = []

    def runtime(owner, **options):
        created = factory(owner, **options)
        captured.append(created)
        ready = created.train_ready

        def changed(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if applied and failure == "postcommit":
                raise RuntimeError("paid return observer failed")
            if applied and failure == "bad_receipt":
                return (replace(applied[0], arrived_at=4),)
            return applied

        monkeypatch.setattr(created, "train_ready", changed)
        return created

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    if failure == "native_proposal":

        def fail(*args, **kwargs):
            raise ValueError("native proposal failed")

        monkeypatch.setattr(pc, "_propose_update", fail)
    result = demo.run_continuous_correction_update(source, make_batch(), now=3)
    report = result.report
    assert report["complete"] is False and result.candidate is None
    assert report["progress"]["updates_completed"] == (0 if failure == "native_proposal" else 1)
    assert len(report["events"]) == 1 and "work" not in report
    if failure == "native_proposal":
        assert report["native_updates"] == report["applied_updates"] == []
        assert report["failed_native_work"]["inference_returns"] == 1
        assert report["failed_native_work"]["parameter_proposals_attempted"] == 1
        assert report["failed_native_work"]["parameter_proposals_completed"] == 0
    else:
        assert len(report["native_updates"]) == len(report["applied_updates"]) == 1
        assert report["native_updates"][0]["work"]["published_updates"] == 1
        assert report["applied_updates"][0]["arrived_at"] == 3
    assert_state(initial, source.snapshot_state())
    assert_state(initial, captured[0].actor.snapshot_state().state)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "fault",
    [
        "coherent_wake",
        "facade_identity",
        "late_source_getter",
        "counter_type",
        "budget_identity",
        "timer_identity",
    ],
)
def test_should_refuse_opaque_preupdate_owner_corruption_before_native_work(
    sources, monkeypatch, fault
):
    source = sources["active-cpc"].fork()
    factory = demo.ActorShadowRuntime
    captured = []

    def runtime(owner, **options):
        created = factory(owner, **options)
        captured.append(created)
        if fault == "late_source_getter":
            snapshot = source.snapshot_state
            changed: list[bool] = []
            reads: list[bool] = []

            def altered(*args, **kwargs):
                state = snapshot()
                reads.append(True)
                # Skip the detached-readout fork; invalidate the actor only
                # after its public snapshot was observed by the owner guard.
                if len(reads) > 1 and not changed:
                    coherent_wake(native_model(created.actor._learner))
                    changed.append(True)
                return state

            monkeypatch.setattr(source, "snapshot_state", altered)
        else:
            label = created.record_label

            def altered(*args, **kwargs):
                result = label(*args, **kwargs)
                if fault == "coherent_wake":
                    coherent_wake(native_model(created._candidate))
                elif fault == "facade_identity":
                    facade = cast(RecordedContinuousPC, created._candidate)
                    facade._learner = facade._learner.fork()
                elif fault == "counter_type":
                    options["budget"].updates_completed = 0.0
                elif fault == "budget_identity":
                    options["budget"].budget = replace(options["budget"].budget)
                else:
                    timer = options["budget"].clock
                    options["budget"].clock = lambda: timer()
                return result

            monkeypatch.setattr(created, "record_label", altered)
        return created

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(
        pc.NumpyContinuousPCLearner,
        "train_batch_with_work",
        lambda *_: pytest.fail("corrupt owner admitted native update"),
    )
    result = demo.run_continuous_correction_update(source, make_batch(), now=3)
    assert result.candidate is None and result.report["complete"] is False
    assert result.report["progress"]["updates_completed"] == 0
    assert result.report["native_updates"] == result.report["applied_updates"] == []


def test_should_refuse_coherent_readout_wake_changed_by_final_clock_getter(sources, monkeypatch):
    source = sources["active-cpc"].fork()
    initial = source.snapshot_state()
    factory, restore = demo.ActorShadowRuntime, pc.NumpyContinuousPCLearner.restore_state
    detached: list[Any] = []
    changed: list[bool] = []

    def restored(owner, state):
        result = restore(owner, state)
        if state.completed_updates == initial.completed_updates + 8:
            detached.append(owner)
        return result

    def runtime(owner, **options):
        created = factory(owner, **options)
        now = options["clock"].now

        def late():
            value = now()
            if detached and not changed:
                coherent_wake(detached[-1])
                changed.append(True)
            return value

        monkeypatch.setattr(options["clock"], "now", late)
        return created

    monkeypatch.setattr(pc.NumpyContinuousPCLearner, "restore_state", restored)
    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_correction_update(source, make_batch(), now=3)
    assert changed and result.candidate is None and result.report["complete"] is False
    assert result.report["progress"]["updates_completed"] == 8
    assert len(result.report["native_updates"]) == len(result.report["applied_updates"]) == 8
    assert "work" not in result.report
    assert_state(initial, source.snapshot_state())


def test_should_refuse_returned_native_work_rewritten_by_later_source_getter(sources, monkeypatch):
    source = sources["active-cpc"].fork()
    initial = source.snapshot_state()
    factory, snapshot = demo.ActorShadowRuntime, source.snapshot_state
    changed: list[bool] = []

    def runtime(owner, **options):
        created = factory(owner, **options)

        def observed():
            state = snapshot()
            if len(owner._reports) == 1 and not changed:
                owner._reports[0]["work"]["normalization_returns"] = 0
                changed.append(True)
            return state

        monkeypatch.setattr(source, "snapshot_state", observed)
        return created

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_correction_update(source, make_batch(), now=3)
    assert changed and result.report["complete"] is False and result.candidate is None
    assert result.report["progress"]["updates_completed"] == 1
    assert len(result.report["native_updates"]) == len(result.report["applied_updates"]) == 1
    assert "work" not in result.report
    assert_state(initial, snapshot())


def test_should_preserve_eight_native_returns_when_final_receipt_observer_raises(
    sources, monkeypatch
):
    source = sources["active-cpc"].fork()
    initial = source.snapshot_state()
    factory, restore = demo.ActorShadowRuntime, pc.NumpyContinuousPCLearner.restore_state
    applied = getattr(factory, "applied_updates")
    detached, captured = [], []

    def runtime(owner, **options):
        created = factory(owner, **options)
        captured.append(created)
        return created

    def restored(owner, state):
        result = restore(owner, state)
        if state.completed_updates == initial.completed_updates + 8:
            detached.append(owner)
        return result

    def observed(owner):
        returned = applied.fget(owner)
        if detached:
            raise RuntimeError("final receipt observation failed")
        return returned

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(pc.NumpyContinuousPCLearner, "restore_state", restored)
    monkeypatch.setattr(factory, "applied_updates", property(observed))
    result = demo.run_continuous_correction_update(source, make_batch(), now=3)
    report = result.report
    assert report["complete"] is False and result.candidate is None
    assert len(report["native_updates"]) == 8
    assert report["applied_updates"] is report["progress"] is None
    assert "final receipt observation failed" in report["metadata_error"]
    assert "work" not in report
    assert captured[0]._budget.updates_completed == len(captured[0]._inbox._applied) == 8
    assert_state(initial, source.snapshot_state())


def test_should_project_only_arrived_raw_labels_for_native_pc_updates(monkeypatch):
    owner = new_owner("pc", 21)
    initial = owner.snapshot_state()
    batch = make_batch(789)
    original = asdict(batch)
    captured = []
    train = pc.NumpyContinuousPCLearner.train_batch_with_work

    def observed(model, features, targets):
        captured.append((features.copy(), targets.copy()))
        return train(model, features, targets)

    monkeypatch.setattr(pc.NumpyContinuousPCLearner, "train_batch_with_work", observed)
    result = demo.run_continuous_correction_update(
        owner, batch, now=3, observation_projection=public_vector_projection()
    )
    assert result.report["complete"] is True and len(captured) == 8
    assert type(result.candidate) is pc.NumpyContinuousPCLearner
    for features, targets in captured:
        assert features.shape == (2, 21) and targets.shape == (2, 4)
        np.testing.assert_array_equal(features[:, 0], [0.0, 1.0])
        np.testing.assert_array_equal(targets, [[0.5, 0.0, 1.0, 1.0]] * 2)
    metadata = result.report["input_projection"]
    assert metadata["source_spec"]["observation_dim"] == 789
    assert metadata["model_spec"]["observation_dim"] == 21
    assert asdict(batch) == original
    assert_state(initial, owner.snapshot_state())


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environments, predictions = [], []
    predict = pc.NumpyContinuousPCLearner.predict

    def factory():
        environment = FakeEnvironment()
        environments.append(environment)
        return environment

    def observed(owner, features):
        result = predict(owner, features)
        predictions.append(result.copy())
        return result

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-pc-corrections")
    monkeypatch.setattr(
        queries,
        "_geometry",
        lambda _: PickCubeGeometry((0.0, 0.0, 0.0), (0.05, 0.0, 0.02), (0.1, 0.0, 0.2)),
    )
    monkeypatch.setattr(pc.NumpyContinuousPCLearner, "predict", observed)
    return environments, predictions, tmp_path


def test_should_collect_student19_then_update_privately_and_roll_frozen_candidate17_19(
    sources, boundary
):
    environments, predictions, directory = boundary
    source = sources["active-cpc"].fork()
    initial = source.snapshot_state()
    destination = directory / "recipe"
    summary = smoke.run_corrective_adaptation(
        source, output_directory=destination, visual_observation=False
    )
    assert summary["complete"] is summary["source_state_exact"] is True
    assert [environment.reset_calls for environment in environments] == [[19], [17], [19]]
    assert len(predictions) == 6 and len(environments) == 3
    assert all(
        environment.close_calls == 1 and len(environment.commands) == 2
        for environment in environments
    )
    for index, environment in enumerate(environments):
        for step, command in enumerate(environment.commands):
            np.testing.assert_array_equal(command, predictions[index * 2 + step].astype(np.float32))
    collection = json.loads((destination / "collection.json").read_text(encoding="utf-8"))
    update = json.loads((destination / "update.json").read_text(encoding="utf-8"))
    assert (
        collection["policy"] == "frozen_continuous_pc"
        and len(collection["corrections"]["labels"]) == 2
    )
    assert (
        update["protocol_id"] == "continuous_pc_correction_8_v1"
        and len(update["native_updates"]) == 8
    )
    assert update["progress"]["updates_completed"] == 8
    for seed in (17, 19):
        report = json.loads((destination / f"updated-{seed}.json").read_text(encoding="utf-8"))
        assert report["source_updates"] == 10 and report["source_training_rows"] == 20
        assert report["frozen_state_exact"] is report["original_owner_exact"] is True
        assert report["source_wake_updates"] == 10 and report["optimizer_updates"] == 0
    assert_state(initial, source.snapshot_state())
    with pytest.raises(FileExistsError):
        smoke.run_corrective_adaptation(
            source, output_directory=destination, visual_observation=False
        )
    assert len(environments) == 3


def test_should_retain_returned_collection_and_stop_recipe_if_source_wake_changes(
    sources, monkeypatch, tmp_path
):
    source = sources["active-cpc"].fork()
    corrections = make_batch()
    calls = []
    collected = smoke.FrozenRolloutResult(
        {
            "complete": True,
            "episode": asdict(corrections.episode),
            "corrections": {
                "labels": [asdict(row) for row in corrections.labels],
                "batch": None,
                "teacher_actions": 2,
            },
        },
        corrections,
    )

    def returned(output, **options):
        calls.append((output.name, options["train_seed"]))
        coherent_wake(source)
        return collected

    monkeypatch.setattr(smoke, "run_frozen_rollout", returned)
    monkeypatch.setattr(
        smoke,
        "run_continuous_correction_update",
        lambda *args, **kwargs: pytest.fail("changed source admitted private update"),
    )
    summary = smoke.run_corrective_adaptation(
        source, output_directory=tmp_path / "failed", visual_observation=False
    )
    assert summary["complete"] is summary["source_state_exact"] is False
    assert summary["collection"]["complete"] is True
    assert summary["collection"]["queries"]["teacher_actions"] == 2
    assert summary["update"] is None and summary["rollouts"] == []
    assert calls == [("collection.json", 19)]
