"""Private corrective wake updates with separately charged fixed TRAIN rehearsal.

Per platform: four D3/H2/A4 initializers (seven constructor entries including PC
delegation),56 normal values;75 published native updates/134 rows, zero fits.
PC41 updates/70 rows bound205 energy entries/164 latent proposals; BP34 updates/
64 rows. The only external predictions are six rows in three fake scenes/six
controls. All other failures stop before native training. No real simulator.
"""

from dataclasses import replace
import json
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import continuous_correction_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.app import continuous_stream_rehearsal as rehearsal
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_corrections import PickCubeCorrectionBatch, PickCubeCorrectionLabel
from src.core.pick_cube_action_history import action_history_training_episode
from src.core.pick_cube_readiness_corrections import (
    ReadinessCorrectionBatch,
    ReadinessCorrectionLabel,
)
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries
from test_maniskill_bc_smoke import FakeEnvironment


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")


def spec():
    return ContinuousActionSpec(3, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)


def reference():
    positions = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0))
    targets = ((0.25, -0.5, 0.125, -1.0), (-0.25, 0.25, -0.5, 1.0))
    return ContinuousEpisode(
        "reference-A",
        "pick-cube",
        "train",
        spec(),
        tuple(
            ContinuousTransition(
                i, positions[i], action, float(i == 1), positions[i + 1], i == 1, False
            )
            for i, action in enumerate(targets)
        ),
    )


def reference_one():
    original = reference()
    return replace(original, transitions=(replace(original.transitions[0], terminated=True),))


def batch():
    rows = tuple(
        ContinuousTransition(
            i,
            (float(i), 0.0, 0.0),
            (0.125, -0.25, 0.5, -0.5),
            float(i == 1),
            (float(i + 1), 0.0, 0.0),
            i == 1,
            False,
        )
        for i in range(2)
    )
    episode = ContinuousEpisode("pick-cube-smoke-19", "pick-cube-train-19", "train", spec(), rows)
    return PickCubeCorrectionBatch(
        19,
        episode,
        tuple(
            PickCubeCorrectionLabel(i, row.observation, (0.5, 0.0, 1.0, 1.0), "align")
            for i, row in enumerate(rows)
        ),
    )


def assert_state(first, second):
    assert type(first) is type(second) and first.policy == second.policy
    assert (first.completed_updates, first.training_examples) == (
        second.completed_updates,
        second.training_examples,
    )
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)
    a, b = getattr(first, "wake_state", None), getattr(second, "wake_state", None)
    if a is None:
        assert b is None
    else:
        assert b is not None and a.wake_updates == b.wake_updates
        np.testing.assert_array_equal(a.chemical_state, b.chemical_state)
        np.testing.assert_array_equal(a.plasticity_state, b.plasticity_state)


@pytest.fixture(scope="module")
def sources():
    result = {}
    features = reference().transitions
    x = np.asarray([row.observation for row in features], dtype=np.float64)
    y = np.asarray([row.action for row in features], dtype=np.float64)
    for family in FAMILIES:
        base = bp.ContinuousBCPolicy(
            spec(),
            hidden_dim=2,
            seed=17,
            observation_standardization=ObservationStandardization(
                (1.0, 0.0, 0.0), (1.0, 1.0, 1.0), 2
            ),
        )
        if family == "bp":
            owner: Any = bp.NumpyContinuousBCLearner(base)
        else:
            config = (
                None
                if family == "pc"
                else CoupledCircadianConfig.matched_pc_control()
                if family == "neutral-cpc"
                else CoupledCircadianConfig()
            )
            owner = pc.NumpyContinuousPCLearner(pc.ContinuousPCPolicy(base, circadian=config))
        owner.train_batch(x, y)
        assert any(np.any(value != 0) for value in owner.snapshot_state().velocity)
        result[family] = owner
    return result


@pytest.fixture(scope="module")
def completed(sources):
    records: dict[str, Any] = {}
    runtime_factory, helper_factory = demo.ActorShadowRuntime, demo.ContinuousStreamRehearsal
    bp_train, pc_train = (
        bp.NumpyContinuousBCLearner.train_batch,
        pc.NumpyContinuousPCLearner.train_batch_with_work,
    )
    with pytest.MonkeyPatch.context() as patch:

        def runtime(*args, **options):
            value = runtime_factory(*args, **options)
            records[family].update(runtime=value, wake=options["budget"])
            return value

        def helper(*args, **options):
            value = helper_factory(*args, **options)
            records[family].update(helper=value, replay=options["replay_budget"])
            return value

        def record_train(original, owner, features, targets):
            records[family]["trains"].append((features.copy(), targets.copy()))
            return original(owner, features, targets)

        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(demo, "ContinuousStreamRehearsal", helper)
        patch.setattr(
            bp.NumpyContinuousBCLearner,
            "train_batch",
            lambda owner, x, y: record_train(bp_train, owner, x, y),
        )
        patch.setattr(
            pc.NumpyContinuousPCLearner,
            "train_batch_with_work",
            lambda owner, x, y: record_train(pc_train, owner, x, y),
        )
        for family in FAMILIES:
            owner = sources[family]
            records[family] = dict(source=owner, initial=owner.snapshot_state(), trains=[])
            records[family]["result"] = demo.run_continuous_correction_update(
                owner,
                batch(),
                now=3,
                replay_plan=rehearsal.ContinuousReplayPlan(2, 4),
                reference_episodes=(reference_one(),),
            )
    return records


@pytest.mark.parametrize("family", FAMILIES)
def test_should_continue_all_native_state_with_eight_wakes_and_four_separately_charged_replays(
    completed, family
):
    record = completed[family]
    result, runtime, initial = record["result"], record["runtime"], record["initial"]
    assert result.report["complete"] is True and result.candidate is not None
    final = result.candidate.snapshot_state()
    assert final.completed_updates == initial.completed_updates + 12 == 13
    assert final.training_examples == initial.training_examples + 20 == 22
    assert_state(initial, record["source"].snapshot_state())
    assert_state(initial, runtime.actor.snapshot_state().state)
    assert_state(final, runtime.candidate_snapshot().state)
    assert final.policy == initial.policy
    if family in ("neutral-cpc", "active-cpc"):
        assert final.wake_state.wake_updates == 13
    assert record["wake"] is not record["replay"]
    assert record["wake"].updates_completed == 8 and record["wake"].replay_examples_completed == 0
    assert (
        record["replay"].updates_completed == 0 and record["replay"].replay_examples_completed == 4
    )
    assert len(runtime.applied_updates) == 8 and len(runtime.applied_consolidations) == 4
    report = result.report
    assert (
        report["progress"]["updates_completed"] == 8
        and report["progress"]["replay_examples_completed"] == 0
    )
    assert report["work"]["optimizer_updates"] == 12 and report["work"]["training_rows"] == 20
    assert report["work"]["reference_replay_rows"] == 4
    assert report["replay"]["charged_exposures"] == 4 and report["replay"]["committed_updates"] == 4
    assert report["replay"]["bank_rows"] == 1
    assert report["replay"]["bank_episode_ids"] == ["reference-A"]
    assert len(report["applied_updates"]) == 8 and len(report["applied_consolidations"]) == 4
    assert all(
        row["observed_at"] == 2 and row["arrived_at"] == row["applied_at"] == 3
        for row in report["applied_updates"]
    )
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("family", FAMILIES)
def test_should_train_replay_only_on_the_declared_bank_and_wakes_only_on_arrived_targets(
    completed, family
):
    trains = completed[family]["trains"]
    assert len(trains) == 12
    wake_x, wake_y = batch().training_pairs(now=3)
    replay_x = tuple(row.observation for row in reference_one().transitions)
    replay_y = tuple(row.action for row in reference_one().transitions)
    for index, (features, targets) in enumerate(trains):
        x, y = (replay_x, replay_y) if index % 3 == 2 else (wake_x, wake_y)
        np.testing.assert_array_equal(features, x)
        np.testing.assert_array_equal(targets, y)
    if family != "bp":
        rows = completed[family]["result"].report["native_updates"]
        assert len(rows) == 12 and [row["training_rows"] for row in rows] == [2, 2, 1] * 4
        assert sum(row["work"]["published_updates"] for row in rows) == 12
        work = completed[family]["result"].report["work"]
        assert work["native_update_counts"]["normalization_returns"] == 12
        assert work["native_event_row_weights"]["normalization_returns"] == 20


def test_should_keep_pc_and_neutral_common_parameters_momentum_and_receipts_exact(completed):
    first, second = (completed[family]["result"] for family in ("pc", "neutral-cpc"))
    a, b = first.candidate.snapshot_state(), second.candidate.snapshot_state()
    assert replace(a.policy, circadian=None) == replace(b.policy, circadian=None)
    for left, right in zip(a.parameters + a.velocity, b.parameters + b.velocity, strict=True):
        np.testing.assert_array_equal(left, right)
    assert first.report["applied_updates"] == second.report["applied_updates"]
    assert first.report["applied_consolidations"] == second.report["applied_consolidations"]


def forbidden(*args, **kwargs):
    pytest.fail("invalid rehearsal admitted private model work")


@pytest.mark.parametrize(
    "fault",
    [
        "unrequested_bank",
        "empty",
        "list",
        "final",
        "probe",
        "spec",
        "duplicate",
        "quota",
        "incomplete",
    ],
)
def test_should_refuse_invalid_or_unreleased_reference_bank_before_any_fork(
    sources, monkeypatch, fault
):
    owner = sources["bp"].fork()
    monkeypatch.setattr(owner, "fork", forbidden)
    monkeypatch.setattr(demo, "ActorShadowRuntime", forbidden)
    plan: Any = rehearsal.ContinuousReplayPlan(2, 8)
    bank: Any = (reference(),)
    if fault == "unrequested_bank":
        plan = None
    elif fault == "empty":
        bank = ()
    elif fault == "list":
        bank = list(bank)
    elif fault in ("final", "probe"):
        bank = (replace(bank[0], role="final_test" if fault == "final" else "probe_train"),)
    elif fault == "spec":
        bank = (
            replace(
                bank[0],
                spec=replace(bank[0].spec, action_units=("different", *native.ACTION_UNITS[1:])),
            ),
        )
    elif fault == "duplicate":
        bank = (bank[0], bank[0])
    elif fault == "quota":
        plan = rehearsal.ContinuousReplayPlan(2, 1)
    else:
        object.__setattr__(bank[0].transitions[-1], "terminated", False)
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(
            owner, batch(), now=3, replay_plan=plan, reference_episodes=bank
        )


def test_should_refuse_nearest_replay_before_reading_policy_or_allocating_private_models():
    # The type gate needs no initialized nearest state, model constructor or append.
    owner = object.__new__(demo.NumpyNearestContinuousBCLearner)
    setattr(owner, "fork", forbidden)
    with pytest.raises(ValueError, match="neural"):
        demo.run_continuous_correction_update(
            owner,
            batch(),
            now=3,
            replay_plan=rehearsal.ContinuousReplayPlan(2, 8),
            reference_episodes=(reference(),),
        )


def test_should_refuse_false_encoded_history_tail_with_valid_model_spec_before_private_fork(
    sources, monkeypatch
):
    owner = sources["bp"].fork()
    old = owner.snapshot_state()
    # Declare a tiny deterministic12D fixture via an RNG-free owned fork/restore;
    # no claim is made that its copied non-input matrices were trained in12D.
    policy = replace(
        old.policy,
        spec=replace(old.policy.spec, observation_dim=12),
        observation_standardization=None,
    )
    owner._policy = policy
    owner.restore_state(
        replace(
            old,
            policy=policy,
            parameters=(np.zeros((12, 2), dtype=np.float64), *old.parameters[1:]),
            velocity=(np.zeros((12, 2), dtype=np.float64), *old.velocity[1:]),
        )
    )

    def raw_episode(value):
        return replace(
            value,
            spec=replace(value.spec, observation_dim=42),
            transitions=tuple(
                replace(
                    row,
                    observation=(*row.observation, *(0.0,) * 39),
                    next_observation=(*row.next_observation, *(0.0,) * 39),
                )
                for row in value.transitions
            ),
        )

    raw = raw_episode(batch().episode)
    arrived = ReadinessCorrectionBatch(
        19,
        raw,
        tuple(
            ReadinessCorrectionLabel(i, row.observation, (0.5, 0.0, 1.0, 1.0), "align")
            for i, row in enumerate(raw.transitions)
        ),
    )
    recipe = PickCubeTaskSpaceHistoryRecipe()
    bank = action_history_training_episode(raw_episode(reference()), recipe)
    first = bank.transitions[0]
    object.__setattr__(first, "observation", (*first.observation[:-1], 0.5))
    monkeypatch.setattr(owner, "fork", forbidden)
    monkeypatch.setattr(demo, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError, match="history"):
        demo.run_continuous_correction_update(
            owner,
            arrived,
            now=3,
            action_history_recipe=recipe,
            replay_plan=rehearsal.ContinuousReplayPlan(2, 8),
            reference_episodes=(bank,),
        )


def test_should_stop_before_fifth_wake_when_second_replay_cannot_fit_its_original_allowance(
    sources,
):
    source = sources["bp"].fork()
    initial = source.snapshot_state()
    result = demo.run_continuous_correction_update(
        source,
        batch(),
        now=3,
        replay_plan=rehearsal.ContinuousReplayPlan(2, 2),
        reference_episodes=(reference(),),
    )
    assert result.candidate is None and result.report["complete"] is False
    assert result.report["progress"]["updates_completed"] == 4
    assert len(result.report["applied_updates"]) == 4
    assert len(result.report["applied_consolidations"]) == 1
    assert result.report["replay"]["charged_exposures"] == 2
    assert result.report["replay"]["committed_updates"] == 1
    assert_state(initial, source.snapshot_state())


@pytest.mark.parametrize("fault", ["plan", "budget", "timer"])
def test_should_refuse_replaced_authority_before_the_first_native_wake(sources, monkeypatch, fault):
    original = demo.ContinuousStreamRehearsal
    captured = []

    def factory(*args, **options):
        helper = original(*args, **options)
        captured.append(helper)
        before = helper.before_update

        def changed(runtime, updates):
            if fault == "plan":
                helper._plan = replace(helper._plan)
            elif fault == "budget":
                helper._replay.session = replace_session(helper._replay.session)
            else:
                helper._replay.session.clock = lambda: 0.0
            return before(runtime, updates)

        monkeypatch.setattr(helper, "before_update", changed)
        return helper

    monkeypatch.setattr(demo, "ContinuousStreamRehearsal", factory)
    result = demo.run_continuous_correction_update(
        sources["bp"].fork(),
        batch(),
        now=3,
        replay_plan=rehearsal.ContinuousReplayPlan(2, 8),
        reference_episodes=(reference(),),
    )
    assert result.candidate is None and result.report["complete"] is False
    assert (
        result.report["progress"]["updates_completed"] == 0
        and result.report["applied_updates"] == []
    )
    assert result.report["replay"]["charged_exposures"] == 0
    assert captured[0].report["stopped"] is True


def replace_session(session):
    from copy import copy

    return copy(session)


def test_should_recheck_preceding_replay_receipt_before_a_third_wake(sources, monkeypatch):
    original = demo.ContinuousStreamRehearsal

    def factory(*args, **options):
        helper = original(*args, **options)
        before = helper.before_update

        def changed(runtime, updates):
            if updates == 2:
                object.__setattr__(runtime._consolidations[0], "event_id", "replaced-paid-receipt")
            return before(runtime, updates)

        monkeypatch.setattr(helper, "before_update", changed)
        return helper

    monkeypatch.setattr(demo, "ContinuousStreamRehearsal", factory)
    source = sources["bp"].fork()
    initial = source.snapshot_state()
    result = demo.run_continuous_correction_update(
        source,
        batch(),
        now=3,
        replay_plan=rehearsal.ContinuousReplayPlan(2, 8),
        reference_episodes=(reference(),),
    )
    assert result.candidate is None and result.report["complete"] is False
    assert result.report["progress"]["updates_completed"] == 2
    assert len(result.report["applied_updates"]) == 2
    assert (
        result.report["replay"]["charged_exposures"] == 2
        and result.report["replay"]["committed_updates"] == 1
    )
    assert_state(initial, source.snapshot_state())


def test_should_bind_postwake_state_before_a_helper_callback_can_legitimize_changed_parameters(
    sources, monkeypatch
):
    original = demo.ContinuousStreamRehearsal

    def factory(*args, **options):
        helper = original(*args, **options)
        after = helper.after_update

        def changed(runtime, updates, **kwargs):
            state = runtime._candidate.snapshot_state()
            state.parameters[0][0, 0] += 0.125
            runtime._candidate.restore_state(state)
            return after(runtime, updates, **kwargs)

        monkeypatch.setattr(helper, "after_update", changed)
        return helper

    monkeypatch.setattr(demo, "ContinuousStreamRehearsal", factory)
    source = sources["bp"].fork()
    initial = source.snapshot_state()
    result = demo.run_continuous_correction_update(
        source,
        batch(),
        now=3,
        replay_plan=rehearsal.ContinuousReplayPlan(2, 8),
        reference_episodes=(reference(),),
    )
    assert result.candidate is None and result.report["complete"] is False
    assert result.report["progress"]["updates_completed"] == 1
    assert (
        len(result.report["applied_updates"]) == 1 and result.report["applied_consolidations"] == []
    )
    assert result.report["replay"]["charged_exposures"] == 0
    assert_state(initial, source.snapshot_state())


def test_should_retain_sealed_pc_replay_return_when_later_source_getter_changes_live_work(
    sources, monkeypatch
):
    owner = sources["pc"].fork()
    initial = owner.snapshot_state()
    snapshot, helper_factory = owner.snapshot_state, demo.ContinuousStreamRehearsal
    captured: dict[str, Any] = dict(reports=None, changed=False)

    def helper(*args, **options):
        captured["reports"] = args[0]._reports
        return helper_factory(*args, **options)

    def observed():
        state = snapshot()
        reports = captured["reports"]
        if reports is not None and len(reports) == 2 and not captured["changed"]:
            reports[-1]["work"]["normalization_attempts"] += 1
            captured["changed"] = True
        return state

    monkeypatch.setattr(demo, "ContinuousStreamRehearsal", helper)
    monkeypatch.setattr(owner, "snapshot_state", observed)
    result = demo.run_continuous_correction_update(
        owner,
        batch(),
        now=3,
        replay_plan=rehearsal.ContinuousReplayPlan(1, 16),
        reference_episodes=(reference(),),
    )
    assert captured["changed"] is True
    assert result.candidate is None and result.report["complete"] is False
    assert (
        result.report["progress"]["updates_completed"] == 1
        and len(result.report["applied_updates"]) == 1
    )
    assert result.report["replay"]["charged_exposures"] == 2
    sealed = result.report["sealed_replay_native_updates"]
    assert len(sealed) == 2 and sealed[-1]["work"]["normalization_attempts"] == 1
    assert sealed[-1]["training_rows"] == 2 and sealed[-1]["work"]["published_updates"] == 1
    assert_state(initial, owner.snapshot_state())


def test_should_forward_bank_only_to_private_update_and_run_three_frozen_fake_scenes(
    sources, monkeypatch, tmp_path
):
    environments, calls, update_calls = [], [], []

    def factory():
        environment = FakeEnvironment()
        environments.append(environment)
        return environment

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-reference-rehearsal")
    monkeypatch.setattr(
        queries,
        "_geometry",
        lambda _: PickCubeGeometry((0.0, 0.0, 0.0), (0.05, 0.0, 0.02), (0.1, 0.0, 0.2)),
    )
    native_rollout, update = smoke.run_frozen_rollout, smoke.run_continuous_correction_update

    def rollout(*args, **options):
        calls.append(options)
        return native_rollout(*args, **options)

    def updating(*args, **options):
        update_calls.append(options)
        return update(*args, **options)

    monkeypatch.setattr(smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(smoke, "run_continuous_correction_update", updating)
    source = sources["bp"].fork()
    initial = source.snapshot_state()
    plan, bank = rehearsal.ContinuousReplayPlan(2, 8), (reference(),)
    result = smoke.run_corrective_adaptation(
        source,
        output_directory=tmp_path / "recipe",
        visual_observation=False,
        replay_plan=plan,
        reference_episodes=bank,
    )
    assert result["complete"] is True
    assert [call["train_seed"] for call in calls] == [19, 17, 19]
    assert all("replay_plan" not in call and "reference_episodes" not in call for call in calls)
    assert update_calls[0]["replay_plan"] == plan and update_calls[0]["reference_episodes"] == bank
    assert result["update"]["replay"]["charged_exposures"] == 8
    assert len(environments) == 3 and all(
        len(environment.commands) == 2 and environment.close_calls == 1
        for environment in environments
    )
    assert_state(initial, source.snapshot_state())
