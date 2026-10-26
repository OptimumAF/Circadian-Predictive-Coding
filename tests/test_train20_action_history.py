"""TRAIN20 uses unchanged causal47D/TS12 history, without model or native IO.

Six cached three-row views plus tiny causality/refusal cases. Prospective bound:
200 scalar history projections and100 history advances, including failed work.
No teacher qualification is inferred from a core synthetic TRAIN episode.
"""

from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from src.core import pick_cube_action_history as history
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from test_pick_cube_action_history import ACTIONS, episode


VIEWS = (
    ("raw47", "executed"),
    ("raw47", "zero"),
    ("canonical47", "executed"),
    ("canonical47", "zero"),
    ("taskspace12", "executed"),
    ("taskspace12", "zero"),
)
EXECUTED_VIEWS = ("raw47", "canonical47", "taskspace12")


def source20(actions=ACTIONS):
    # The existing tiny helper deliberately defaults its non17 context to19.
    # This is an explicitly declared synthetic20 record, not relabeled evidence.
    return replace(episode(20, actions), context="pick-cube-train-20")


def recipe(view, mode="executed"):
    return (
        PickCubeTaskSpaceHistoryRecipe(mode=mode)
        if view == "taskspace12"
        else history.PickCubeActionHistoryRecipe(
            mode=mode, canonical_quaternions=view == "canonical47"
        )
    )


@pytest.fixture(scope="module", autouse=True)
def bounded_work():
    counts = dict(
        projection_attempts=0,
        projection_returns=0,
        emitted_feature_values=0,
        advance_attempts=0,
        advance_returns=0,
    )
    project, advance = history.action_history_observation, history.advance_action_history

    def projected(*args, **kwargs):
        counts["projection_attempts"] += 1
        returned = project(*args, **kwargs)
        counts["projection_returns"] += 1
        counts["emitted_feature_values"] += len(returned)
        return returned

    def advanced(*args, **kwargs):
        counts["advance_attempts"] += 1
        returned = advance(*args, **kwargs)
        counts["advance_returns"] += 1
        return returned

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(history, "action_history_observation", projected)
        patch.setattr(history, "advance_action_history", advanced)
        yield counts
    assert counts["projection_attempts"] <= 200 and counts["advance_attempts"] <= 100


@pytest.fixture(scope="module")
def encoded():
    source = source20()
    original = asdict(source)
    values = {
        (view, mode): history.action_history_training_episode(source, recipe(view, mode))
        for view, mode in VIEWS
    }
    assert asdict(source) == original
    return source, values


@pytest.mark.parametrize("view,mode", VIEWS)
def test_should_encode_train20_with_original_metadata_and_exact_before_after_history(
    encoded, view, mode
):
    source, values = encoded
    actual = values[(view, mode)]
    config = recipe(view, mode)
    assert actual.episode_id == source.episode_id == "arbitrary-train-20"
    assert (
        actual.context == source.context == "pick-cube-train-20"
        and actual.role == source.role == "train"
    )
    assert actual.spec.action_units == source.spec.action_units
    assert (
        actual.spec.lower_bounds == source.spec.lower_bounds
        and actual.spec.upper_bounds == source.spec.upper_bounds
    )
    assert actual.spec.observation_dim == (12 if view == "taskspace12" else 47)
    before_tails = ((0.0,) * 5, (*ACTIONS[0], 1 / 6), (*ACTIONS[1], 2 / 6))
    after_tails = ((*ACTIONS[0], 1 / 6), (*ACTIONS[1], 2 / 6), (*ACTIONS[2], 0.0))
    for index, (row, original) in enumerate(
        zip(actual.transitions, source.transitions, strict=True)
    ):
        assert (row.step, row.action, row.reward, row.terminated, row.truncated) == (
            original.step,
            original.action,
            original.reward,
            original.terminated,
            original.truncated,
        )
        assert row.observation[-5:] == ((0.0,) * 5 if mode == "zero" else before_tails[index])
        assert row.next_observation[-5:] == ((0.0,) * 5 if mode == "zero" else after_tails[index])
        for output, raw in (
            (row.observation, original.observation),
            (row.next_observation, original.next_observation),
        ):
            if view == "raw47":
                assert output[:42] == raw
            elif view == "canonical47":
                assert (
                    output[:22] == raw[:22]
                    and output[26:32] == raw[26:32]
                    and output[36:42] == raw[36:42]
                )
                assert output[22:26] == (1.0, 0.0, 0.0, 0.0)
                assert output[32:36] == (0.0, 1.0, 0.0, 0.0)
            else:
                assert output[:7] == (raw[18],) + tuple(value / 0.1 for value in raw[36:42])
        if index:
            assert actual.transitions[index - 1].next_observation == row.observation
    admitted = history.validate_action_history_training_episode(actual, config)
    assert admitted == actual and admitted is not actual
    assert source == source20()


@pytest.mark.parametrize("view", EXECUTED_VIEWS)
def test_should_ignore_future_actions_and_restart_causal_state_for_each_train20_episode(view):
    config = recipe(view)
    first = history.action_history_training_episode(source20(), config)
    future = history.action_history_training_episode(
        source20((*ACTIONS[:2], (0.0, 0.0, 0.0, -1.0))), config
    )
    assert tuple(row.observation for row in first.transitions) == tuple(
        row.observation for row in future.transitions
    )
    assert first.transitions[-1].next_observation[-5:] == (*ACTIONS[-1], 0.0)
    assert future.transitions[-1].next_observation[-5:] == (0.0, 0.0, 0.0, -1.0, 3 / 6)
    current = history.action_history_training_episode(
        source20(((0.0, 0.0, 0.0, 1.0), *ACTIONS[1:])), config
    )
    assert (
        first.transitions[0].observation
        == future.transitions[0].observation
        == current.transitions[0].observation
    )
    assert current.transitions[0].observation[-5:] == (0.0,) * 5
    assert current.transitions[1].observation[-5:] == (0.0, 0.0, 0.0, 1.0, 0.0)


def forbidden(*args, **kwargs):
    pytest.fail("unadmitted source entered action-history projection or validation")


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "inner_guard", "final_test"])
def test_should_refuse_non_train_role_before_constructing_or_validating_features(monkeypatch, role):
    source = replace(source20(), role=role)
    monkeypatch.setattr(history, "_action_history_episode", forbidden)
    monkeypatch.setattr(history, "_validate_encoded_history", forbidden)
    for config in (recipe("raw47"), recipe("taskspace12")):
        with pytest.raises(ValueError):
            history.action_history_training_episode(source, config)
        with pytest.raises(ValueError):
            history.validate_action_history_training_episode(source, config)


@pytest.mark.parametrize(
    "context",
    [
        "pick-cube-train-21",
        "pick-cube-reserved-probe-23-24",
        "pick-cube-inner-guard-31",
        "pick-cube-inner-guard-37",
        "pick-cube-train-23",
        "pick-cube-train-24",
        "pick-cube-train-29",
    ],
)
def test_should_refuse_unknown_or_reserved_context_before_projection(monkeypatch, context):
    source = replace(source20(), context=context)
    monkeypatch.setattr(history, "_action_history_episode", forbidden)
    for config in (recipe("raw47"), recipe("taskspace12")):
        with pytest.raises(ValueError):
            history.action_history_training_episode(source, config)


@pytest.mark.parametrize("view", EXECUTED_VIEWS)
@pytest.mark.parametrize("side", ["before", "after"])
def test_should_refuse_tampered_encoded_train20_action_tail_without_changing_valid_view(
    encoded, view, side
):
    valid = encoded[1][(view, "executed")]
    original = deepcopy(valid)
    damaged = deepcopy(valid)
    if side == "before":
        row = damaged.transitions[0]
        object.__setattr__(row, "observation", (*row.observation[:-5], *ACTIONS[0], 1 / 6))
    else:
        row = damaged.transitions[-1]
        object.__setattr__(
            row, "next_observation", (*row.next_observation[:-5], *ACTIONS[1], 2 / 6)
        )
    with pytest.raises(ValueError):
        history.validate_action_history_training_episode(damaged, recipe(view))
    assert valid == original


@pytest.mark.parametrize("view", EXECUTED_VIEWS)
def test_should_refuse_unquantized_train20_action_instead_of_repairing_it(view):
    source = source20(((0.1, 0.0, 0.0, -1.0), *ACTIONS[1:]))
    original = deepcopy(source)
    with pytest.raises(ValueError):
        history.action_history_training_episode(source, recipe(view))
    assert source == original
