"""Explicit action decoding and paid control receipts, with literal records only."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
import math
from typing import Any

import pytest

from src.app.action_decoding import ActionDecodingLedger
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_action_decoding import (
    PickCubeBinaryGripperRecipe,
    decode_gripper_action,
    validate_gripper_recipe,
)


PREDICTION = (0.25, -0.5, 0.0, -0.125)
COMMAND = (0.25, -0.5, 0.0, -1.0)


def record(step=0, *, action=COMMAND, done=True):
    return ContinuousTransition(
        step,
        (float(step),) + (0.0,) * 41,
        action,
        float(done),
        (float(step + 1),) + (0.0,) * 41,
        done,
        False,
    )


def accepted(ledger, step=0, *, done=True, records=None):
    actual = ledger.report()
    assert ledger.select(step, PREDICTION, actual) == COMMAND
    ledger.attempt(ledger.report())
    row = record(step, done=done)
    ledger.returned(row, ledger.report())
    if records is None:
        records = []
    records.append(row)
    ledger.check(ledger.report(), tuple(records))
    return row


@pytest.mark.parametrize(
    "gripper,expected",
    [
        (-1.0, -1.0),
        (-0.125, -1.0),
        (-1e-50, -1.0),
        (-0.0, 1.0),
        (0.0, 1.0),
        (0.125, 1.0),
        (1.0, 1.0),
    ],
)
def test_should_decode_only_the_gripper_and_preserve_xyz_signed_zero(gripper, expected):
    action = (-0.0, 0.5, -0.5, gripper)
    decoded = decode_gripper_action(action, PickCubeBinaryGripperRecipe())
    assert decoded == (-0.0, 0.5, -0.5, expected)
    assert math.copysign(1.0, decoded[0]) == -1.0
    assert action[3] == gripper


def test_should_own_a_single_fixed_recipe_field():
    recipe: Any = PickCubeBinaryGripperRecipe()
    owned = validate_gripper_recipe(recipe)
    assert owned == recipe and owned is not recipe
    assert asdict(owned) == {"definition": "pick_cube_binary_gripper_zero_open_v1"}
    with pytest.raises(FrozenInstanceError):
        setattr(recipe, "definition", "different")


@pytest.mark.parametrize("fault", ["foreign", "subclass", "corrupt"])
def test_should_refuse_foreign_or_changed_recipe_definitions(fault):
    recipe: Any = PickCubeBinaryGripperRecipe()
    if fault == "foreign":
        recipe = "pick_cube_binary_gripper_zero_open_v1"
    elif fault == "subclass":

        class Other(PickCubeBinaryGripperRecipe):
            pass

        recipe = object.__new__(Other)
        object.__setattr__(recipe, "definition", "pick_cube_binary_gripper_zero_open_v1")
    else:
        object.__setattr__(recipe, "definition", True)
    with pytest.raises(ValueError):
        validate_gripper_recipe(recipe)


@pytest.mark.parametrize(
    "action",
    [
        [0.0, 0.0, 0.0, -1.0],
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 0.0, 1),
        (0.0, 0.0, 0.0, True),
        (float("nan"), 0.0, 0.0, -1.0),
        (0.0, 0.0, 0.0, float("inf")),
        (1.001, 0.0, 0.0, -1.0),
        (0.0, 0.0, 0.0, -1.001),
    ],
)
def test_should_refuse_non_native_or_nonfinite_normalized_actions(action):
    with pytest.raises(ValueError):
        decode_gripper_action(action, PickCubeBinaryGripperRecipe())


def test_should_keep_prediction_selected_command_and_native_return_separate_and_owned():
    ledger = ActionDecodingLedger(PickCubeBinaryGripperRecipe())
    records: list[ContinuousTransition] = []
    first = accepted(ledger, done=False, records=records)
    second = accepted(ledger, 1, records=records)
    actual = ledger.report()
    assert actual["controls"] == [
        dict(
            step=i,
            prediction=PREDICTION,
            selected_command=COMMAND,
            control_attempted=True,
            returned_action=COMMAND,
        )
        for i in range(2)
    ]
    assert actual["work"] == dict(
        decode_attempts=2,
        decode_returns=2,
        prediction_values=8,
        copied_xyz_values=6,
        gripper_decisions=2,
        control_attempts=2,
        control_returns=2,
        validated_control_returns=2,
        returned_action_values=8,
    )
    units = ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper")
    episode = ContinuousEpisode(
        "literal",
        "pick-cube",
        "train",
        ContinuousActionSpec(42, units, (-1.0,) * 4, (1.0,) * 4),
        (first, second),
    )
    ledger.verify_episode(episode, actual)
    detached = deepcopy(actual)
    detached["controls"][0]["prediction"] = (0.0,) * 4
    assert ledger.report() == actual


def test_should_charge_a_returned_control_before_refusing_its_wrong_native_action():
    ledger = ActionDecodingLedger(PickCubeBinaryGripperRecipe())
    ledger.select(0, PREDICTION, ledger.report())
    ledger.attempt(ledger.report())
    with pytest.raises(ValueError):
        ledger.returned(record(action=(0.25, -0.5, 0.0, 1.0)), ledger.report())
    actual = ledger.report()
    assert actual["work"]["control_attempts"] == actual["work"]["control_returns"] == 1
    assert actual["work"]["validated_control_returns"] == 0
    with pytest.raises(ValueError):
        ledger.select(1, PREDICTION, actual)
    assert ledger.report()["work"]["control_returns"] == 1


def test_should_permanently_refuse_a_rewritten_paid_public_receipt_even_if_repaired():
    ledger = ActionDecodingLedger(PickCubeBinaryGripperRecipe())
    row = accepted(ledger)
    original = ledger.report()
    changed = deepcopy(original)
    changed["controls"][0]["prediction"] = (0.0,) * 4
    with pytest.raises(ValueError):
        ledger.check(changed, (row,))
    with pytest.raises(ValueError):
        ledger.check(original, (row,))
    assert ledger.report()["work"]["control_returns"] == 1


def test_should_bound_decoding_and_native_controls_at_fifty_without_an_extra_paid_attempt():
    ledger = ActionDecodingLedger(PickCubeBinaryGripperRecipe())
    records: list[ContinuousTransition] = []
    for step in range(50):
        accepted(ledger, step, done=step == 49, records=records)
    before = ledger.report()
    assert len(before["controls"]) == before["work"]["control_returns"] == 50
    with pytest.raises(ValueError):
        ledger.select(50, PREDICTION, before)
    after = ledger.report()
    assert after["work"] == before["work"]
