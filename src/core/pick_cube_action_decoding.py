"""Explicit binary-gripper serving of an already computed normalized command.

Preserve continuous XYZ and choose an endpoint for axis3 using one fixed rule.
No learner, labels, threshold fitting, simulator, native rounding or IO occurs.
Soft continuous serving remains a separate valid policy.
"""

from dataclasses import dataclass
import math


DEFINITION = "pick_cube_binary_gripper_zero_open_v1"


@dataclass(frozen=True)
class PickCubeBinaryGripperRecipe:
    definition: str = DEFINITION

    def __post_init__(self) -> None:
        if type(self.definition) is not str or self.definition != DEFINITION:
            raise ValueError("binary gripper recipe requires the fixed zero-open definition")


def validate_gripper_recipe(value: PickCubeBinaryGripperRecipe) -> PickCubeBinaryGripperRecipe:
    if type(value) is not PickCubeBinaryGripperRecipe or set(vars(value)) != {"definition"}:
        raise ValueError("expected an exact binary gripper recipe with its original field")
    return PickCubeBinaryGripperRecipe(value.definition)


def _normalized_action(action: tuple[float, ...]) -> tuple[float, ...]:
    if (
        type(action) is not tuple
        or len(action) != 4
        or any(type(value) is not float or not math.isfinite(value) for value in action)
        or any(not -1.0 <= value <= 1.0 for value in action)
    ):
        raise ValueError("decoding requires exactly four finite native floats in[-1,+1]")
    return tuple(value for value in action)


def decode_gripper_action(
    action: tuple[float, ...], recipe: PickCubeBinaryGripperRecipe
) -> tuple[float, ...]:
    """Keep XYZ exactly; negative grip closes, while either signed zero opens."""
    validate_gripper_recipe(recipe)
    owned = _normalized_action(action)
    return (*owned[:3], -1.0 if owned[3] < 0.0 else 1.0)
