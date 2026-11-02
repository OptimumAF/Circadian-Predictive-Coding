"""Descriptive native-command fidelity for bounded, already admitted TRAIN rows.

Consumes returned actions/targets and stage labels, never a learner or raw file.
No predictions, training, role authority, simulator, tuning or success decision.
Why: a small aggregate residual can still hide a wrong direction or zero command.
"""

from dataclasses import asdict, dataclass
import struct

import numpy as np

from src.core import continuous_bc as bc
from src.core.continuous_bc_diagnostics import _stage_labels

DEFINITION = "continuous_native_command_fidelity_v1"
UNITS = ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper")


@dataclass
class CommandFidelityWork:
    conversion_attempts: int = 0
    conversion_returns: int = 0
    equality_comparisons: int = 0
    absolute_error_coordinates: int = 0
    translation_direction_comparisons: int = 0
    gripper_margin_products: int = 0
    gripper_sign_classifications: int = 0
    known_owned_numeric_bytes: int = 0


def _native(values, work):
    result = np.empty_like(values)
    work.known_owned_numeric_bytes += result.nbytes
    for index in np.ndindex(values.shape):
        work.conversion_attempts += 1
        result[index] = struct.unpack("<f", struct.pack("<f", float(values[index])))[0]
        work.conversion_returns += 1
    return result


def _sign(value):
    return int(value > 0) - int(value < 0)


def _admit(predictions, targets, spec, stages, binary_gripper_axis, work):
    if type(work) is not CommandFidelityWork or any(
        type(value) is not int or value < 0 for value in asdict(work).values()
    ):
        raise ValueError("command work requires exact nonnegative native counters")
    owned = bc._spec(spec)
    if (
        owned.action_units != UNITS
        or owned.lower_bounds != (-1.0,) * 4
        or owned.upper_bounds != (1.0,) * 4
    ):
        raise ValueError("command fidelity requires four declared normalized root/gripper axes")
    if binary_gripper_axis is not None and (
        type(binary_gripper_axis) is not int or binary_gripper_axis != 3
    ):
        raise ValueError("endpoint gripper description requires exact axis3 or None")
    if type(predictions) is not np.ndarray or predictions.ndim != 2:
        raise ValueError("command predictions require exact finite float64 rows")
    labels = _stage_labels(stages, len(predictions))
    predicted = bc._targets(predictions, len(predictions), owned)
    work.known_owned_numeric_bytes += predicted.nbytes
    actual = bc._targets(targets, len(predicted), owned)
    work.known_owned_numeric_bytes += actual.nbytes
    bc._binary_gripper_targets(actual, owned, binary_gripper_axis)
    return predicted, actual, labels


def continuous_command_fidelity(
    predictions,
    targets,
    spec,
    stages,
    *,
    binary_gripper_axis=None,
    work: CommandFidelityWork | None = None,
) -> dict:
    """Describe rounded commands; no hard gripper threshold is applied to policy.

    Directions are negative/zero/positive after native32 conversion; both signed
    zeros compare equal. Endpoint target sign times ANALOG gripper command is a
    descriptive margin only. Counters accumulate, including partial conversion
    failures; stage indices refer to the original rows, not reordered subsets.
    """
    work = CommandFidelityWork() if work is None else work
    predicted, actual, labels = _admit(
        predictions, targets, spec, stages, binary_gripper_axis, work
    )
    native_predicted, native_actual = _native(predicted, work), _native(actual, work)
    rows = len(predicted)
    errors = np.empty_like(predicted)
    work.known_owned_numeric_bytes += errors.nbytes
    mismatch: list[int] = []
    directions: list[list[int]] = [[], [], []]
    margins = []
    negative, zero = [], []
    for row in range(rows):
        exact = True
        for axis in range(4):
            work.equality_comparisons += 1
            exact = bool(native_predicted[row, axis] == native_actual[row, axis]) and exact
            work.absolute_error_coordinates += 1
            errors[row, axis] = abs(native_predicted[row, axis] - native_actual[row, axis])
            if axis < 3:
                work.translation_direction_comparisons += 1
                if _sign(native_predicted[row, axis]) != _sign(native_actual[row, axis]):
                    directions[axis].append(row)
        if not exact:
            mismatch.append(row)
        if binary_gripper_axis is not None:
            work.gripper_margin_products += 1
            margin = float(native_actual[row, 3] * native_predicted[row, 3])
            margins.append(margin)
            work.gripper_sign_classifications += 1
            if margin < 0:
                negative.append(row)
            elif margin == 0:
                zero.append(row)
    stages_report = []
    for label in dict.fromkeys(labels):
        indices = [row for row, value in enumerate(labels) if value == label]
        stages_report.append(
            dict(
                label=label,
                row_indices=indices,
                native_mismatch_count=sum(row in mismatch for row in indices),
                translation_direction_mismatch_counts=[
                    sum(row in axis for row in indices) for axis in directions
                ],
                gripper_nonpositive_count=(
                    sum(margins[row] <= 0 for row in indices) if margins else None
                ),
            )
        )
    return dict(
        definition=DEFINITION,
        rows=rows,
        native_mismatch_rows=mismatch,
        native_axis_errors=[
            dict(
                axis=axis,
                mean_absolute_error=float(np.mean(errors[:, axis])),
                maximum_absolute_error=float(np.max(errors[:, axis])),
                maximum_error_row=int(np.argmax(errors[:, axis])),
            )
            for axis in range(4)
        ],
        translation_direction_mismatch_rows=directions,
        gripper=(
            dict(
                axis=3,
                meaning="target-sign times analog native command; no policy threshold",
                signed_native_margins=margins,
                positive_rows=rows - len(negative) - len(zero),
                zero_rows=zero,
                negative_rows=negative,
            )
            if binary_gripper_axis is not None
            else None
        ),
        stages=stages_report,
        work=asdict(work),
        work_scope="cumulative conversion/comparison/owned action-array payload; excludes validation, summaries, copies outside this helper, Python/JSON and RSS; no model arithmetic",
    )
