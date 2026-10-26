"""Bounded continuous-action residuals and MLP conditioning summaries.

Inputs are owned float64 actions, declared controller bounds and row stage labels.
Conditioning additionally reads standardized features and actual objective
gradients/activations. Outputs are immutable scalar summaries, not model updates or evaluation
selection. No learner, role conversion, simulator, RNG, persistence or IO here.
"""

from dataclasses import asdict, dataclass
import math
import struct

import numpy as np

from src.core import continuous_bc as bc
from src.core.continuous_demonstration import ContinuousActionSpec

SATURATION_THRESHOLD = 0.99
INPUT_TARGET_CONSISTENCY_DEFINITION = "continuous_input_target_exact_consistency_v1"


@dataclass(frozen=True)
class ValueMagnitude:
    values: int
    maximum_absolute: float
    rms: float
    zero_values: int


@dataclass(frozen=True)
class TanhSaturation:
    values: int
    saturated_values: int
    saturated_fraction: float
    minimum_derivative: float
    mean_derivative: float


@dataclass(frozen=True)
class AxisSaturation:
    index: int
    unit: str
    active: bool
    saturation: TanhSaturation | None


@dataclass(frozen=True)
class GradientSummary:
    name: str
    values: int
    l2_norm: float
    maximum_absolute: float
    zero_values: int


@dataclass(frozen=True)
class ConditioningWork:
    forward_passes: int
    backward_passes: int
    input_rows: int
    input_dimension: int
    hidden_dimension: int
    output_dimension: int
    input_values: int
    hidden_values: int
    output_values: int
    parameter_values: int
    gradient_values: int
    gradient_numeric_bytes: int
    dense_multiply_accumulates: int


@dataclass(frozen=True)
class ContinuousBCConditioning:
    rows: int
    feature_magnitude: ValueMagnitude
    hidden_saturation: TanhSaturation
    output_saturation: tuple[AxisSaturation, ...]
    objective_definition: str
    objective_loss: float
    gradients: tuple[GradientSummary, ...]
    work: ConditioningWork


@dataclass(frozen=True)
class TrainingConditioningDiagnostic:
    conditioning: ContinuousBCConditioning
    standardization_calls: int
    standardization_rows: int
    standardization_values: int
    standardization_applied: bool


def _magnitude(values: bc.Array) -> ValueMagnitude:
    maximum = float(np.max(np.abs(values)))
    # Why: squaring finite inputs directly can overflow even when RMS is finite.
    rms = maximum * math.sqrt(float(np.mean((values / maximum) ** 2))) if maximum else 0.0
    if not math.isfinite(rms):
        raise FloatingPointError("diagnostic RMS is not representable")
    return ValueMagnitude(values.size, maximum, rms, int(np.count_nonzero(values == 0)))


def _gradient_summary(name: str, values: bc.Array) -> GradientSummary:
    magnitude = _magnitude(values)
    maximum = magnitude.maximum_absolute
    norm = maximum * math.sqrt(float(np.sum((values / maximum) ** 2))) if maximum else 0.0
    if not math.isfinite(norm):
        raise FloatingPointError("diagnostic gradient L2 norm is not representable")
    return GradientSummary(name, values.size, norm, maximum, magnitude.zero_values)


def _saturation(values: bc.Array) -> TanhSaturation:
    derivative = 1 - values**2
    count = int(np.count_nonzero(np.abs(values) >= SATURATION_THRESHOLD))
    return TanhSaturation(
        values.size,
        count,
        count / values.size,
        float(np.min(derivative)),
        float(np.mean(derivative)),
    )


def continuous_bc_conditioning(
    features: bc.Array,
    targets: bc.Array,
    parameters: bc.Parameters,
    spec: ContinuousActionSpec,
    *,
    binary_gripper_axis: int | None = None,
) -> ContinuousBCConditioning:
    """Describe standardized inputs and actual objective derivatives, without updates.

    Two forwards/one backward reuse the working kernels without refactoring them.
    Dense MACs exclude elementwise math/reductions/validation/copies. Tanh output
    saturation is descriptive even when binary CE bypasses that loss derivative.
    Successful work counts do not imply a failed/unreturned call consumed no work.
    """
    owned_spec = bc._spec(spec)
    values = bc._features(features, owned_spec)
    actions = bc._targets(targets, len(values), owned_spec)
    owned = bc._parameters(parameters, owned_spec)
    bc._binary_gripper_targets(actions, owned_spec, binary_gripper_axis)
    result = bc.continuous_bc_loss_and_gradients(
        values, actions, owned, owned_spec, binary_gripper_axis=binary_gripper_axis
    )
    hidden, normalized, _, _ = bc._forward(values, owned, owned_spec)
    _, _, _, _, active = bc._geometry(owned_spec)
    rows, dimension = values.shape
    width, axes = owned[0].shape[1], len(owned_spec.action_units)
    parameter_values = sum(value.size for value in owned)
    outputs = tuple(
        AxisSaturation(
            index,
            unit,
            bool(active[index]),
            _saturation(normalized[:, index]) if active[index] else None,
        )
        for index, unit in enumerate(owned_spec.action_units)
    )
    gradients = tuple(
        _gradient_summary(name, value)
        for name, value in zip(("w1", "b1", "w2", "b2"), result.gradients, strict=True)
    )
    return ContinuousBCConditioning(
        rows,
        _magnitude(values),
        _saturation(hidden),
        outputs,
        bc.OBJECTIVE if binary_gripper_axis is None else bc.MIXED_OBJECTIVE,
        result.loss,
        gradients,
        ConditioningWork(
            2,
            1,
            rows,
            dimension,
            width,
            axes,
            values.size,
            hidden.size,
            normalized.size,
            parameter_values,
            parameter_values,
            parameter_values * 8,
            rows * (3 * dimension * width + 4 * width * axes),
        ),
    )


@dataclass(frozen=True)
class AxisResidual:
    index: int
    unit: str
    active: bool
    normalized_bias: float
    normalized_mae: float
    normalized_rmse: float
    normalized_half_mse: float
    maximum_absolute_normalized_error: float


@dataclass(frozen=True)
class ResidualSummary:
    rows: int
    active_axes: int
    normalized_half_mse: float
    axes: tuple[AxisResidual, ...]


@dataclass(frozen=True)
class StageResidual:
    label: str
    row_indices: tuple[int, ...]
    summary: ResidualSummary


@dataclass(frozen=True)
class ContinuousBCResiduals:
    overall: ResidualSummary
    stages: tuple[StageResidual, ...]


@dataclass(frozen=True)
class TrainingResidualDiagnostic:
    residuals: ContinuousBCResiduals
    prediction_calls: int
    prediction_rows: int
    prediction_values: int
    prediction_bytes: int


def _stage_labels(stages: tuple[str, ...], rows: int) -> tuple[str, ...]:
    if type(stages) is not tuple or len(stages) != rows or not 1 <= rows <= 256:
        raise ValueError("stages must be an exact tuple matching1..256 action rows")
    for label in stages:
        if type(label) is not str or not label or label != label.strip():
            raise ValueError("stage labels must be nonempty native text without edge whitespace")
        try:
            size = len(label.encode("utf8"))
        except UnicodeEncodeError as error:
            raise ValueError("stage labels must be valid UTF-8") from error
        if size > 32:
            raise ValueError("stage labels exceed32UTF8bytes")
    if len(set(stages)) > 16:
        raise ValueError("at most16 distinct stage labels are supported")
    return tuple(stages)


def _summary(errors, spec, active) -> ResidualSummary:
    axes = []
    for index, unit in enumerate(spec.action_units):
        values = errors[:, index]
        squared_mean = float(np.mean(values * values))
        axes.append(
            AxisResidual(
                index,
                unit,
                bool(active[index]),
                float(np.mean(values)),
                float(np.mean(np.abs(values))),
                float(np.sqrt(squared_mean)),
                0.5 * squared_mean,
                float(np.max(np.abs(values))),
            )
        )
    count = int(np.count_nonzero(active))
    objective = float(0.5 * np.mean(errors[:, active] ** 2)) if count else 0.0
    return ResidualSummary(len(errors), count, objective, tuple(axes))


def continuous_bc_residuals(
    predictions: bc.Array,
    targets: bc.Array,
    spec: ContinuousActionSpec,
    stages: tuple[str, ...],
) -> ContinuousBCResiduals:
    """Summarize all supplied physical actions; positive bias means overprediction.

    Half-MSE uses active axes only. Constant axes report zero with active=False.
    Normalization is of physical outputs; finite-precision output rounding may
    differ from the learner's latent tanh objective at extreme narrow bounds.
    """
    owned = bc._spec(spec)
    if type(predictions) is not np.ndarray or predictions.ndim != 2:
        raise ValueError("predictions must be a finite float64 action matrix")
    rows = len(predictions)
    labels = _stage_labels(stages, rows)
    predicted = bc._targets(predictions, rows, owned)
    actual = bc._targets(targets, rows, owned)
    _, _, center, half, active = bc._geometry(owned)
    errors = np.zeros_like(predicted)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        # Why: subtracting opposite physical endpoints can overflow. Normalize
        # each bounded action first, then subtract dimensionless coordinates.
        errors[:, active] = (predicted[:, active] - center[active]) / half[active] - (
            actual[:, active] - center[active]
        ) / half[active]
    groups = []
    for label in dict.fromkeys(labels):
        indices = tuple(i for i, value in enumerate(labels) if value == label)
        groups.append(StageResidual(label, indices, _summary(errors[list(indices)], owned, active)))
    return ContinuousBCResiduals(_summary(errors, owned, active), tuple(groups))


@dataclass
class _InputTargetConsistencyWork:
    native_conversion_attempts: int = 0
    native_conversion_returns: int = 0
    pair_attempts: int = 0
    pair_returns: int = 0
    input_coordinate_differences: int = 0
    target_coordinate_differences: int = 0
    native_target_coordinate_differences: int = 0
    nearest_comparisons: int = 0


@dataclass(frozen=True)
class InputTargetPair:
    """Signed target deltas are second-row minus first-row, before and after native32."""

    first_row: int
    second_row: int
    input_l2: float
    target_delta: tuple[float, ...]
    native_target_delta: tuple[float, ...]
    target_l2: float
    native_target_l2: float
    duplicate_input: bool
    raw_target_conflict: bool
    native_target_conflict: bool
    opposite_gripper: bool


@dataclass(frozen=True)
class ContinuousInputTargetConsistency:
    definition: str
    rows: int
    input_dimension: int
    action_dimension: int
    pairs: tuple[InputTargetPair, ...]
    nearest_other_rows: tuple[int | None, ...]
    duplicate_input_pairs: int
    raw_conflict_pairs: int
    native_conflict_pairs: int
    opposite_gripper_pairs: int
    work: dict[str, int | str]


def _native_consistency_targets(actions, work):
    encoded = []
    for row in actions:
        values = []
        for value in row:
            work.native_conversion_attempts += 1
            try:
                native = struct.unpack("!f", struct.pack("!f", float(value)))[0]
            except (OverflowError, struct.error) as error:
                raise FloatingPointError("target is unrepresentable as native float32") from error
            if not math.isfinite(native):
                raise FloatingPointError("native float32 target is nonfinite")
            work.native_conversion_returns += 1
            values.append(native)
        encoded.append(tuple(values))
    return tuple(encoded)


def _consistency_delta(first, second, work, counter):
    differences = []
    for left, right in zip(first, second, strict=True):
        setattr(work, counter, getattr(work, counter) + 1)
        value = float(right) - float(left)
        if not math.isfinite(value):
            raise FloatingPointError("input/target coordinate difference is unrepresentable")
        differences.append(value)
    return tuple(differences)


def _consistency_norm(differences):
    result = math.hypot(*differences)
    if not math.isfinite(result):
        raise FloatingPointError("input/target L2 norm is unrepresentable")
    return result


def continuous_input_target_consistency(
    features: bc.Array,
    targets: bc.Array,
    spec: ContinuousActionSpec,
    *,
    _work: _InputTargetConsistencyWork | None = None,
) -> ContinuousInputTargetConsistency:
    """Keep every pair; exact duplicate inputs and rounding conflicts are separate.

    Why: teacher completion may succeed while the fitted policy fails. Check
    supervision ambiguity before admitting further training, without choosing a
    proximity threshold or treating rounded distance zero as exact input equality.
    Supplied work accumulates without reset and remains visible on any failure.
    """
    work = _InputTargetConsistencyWork() if _work is None else _work
    if type(work) is not _InputTargetConsistencyWork or any(
        type(value) is not int or value < 0 for value in asdict(work).values()
    ):
        raise ValueError("consistency work requires exact nonnegative native counters")
    owned = bc._spec(spec)
    if (
        len(owned.action_units) != 4
        or owned.lower_bounds != (-1.0,) * 4
        or owned.upper_bounds != (1.0,) * 4
    ):
        raise ValueError(
            "input/target consistency requires four normalized(-1,+1) axes/gripperaxis3"
        )
    if type(features) is not np.ndarray or features.ndim != 2 or not 1 <= len(features) <= 100:
        raise ValueError("input/target consistency requires1..100 feature rows")
    values = bc._features(features, owned)
    actions = bc._targets(targets, len(values), owned)
    bc._binary_gripper_targets(actions, owned, 3)
    native = _native_consistency_targets(actions, work)
    rows = len(values)
    pairs = []
    nearest: list[tuple[float, int] | None] = [None] * rows
    for first in range(rows):
        for second in range(first + 1, rows):
            work.pair_attempts += 1
            distance = _consistency_norm(
                _consistency_delta(
                    values[first], values[second], work, "input_coordinate_differences"
                )
            )
            target_delta = _consistency_delta(
                actions[first], actions[second], work, "target_coordinate_differences"
            )
            native_delta = _consistency_delta(
                native[first], native[second], work, "native_target_coordinate_differences"
            )
            duplicate = bool(np.array_equal(values[first], values[second]))
            raw_difference = not bool(np.array_equal(actions[first], actions[second]))
            native_difference = native[first] != native[second]
            pair = InputTargetPair(
                first,
                second,
                distance,
                target_delta,
                native_delta,
                _consistency_norm(target_delta),
                _consistency_norm(native_delta),
                duplicate,
                duplicate and raw_difference,
                duplicate and native_difference,
                bool(actions[first, 3] != actions[second, 3]),
            )
            pairs.append(pair)
            work.pair_returns += 1
            for row, other in ((first, second), (second, first)):
                work.nearest_comparisons += 1
                choice = (distance, other)
                previous = nearest[row]
                if previous is None or choice < previous:
                    nearest[row] = choice
    counts: dict[str, int | str] = dict(asdict(work))
    counts.update(
        owned_input_numeric_bytes=values.nbytes,
        owned_target_numeric_bytes=actions.nbytes,
        logical_owned_array_bytes=values.nbytes + actions.nbytes,
        native_target_float32_numeric_bytes=rows * 4 * 4,
        owned_native_target_numeric_bytes=rows * 4 * 8,
        serialized_pair_scalar_values=17 * len(pairs),
        serialized_nearest_row_values=rows,
        scope="supplied counters accumulate; payload sizes describe this call; decoded native32 targets use logical float64 values; excludes validation, copies, Python containers, JSON keys and RSS",
    )
    return ContinuousInputTargetConsistency(
        INPUT_TARGET_CONSISTENCY_DEFINITION,
        rows,
        owned.observation_dim,
        4,
        tuple(pairs),
        tuple(None if value is None else value[1] for value in nearest),
        sum(pair.duplicate_input for pair in pairs),
        sum(pair.raw_target_conflict for pair in pairs),
        sum(pair.native_target_conflict for pair in pairs),
        sum(pair.opposite_gripper for pair in pairs),
        counts,
    )
