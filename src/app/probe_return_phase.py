"""Admit fixed old-TRAIN return inputs and protect the preceding paid probe.

Inputs are the original bounded probe/monitor cohorts and two independent TRAIN
templates. Helpers validate the combined envelope and exact corresponding data,
then seal the existing small report while the caller reuses its learner loop.
No learner, updates, objective, budget renewal, persistence or final-data access
is supplied here.
"""

from copy import deepcopy
from operator import is_, itemgetter
from typing import Any, Callable

import numpy as np

from src.app.coupled_attribution import AttributionBatch, _same_value, _validate_batch


def budget_binding(budget) -> tuple[Any, ...]:
    """Keep original controls fixed while measured clocks and RSS may advance."""
    return (
        id(budget.budget),
        id(budget.clock),
        budget.started_at,
        id(budget.process_rss_sampler),
        id(budget.progress),
    )


def validate_return_envelope(
    training, evaluation, monitors, templates, *, short_affine_return=False
) -> None:
    if type(short_affine_return) is not bool:
        raise ValueError("short_affine_return must be an exact bool")
    if type(templates) is not tuple or len(templates) != 2:
        raise ValueError("return_training requires exactly two TRAIN templates")
    if (
        type(monitors) is not tuple
        or len(monitors) != 2
        or len(training) != (2 if short_affine_return else 16)
        or len(evaluation) != 2
    ):
        raise ValueError(
            "A return requires sixteen F updates, two F evaluations and two TRAIN readouts"
        )
    if short_affine_return and any(
        type(b) is not AttributionBatch
        or type(b.features) is not np.ndarray
        or b.features.dtype != np.float64
        or b.features.shape != (rows, 296)
        or type(b.targets) is not np.ndarray
        or b.targets.dtype != np.float64
        or b.targets.shape != (rows, 1)
        or type(b.num_classes) is not int
        or b.num_classes != 7
        or b.episode is not None
        for batches, rows in ((training, 9), (evaluation, 9), (monitors, 18), (templates, 18))
        for b in batches
    ):
        raise ValueError("short affine return requires fixed raw296/K7 F9/oldTRAIN18 rows")
    original = (*training, *evaluation, *monitors)
    arrays = [array for batch in original for array in (batch.features, batch.targets)]
    for template, monitor in zip(templates, monitors, strict=True):
        if (
            type(template) is not AttributionBatch
            or type(template.role) is not str
            or template.role != "train"
            or type(template.features) is not np.ndarray
            or template.features.dtype != np.float64
            or template.features.shape != monitor.features.shape
            or type(template.targets) is not np.ndarray
            or template.targets.dtype != np.float64
            or template.targets.shape != monitor.targets.shape
            or template.num_classes != monitor.num_classes
            or (template.episode is None) != (monitor.episode is None)
        ):
            raise ValueError(
                "return templates must match corresponding TRAIN readout shapes/classes/episode mode"
            )
        arrays.extend((template.features, template.targets))
    if sum(array.size for array in arrays) > 65536:
        raise ValueError("probe, monitor and return arrays exceed 65536 aggregate entries")
    if any(
        np.shares_memory(left, right)
        for index, left in enumerate(arrays)
        for right in arrays[index + 1 :]
    ):
        raise ValueError("return inputs require disjoint storage from all probe and monitor arrays")


def validate_return_values(training, evaluation, monitors, templates) -> None:
    seen = {row_id for batch in (*training, *evaluation, *monitors) for row_id in batch.row_ids}
    for template, monitor in zip(templates, monitors, strict=True):
        _validate_batch(template, "train", seen)
        if (
            type(template.context) is not str
            or template.context != monitor.context
            or not np.array_equal(template.features, monitor.features)
            or not np.array_equal(template.targets, monitor.targets)
            or (
                template.episode is not None
                and monitor.episode is not None
                and not _same_value(template.episode.transitions, monitor.episode.transitions)
            )
        ):
            raise ValueError(
                "return template values must equal corresponding original TRAIN readouts"
            )


def _report_containers(value: Any, path: tuple[Any, ...] = ()):
    """Only this already bounded scalar report is sealed, never opaque state."""
    if type(value) in (dict, list, tuple):
        yield path, value
        items = value.items() if type(value) is dict else enumerate(value)
        for key, child in items:
            yield from _report_containers(child, (*path, key))


def _same_scalar_report(left: Any, right: Any, owners: dict[tuple[Any, ...], Any]) -> bool:
    """Prove pure scalar report values and original container owners together.

    False requests the original general proof, including its error precedence.
    Opaque comparisons can execute code, so they must keep the two-pass proof.
    """
    visited = 0

    def match(actual: Any, expected: Any, path: tuple[Any, ...]) -> bool:
        nonlocal visited
        kind = type(actual)
        if kind is not type(expected):
            return False
        if kind is str or kind is int or kind is float or kind is bool or actual is None:
            return bool(actual == expected)
        if kind is not dict and kind is not list and kind is not tuple:
            return False
        if path not in owners or owners[path] is not actual:
            return False
        visited += 1
        if kind is dict:
            # Check keys before equality/lookup: custom keys may execute code.
            if any(type(key) is not str for key in actual) or any(
                type(key) is not str for key in expected
            ):
                return False
            return actual.keys() == expected.keys() and all(
                match(actual[key], expected[key], (*path, key)) for key in actual
            )
        return len(actual) == len(expected) and all(
            match(a, b, (*path, index)) for index, (a, b) in enumerate(zip(actual, expected))
        )

    # Missing original paths must fall back too, even if the caller changed
    # both returned expected values and report values to the same new shape.
    return match(left, right, ()) and visited == len(owners)


def _bind_scalar_report_check(report, expected, owners) -> Callable[[], bool] | None:
    """Bind structural reads, never a cached validity result.

    Why: repeated Python recursion over every numeric game feature dominates
    large paid reports. Builtin scalar leaves are immutable: reading every child
    identity proves the sealed values and owners still hold, without repeating
    nested equality. Any drift falls back to the original dynamic proof.
    """
    # Deepcopy can execute opaque user code before owners are recorded. Prove
    # initial equality too, even if that code leaves only builtin values behind.
    # This also keeps NaN and all opaque leaves on the original path.
    if not _same_scalar_report(report, expected, dict(owners)):
        return None
    plans = []
    values: Callable[..., Any]
    for path, actual in owners:
        trusted = expected
        for key in path:
            trusted = trusted[key]
        kind = type(actual)
        if kind is not dict and kind is not list and kind is not tuple:
            return None
        if kind is dict:
            keys = tuple(actual)
            if any(type(key) is not str for key in keys):
                return None
            get: Any = itemgetter(*keys) if keys else None
            if len(keys) == 1:
                values = lambda owner, get=get: (get(owner),)
            elif keys:
                values = get
            else:
                values = lambda owner: ()
            key_set = frozenset(keys)
        else:
            values, key_set = tuple, None
        original_values, trusted_values = values(actual), values(trusted)
        plans.append((actual, trusted, kind, values, key_set, original_values, trusted_values))

    def check() -> bool:
        for actual, trusted, kind, values, key_set, original_values, baseline in plans:
            if type(actual) is not kind or type(trusted) is not kind:
                return False
            if kind is dict:
                if (
                    any(type(key) is not str for key in actual)
                    or any(type(key) is not str for key in trusted)
                    or actual.keys() != key_set
                    or trusted.keys() != key_set
                ):
                    return False
            elif len(actual) != len(original_values) or len(trusted) != len(baseline):
                return False
            actual_values, trusted_values = values(actual), values(trusted)
            if not all(map(is_, actual_values, original_values)) or not all(
                map(is_, trusted_values, baseline)
            ):
                return False
        return True

    return check


def paid_report_guard(report: dict[str, Any]) -> tuple[Callable[[], None], dict[str, Any]]:
    """Refuse value edits and equal-valued container replacements in paid evidence."""
    expected = deepcopy(report)
    owners = tuple(_report_containers(report))
    # Why: bounded scalar game reports dominate measured guard cost. Index only
    # primitive paths; hashing an opaque key would add a new external callback.
    scalar_owners = (
        dict(owners)
        if all(type(key) is str or type(key) is int for path, _ in owners for key in path)
        else None
    )
    scalar_check = (
        _bind_scalar_report_check(report, expected, owners) if scalar_owners is not None else None
    )

    def guard() -> None:
        if scalar_check is not None and scalar_check():
            return
        if scalar_owners is not None and _same_scalar_report(report, expected, scalar_owners):
            return
        if not _same_value(report, expected):
            raise ValueError("preceding paid probe report changed during A return")
        for path, owner in owners:
            current: Any = report
            for key in path:
                current = current[key]
            if current is not owner:
                raise ValueError("preceding paid probe report was replaced during A return")

    return guard, expected


def validate_return_zero(original, returned, monitors) -> None:
    previous = original["checkpoints"][-1]
    old_monitor = original["training_readouts"]["checkpoints"][-1]
    current_monitor = monitors.checkpoints[-1]
    for before, after in ((previous, returned.checkpoints[-1]), (old_monitor, current_monitor)):
        if not _same_value(before["scores"], after["scores"]) or not _same_value(
            before.get("decision_diagnostics"), after.get("decision_diagnostics")
        ):
            raise ValueError("return0 predictions differ from the paid F16/A16 readouts")
