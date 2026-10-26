"""Paired disposable velocity intervention, composing the existing probe runner.

Inputs are an owned checkpoint, validated single-scope plan and runner callback.
Paired output contains reports and preparation costs; TRAIN preparation can
return a private velocity-only fork. No fit, RNG, optimizer math change,
promotion, final access or tuning.
This leaf does not import the probe module: its caller supplies that existing
runner, avoiding another orchestration implementation or circular dependency.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.continuous_learnability_modes import same_continuous_state
from src.adapters.numpy_continuous_bc import (
    NumpyContinuousBCLearner,
    _snapshot as validate_bc_snapshot,
)
from src.adapters.numpy_continuous_pc import (
    NumpyContinuousPCLearner,
    _snapshot as validate_pc_snapshot,
)


def _native_state(owner, kind, expected=None):
    if type(owner) is not kind or kind not in (
        NumpyContinuousBCLearner,
        NumpyContinuousPCLearner,
    ):
        raise ValueError("paired probe requires the unchanged exact continuous owner")
    state = (
        validate_pc_snapshot(owner._state, owner._policy)
        if isinstance(owner, NumpyContinuousPCLearner)
        else validate_bc_snapshot(owner._state, owner._policy)
    )
    if expected is not None and not same_continuous_state(expected, state):
        raise RuntimeError("paired probe changed checkpoint/control source")
    return state


def _observe_state(owner, kind, expected=None):
    if type(owner) is not kind:
        raise ValueError("paired probe owner type changed")
    observed, policy = owner.snapshot_state(), owner.policy
    actual = _native_state(owner, kind, expected)
    if observed.policy != policy or not same_continuous_state(observed, actual):
        raise RuntimeError("paired probe state changed during public observation")
    return actual


def _zero_velocity_snapshot(state, work):
    """Share the original four-array operation and its paid-prefix accounting."""
    velocity = []
    for value in state.velocity:
        work["velocity_zero_array_attempts"] += 1
        zeroed = np.zeros_like(value)
        velocity.append(zeroed)
        work["velocity_zero_array_returns"] += 1
        work["velocity_values_zeroed"] += zeroed.size
        work["velocity_numeric_bytes_zeroed"] += zeroed.nbytes
    return replace(state, velocity=tuple(velocity))


@dataclass(frozen=True)
class ZeroVelocityPreparation:
    report: dict[str, Any]
    candidate: NumpyContinuousBCLearner | None


def _bc_birth(owner):
    state, policy = owner._state, owner._policy
    return (
        policy,
        policy.observation_standardization,
        state,
        state.parameters,
        state.velocity,
        *state.parameters,
        *state.velocity,
    )


def _require_bc_birth(owner, birth, expected):
    if any(a is not b for a, b in zip(_bc_birth(owner), birth, strict=True)):
        raise RuntimeError("zero-velocity original BC owner was replaced")
    _native_state(owner, NumpyContinuousBCLearner, expected)


def _require_bc_disjoint(source, control):
    if any(
        np.shares_memory(first, second)
        for first in (*source._state.parameters, *source._state.velocity)
        for second in (*control._state.parameters, *control._state.velocity)
    ):
        raise RuntimeError("zero-velocity private model arrays alias original source")


def prepare_zero_velocity_bc(source, *, check_boundary=None) -> ZeroVelocityPreparation:
    """Own a private TRAIN BC velocity intervention without updates or renewal.

    The returned learner differs only in optimizer velocity. This is preparation,
    not learning, policy selection, collection or authority to train its data.
    """
    kind = NumpyContinuousBCLearner
    if type(source) is not kind:
        raise ValueError("zero-velocity TRAIN preparation requires exact native BC")
    if check_boundary is not None and not callable(check_boundary):
        raise ValueError("check_boundary must be callable or None")
    source_birth = _bc_birth(source)
    before = _native_state(source, kind)
    if type(before.policy.training_role) is not str or before.policy.training_role != "train":
        raise ValueError("zero-velocity preparation admits only a TRAIN source")
    work: dict[str, Any] = dict(
        source_fork_attempts=0,
        source_fork_returns=0,
        additional_source_owners=0,
        velocity_zero_array_attempts=0,
        velocity_zero_array_returns=0,
        velocity_values_zeroed=0,
        velocity_numeric_bytes_zeroed=0,
        restore_attempts=0,
        restore_returns=0,
        fits=0,
        parameter_initializations=0,
        seeded_normal_values=0,
        parameter_momentum_numeric_bytes_per_source=sum(
            value.nbytes for value in before.parameters + before.velocity
        ),
        scope="selected preparation payload/events; snapshots, validation, copies and Python/RSS additional",
    )
    report: dict[str, Any] = dict(
        definition="continuous_private_zero_velocity_bc_v1",
        complete=False,
        source_unchanged=False,
        initial_counts=dict(
            updates=before.completed_updates, training_rows=before.training_examples
        ),
        work=work,
    )
    control = control_birth = expected = None

    def pure():
        _require_bc_birth(source, source_birth, before)
        if control is not None:
            if type(control) is not kind or control is source:
                raise RuntimeError("zero-velocity control owner changed")
            _require_bc_birth(control, control_birth, expected)
            _require_bc_disjoint(source, control)

    def boundary():
        if check_boundary is not None:
            check_boundary()
        pure()
        _observe_state(source, kind, before)
        if control is not None:
            _observe_state(control, kind, expected)
        if check_boundary is not None:
            check_boundary()
        # A later public getter can change an earlier owner; close all owners
        # using native fields and the original identity pins after callbacks.
        pure()

    try:
        boundary()
        work["source_fork_attempts"] += 1
        control = source.fork()
        work["source_fork_returns"] += 1
        if type(control) is not kind or control is source:
            raise ValueError("zero-velocity preparation requires a distinct exact BC fork")
        work["additional_source_owners"] += 1
        control_birth = _bc_birth(control)
        expected = before
        pure()
        boundary()
        zero_expected = _zero_velocity_snapshot(expected, work)
        if any(np.any(value != 0.0) for value in zero_expected.velocity):
            raise RuntimeError("zero-velocity operation returned nonzero velocity")
        boundary()
        # Why: restore callbacks must not receive the independent expected state
        # and thereby redefine what clearing inherited velocity means.
        argument = validate_bc_snapshot(zero_expected, zero_expected.policy)
        work["restore_attempts"] += 1
        control.restore_state(argument)
        work["restore_returns"] += 1
        _require_bc_birth(source, source_birth, before)
        if (
            control._policy is not control_birth[0]
            or control._policy.observation_standardization is not control_birth[1]
        ):
            raise RuntimeError("zero-velocity restore changed original policy owner")
        _native_state(control, kind, zero_expected)
        control_birth = _bc_birth(control)
        expected = zero_expected
        boundary()
        report.update(complete=True, source_unchanged=True)
        pure()
        return ZeroVelocityPreparation(deepcopy(report), control)
    except Exception as error:
        report.update(
            complete=False, source_unchanged=False, error=f"{type(error).__name__}: {error}"
        )
        return ZeroVelocityPreparation(deepcopy(report), None)


def _require_completed(report, before, plan):
    """Validate the preceding returned success before admitting another scope."""
    updates = plan.updates
    counts = (
        report["source_updates"],
        report["source_training_examples"],
        report["progress"]["updates_completed"],
        report["progress"]["replay_examples_completed"],
        report["progress"]["next_tick"],
        *(point["updates"] for point in report["checkpoints"]),
    )
    if (
        any(type(value) is not int for value in counts)
        or report.get("complete") is not True
        or report.get("receipt_observation_complete") is not True
        or report["source_updates"] != before.completed_updates
        or report["source_training_examples"] != before.training_examples
        or type(report["progress"]["updates_completed"]) is not int
        or report["progress"]["updates_completed"] != updates
        or report["progress"]["replay_examples_completed"] != 0
        or report["progress"]["next_tick"] != 2 * updates
        or len(report["applied_updates"]) != updates
        or len(report["native_updates"]) != updates
        or len(report["events"]) != updates
        or not all(
            event["success_validated"] is True and event["deferred_exact"] is True
            for event in report["events"]
        )
        or len(report["checkpoints"]) != 2
        or not all(point["complete"] is True for point in report["checkpoints"])
        or [point["updates"] for point in report["checkpoints"]]
        != [before.completed_updates, before.completed_updates + updates]
    ):
        raise RuntimeError(
            "paired probe preceding inherited/control single-scope success is incomplete"
        )
    returned = [item for event in report["events"] for item in event["returned_receipts"]]
    if json.dumps(returned, sort_keys=True, allow_nan=False) != json.dumps(
        report["applied_updates"], sort_keys=True, allow_nan=False
    ):
        raise RuntimeError("paired probe preceding receipt sequence changed")
    for index, (receipt, native) in enumerate(zip(returned, report["native_updates"], strict=True)):
        expected = dict(
            update_number=index + 1,
            observed_at=2 * index,
            arrived_at=2 * index + 1,
            applied_at=2 * index + 1,
        )
        if (
            any(
                type(receipt[key]) is not int or receipt[key] != value
                for key, value in expected.items()
            )
            or receipt["episode_id"] != plan.training.episode_id
            or receipt["event_id"] != f"probe-label-{index}"
            or receipt["sample_id"] != f"probe-batch-{index}"
            or receipt["actor_version"] != "probe-actor"
            or receipt["learner_version"] != "probe-candidate"
            or native["diagnostic"] != receipt["diagnostic"]
            or type(native["training_rows"]) is not int
            or native["training_rows"] != len(plan.training.transitions)
        ):
            raise RuntimeError("paired probe preceding native receipt changed")


def _contrast(inherited, control):
    first, second = inherited["metrics"], control["metrics"]
    result = dict(
        direction="zero_velocity minus inherited; acquisition gains larger better, old losses smaller better",
        fresh_acquisition_gain=second["fresh_acquisition_gain"] - first["fresh_acquisition_gain"],
        old_retention_loss={
            key: second["old_retention_loss"][key] - value
            for key, value in first["old_retention_loss"].items()
        },
    )
    if "fresh_training_gain" in first:
        result["fresh_training_gain"] = second["fresh_training_gain"] - first["fresh_training_gain"]
    values = [result["fresh_acquisition_gain"], *result["old_retention_loss"].values()]
    if "fresh_training_gain" in result:
        values.append(result["fresh_training_gain"])
    if not all(math.isfinite(value) for value in values):
        raise ValueError("paired probe signed contrast is not finite")
    return result


def run_zero_velocity_pair(source, plan, *, training_readouts=(), check_boundary=None, run_probe):
    """Two independent local allowances; source global counters/wake never reset."""
    started = monotonic()
    kind = type(source)
    if kind not in (NumpyContinuousBCLearner, NumpyContinuousPCLearner):
        raise ValueError("paired probe requires an exact continuous source")
    if check_boundary is not None and not callable(check_boundary):
        raise ValueError("check_boundary must be callable or None")
    before = _observe_state(source, kind)
    control = prepared = None
    inherited_return = controlled_return = None
    work: dict[str, Any] = dict(
        source_fork_attempts=0,
        source_fork_returns=0,
        additional_source_owners=0,
        velocity_zero_array_attempts=0,
        velocity_zero_array_returns=0,
        velocity_values_zeroed=0,
        velocity_numeric_bytes_zeroed=0,
        restore_attempts=0,
        restore_returns=0,
        fits=0,
        parameter_initializations=0,
        seeded_normal_values=0,
        parameter_momentum_numeric_bytes_per_source=sum(
            array.nbytes for array in before.parameters + before.velocity
        ),
        scope="selected zeroing payload/events; each nested probe reports its own three extra owners and paid work; excludes full snapshot/gate validation, copies, elementwise, metadata, temporaries and process RSS",
    )
    report: dict[str, Any] = dict(
        complete=False,
        definition="continuous_zero_velocity_pair_v1",
        source_updates=before.completed_updates,
        source_training_examples=before.training_examples,
        inherited=None,
        zero_velocity=None,
        preparation_work=work,
        initial_scores_exact=False,
    )

    def boundary():
        if check_boundary is not None:
            check_boundary()
        _observe_state(source, kind, before)
        if control is not None:
            _observe_state(control, kind, prepared)
        if check_boundary is not None:
            check_boundary()
        # Why: the last public observer may mutate a different owner. Finish
        # all owners from their native fields after every public callback.
        _native_state(source, kind, before)
        if control is not None:
            _native_state(control, kind, prepared)
        # Keep the originally observed paid report even when a later callback
        # replaces the runner's returned object. Metadata only; no model work.
        for name, returned in (
            ("inherited", inherited_return),
            ("zero_velocity", controlled_return),
        ):
            if returned is not None and json.dumps(
                returned, sort_keys=True, allow_nan=False
            ) != json.dumps(report[name], sort_keys=True, allow_nan=False):
                raise RuntimeError(f"paired probe preceding {name} evidence changed")

    try:
        boundary()
        inherited_return = run_probe(
            source, plan, training_readouts=training_readouts, check_boundary=boundary
        )
        inherited = report["inherited"] = deepcopy(inherited_return)
        boundary()
        _require_completed(inherited, before, plan)
        work["source_fork_attempts"] += 1
        control = source.fork()
        work["source_fork_returns"] += 1
        if control is source or type(control) is not kind:
            raise ValueError("zero-velocity control requires a detached exact source fork")
        work["additional_source_owners"] += 1
        prepared = _observe_state(control, kind, before)
        boundary()
        prepared = _zero_velocity_snapshot(prepared, work)
        # The callback never receives our private expected state: an otherwise
        # coherent mutation of its argument must not redefine the intervention.
        if isinstance(control, NumpyContinuousPCLearner):
            pc_argument = validate_pc_snapshot(prepared, prepared.policy)
            work["restore_attempts"] += 1
            control.restore_state(pc_argument)
        else:
            bc_argument = validate_bc_snapshot(prepared, prepared.policy)
            work["restore_attempts"] += 1
            control.restore_state(bc_argument)
        work["restore_returns"] += 1
        boundary()
        controlled_return = run_probe(
            control, plan, training_readouts=training_readouts, check_boundary=boundary
        )
        controlled = report["zero_velocity"] = deepcopy(controlled_return)
        boundary()
        _require_completed(controlled, prepared, plan)
        if inherited["checkpoints"][0] != controlled["checkpoints"][0]:
            raise RuntimeError("velocity-only control changed initial physical score checkpoint")
        report["initial_scores_exact"] = True
        report["contrast"] = _contrast(inherited, controlled)
        boundary()
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["elapsed_seconds"] = monotonic() - started
    return report
