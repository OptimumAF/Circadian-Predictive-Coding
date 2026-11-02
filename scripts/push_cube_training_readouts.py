"""Bounded TRAIN-only conditioning and physical residual readouts.

Own fixed four-group, corrective five-group or explicit aggregated six-group TRAIN
inputs through the
existing detached phase boundary. This helper
neither trains nor initializes a learner, selects examples, fits statistics or runs
physics. Objective losses are native objective-specific; physical residuals are
reported separately so a mixed loss is never presented as MSE.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any

import numpy as np

from src.adapters.continuous_learnability_modes import (
    diagnose_conditioning,
    diagnose_residual,
    same_continuous_state,
)
from src.adapters.continuous_phase_readouts import PhaseReadouts
from src.adapters.numpy_continuous_bc import (
    NumpyContinuousBCLearner,
    _snapshot,
)
from src.core import continuous_bc as bc
from src.core import continuous_command_fidelity as commands
from src.app.probe_return_phase import paid_report_guard
from scripts.push_cube_correction_weights import COMPLETE_B22_GROUPS, _own_complete_b22_groups
from src.core.continuous_demonstration import (
    ContinuousEpisode,
    copy_continuous_episode,
)

GROUPS = ("push", "a", "b", "train20")
CORRECTIVE_GROUPS = ("corrections", "push", "a", "b", "train20")
AGGREGATED_GROUPS = ("current_corrections", "corrections", "push", "a", "b", "train20")
DEFINITION = "continuous_per_group_training_readouts_v1"


def _json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _native_state(learner: NumpyContinuousBCLearner):
    return _snapshot(learner._state, learner._policy)


def _birth(learner: NumpyContinuousBCLearner) -> tuple[Any, ...]:
    state = learner._state
    return (
        learner._policy,
        state,
        state.parameters,
        state.velocity,
        *state.parameters,
        *state.velocity,
    )


def _require_owner(learner, birth, state) -> None:
    current = _birth(learner)
    if any(a is not b for a, b in zip(current, birth, strict=True)):
        raise RuntimeError("TRAIN readout native owner was replaced")
    if not same_continuous_state(_native_state(learner), state):
        raise RuntimeError("TRAIN readout native state changed")


def _admit(
    learner,
    groups,
    *,
    corrective: bool = False,
    aggregated: bool = False,
    complete_b22: bool = False,
):
    if type(corrective) is not bool:
        raise ValueError("corrective must be an exact bool")
    if type(aggregated) is not bool:
        raise ValueError("aggregated must be an exact bool")
    if aggregated and not corrective:
        raise ValueError("aggregated readouts require corrective=True")
    if type(complete_b22) is not bool or (complete_b22 and not (corrective and aggregated)):
        raise ValueError("complete_b22 requires an exact bool and corrective+aggregated scopes")
    if complete_b22:
        _own_complete_b22_groups(groups)
    names = (
        COMPLETE_B22_GROUPS
        if complete_b22
        else AGGREGATED_GROUPS
        if aggregated
        else CORRECTIVE_GROUPS
        if corrective
        else GROUPS
    )
    maximum = 220 if complete_b22 else 198 if aggregated else 148 if corrective else 130
    if type(learner) is not NumpyContinuousBCLearner:
        raise ValueError("readouts require an exact native continuous BP learner")
    state = _native_state(learner)
    policy = state.policy
    axis = policy.binary_gripper_axis
    if (
        type(policy.training_role) is not str
        or policy.training_role != "train"
        or policy.spec.observation_dim > 64
        or policy.hidden_dim > 64
        or not (axis is None or (type(axis) is int and axis == 3))
    ):
        raise ValueError("readouts require bounded TRAIN common or axis3 mixed BP")
    if type(groups) is not tuple or len(groups) != len(names):
        raise ValueError(
            "groups must be an exact six-group tuple"
            if aggregated
            else "groups must be an exact five-group tuple"
            if corrective
            else "groups must be an exact four-group tuple"
        )
    owned = []
    for expected, item in zip(names, groups, strict=True):
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or item[0] != expected
            or type(item[1]) is not ContinuousEpisode
        ):
            raise ValueError(
                "groups must be ordered current_corrections, corrections, push, a, b, train20 episodes"
                if aggregated
                else "groups must be ordered corrections, push, a, b, train20 episodes"
                if corrective
                else "groups must be ordered push, a, b, train20 episodes"
            )
        episode = item[1]
        if type(episode.role) is not str or episode.role != "train" or episode.spec != policy.spec:
            raise ValueError("readout episodes must be TRAIN with the learner spec")
        episode = copy_continuous_episode(episode)
        if not 1 <= len(episode.transitions) <= maximum:
            raise ValueError(f"each readout group must contain1..{maximum} rows")
        owned.append((expected, episode))
    if corrective and len({episode.episode_id for _, episode in owned}) != len(owned):
        raise ValueError("corrective readout groups require distinct episode IDs")
    if sum(len(episode.transitions) for _, episode in owned) > maximum:
        raise ValueError(f"readout groups exceed{maximum} total rows")
    return tuple(owned), state


def _work() -> dict[str, Any]:
    counters = (
        "conditioning_attempts",
        "conditioning_returns",
        "conditioning_attempted_rows",
        "conditioning_returned_rows",
        "residual_attempts",
        "residual_returns",
        "residual_attempted_rows",
        "residual_returned_rows",
        "forward_passes",
        "forward_rows",
        "gradient_evaluations",
        "gradient_rows",
        "gradient_values",
        "gradient_numeric_bytes",
        "normalization_calls",
        "normalization_rows",
        "normalization_values",
        "prediction_calls",
        "prediction_rows",
        "prediction_values",
        "prediction_bytes",
        "known_dense_multiply_accumulates",
    )
    return dict.fromkeys(counters, 0) | {
        "unknown_unreturned_work": False,
        "known_math_source_derived": True,
        "work_scope": "returned native math; excludes copies, validation and elementwise work",
    }


def _charge(
    work: dict[str, Any],
    kind: str,
    result: dict[str, Any],
    rows: int,
    dimension: int,
    hidden: int,
    axes: int,
) -> None:
    if kind == "conditioning":
        native = result["conditioning"]["work"]
        work["forward_passes"] += native["forward_passes"]
        work["forward_rows"] += native["forward_passes"] * rows
        work["gradient_evaluations"] += native["backward_passes"]
        work["gradient_rows"] += native["backward_passes"] * rows
        work["gradient_values"] += native["gradient_values"]
        work["gradient_numeric_bytes"] += native["gradient_numeric_bytes"]
        work["known_dense_multiply_accumulates"] += native["dense_multiply_accumulates"]
        work["normalization_calls"] += result["standardization_calls"]
        work["normalization_rows"] += result["standardization_rows"]
        work["normalization_values"] += result["standardization_values"]
    else:
        calls = result["prediction_calls"]
        work["prediction_calls"] += calls
        work["prediction_rows"] += result["prediction_rows"]
        work["prediction_values"] += result["prediction_values"]
        work["prediction_bytes"] += result["prediction_bytes"]
        # Why: residual prediction uses the existing full forward and input
        # standardization once, including identity normalization when unfitted.
        work["forward_passes"] += calls
        work["forward_rows"] += result["prediction_rows"]
        work["normalization_calls"] += calls
        work["normalization_rows"] += result["prediction_rows"]
        work["normalization_values"] += result["prediction_rows"] * dimension
        work["known_dense_multiply_accumulates"] += rows * (dimension * hidden + hidden * axes)


def run_training_readout(
    learner,
    groups,
    *,
    label: str,
    guard: Callable[[], None],
    corrective: bool = False,
    aggregated: bool = False,
    complete_b22: bool = False,
    command_fidelity: bool = False,
) -> dict[str, Any]:
    """Return all declared TRAIN diagnostics, retaining a paid partial on failure.

    Each invocation owns a fresh one-observation PhaseReadouts boundary. It does
    not renew training authority, and its returned fork never leaves this call.
    Why: corrective=True retains all50 correction and98 existing TRAIN rows
    together; aggregated=True adds the current50 without dropping old data.
    complete_b22=True explicitly adds the original22-row B19 demonstration.
    Existing four/five/six-group scopes remain bounded to130/148/198 rows.
    """
    report: dict[str, Any] = {
        "definition": DEFINITION,
        "label": label,
        "complete": False,
        "error": None,
        "groups": {},
        "operations": [],
        "work": _work(),
        "phase_readouts": None,
    }
    phase = None
    paid_payloads: list[tuple[str, str, str]] = []
    paid_owners: list[tuple[str, str, Any, Any]] = []
    work = report["work"]
    command_work = commands.CommandFidelityWork()
    groups_view, operations_view = report["groups"], report["operations"]
    group_owners: list[tuple[str, Any, int]] = []
    issued_operations: list[tuple[Any, Any]] = []
    try:
        if type(command_fidelity) is not bool:
            raise ValueError("command_fidelity must be an exact bool")
        if type(corrective) is not bool:
            raise ValueError("corrective must be an exact bool")
        if type(aggregated) is not bool:
            raise ValueError("aggregated must be an exact bool")
        if aggregated and not corrective:
            raise ValueError("aggregated readouts require corrective=True")
        if type(complete_b22) is not bool or (complete_b22 and not (corrective and aggregated)):
            raise ValueError("complete_b22 requires an exact bool and corrective+aggregated scopes")
        if (
            type(label) is not str
            or not label
            or label != label.strip()
            or len(label.encode("utf8")) > 64
            or not callable(guard)
        ):
            raise ValueError("label must be bounded native text and guard callable")
        if type(learner) is not NumpyContinuousBCLearner:
            raise ValueError("readouts require an exact native continuous BP learner")
        source_birth = _birth(learner)
        initial_state = _native_state(learner)
        if complete_b22:
            owned, source_state = _admit(
                learner, groups, corrective=True, aggregated=True, complete_b22=True
            )
        elif aggregated:
            owned, source_state = _admit(learner, groups, corrective=True, aggregated=True)
        else:
            owned, source_state = (
                _admit(learner, groups, corrective=True) if corrective else _admit(learner, groups)
            )
        _require_owner(learner, source_birth, initial_state)
        original_groups = _json([asdict(episode) for _, episode in groups])
        owned_groups = _json([asdict(episode) for _, episode in owned])
        raw_results: list[tuple[Any, str]] = []
        payloads: list[tuple[dict[str, Any], str]] = []
        operations: list[tuple[dict[str, Any], str]] = []
        captures: list[tuple[Any, Any, Any, Any]] = []
        if command_fidelity:
            work.update(
                command_attempts=0,
                command_returns=0,
                command_rows=0,
                prediction_capture_numeric_bytes=0,
                command_fidelity=asdict(command_work),
            )
            report["command_fidelity_enabled"] = True
        work_view = None
        work_guard = None

        def publish_work():
            nonlocal work_view, work_guard
            if command_fidelity:
                if work_guard is not None:
                    if report["work"] is not work_view:
                        raise RuntimeError("TRAIN readout paid work view replaced")
                    work_guard()
                work_view = deepcopy(work)
                report["work"] = work_view
                work_guard, _ = paid_report_guard({"work": work_view})

        publish_work()
        policy = source_state.policy
        report["native_objective"] = (
            bc.OBJECTIVE if policy.binary_gripper_axis is None else bc.MIXED_OBJECTIVE
        )
        report["physical_objective"] = "normalized_active_axis_half_mse"
        report["source_updates"] = source_state.completed_updates
        report["source_training_examples"] = source_state.training_examples
        report["group_order"] = [name for name, _ in owned]

        def check() -> None:
            guard()
            if command_fidelity:
                if (
                    report["groups"] is not groups_view
                    or report["operations"] is not operations_view
                ):
                    raise RuntimeError("TRAIN readout paid group or operation holder replaced")
                if (
                    len(groups_view) != len(group_owners)
                    or len(operations_view) != len(issued_operations)
                    or any(
                        groups_view.get(name) is not owner or owner.get("rows") != rows
                        for name, owner, rows in group_owners
                    )
                    or any(
                        actual is not owner or actual != trusted
                        for actual, (owner, trusted) in zip(
                            operations_view, issued_operations, strict=True
                        )
                    )
                ):
                    raise RuntimeError("TRAIN readout paid groups or operations changed")
            _require_owner(learner, source_birth, source_state)
            if (
                _json([asdict(episode) for _, episode in groups]) != original_groups
                or _json([asdict(episode) for _, episode in owned]) != owned_groups
            ):
                raise RuntimeError("TRAIN diagnostic inputs changed")
            for native, seal in raw_results:
                if _json(asdict(native)) != seal:
                    raise RuntimeError("preceding native diagnostic changed")
            for payload, seal in payloads:
                if _json(payload) != seal:
                    raise RuntimeError("preceding diagnostic payload changed")
            for operation, seal in operations:
                if _json(operation) != seal:
                    raise RuntimeError("preceding diagnostic operation changed")
            if command_fidelity:
                if report["work"] is not work_view:
                    raise RuntimeError("TRAIN readout paid work view replaced")
                if work_guard is None:
                    raise RuntimeError("TRAIN readout paid work witness missing")
                work_guard()
                accumulator = asdict(command_work)
                if accumulator != work["command_fidelity"] or any(
                    type(value) is not int for value in accumulator.values()
                ):
                    raise RuntimeError("command accumulator changed before further model work")
                for prediction, target, before_prediction, before_target in captures:
                    if (
                        prediction.dtype != np.float64
                        or target.dtype != np.float64
                        or not np.array_equal(prediction, before_prediction)
                        or not np.array_equal(target, before_target)
                    ):
                        raise RuntimeError("captured TRAIN prediction or target changed")
                for name, kind, seal in paid_payloads:
                    if _json(report["groups"][name][kind]) != seal:
                        raise RuntimeError("preceding paid TRAIN payload changed")
                for name, kind, owner, witness in paid_owners:
                    if report["groups"][name][kind] is not owner:
                        raise RuntimeError("preceding paid TRAIN payload replaced")
                    witness()

        def observer(detached, metadata):
            birth = _birth(detached)
            expected = _native_state(detached)

            def boundary() -> None:
                check()
                _require_owner(detached, birth, expected)

            for name, episode in owned:
                prediction_capture: list[Any] = []

                def capture(diagnostic, predictions, targets, labels):
                    if prediction_capture:
                        raise RuntimeError("TRAIN prediction capture repeated")
                    prediction_capture.append((diagnostic, predictions, targets, labels))
                    captures.append((predictions, targets, predictions.copy(), targets.copy()))
                    work["prediction_capture_numeric_bytes"] += predictions.nbytes + targets.nbytes

                rows = len(episode.transitions)
                group: dict[str, Any] = {"rows": rows}
                report["groups"][name] = group
                if command_fidelity:
                    group_owners.append((name, group, rows))
                for kind in ("conditioning", "residual"):
                    boundary()
                    entry = {
                        "group": name,
                        "kind": kind,
                        "rows": rows,
                        "attempted": True,
                        "returned": False,
                        "validated": False,
                    }
                    report["operations"].append(entry)
                    if command_fidelity:
                        issued_operations.append((entry, deepcopy(entry)))
                    work[kind + "_attempts"] += 1
                    work[kind + "_attempted_rows"] += rows
                    work["unknown_unreturned_work"] = True
                    publish_work()
                    if kind == "conditioning":
                        result = diagnose_conditioning(detached, episode)
                    else:
                        stages = tuple("initial" if i == 0 else "remainder" for i in range(rows))
                        result = (
                            diagnose_residual(
                                detached, episode, stages, prediction_observer=capture
                            )
                            if command_fidelity
                            else diagnose_residual(detached, episode, stages)
                        )
                    entry["returned"] = True
                    if command_fidelity:
                        issued_operations[-1][1]["returned"] = True
                    work[kind + "_returns"] += 1
                    work[kind + "_returned_rows"] += rows
                    # Retain returned payload and charges before any later
                    # callback-capable guard. JSON owns tuples as native lists.
                    raw = asdict(result)
                    group[kind] = raw
                    _charge(
                        work,
                        kind,
                        raw,
                        rows,
                        policy.spec.observation_dim,
                        policy.hidden_dim,
                        len(policy.spec.action_units),
                    )
                    seal = _json(raw)
                    group[kind] = json.loads(seal)
                    raw_results.append((result, seal))
                    payloads.append((raw, seal))
                    payloads.append((group[kind], seal))
                    if command_fidelity:
                        paid_payloads.append((name, kind, seal))
                        witness, _ = paid_report_guard({"payload": group[kind]})
                        paid_owners.append((name, kind, group[kind], witness))
                    work["unknown_unreturned_work"] = False
                    publish_work()
                    boundary()
                    entry["validated"] = True
                    if command_fidelity:
                        issued_operations[-1][1]["validated"] = True
                    operations.append((entry, _json(entry)))
                if command_fidelity:
                    boundary()
                    if len(prediction_capture) != 1 or prediction_capture[0][0] is not result:
                        raise RuntimeError(
                            "TRAIN residual prediction capture differs from paid return"
                        )
                    _, predictions, targets, labels = prediction_capture[0]
                    entry = dict(
                        group=name,
                        kind="command_fidelity",
                        rows=rows,
                        attempted=True,
                        returned=False,
                        validated=False,
                    )
                    report["operations"].append(entry)
                    issued_operations.append((entry, deepcopy(entry)))
                    work["command_attempts"] += 1
                    work["unknown_unreturned_work"] = True
                    publish_work()
                    prior_command_work = deepcopy(work["command_fidelity"])
                    # Why: the helper's mutable accumulator is not authority to
                    # forget paid earlier groups. Expected successful deltas
                    # follow the fixed4-axis conversion/comparison contract.
                    delta = dict(
                        conversion_attempts=8 * rows,
                        conversion_returns=8 * rows,
                        equality_comparisons=4 * rows,
                        absolute_error_coordinates=4 * rows,
                        translation_direction_comparisons=3 * rows,
                        gripper_margin_products=rows if policy.binary_gripper_axis == 3 else 0,
                        gripper_sign_classifications=rows if policy.binary_gripper_axis == 3 else 0,
                        known_owned_numeric_bytes=160 * rows,
                    )
                    expected_command_work = {
                        key: prior_command_work[key] + value for key, value in delta.items()
                    }
                    try:
                        command = commands.continuous_command_fidelity(
                            predictions,
                            targets,
                            policy.spec,
                            labels,
                            binary_gripper_axis=policy.binary_gripper_axis,
                            work=command_work,
                        )
                    except Exception:
                        observed = asdict(command_work)
                        if all(
                            type(observed[key]) is int
                            and prior_command_work[key]
                            <= observed[key]
                            <= expected_command_work[key]
                            for key in delta
                        ):
                            work["command_fidelity"] = observed
                        else:
                            work["command_unvalidated_work_unknown"] = True
                        raise
                    group["command_fidelity"] = command
                    entry["returned"] = True
                    issued_operations[-1][1]["returned"] = True
                    work["command_returns"] += 1
                    work["command_rows"] += rows
                    work["unknown_unreturned_work"] = False
                    command["input"] = dict(
                        episode_id=episode.episode_id,
                        context=episode.context,
                        role=episode.role,
                    )
                    command_seal = _json(command)
                    paid_payloads.append((name, "command_fidelity", command_seal))
                    witness, _ = paid_report_guard({"payload": command})
                    paid_owners.append((name, "command_fidelity", command, witness))
                    returned_work = command.get("work")
                    if returned_work == expected_command_work and all(
                        type(value) is int for value in returned_work.values()
                    ):
                        work["command_fidelity"] = deepcopy(returned_work)
                    else:
                        work["command_unvalidated_work_unknown"] = True
                        raise RuntimeError(
                            "returned command work differs from original paid row deltas"
                        )
                    accumulator = asdict(command_work)
                    if accumulator != returned_work or any(
                        type(value) is not int for value in accumulator.values()
                    ):
                        raise RuntimeError("command accumulator changed after paid return")
                    publish_work()
                    boundary()
                    entry["validated"] = True
                    issued_operations[-1][1]["validated"] = True
                    operations.append((entry, _json(entry)))
            boundary()
            return json.loads(
                _json(
                    {
                        "label": label,
                        "groups": report["groups"],
                        "operations": report["operations"],
                        "work": work,
                        "native_objective": report["native_objective"],
                        "physical_objective": report["physical_objective"],
                    }
                )
            )

        phase = PhaseReadouts(observer)
        phase.observe(
            learner,
            {
                "checkpoint_index": 0,
                "phase": "initial",
                "updates": source_state.completed_updates,
                "training_examples": source_state.training_examples,
            },
            check,
        )
        report["phase_readouts"] = phase.report
        check()
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {str(error)[:2000]}"
        if phase is not None:
            report["phase_readouts"] = phase.report
        if command_fidelity is True:
            report["work"] = deepcopy(work)
            report["groups"] = {name: dict(rows=rows) for name, _, rows in group_owners}
            report["operations"] = [deepcopy(trusted) for _, trusted in issued_operations]
            for name, kind, seal in paid_payloads:
                report["groups"].setdefault(name, {})[kind] = json.loads(seal)
    return deepcopy(report)
