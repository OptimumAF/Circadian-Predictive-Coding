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


def _admit(learner, groups, *, corrective: bool = False, aggregated: bool = False):
    if type(corrective) is not bool:
        raise ValueError("corrective must be an exact bool")
    if type(aggregated) is not bool:
        raise ValueError("aggregated must be an exact bool")
    if aggregated and not corrective:
        raise ValueError("aggregated readouts require corrective=True")
    names = AGGREGATED_GROUPS if aggregated else CORRECTIVE_GROUPS if corrective else GROUPS
    maximum = 198 if aggregated else 148 if corrective else 130
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
) -> dict[str, Any]:
    """Return all declared TRAIN diagnostics, retaining a paid partial on failure.

    Each invocation owns a fresh one-observation PhaseReadouts boundary. It does
    not renew training authority, and its returned fork never leaves this call.
    Why: corrective=True retains all50 correction and98 existing TRAIN rows
    together; aggregated=True adds the current50 without dropping old data.
    Existing four/five-group scopes remain bounded to130/148 rows.
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
    try:
        if type(corrective) is not bool:
            raise ValueError("corrective must be an exact bool")
        if type(aggregated) is not bool:
            raise ValueError("aggregated must be an exact bool")
        if aggregated and not corrective:
            raise ValueError("aggregated readouts require corrective=True")
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
        if aggregated:
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
        work = report["work"]
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

        def observer(detached, metadata):
            birth = _birth(detached)
            expected = _native_state(detached)

            def boundary() -> None:
                check()
                _require_owner(detached, birth, expected)

            for name, episode in owned:
                rows = len(episode.transitions)
                group: dict[str, Any] = {"rows": rows}
                report["groups"][name] = group
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
                    work[kind + "_attempts"] += 1
                    work[kind + "_attempted_rows"] += rows
                    work["unknown_unreturned_work"] = True
                    if kind == "conditioning":
                        result = diagnose_conditioning(detached, episode)
                    else:
                        stages = tuple("initial" if i == 0 else "remainder" for i in range(rows))
                        result = diagnose_residual(detached, episode, stages)
                    entry["returned"] = True
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
                    work["unknown_unreturned_work"] = False
                    boundary()
                    entry["validated"] = True
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
    return deepcopy(report)
