"""Compose explicit CPU adaptation through the existing memory stream/probe.

Owns typed original reserved episodes, finite reports and original TRAIN return
selection. No files, simulator, source adoption or adapted learner export.
"""

from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any

from src.adapters.continuous_probe import (
    ContinuousProbePlan,
    copy_probe_plan,
    run_retained_nearest_probe,
)
from src.adapters.continuous_stream_sequence import run_sequence
from src.adapters.numpy_retained_nearest_bc import pin_retained_native, require_retained_native
from src.app.probe_return_phase import paid_report_guard
from src.core.pick_cube_action_history import validate_action_history_probe_episode
from src.core.pick_cube_probe_context import pick_cube_probe_binding
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.cube_taskspace_state import cube_taskspace_spec

RETURN_GROUPS = {"pick17": 0, "pick19": 6, "pick20": 2, "push17": 3}


def _plan(value):
    if (
        type(value) is not ContinuousProbePlan
        or value.updates != 1
        or value.diagnostics
        or value.zero_velocity_control
        or value.optimizer != "sgd"
    ):
        raise ValueError("memory adaptation requires one plain reserved-pair update")
    owned = copy_probe_plan(value)
    for role, episode in (("probe_train", owned.training), ("probe_test", owned.evaluation)):
        binding = pick_cube_probe_binding(role)
        validate_action_history_probe_episode(episode, PickCubeTaskSpaceHistoryRecipe())
        if (
            episode.role != binding.role
            or episode.context != binding.context
            or episode.episode_id != binding.episode_id
            or len(episode.transitions) != 18
            or episode.spec != cube_taskspace_spec()
        ):
            raise ValueError("memory adaptation requires the original23/24 typed pair")
    return owned


def run_adaptation(
    payload,
    plan,
    *,
    retention,
    max_prototypes,
    return_task=None,
    endpoint_observer=None,
    check_boundary=None,
):
    from src.adapters.taskspace_memory_demo import _admit, _retention_options

    retention = _retention_options(retention, max_prototypes)
    if retention is None:
        raise ValueError("memory adaptation requires explicit retained memory")
    retention = deepcopy(retention)

    if return_task is not None and (
        type(return_task) is not str or return_task not in RETURN_GROUPS
    ):
        raise ValueError("return_task must be pick17, pick19, pick20, push17 or None")
    if endpoint_observer is not None and not callable(endpoint_observer):
        raise ValueError("endpoint_observer must be callable or None")
    if check_boundary is not None and not callable(check_boundary):
        raise ValueError("check_boundary must be callable or None")
    plan = _plan(plan)
    episodes, _, provenance = _admit(payload)
    plan = copy_probe_plan(plan, episodes)
    old = tuple(episodes[index] for index in (0, 6, 2, 3))
    returning = None if return_task is None else episodes[RETURN_GROUPS[return_task]]
    inputs = lambda: json.dumps(
        dict(
            episodes=[asdict(e) for e in episodes], plan=asdict(plan), retention=asdict(retention)
        ),
        sort_keys=True,
        allow_nan=False,
    )
    input_seal = inputs()
    report: dict[str, Any] = dict(
        definition="taskspace_memory_adaptation_v1",
        mode="isolated_retained_adaptation",
        classification="CPU fit/interference/return scores; native outcomes unmeasured",
        complete=False,
        error=None,
        stream=None,
        adaptation=None,
        native_outcomes=None,
        provenance=provenance,
        retention=asdict(retention),
        max_prototypes=max_prototypes,
        return_task=return_task,
        unreturned_private_work_unknown=False,
    )
    header_keys = (
        "definition",
        "mode",
        "classification",
        "native_outcomes",
        "provenance",
        "retention",
        "max_prototypes",
        "return_task",
    )
    header = lambda: json.dumps(
        {key: report[key] for key in header_keys}, sort_keys=True, allow_nan=False
    )
    header_seal = header()
    final_seen = False
    paid_guard = trusted = None

    def boundary():
        if check_boundary is not None:
            check_boundary()
        if inputs() != input_seal or header() != header_seal:
            raise RuntimeError("memory adaptation original inputs/header changed")
        if paid_guard is not None:
            paid_guard()

    def observe(source, metadata):
        nonlocal final_seen, paid_guard, trusted
        boundary()
        if metadata["phase"] != "after_release" or metadata["packet_index"] != 5:
            return {}
        if final_seen:
            raise RuntimeError("memory adaptation final checkpoint repeated")
        final_seen = True
        pin = pin_retained_native(source)
        rows = sum(len(e.transitions) for e in episodes)
        seal = json.dumps(metadata, sort_keys=True, allow_nan=False)
        if (
            pin.state.completed_updates != 7
            or pin.state.training_examples != rows
            or metadata["known_episode_ids"] != [e.episode_id for e in episodes]
            or metadata["prototype_rows"] != pin.state.retained_rows
        ):
            raise RuntimeError("memory adaptation source differs from original seven TRAIN packets")

        def guard():
            boundary()
            require_retained_native(source, pin)
            if json.dumps(metadata, sort_keys=True, allow_nan=False) != seal:
                raise RuntimeError("memory adaptation checkpoint metadata changed")

        report["unreturned_private_work_unknown"] = True
        returned = run_retained_nearest_probe(
            source,
            plan,
            training_readouts=old,
            return_training=returning,
            check_boundary=guard,
            endpoint_observer=endpoint_observer,
        )
        report["adaptation"] = returned  # Paid birth before any later callback.
        report["unreturned_private_work_unknown"] = False
        paid_guard, trusted = paid_report_guard({"adaptation": returned})
        # Include holder identity; equal replacement is not the original paid return.
        original_guard = paid_guard

        def require_paid():
            if report["adaptation"] is not returned:
                raise RuntimeError("paid memory adaptation return replaced")
            original_guard()

        paid_guard = require_paid
        guard()
        if returned.get("complete") is not True:
            raise RuntimeError(f"memory adaptation incomplete: {returned.get('error')}")
        return dict(private_updates=1, old_train_return_updates=int(returning is not None))

    try:
        report["stream"] = run_sequence(
            episodes[0],
            episodes[1:],
            observe,
            fixed_input=True,
            maximum_releases=6,
            retention=retention,
            max_prototypes=max_prototypes,
        )
        boundary()
        if not final_seen or report["stream"].get("complete") is not True:
            raise RuntimeError(
                f"memory adaptation stream incomplete: {report['stream'].get('error')}"
            )
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if trusted is not None:
            report["adaptation"] = trusted["adaptation"]
    return deepcopy(report)
