"""Observe one final isolated adaptation endpoint through existing native ports.

Owns five finite paid receipts/captures. No learner updates, new native authority,
before/after native comparison, source adoption or adapted model export.
"""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.maniskill_continuous_smoke import run_frozen_rollout
from scripts.push_cube_continuation_study import roll_push
from scripts.push_cube_memory_control import _encoded, _rejected, _require_native, _summary
from scripts.taskspace_memory_native import _read_capture, MAX_REPORT_BYTES
from src.adapters.taskspace_memory_adaptation import _plan, RETURN_GROUPS
from src.adapters.taskspace_memory_demo import _admit, _own_json, _retention_options, run_demo
from src.adapters.numpy_retained_nearest_bc import pin_retained_native, require_retained_native
from src.app.probe_return_phase import paid_report_guard
from src.core.pick_cube_probe_context import pick_cube_probe_binding
from src.core.pick_cube_train_context import pick_cube_training_context
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
import json

SCENES = (
    ("pick17", "pick", 17),
    ("probe23", "probe", 23),
    ("pick19", "pick", 19),
    ("pick20", "pick", 20),
    ("push17", "push", 17),
)


def native_adaptation_outputs(output: Path) -> tuple[Path, ...]:
    paths = (
        output,
        output.with_suffix(".failure.json"),
        *(
            output.parent / f"{output.stem}-{name}.json"
            for name, task, _ in SCENES
            if task != "push"
        ),
    )
    if (
        len(set(paths)) != 6
        or not output.parent.is_dir()
        or any(path.exists() or path.is_symlink() for path in paths)
    ):
        raise ValueError(
            "native adaptation requires six distinct fresh outputs in an existing directory"
        )
    return paths


def run_native_adaptation(
    payload, plan, output, *, retention, max_prototypes=256, return_task=None, input_receipts=None
):
    output, failure, *pick_paths = native_adaptation_outputs(output)
    rule = _retention_options(retention, max_prototypes)
    if rule is None:
        raise ValueError("native adaptation requires explicit retained memory")
    rule = deepcopy(rule)
    if return_task is not None and (
        type(return_task) is not str or return_task not in RETURN_GROUPS
    ):
        raise ValueError("native return task must name one original old TRAIN packet")
    plan = _plan(plan)
    payload = _own_json(payload)
    episodes, _, _ = _admit(payload)
    expected_rows = sum(len(e.transitions) for e in episodes) + 18
    if return_task is not None:
        expected_rows += len(episodes[RETURN_GROUPS[return_task]].transitions)
    expected_updates = 8 + int(return_task is not None)
    phases = ("before_probe", "after_probe") + (("after_return",) if return_task else ())
    paths = {
        name: path
        for (name, task, _), path in zip(
            (row for row in SCENES if row[1] != "push"), pick_paths, strict=True
        )
    }
    report: dict[str, Any] = dict(
        definition="taskspace_memory_native_adaptation_v1",
        complete=False,
        error=None,
        classification="final known TRAIN scenes only; native before/after change unmeasured",
        observed_native_phase=phases[-1],
        unobserved_native_phases=list(phases[:-1]),
        retention=dict(rule=asdict(rule), max_prototypes=max_prototypes),
        return_task=return_task,
        input_receipts=deepcopy(input_receipts),
        workflow=None,
        native=[],
        outcomes={},
        attempted=[],
        returned=[],
        cpu_endpoint_phases=[],
        unreturned_native_work_unknown=False,
    )
    identity_keys = (
        "definition",
        "classification",
        "observed_native_phase",
        "unobserved_native_phases",
        "retention",
        "return_task",
        "input_receipts",
    )
    identity_seal = _encoded({key: report[key] for key in identity_keys})
    input_seal = _encoded(dict(payload=payload, plan=asdict(plan), rule=asdict(rule)))
    native, attempted, returned, seen = (
        report[key] for key in ("native", "attempted", "returned", "cpu_endpoint_phases")
    )
    proof: list[Any] = []
    attempts: list[str] = []
    returns: list[str] = []
    phase_count = 0
    outcome_seal = _encoded({})
    workflow_guard = trusted_workflow = None
    saved_main = readout = pin = None

    def pure():
        if readout is not None:
            require_retained_native(readout, pin)
        if (
            _encoded({key: report[key] for key in identity_keys}) != identity_seal
            or _encoded(dict(payload=payload, plan=asdict(plan), rule=asdict(rule))) != input_seal
        ):
            raise RuntimeError("native adaptation original identity or inputs changed")
        if (
            report["native"] is not native
            or len(native) != len(proof)
            or report["attempted"] is not attempted
            or attempted != attempts
            or report["returned"] is not returned
            or returned != returns
            or report["cpu_endpoint_phases"] is not seen
            or seen != list(phases[:phase_count])
            or _encoded(report["outcomes"]) != outcome_seal
        ):
            raise RuntimeError("native adaptation paid holders, phases or outcomes changed")
        for index, (entry, raw, seal, _, _, holder) in enumerate(proof):
            if (
                native[index] is not entry
                or entry["scene"] != SCENES[index][0]
                or entry["report"] is not raw
                or _encoded(raw) != seal
                or holder is not None
                and holder.report is not raw
            ):
                raise RuntimeError("preceding paid native return changed or was replaced")
        if workflow_guard is not None:
            workflow_guard()

    def guard():
        pure()
        if saved_main is not None and _read_capture(output, MAX_REPORT_BYTES) != saved_main:
            raise RuntimeError("preceding main capture changed or was replaced")
        for _, _, _, path, saved, _ in proof:
            if saved is not None and _read_capture(path, 1024**2) != saved:
                raise RuntimeError("preceding Pick capture changed or was replaced")
        pure()

    def save():
        nonlocal saved_main
        guard()
        raw = _encoded(report) + b"\n"
        if len(raw) > MAX_REPORT_BYTES:
            raise ValueError("native adaptation main capture exceeds3MiB")
        previous_id = None if saved_main is None else saved_main[1]
        with output.open("xb" if saved_main is None else "wb") as handle:
            handle.write(raw)
        current = _read_capture(output, MAX_REPORT_BYTES)
        if current[0] != raw or previous_id is not None and current[1] != previous_id:
            raise RuntimeError("main capture write changed bytes or replaced its original file")
        saved_main = current
        guard()
        if _encoded(report) + b"\n" != raw:
            raise RuntimeError("capture IO changed the paid report")

    def endpoint(child, phase, check):
        nonlocal phase_count, readout, pin, outcome_seal
        check(guard)
        if phase_count >= len(phases) or phase != phases[phase_count]:
            raise RuntimeError("CPU adaptation endpoints repeated or reordered")
        phase_count += 1
        seen.append(phase)
        if phase != phases[-1]:
            return
        readout, pin = child, pin_retained_native(child)
        if (
            pin.state.completed_updates != expected_updates
            or pin.state.training_examples != expected_rows
        ):
            raise RuntimeError("native endpoint differs from original acquisition/return dose")
        for name, task, seed in SCENES:
            check(guard)
            attempted.append(name)
            attempts.append(name)
            save()
            check(guard)  # IO cannot redefine preceding evidence before the next scene.

            def observe():
                nonlocal outcome_seal
                path, holder = paths.get(name), None
                if path is None:
                    raw = roll_push(child, label="isolated-memory-final", seed=seed)
                else:
                    opts: dict[str, Any] = (
                        dict(probe_train_seed=23) if task == "probe" else dict(train_seed=seed)
                    )
                    holder = run_frozen_rollout(
                        path,
                        learner=child,
                        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
                        **opts,
                    )
                    raw = holder.report
                returned.append(name)
                returns.append(name)
                seal = _encoded(_rejected(raw))
                entry = dict(scene=name, report=raw)
                native.append(entry)
                proof.append((entry, raw, seal, path, None, holder))  # Birth before later IO.
                guard()
                saved = None if path is None else _read_capture(path, 1024**2)
                guard()
                proof[-1] = (entry, raw, seal, path, saved, holder)
                if path is not None:
                    if holder is None or saved is None:
                        raise RuntimeError("Pick receipt owner or saved capture missing")
                    if holder.corrections is not None or _encoded(json.loads(saved[0])) != seal:
                        raise RuntimeError("saved native receipt differs from its original birth")
                _require_native(
                    raw,
                    push=task == "push",
                    updates=expected_updates,
                    rows=expected_rows,
                    prototype_rows=pin.state.retained_rows,
                )
                if (
                    raw["role"] != ("probe_train" if task == "probe" else "train")
                    or raw["seed"] != seed
                ):
                    raise RuntimeError("native role or original reset changed")
                if task == "probe":
                    binding = pick_cube_probe_binding("probe_train")
                    if (
                        raw["episode"]["context"] != binding.context
                        or raw["episode"]["episode_id"] != "pick-cube-probe-train-smoke-23"
                        or raw.get("probe_train_seed") != 23
                        or raw.get("scoring_only") is not True
                    ):
                        raise RuntimeError("reserved23 inference identity or authority changed")
                elif raw["episode"]["context"] != (
                    f"push-cube-train-{seed}"
                    if task == "push"
                    else pick_cube_training_context(seed)
                ):
                    raise RuntimeError("native original TRAIN context changed")
                guard()
                report["outcomes"][name] = _summary(raw, push=task == "push")
                outcome_seal = _encoded(report["outcomes"])
                save()

            check(observe)

    def restore_paid():
        report.update(json.loads(identity_seal))
        report["native"] = [
            dict(scene=SCENES[i][0], report=json.loads(p[2])) for i, p in enumerate(proof)
        ]
        report["outcomes"] = json.loads(outcome_seal)
        report.update(
            attempted=list(attempts),
            returned=list(returns),
            cpu_endpoint_phases=list(phases[:phase_count]),
        )
        if trusted_workflow is not None:
            report["workflow"] = trusted_workflow["workflow"]

    integrity_failed = False
    try:
        save()
        value = run_demo(
            payload,
            adaptation_probe=plan,
            retention=rule,
            max_prototypes=max_prototypes,
            return_task=return_task,
            endpoint_observer=endpoint,
            check_boundary=guard,
        )
        report["workflow"] = value
        paid_guard, trusted_workflow = paid_report_guard({"workflow": value})

        def require_workflow():
            if report["workflow"] is not value:
                raise RuntimeError("paid CPU adaptation workflow replaced")
            paid_guard()

        workflow_guard = require_workflow
        guard()
        if (
            not value["complete"]
            or len(proof) != 5
            or len(report["outcomes"]) != 5
            or phase_count != len(phases)
        ):
            raise RuntimeError(f"native adaptation workflow incomplete: {value.get('error')}")
        report["complete"] = True
    except Exception as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        try:
            guard()
        except Exception as integrity_error:
            report["integrity_error"] = f"{type(integrity_error).__name__}: {integrity_error}"
            integrity_failed = True
    report["unreturned_native_work_unknown"] = attempts != returns
    try:
        if integrity_failed:
            raise RuntimeError("original native evidence failed integrity; preserve main")
        save()
    except Exception as error:
        report.update(complete=False, capture_error=f"{type(error).__name__}: {error}")
        restore_paid()
        raw = _encoded(report) + b"\n"
        if len(raw) > MAX_REPORT_BYTES:
            raise ValueError("native adaptation failure exceeds3MiB") from error
        with failure.open("xb") as handle:
            handle.write(raw)
        return json.loads(raw)  # Frozen failure birth, not a holder changed by IO.
    return deepcopy(report)
