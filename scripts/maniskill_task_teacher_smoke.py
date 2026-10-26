"""One TRAIN, guard-only or fixed reserved-probe teacher, no learner.

Usage: python -m scripts.maniskill_task_teacher_smoke --output fresh-teacher.json
Requires optional pinned CPU stack and a separately frozen local native budget.
"""

import argparse
from dataclasses import asdict
import faulthandler
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
import time
from typing import Any

from src.infra.maniskill_task_teacher import (
    collect_task_teacher,
    collect_guard_teacher,
    collect_readiness_teacher,
    collect_probe_teacher,
    TaskTeacherResult,
    ObservedGraspTaskTeacherResult,
    GoalConsistentObservedGraspTaskTeacherResult,
    ReservedProbeTaskTeacherResult,
)
from src.core.pick_cube_probe_context import pick_cube_probe_binding
from src.core.pick_cube_train_context import TRAIN_SEEDS, pick_cube_training_context
from src.core.pick_cube_readiness_teacher import (
    DEFINITION as READINESS_DEFINITION,
    OBSERVED_GRASP_DEFINITION,
    GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION,
)
from src.core.pick_cube_guard_context import pick_cube_guard_context
from src.core.experience import require_tick
from src.core.pick_cube_mass_context import pick_cube_mass_context, require_mass_binding
from src.core.pick_cube_camera_context import pick_cube_camera_context, require_camera_binding
from src.core.pick_cube_lighting_context import (
    pick_cube_lighting_context,
    require_lighting_binding,
    validate_pick_cube_lighting_context,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--train-seed", type=int, choices=TRAIN_SEEDS)
    selection.add_argument("--guard-seed", type=int, choices=(31, 37))
    selection.add_argument("--probe-role", choices=("probe_train", "probe_test"))
    parser.add_argument("--labels-arrived-at", type=int)
    parser.add_argument("--visual-observation", action="store_true")
    parser.add_argument("--camera-drift", action="store_true")
    parser.add_argument("--mass-drift", action="store_true")
    parser.add_argument("--lighting-drift", action="store_true")
    parser.add_argument("--geometry-readiness", action="store_true")
    parser.add_argument("--observed-grasp", action="store_true")
    parser.add_argument("--goal-consistent", action="store_true")
    args = parser.parse_args()
    if args.probe_role is not None and (
        not args.geometry_readiness
        or not args.observed_grasp
        or not args.goal_consistent
        or args.visual_observation
        or args.camera_drift
        or args.mass_drift
        or args.lighting_drift
        or args.labels_arrived_at is not None
    ):
        parser.error(
            "--probe-role requires raw-state --geometry-readiness --observed-grasp --goal-consistent without interventions or guard labels"
        )
    if args.goal_consistent and not args.observed_grasp:
        parser.error("--goal-consistent requires --observed-grasp")
    if args.observed_grasp and (not args.geometry_readiness or args.visual_observation):
        parser.error("--observed-grasp requires --geometry-readiness with raw state observations")
    if args.geometry_readiness and (
        args.guard_seed is not None
        or args.labels_arrived_at is not None
        or args.camera_drift
        or args.mass_drift
        or args.lighting_drift
    ):
        parser.error("--geometry-readiness requires TRAIN without guard labels or interventions")
    if args.guard_seed is not None:
        if args.labels_arrived_at is None:
            parser.error("--guard-seed requires --labels-arrived-at")
        try:
            require_tick(args.labels_arrived_at, "guard label arrival")
        except ValueError as error:
            parser.error(str(error))
        seed, role = args.guard_seed, "inner_guard"
        context = pick_cube_guard_context(seed)
    elif args.probe_role is not None:
        allocation = pick_cube_probe_binding(args.probe_role)
        seed, role, context = allocation.seed, allocation.role, allocation.context
    else:
        if args.labels_arrived_at is not None:
            parser.error("--labels-arrived-at is for guard collection only")
        seed, role = 17 if args.train_seed is None else args.train_seed, "train"
        context = pick_cube_training_context(seed)
    camera = None
    if args.camera_drift:
        camera = pick_cube_camera_context()
        context = camera.context
        try:
            require_camera_binding(
                camera,
                context=context,
                seed=seed,
                role=role,
                visual_observation=args.visual_observation,
            )
        except ValueError as error:
            parser.error(str(error))
    mass = None
    if args.mass_drift:
        if camera is not None:
            parser.error("mass and camera interventions require separate declarations")
        mass = pick_cube_mass_context()
        context = mass.context
        try:
            require_mass_binding(mass, context=context, seed=seed, role=role)
        except ValueError as error:
            parser.error(str(error))
    lighting = None
    if args.lighting_drift:
        if camera is not None or mass is not None:
            parser.error("lighting, camera and mass interventions require separate declarations")
        lighting = pick_cube_lighting_context()
        context = lighting.context
        try:
            require_lighting_binding(
                lighting,
                context=context,
                seed=seed,
                role=role,
                visual_observation=args.visual_observation,
            )
        except ValueError as error:
            parser.error(str(error))
    with args.output.open("x", encoding="utf8") as handle:
        faulthandler.enable()
        logger = logging.getLogger("src.infra.maniskill_task_teacher")
        previous_level, previous_propagation = logger.level, logger.propagate
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        start = time.monotonic()
        try:
            options: dict[str, Any] = {}
            if role == "inner_guard":
                options.update(guard_seed=seed, labels_arrived_at=args.labels_arrived_at)
            elif seed != 17:
                options["train_seed"] = seed
            if args.visual_observation:
                options["visual_observation"] = True
            if camera is not None:
                options["camera_context"] = camera
            if mass is not None:
                options["mass_context"] = mass
            if lighting is not None:
                options["lighting_context"] = lighting
            if args.observed_grasp:
                options["observed_grasp"] = True
            if args.goal_consistent:
                options["goal_consistent"] = True
            result = (
                collect_probe_teacher(role=role)
                if args.probe_role is not None
                else collect_readiness_teacher(**options)
                if args.geometry_readiness
                else collect_guard_teacher(**options)
                if role == "inner_guard"
                else collect_task_teacher(**options)
            )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            logger.propagate = previous_propagation
            handler.close()
        payload = asdict(result)
        if result.teacher_definition is None:
            payload.pop("teacher_definition")
        if result.camera_context is None:
            # Why: retain the established default/guard JSON schema exactly.
            payload.pop("camera_context")
        if mass is None and all(
            getattr(result, name) is None for name in ("mass_context", "mass_readback", "mass_work")
        ):
            for name in ("mass_context", "mass_readback", "mass_work"):
                payload.pop(name)
        if lighting is None and all(
            getattr(result, name) is None
            for name in ("lighting_context", "lighting_readback", "lighting_work")
        ):
            for name in ("lighting_context", "lighting_readback", "lighting_work"):
                payload.pop(name)
        versions: dict[str, str | None] = {}
        for name in ("mani_skill", "sapien", "gymnasium", "torch", "numpy"):
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                versions[name] = None
        qualified = result.qualified
        expected_definition = (
            GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
            if args.goal_consistent
            else OBSERVED_GRASP_DEFINITION
            if args.observed_grasp
            else READINESS_DEFINITION
        )
        expected_result = (
            ReservedProbeTaskTeacherResult
            if args.probe_role is not None
            else GoalConsistentObservedGraspTaskTeacherResult
            if args.goal_consistent
            else ObservedGraspTaskTeacherResult
            if args.observed_grasp
            else TaskTeacherResult
        )
        if args.geometry_readiness and (
            result.teacher_definition != expected_definition or type(result) is not expected_result
        ):
            qualified = False
            payload["readiness_request_error"] = (
                "returned teacher family or definition differs from requested geometry readiness"
            )
        if (
            role == "train"
            and camera is None
            and mass is None
            and lighting is None
            and result.candidate_episode is not None
        ):
            # Why: CLI metadata must not qualify a different paid reset returned
            # by the collector. Retain that report while refusing its release.
            prefix = (
                "goal-consistent-observed-grasp"
                if args.goal_consistent
                else "observed-grasp"
                if args.observed_grasp
                else "readiness"
                if args.geometry_readiness
                else "task"
            )
            train_episode = result.candidate_episode
            if (train_episode.role, train_episode.context, train_episode.episode_id) != (
                role,
                context,
                f"pick-cube-{prefix}-teacher-{seed}",
            ):
                qualified = False
                payload["train_request_error"] = (
                    "returned teacher differs from the requested TRAIN allocation"
                )
        if args.probe_role is not None:
            try:
                if type(result) is not ReservedProbeTaskTeacherResult:
                    raise ValueError("reserved probe request returned another result family")
                episode = result.candidate_episode
                requested = pick_cube_probe_binding(args.probe_role)
                if (
                    type(result.probe_seed) is not int
                    or result.probe_seed != requested.seed
                    or (
                        episode is not None
                        and (episode.role, episode.context, episode.episode_id)
                        != (requested.role, requested.context, requested.episode_id)
                    )
                ):
                    raise ValueError(
                        "returned probe differs from the requested reserved allocation"
                    )
            except (ValueError, TypeError, AttributeError) as error:
                qualified = False
                payload["probe_request_error"] = str(error)
        if lighting is not None:
            try:
                returned_lighting = result.lighting_context
                if (
                    returned_lighting is None
                    or validate_pick_cube_lighting_context(returned_lighting) != lighting
                ):
                    raise ValueError("returned teacher lighting differs from the requested context")
            except (ValueError, TypeError, AttributeError) as error:
                qualified = False
                payload["lighting_request_error"] = str(error)
        payload.update(
            qualified=qualified,
            task="PickCube-v1",
            controller="pd_ee_delta_pos",
            role=role,
            seed=seed,
            context=context,
            optimizer_updates=0,
            constructor_reset="official_seed2022_reconfigure",
            explicit_reset=f"seed{seed}_if_port_construction_reaches_reset",
            reset_physics_steps=None,
            versions=versions,
            elapsed_seconds=time.monotonic() - start,
        )
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "candidate_episode"}))
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
