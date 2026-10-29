"""Bounded native TRAIN pair collection for declared G, A4 and red-wall A3.

Reuse one teacher/cleanup/provenance path across workflows and controller
validation without introducing a cycle between script orchestrators. No learner,
chooser, source selection, probe/final access or independent budget belongs here.
"""

from dataclasses import asdict, replace


def collect_train_pair(check_boundary, reports, *, layout=None):
    """Collect fresh G (default) or explicit A4 TRAIN through native teacher ports.

    A4 retains its own observed length, bounded by16, instead of inheriting the
    original G nine-row contract. No saved episode or learner enters this port.
    """
    from src.core.minigrid_key_door import (
        CueKeyDoorTeacher,
        MiniGridKeyDoorLayout,
        predefined_recurring_train_layout,
    )
    from src.core.minigrid_public_history import encode_public_history_episode
    from src.infra.minigrid_key_door import MiniGridKeyDoor

    original = predefined_recurring_train_layout()
    if layout is None:
        layout = original
    if type(layout) is not MiniGridKeyDoorLayout or set(vars(layout)) != set(asdict(original)):
        raise ValueError("TRAIN collection requires an exact G or A4 layout")
    owned = replace(layout)
    is_g = owned == original
    is_red = owned == MiniGridKeyDoorLayout("A", branch_row=3, wall_color="red")
    if not is_g and not is_red and owned != MiniGridKeyDoorLayout("A", branch_row=4):
        raise ValueError("TRAIN collection supports only original G, fixed A4 or fixed red-wall A3")
    identity_prefix = "G_train" if is_g else "A3-redwall-train" if is_red else "A4-train"
    sources = []
    for cue in ("blue", "red"):
        check_boundary()
        identity = f"minigrid-source-{identity_prefix}-{cue}"
        try:
            environment = MiniGridKeyDoor(
                owned,
                cue,
                seed=17,
                max_steps=16,
                role="train",
                episode_id=identity,
            )
        except Exception as error:
            reports.append(
                dict(
                    status="construction_failed",
                    role="train",
                    context=owned.context,
                    cue=cue,
                    episode_id=identity,
                    closed=None,
                    environment_steps_attempted=None,
                    environment_steps=None,
                    failure_work_may_be_unobserved=True,
                    reason=f"{type(error).__name__}: {error}"[:512],
                )
            )
            raise
        problem = None
        try:
            # Why: a teacher-construction failure must still close the already
            # created environment and retain its physical report.
            teacher = CueKeyDoorTeacher()
            for _ in range(16):
                check_boundary()
                action = teacher.choose_action(environment.observation)
                check_boundary()
                environment.step(action)
                check_boundary()
                if environment.terminal:
                    break
            episode = environment.to_demonstration()
            if (
                ((is_g or is_red) and len(episode.transitions) != 9)
                or not 1 <= len(episode.transitions) <= 16
                or environment.report["total_return"] <= 0
            ):
                raise ValueError(
                    "original G teacher must complete nine steps with positive return"
                    if is_g
                    else "A4 teacher must complete1..16 steps with positive return"
                )
            if (episode.role, episode.context, episode.episode_id) != (
                "train",
                owned.context,
                identity,
            ):
                raise ValueError("fresh TRAIN episode role/context/identity changed")
        except Exception as error:
            problem = error
            raise
        finally:
            cleanup = None
            try:
                environment.close()
            except Exception as error:
                cleanup = error
            try:
                observed = environment.report
            except Exception as error:
                observed = dict(failure_work_may_be_unobserved=True)
                cleanup = cleanup or error
            if problem is not None:
                observed["primary_error"] = f"{type(problem).__name__}: {problem}"[:512]
            if cleanup is not None:
                observed["cleanup_error"] = f"{type(cleanup).__name__}: {cleanup}"[:512]
            reports.append(observed)
            if cleanup is not None and problem is None:
                raise cleanup
        # Why: the next source is admitted only after the original native report
        # confirms cleanup. Original TRAIN role and episode identity are retained.
        check_boundary()
        if observed.get("status") != "complete" or observed.get("closed") is not True:
            raise ValueError("original G source must have a complete closed native record")
        if not is_g and (
            observed.get("layout") != asdict(owned)
            or (
                observed.get("role"),
                observed.get("context"),
                observed.get("episode_id"),
                observed.get("cue"),
            )
            != ("train", owned.context, identity, cue)
            or observed.get("terminated") is not True
            or observed.get("truncated") is not False
            or observed.get("failure_work_may_be_unobserved") is not False
            or observed.get("environment_steps") != len(episode.transitions)
            or observed.get("environment_steps_attempted") != len(episode.transitions)
            or type(observed.get("total_return")) not in (int, float)
            or observed.get("total_return") != sum(t.reward for t in episode.transitions)
            or observed.get("total_return", 0) <= 0
        ):
            raise ValueError("fresh A4 TRAIN episode and closed native record disagree")
        if is_red:
            from src.core.demonstration_episode import DemonstrationTransition

            if observed.get("seed") != 17 or observed.get("max_steps") != 16:
                raise ValueError("red-wall TRAIN collection must retain seed17/max16")
            rows = observed.get("transitions", ())
            if len(rows) != 9 or any(r.get("applied") is not True for r in rows):
                raise ValueError("complete returned red-wall transitions required")
            native = tuple(
                DemonstrationTransition(
                    i,
                    tuple(r["observation"]),
                    r["action"],
                    r["reward"],
                    tuple(r["next_observation"]),
                    r["terminated"],
                    7,
                )
                for i, r in enumerate(rows)
            )
            if native != episode.transitions:
                raise ValueError("red-wall packet differs from its native collection record")
        sources.append(encode_public_history_episode(episode))
    return tuple(sources)
