"""Owned recorded readiness stages for existing TRAIN residual diagnostics.

Bind trusted complete demonstration episodes to their actual command phases.
External files still require the existing teacher qualification reader. No phase
inference, qualification, learner inputs, loss weighting, model, timer or IO is
provided; metadata contains no duplicate observation/action payload.
"""

from dataclasses import dataclass
from typing import Any

from src.core.continuous_demonstration import ContinuousEpisode, copy_continuous_episode
from src.core.pick_cube_corrections import _UNITS
from src.core.pick_cube_readiness_corrections import (
    ReadinessCorrectionLabel,
    _observed_grasp_flags,
)
from src.core.pick_cube_readiness_teacher import (
    DEFINITION as READINESS_DEFINITION,
    OBSERVED_GRASP_DEFINITION,
)
from src.core.pick_cube_train_context import pick_cube_training_context


@dataclass(frozen=True)
class ReadinessStageBinding:
    episode: ContinuousEpisode
    stages: tuple[str, ...]

    def __post_init__(self) -> None:
        selected = _observed_stage_episode(self.episode)
        flags = _observed_grasp_flags(self.episode) if selected else None
        owned = copy_continuous_episode(self.episode)
        identities = {
            f"pick-cube-{family}-teacher-{seed}": pick_cube_training_context(seed)
            for seed in (17, 19)
            for family in ("readiness", "observed-grasp")
        }
        # Why: retain the collected TRAIN20 demonstration without widening the
        # ordered warm17/new19 diagnostic pair or admitting reserved identities.
        identities["pick-cube-observed-grasp-teacher-20"] = pick_cube_training_context(20)
        if (
            owned.role != "train"
            or owned.episode_id not in identities
            or owned.context != identities[owned.episode_id]
            or owned.spec.action_units != _UNITS
            or owned.spec.lower_bounds != (-1.0,) * 4
            or owned.spec.upper_bounds != (1.0,) * 4
            or len(owned.transitions) > 50
        ):
            raise ValueError(
                "stage binding requires readiness TRAIN17/19 or observed-grasp TRAIN20"
            )
        if type(self.stages) is not tuple or len(self.stages) != len(owned.transitions):
            raise ValueError("stage binding requires one exact recorded stage per transition")
        # The readiness label validator checks stage names and command semantics
        # without assuming an elapsed schedule or granting label-release authority.
        stages = tuple(
            ReadinessCorrectionLabel(row.step, row.observation, row.action, stage).stage
            for row, stage in zip(owned.transitions, self.stages, strict=True)
        )
        if flags is not None and any(
            (stage in ("lift", "settle")) is not flag
            for stage, flag in zip(stages, flags, strict=True)
        ):
            raise ValueError("observed-grasp stages differ from their current official flags")
        object.__setattr__(self, "episode", owned)
        object.__setattr__(self, "stages", stages)


def _observed_stage_episode(episode) -> bool:
    return (
        type(episode) is ContinuousEpisode
        and type(episode.episode_id) is str
        and episode.episode_id.startswith("pick-cube-observed-grasp-teacher-")
    )


def _copy_stage_episode(episode):
    # Why: re-owning a forged int/bool flag must not normalize it into evidence.
    if _observed_stage_episode(episode):
        _observed_grasp_flags(episode)
    return copy_continuous_episode(episode)


def _copy_pair(
    bindings: tuple[ReadinessStageBinding, ReadinessStageBinding],
) -> tuple[ReadinessStageBinding, ReadinessStageBinding]:
    if type(bindings) is not tuple or len(bindings) != 2:
        raise ValueError("readiness stage bindings require an exact warm/new two-tuple")
    if any(type(binding) is not ReadinessStageBinding for binding in bindings):
        raise ValueError("expected exact ReadinessStageBinding records")
    warm = ReadinessStageBinding(bindings[0].episode, bindings[0].stages)
    new = ReadinessStageBinding(bindings[1].episode, bindings[1].stages)
    if any(
        binding.episode.episode_id
        not in (
            f"pick-cube-readiness-teacher-{seed}",
            f"pick-cube-observed-grasp-teacher-{seed}",
        )
        for binding, seed in zip((warm, new), (17, 19), strict=True)
    ):
        raise ValueError("readiness stage bindings must be ordered warm17 then new19")
    return warm, new


def copy_readiness_stage_bindings(
    bindings: tuple[ReadinessStageBinding, ReadinessStageBinding],
    episodes: tuple[ContinuousEpisode, ContinuousEpisode],
) -> tuple[ReadinessStageBinding, ReadinessStageBinding]:
    """Re-own stages only when every episode field matches the supplied inputs."""
    if type(episodes) is not tuple or len(episodes) != 2:
        raise ValueError("readiness stage source episodes require an exact two-tuple")
    expected = tuple(_copy_stage_episode(episode) for episode in episodes)
    owned = _copy_pair(bindings)
    if any(binding.episode != episode for binding, episode in zip(owned, expected, strict=True)):
        raise ValueError("readiness stage bindings differ from the full warm/new source episodes")
    return owned


def readiness_stage_metadata(
    bindings: tuple[ReadinessStageBinding, ReadinessStageBinding],
) -> dict[str, dict[str, Any]]:
    """Return bounded detached phase metadata, revalidating all source records."""
    owned = _copy_pair(bindings)
    return {
        name: dict(
            definition=OBSERVED_GRASP_DEFINITION
            if _observed_stage_episode(binding.episode)
            else READINESS_DEFINITION,
            episode_id=binding.episode.episode_id,
            context=binding.episode.context,
            role=binding.episode.role,
            rows=len(binding.episode.transitions),
            stages=binding.stages,
        )
        for name, binding in zip(("warm", "new"), owned, strict=True)
    }
