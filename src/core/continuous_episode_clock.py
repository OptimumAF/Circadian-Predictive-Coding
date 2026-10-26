"""Bounded public step features for continuous TRAIN demonstrations and actors.

Pure owned feature projection; raw prefixes, controls and completion flags stay
unchanged. No simulator, learner, teacher geometry, future reward, IO, role
conversion or release authority. Zero-clock is an equal-width neutral control.
"""

from dataclasses import dataclass

from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
    _copy_spec,
    _vector,
    copy_continuous_episode,
)


@dataclass(frozen=True)
class EpisodeClock:
    horizon: int
    elapsed: bool

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or not 1 <= self.horizon <= 256:
            raise ValueError("clock horizon must be a native integer in 1..256")
        if type(self.elapsed) is not bool:
            raise ValueError("clock elapsed mode must be an exact bool")


def _clock(value: EpisodeClock) -> EpisodeClock:
    if type(value) is not EpisodeClock:
        raise ValueError("expected an exact EpisodeClock")
    return EpisodeClock(value.horizon, value.elapsed)


def clock_observation(
    observation: tuple[float, ...], step: int, clock: EpisodeClock
) -> tuple[float, ...]:
    """Append current public step/horizon, or zero; terminal next-step is valid."""
    owned = _clock(clock)
    if type(step) is not int or not 0 <= step <= owned.horizon:
        raise ValueError("clock step must be a native integer in 0..horizon")
    raw = _vector(observation, "raw observation", 1023)
    return (*raw, step / owned.horizon if owned.elapsed else 0.0)


def clock_spec(spec: ContinuousActionSpec, clock: EpisodeClock) -> ContinuousActionSpec:
    _clock(clock)
    owned = _copy_spec(spec)
    if owned.observation_dim >= 1024:
        raise ValueError("clock requires room for one additional observation coordinate")
    return ContinuousActionSpec(
        owned.observation_dim + 1, owned.action_units, owned.lower_bounds, owned.upper_bounds
    )


def clock_training_episode(episode: ContinuousEpisode, clock: EpisodeClock) -> ContinuousEpisode:
    """Project a complete TRAIN record only, with causal current/next-step clocks.

    Identity/context/role and every actual control/reward/flag remain unchanged.
    The caller records the explicit feature recipe alongside the derived data.
    """
    if type(episode) is not ContinuousEpisode or episode.role != "train":
        raise ValueError("clock training projection requires a complete TRAIN episode")
    owned = copy_continuous_episode(episode)
    config = _clock(clock)
    if len(owned.transitions) > config.horizon:
        raise ValueError("episode exceeds the declared clock horizon")
    spec = clock_spec(owned.spec, config)
    transitions = tuple(
        ContinuousTransition(
            row.step,
            clock_observation(row.observation, row.step, config),
            row.action,
            row.reward,
            clock_observation(row.next_observation, row.step + 1, config),
            row.terminated,
            row.truncated,
        )
        for row in owned.transitions
    )
    return ContinuousEpisode(owned.episode_id, owned.context, owned.role, spec, transitions)
