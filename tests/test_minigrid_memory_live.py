"""Optional actual-environment contract parity, no native model updates."""

from importlib import metadata

import numpy as np
import pytest

from src.infra.minigrid_memory import (
    MiniGridMemory,
    ENVIRONMENT_ID,
    VERSIONS,
    encode_memory_observation,
)
from scripts.run_minigrid_memory_smoke import UPPER_BRANCH_ACTIONS


def optional_dependencies():
    for name, version in VERSIONS.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            pytest.skip("install requirements-minigrid.txt for actual environment checks")
        if installed != version:
            pytest.skip(f"actual environment check requires {name}=={version}")


@pytest.mark.parametrize("actions", [UPPER_BRANCH_ACTIONS, (0,) * 16])
def test_should_match_each_actual_native_step_without_hidden_state_or_action_translation(actions):
    optional_dependencies()
    import gymnasium as gym
    import minigrid  # noqa: F401

    raw = gym.make(ENVIRONMENT_ID, max_steps=16, render_mode=None)
    try:
        observation, _ = raw.reset(seed=17)
        with MiniGridMemory(17, episode_id="parity17") as environment:
            assert environment.observation == encode_memory_observation(observation)
            assert environment.legal_actions == tuple(range(7))
            count = 0
            for action in actions:
                actual = raw.step(action)
                returned = environment.step(action)
                count += 1
                expected_observation, reward, terminated, truncated, _ = actual
                assert returned == dict(
                    observation=encode_memory_observation(expected_observation),
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                )
                row = environment.report["transitions"][-1]
                np.testing.assert_array_equal(
                    row["raw_next_observation"]["image"], expected_observation["image"]
                )
                assert row["action"] == action and row["applied"]
                if terminated or truncated:
                    break
            report = environment.report
            assert report["environment_steps"] == report["environment_steps_attempted"] == count
            assert report["reset_calls"] == 1 and report["status"] == "complete"
            assert report["terminated"] != report["truncated"]
            if actions == UPPER_BRANCH_ACTIONS:
                episode = environment.to_demonstration()
                assert len(episode.transitions) == count == 7
                assert episode.role == "train" and episode.num_classes == 7
            else:
                assert count == 16 and report["truncated"]
                with pytest.raises(ValueError, match="actual termination"):
                    environment.to_demonstration()
    finally:
        raw.close()
