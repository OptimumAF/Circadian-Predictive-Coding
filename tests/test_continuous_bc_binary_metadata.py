"""Synthetic legacy policy compatibility; no accepted experiment artifact reads."""

from dataclasses import asdict
import importlib
import json
import sys

import numpy as np
import pytest

from src.adapters.numpy_continuous_bc import ContinuousBCPolicy
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.observation_standardization import ObservationStandardization
from src.infra.maniskill_continuous import ACTION_UNITS

if sys.platform == "win32":
    pytest.skip("Legacy native/TRAIN harnesses require Linux resource", allow_module_level=True)


@pytest.fixture(params=["raw", "standardized", "clock", "continued_raw", "continued_standardized"])
def legacy_reader(request, monkeypatch, tmp_path):
    label = request.param
    module_name = {
        "raw": "frozen-bc-train-rollout-20261019.native_case",
        "standardized": "continuous-bc-standardization-20261019.native_case",
        "clock": "continuous-bc-clock-20261019.native_case",
        "continued_raw": "continuous-bc-1024-20261019.continue_training",
        "continued_standardized": "continuous-bc-1024-20261019.continue_training",
    }[label]
    reader = importlib.import_module("artifacts.runs." + module_name)
    monkeypatch.setattr(reader, "source", tmp_path)
    raw_spec = ContinuousActionSpec(2, ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    episode = ContinuousEpisode(
        "synthetic",
        "pick-cube",
        "train",
        raw_spec,
        (
            ContinuousTransition(
                0, (1.0, 2.0), (0.0, 0.0, 0.0, -1.0), 0.0, (2.0, 3.0), False, False
            ),
            ContinuousTransition(1, (2.0, 3.0), (0.0, 0.0, 0.0, 1.0), 1.0, (3.0, 4.0), True, False),
        ),
    )
    dimension = 3 if label == "clock" else 2
    fitted = label in ("standardized", "clock", "continued_standardized")
    transform = (
        ObservationStandardization((0.0,) * dimension, (1.0,) * dimension, 29) if fitted else None
    )
    policy = ContinuousBCPolicy(
        ContinuousActionSpec(dimension, ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4),
        observation_standardization=transform,
    )
    metadata = dict(
        method="zero" if label == "clock" else "standardized" if fitted else "raw",
        complete=True,
        final_updates=256,
        final_training_rows=7424,
        checkpoints=[dict(total_updates=step, full_state_exact=True) for step in (0, 64, 128, 256)],
        policy=json.loads(json.dumps(asdict(policy))),
    )
    if label == "clock":
        metadata["clock_recipe"] = dict(horizon=50, elapsed=False)
    state_file = "bc-256-state.npz" if label == "raw" else f"{metadata['method']}-256-state.npz"
    metadata["state_file"] = state_file
    if label == "raw":
        report = dict(
            complete=True,
            original_owner_exact=True,
            final_total_updates=256,
            final_total_training_rows=7424,
            final_state_file=state_file,
            policy=metadata["policy"],
        )
        path = tmp_path / "train-curve.json"
    else:
        report = dict(complete=True, initial_arrays_and_counters_exact=True, methods=[metadata])
        path = tmp_path / "train-comparison.json"
    shapes = ((dimension, 64), (64,), (64, 4), (4,))
    arrays = {
        f"{prefix}{index}": np.full(shape, index * 0.01, dtype=np.float64)
        for prefix in ("p", "v")
        for index, shape in enumerate(shapes)
    }
    np.savez(tmp_path / state_file, **arrays)

    def restore():
        path.write_text(json.dumps(report), encoding="utf8")
        if label.startswith("continued"):
            return reader.restore_owner(metadata, episode)
        result = (
            reader.restore_final(episode, "zero")
            if label == "clock"
            else reader.restore_final(episode)
        )
        return result[0] if label == "clock" else result

    return reader, metadata, report, path, restore


@pytest.mark.parametrize("presence", ["missing", "explicit_none"])
def test_should_restore_only_default_binary_axis_without_mutating_legacy_metadata(
    legacy_reader, presence
):
    _, metadata, report, path, restore = legacy_reader
    if presence == "missing":
        del metadata["policy"]["binary_gripper_axis"]
    before = json.dumps(report, sort_keys=True)
    model = restore()
    state = model.snapshot_state()
    assert state.policy.binary_gripper_axis is None
    assert state.completed_updates == 256 and state.training_examples == 7424
    assert json.dumps(report, sort_keys=True) == before
    assert json.loads(path.read_text(encoding="utf8")) == json.loads(before)
    assert ("binary_gripper_axis" in metadata["policy"]) is (presence == "explicit_none")


@pytest.mark.parametrize(
    "bad", ["binary_nondefault", "unknown_field", "seed", "learning_rate", "momentum", "role"]
)
def test_should_keep_every_other_legacy_policy_comparison_strict_before_npz(
    legacy_reader, monkeypatch, bad
):
    reader, metadata, report, _, restore = legacy_reader
    policy = metadata["policy"]
    policy.pop("binary_gripper_axis")
    if bad == "binary_nondefault":
        policy["binary_gripper_axis"] = 3
    elif bad == "unknown_field":
        policy["unexpected_new_option"] = None
    else:
        field, value = {
            "seed": ("seed", 18),
            "learning_rate": ("learning_rate", 0.02),
            "momentum": ("momentum", 0.8),
            "role": ("training_role", "probe_train"),
        }[bad]
        policy[field] = value
    before = json.dumps(report, sort_keys=True)
    monkeypatch.setattr(
        reader.np, "load", lambda *_args, **_kwargs: pytest.fail("invalid metadata reached NPZ")
    )
    with pytest.raises((AssertionError, ValueError)):
        restore()
    assert json.dumps(report, sort_keys=True) == before
