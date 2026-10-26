"""Frozen public RGB/proprioception/goal features, without IO or training authority.

Own a128x128 RGB frame, pool nonoverlapping8x8 blocks, and concatenate explicit
public vectors. No object geometry, learned weights, fitting, RNG or simulator.
"""

from dataclasses import dataclass, fields
import math

import numpy as np


@dataclass(frozen=True)
class PickCubeRGBRecipe:
    recipe_id: str = "pick_cube_rgb_pool16_qpos_qvel_goal_v1"
    camera: str = "base_camera"
    frame_shape: tuple[int, int, int] = (128, 128, 3)
    camera_eye: tuple[float, ...] = (0.3, 0.0, 0.6)
    camera_target: tuple[float, ...] = (-0.1, 0.0, 0.1)
    camera_fov: float = math.pi / 2
    camera_near: float = 0.01
    camera_far: float = 100.0
    shader_pack: str = "minimal"
    render_backend: str = "sapien_cpu"
    pixel_layout: str = "HWC_RGB_uint8"
    block_size: int = 8
    pooled_shape: tuple[int, int, int] = (16, 16, 3)
    feature_order: tuple[str, ...] = ("qpos9", "qvel9", "public_goal3", "pooled_RGB768")
    feature_dim: int = 789
    pixel_divisor: float = 255.0


def pick_cube_rgb_recipe() -> PickCubeRGBRecipe:
    """Return fresh immutable metadata for the one fixed supported recipe."""
    return PickCubeRGBRecipe()


def validate_pick_cube_rgb_recipe(value: PickCubeRGBRecipe) -> PickCubeRGBRecipe:
    expected = pick_cube_rgb_recipe()
    if type(value) is not PickCubeRGBRecipe:
        raise ValueError("expected the exact frozen RGB recipe")
    for field in fields(expected):
        actual, target = getattr(value, field.name), getattr(expected, field.name)
        if (
            type(actual) is not type(target)
            or (
                type(target) is tuple
                and (
                    len(actual) != len(target)
                    or any(type(a) is not type(b) for a, b in zip(actual, target, strict=True))
                )
            )
            or actual != target
        ):
            raise ValueError("frozen RGB recipe changed")
    return expected


@dataclass(frozen=True)
class PickCubeRGBEncoding:
    features: tuple[float, ...]
    frame: np.ndarray


def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype("float64")
        or value.shape != (size,)
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be an exact finite float64 vector of{size}")
    return value


def encode_pick_cube_rgb(
    frame: np.ndarray, qpos: np.ndarray, qvel: np.ndarray, goal: np.ndarray
) -> PickCubeRGBEncoding:
    """Encode public inputs only; frame and features own independent values."""
    if (
        type(frame) is not np.ndarray
        or frame.dtype != np.dtype("uint8")
        or frame.shape != (128, 128, 3)
    ):
        raise ValueError("frame must be an exact uint8 HWC_RGB128x128x3 array")
    vectors = (_vector(qpos, 9, "qpos"), _vector(qvel, 9, "qvel"), _vector(goal, 3, "goal"))
    owned = frame.copy(order="C")
    # Why: spatial block means use every pixel without learned fitting or a
    # resize library's implicit interpolation/channel/range conventions.
    pixels = owned.reshape(16, 8, 16, 8, 3).mean(axis=(1, 3), dtype=np.float64) / 255.0
    features = np.concatenate((*vectors, pixels.reshape(-1)))
    return PickCubeRGBEncoding(tuple(float(value) for value in features), owned)
