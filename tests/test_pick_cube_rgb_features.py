"""Independent pixel/block/order oracles for the fixed public RGB recipe."""

from dataclasses import FrozenInstanceError, asdict
import math

import numpy as np
import pytest

from src.core import pick_cube_rgb_features as rgb


def vectors():
    return (
        np.arange(9, dtype=np.float64) - 4.0,
        np.arange(9, dtype=np.float64) * 0.25,
        np.array([0.1, -0.2, 0.3], dtype=np.float64),
    )


def test_should_match_analytical_block_means_rgb_channel_order_and_public_prefix():
    row, column = np.indices((128, 128))
    frame = np.stack((row, column, (row // 8) * 16 + column // 8), axis=2).astype(np.uint8)
    qpos, qvel, goal = vectors()
    result = rgb.encode_pick_cube_rgb(frame, qpos, qvel, goal)
    assert type(result) is rgb.PickCubeRGBEncoding
    assert type(result.features) is tuple and len(result.features) == 789
    assert all(type(value) is float and math.isfinite(value) for value in result.features)
    assert result.features[:9] == tuple(qpos)
    assert result.features[9:18] == tuple(qvel)
    assert result.features[18:21] == tuple(goal)
    for block_row in range(16):
        for block_column in range(16):
            offset = 21 + 3 * (16 * block_row + block_column)
            expected = (
                (8 * block_row + 3.5) / 255,
                (8 * block_column + 3.5) / 255,
                (16 * block_row + block_column) / 255,
            )
            assert result.features[offset : offset + 3] == pytest.approx(expected, abs=1e-15)
    assert min(result.features[21:]) >= 0.0 and max(result.features[21:]) <= 1.0


def test_should_average_each_pixel_once_without_spilling_across_blocks():
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[7, 7, 0] = 255
    frame[8, 16, 2] = 128
    result = rgb.encode_pick_cube_rgb(frame, *vectors())
    pixels = result.features[21:]
    assert pixels[0] == 1 / 64
    assert pixels[3 * (16 + 2) + 2] == pytest.approx(128 / (64 * 255), abs=1e-15)
    assert sum(value != 0.0 for value in pixels) == 2


@pytest.mark.parametrize("value,expected", [(0, 0.0), (255, 1.0)])
def test_should_preserve_exact_black_and_white_pixel_endpoints(value, expected):
    result = rgb.encode_pick_cube_rgb(np.full((128, 128, 3), value, dtype=np.uint8), *vectors())
    assert result.features[21:] == (expected,) * 768


def test_should_own_c_order_frame_and_feature_values_from_noncontiguous_inputs():
    original = np.arange(128 * 128 * 3, dtype=np.int32).astype(np.uint8).reshape(128, 128, 3)
    frame = original[::-1]
    qpos, qvel, goal = vectors()
    result = rgb.encode_pick_cube_rgb(frame, qpos, qvel, goal)
    feature_values = result.features
    assert result.frame.flags.c_contiguous and not np.shares_memory(result.frame, frame)
    np.testing.assert_array_equal(result.frame, frame)
    frame.fill(0)
    qpos.fill(999.0)
    qvel.fill(999.0)
    goal.fill(999.0)
    assert result.features == feature_values
    result.frame.fill(255)
    assert result.features == feature_values
    assert np.all(original == 0)


def test_should_keep_public_vector_values_in_physical_units_without_image_scaling():
    qpos, qvel, goal = vectors()
    qpos[0], qvel[0], goal[0] = 1e308, -1e308, 123.5
    result = rgb.encode_pick_cube_rgb(np.zeros((128, 128, 3), dtype=np.uint8), qpos, qvel, goal)
    assert result.features[0] == 1e308 and result.features[9] == -1e308
    assert result.features[18] == 123.5


@pytest.mark.parametrize(
    "bad", ["list", "float32", "float64", "int32", "batch", "gray", "rgba", "height", "subclass"]
)
def test_should_refuse_ambiguous_frame_type_dtype_layout_or_dimensions(bad):
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    if bad == "list":
        frame = frame.tolist()
    elif bad in ("float32", "float64", "int32"):
        frame = frame.astype({"float32": np.float32, "float64": np.float64, "int32": np.int32}[bad])
    elif bad == "batch":
        frame = frame[None]
    elif bad == "gray":
        frame = frame[:, :, 0]
    elif bad == "rgba":
        frame = np.zeros((128, 128, 4), dtype=np.uint8)
    elif bad == "height":
        frame = frame[:127]
    else:

        class ForeignArray(np.ndarray):
            pass

        frame = frame.view(ForeignArray)
    with pytest.raises(ValueError):
        rgb.encode_pick_cube_rgb(frame, *vectors())


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("bad", ["float32", "integer", "list", "batch", "short", "nan", "inf"])
def test_should_refuse_invalid_public_vector_without_flattening_or_casting(index, bad):
    values = list(vectors())
    value = values[index]
    if bad == "float32":
        value = value.astype(np.float32)
    elif bad == "integer":
        value = value.astype(np.int64)
    elif bad == "list":
        value = value.tolist()
    elif bad == "batch":
        value = value[None]
    elif bad == "short":
        value = value[:-1]
    else:
        value[0] = np.nan if bad == "nan" else np.inf
    values[index] = value
    with pytest.raises(ValueError):
        rgb.encode_pick_cube_rgb(np.zeros((128, 128, 3), dtype=np.uint8), *values)


def test_should_publish_one_fresh_immutable_explicit_recipe_without_rng(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("fixed visual encoding attempted randomness")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    first, second = rgb.pick_cube_rgb_recipe(), rgb.pick_cube_rgb_recipe()
    assert first == second and first is not second
    assert first.recipe_id == "pick_cube_rgb_pool16_qpos_qvel_goal_v1"
    assert first.frame_shape == (128, 128, 3) and first.pooled_shape == (16, 16, 3)
    assert first.feature_dim == 789 and first.block_size == 8 and first.pixel_divisor == 255
    assert first.feature_order == ("qpos9", "qvel9", "public_goal3", "pooled_RGB768")
    assert first.camera == "base_camera" and first.shader_pack == "minimal"
    assert first.camera_eye == (0.3, 0.0, 0.6) and first.camera_target == (-0.1, 0.0, 0.1)
    assert first.camera_fov == math.pi / 2 and first.camera_near == 0.01 and first.camera_far == 100
    assert first.render_backend == "sapien_cpu" and first.pixel_layout == "HWC_RGB_uint8"
    with pytest.raises(FrozenInstanceError):
        first.feature_dim = 1
    assert asdict(second)["feature_dim"] == 789
    assert (
        len(rgb.encode_pick_cube_rgb(np.zeros((128, 128, 3), dtype=np.uint8), *vectors()).features)
        == 789
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_size", 8.0),
        ("feature_dim", np.int64(789)),
        ("pixel_divisor", 255),
        ("camera_near", 0.02),
        ("frame_shape", (128.0, 128, 3)),
        ("camera_eye", (0.3, 0, 0.6)),
        ("shader_pack", "default"),
    ],
)
def test_should_revalidate_recipe_values_and_types_including_equal_numeric_values(field, value):
    original = rgb.pick_cube_rgb_recipe()
    valid = rgb.validate_pick_cube_rgb_recipe(original)
    assert valid == original and valid is not original
    object.__setattr__(original, field, value)
    with pytest.raises(ValueError):
        rgb.validate_pick_cube_rgb_recipe(original)


def test_should_refuse_foreign_recipe_instead_of_accepting_matching_attributes():
    with pytest.raises(ValueError):
        rgb.validate_pick_cube_rgb_recipe(asdict(rgb.pick_cube_rgb_recipe()))
