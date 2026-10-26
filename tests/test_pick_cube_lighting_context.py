"""One fixed ambient-light binding through fake renderer IO; no physics claim."""

from dataclasses import FrozenInstanceError, asdict, replace
from typing import Any, cast

import numpy as np
import pytest

from src.core.pick_cube_camera_context import pick_cube_camera_context
from src.core.pick_cube_mass_context import pick_cube_mass_context
from src.core.pick_cube_lighting_context import (
    PickCubeLightingContext,
    pick_cube_lighting_context,
    require_lighting_binding,
    validate_pick_cube_lighting_context,
)
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.infra import maniskill_continuous as port
from src.infra import maniskill_lighting as lighting
from test_maniskill_continuous import ForeignArray
from test_maniskill_rgb_observations import RGBEnvironment, packet, public_feature_oracle


CONTEXT = "pick-cube-ambient-half-train-17"
ACTION = (0.125, -0.25, 0.5, -1.0)


class Scene:
    def __init__(self, events):
        self.events = events
        self.value = np.array([1.0, 0.5, 0.25], dtype=np.float32)
        self.reads = 0
        self.writes = []
        self.read_error_at = None
        self.write_error = None
        self.on_read = self.after_write = None

    @property
    def ambient_light(self):
        self.events.append("read")
        self.reads += 1
        if self.reads == self.read_error_at:
            raise RuntimeError("ambient read failed")
        if self.on_read is not None:
            self.on_read()
        return self.value

    def set_ambient_light(self, value):
        self.events.append("write")
        self.writes.append(np.array(value, copy=True))
        if self.write_error is not None:
            raise self.write_error
        self.value = np.asarray(value, dtype=np.float32).copy()
        if self.after_write is not None:
            self.after_write()


class Environment(RGBEnvironment):
    def __init__(self):
        super().__init__()
        self.events: list[str] = []
        self.scene = Scene(self.events)
        self.reset_scene = None
        self.refreshes = 0
        self.refresh_error = None
        self.refreshed_packet = packet(20)
        self.after_refresh = self.after_step = None

    def reset(self, *, seed):
        self.events.append("reset")
        result = super().reset(seed=seed)
        if self.reset_scene is not None:
            self.scene = self.reset_scene
        return result

    def get_obs(self):
        self.events.append("refresh")
        self.refreshes += 1
        if self.refresh_error is not None:
            raise self.refresh_error
        if self.after_refresh is not None:
            self.after_refresh()
        return self.refreshed_packet

    def step(self, action):
        self.events.append("step")
        result = super().step(action)
        if self.after_step is not None:
            self.after_step()
        return result

    def close(self):
        self.events.append("close")
        super().close()


@pytest.fixture
def boundary(monkeypatch):
    environment, calls = Environment(), []

    def factory(**kwargs):
        calls.append(kwargs)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    yield environment, calls
    if calls and environment.close_calls == 0:
        environment.close()


def open_lighting(**changes):
    options: dict[str, Any] = dict(
        context=CONTEXT,
        seed=17,
        visual_observation=True,
        lighting_context=pick_cube_lighting_context(),
    )
    options.update(changes)
    return port.ManiSkillPickCubeEpisode("lighting-17", **options)


def test_should_freeze_one_fresh_exact_visual_train17_ambient_declaration():
    source = pick_cube_lighting_context()
    assert asdict(source) == dict(
        context=CONTEXT,
        seed=17,
        ambient_multiplier=0.5,
        semantics="ambient_only_directional_lights_unchanged_by_this_port",
    )
    owned = validate_pick_cube_lighting_context(source)
    assert owned == source and owned is not source and pick_cube_lighting_context() is not source
    with pytest.raises(FrozenInstanceError):
        setattr(source, "seed", 19)
    object.__setattr__(source, "seed", 19)
    assert owned.seed == 17


@pytest.mark.parametrize(
    "changes",
    [
        dict(context="pick-cube"),
        dict(context=CONTEXT + " "),
        dict(context=None),
        dict(seed=19),
        dict(seed=True),
        dict(seed=17.0),
        dict(seed=np.int64(17)),
        dict(ambient_multiplier=1),
        dict(ambient_multiplier=np.float64(0.5)),
        dict(ambient_multiplier=0.25),
        dict(ambient_multiplier=float("nan")),
        dict(semantics="all_lights_scaled"),
    ],
)
def test_should_refuse_changed_or_equal_wrong_type_lighting_declarations(changes):
    with pytest.raises(ValueError):
        validate_pick_cube_lighting_context(PickCubeLightingContext(**changes))


@pytest.mark.parametrize("damage", ["none", "mapping", "subclass", "corrupted"])
def test_should_revalidate_exact_lighting_record_type(damage):
    value: Any = pick_cube_lighting_context()
    if damage == "none":
        value = None
    elif damage == "mapping":
        value = asdict(value)
    elif damage == "subclass":

        class Derived(PickCubeLightingContext):
            pass

        value = Derived()
    else:
        object.__setattr__(value, "ambient_multiplier", 1)
    with pytest.raises(ValueError):
        validate_pick_cube_lighting_context(value)


@pytest.mark.parametrize(
    "changes",
    [
        dict(context="pick-cube"),
        dict(seed=19),
        dict(seed=True),
        dict(seed=17.0),
        dict(role="inner_guard"),
        dict(role="probe_train"),
        dict(role="probe_test"),
        dict(role="final_test"),
        dict(visual_observation=False),
        dict(visual_observation=1),
    ],
)
def test_should_bind_only_explicit_visual_train17(changes):
    options: dict[str, Any] = dict(context=CONTEXT, seed=17, role="train", visual_observation=True)
    options.update(changes)
    with pytest.raises(ValueError):
        require_lighting_binding(pick_cube_lighting_context(), **options)


def test_should_preserve_default_binding_and_refuse_reserved_context_without_metadata():
    assert (
        require_lighting_binding(
            None, context="pick-cube", seed=17, role="train", visual_observation=False
        )
        is None
    )
    with pytest.raises(ValueError):
        require_lighting_binding(
            None, context=CONTEXT, seed=17, role="train", visual_observation=True
        )


@pytest.mark.parametrize(
    "changes",
    [
        dict(context="pick-cube"),
        dict(seed=19),
        dict(seed=True),
        dict(role="probe_train"),
        dict(role="probe_test"),
        dict(role="inner_guard"),
        dict(role="final_test"),
        dict(visual_observation=False),
        dict(visual_observation=1),
        dict(lighting_context=None),
        dict(lighting_context={}),
        dict(camera_context=pick_cube_camera_context()),
        dict(mass_context=pick_cube_mass_context()),
    ],
)
def test_should_refuse_invalid_or_combined_intervention_before_factory(boundary, changes):
    environment, calls = boundary
    with pytest.raises(ValueError):
        open_lighting(**changes)
    assert calls == [] and environment.events == []


def work(reads=3, returned=None, writes=1, written=None, refreshes=1, refreshed=None):
    return lighting.ManiSkillLightingWork(
        reads,
        reads if returned is None else returned,
        writes,
        writes if written is None else written,
        refreshes,
        refreshes if refreshed is None else refreshed,
        1,
    )


def test_should_use_after_reset_scene_discard_old_packet_and_encode_verified_refresh(boundary):
    environment, calls = boundary
    stale_scene = environment.scene
    current = Scene(environment.events)
    current.value[:] = (0.5, 0.75, 1.0)
    environment.reset_scene = current
    session = open_lighting()
    assert calls == [dict(visual_observation=True)] and environment.reset_calls == [17]
    assert stale_scene.reads == 0 and stale_scene.writes == []
    assert environment.events == ["reset", "read", "write", "read", "refresh", "read"]
    np.testing.assert_array_equal(current.writes[0], [0.25, 0.375, 0.5])
    assert session.observation == tuple(public_feature_oracle(environment.refreshed_packet))
    assert session.observation != tuple(public_feature_oracle(environment.reset_packet))
    assert session.spec.observation_dim == 789 and session.spec.action_units == port.ACTION_UNITS
    assert session.observation_recipe == pick_cube_rgb_recipe()
    assert session.lighting_readback.source_ambient_rgb == (0.5, 0.75, 1.0)
    assert session.lighting_readback.applied_ambient_rgb == (0.25, 0.375, 0.5)
    assert session.lighting_work == work()
    assert session.visual_work.observation_attempts == session.visual_work.observation_returns == 1
    assert environment.commands == [] and environment.refreshes == 1
    first = session.step(ACTION)
    last = session.step(ACTION)
    episode = session.episode()
    assert first.observation == tuple(public_feature_oracle(environment.refreshed_packet))
    assert first.next_observation == last.observation
    assert last.truncated and not last.terminated
    assert episode.context == CONTEXT and episode.role == "train"
    assert session.lighting_work == work(reads=8)
    assert session.work == port.ManiSkillWork(2, 2, 10, 0)
    assert session.visual_work.observation_attempts == session.visual_work.observation_returns == 3
    for command in environment.commands:
        np.testing.assert_array_equal(command, np.array([ACTION], dtype=np.float32))
    with pytest.raises(ValueError):
        session.step(ACTION)
    assert current.reads == 8 and len(current.writes) == environment.refreshes == 1


def test_should_own_original_light_readback_context_and_work_without_native_rereads(boundary):
    environment, _ = boundary
    environment.done_at = 1
    source = pick_cube_lighting_context()
    session = open_lighting(lighting_context=source)
    context, readback, costs = (
        session.lighting_context,
        session.lighting_readback,
        session.lighting_work,
    )
    assert type(readback) is lighting.ManiSkillLightingReadback
    assert readback.source_ambient_rgb == (1.0, 0.5, 0.25)
    assert readback.applied_ambient_rgb == (0.5, 0.25, 0.125)
    assert all(
        type(value) is float for value in readback.source_ambient_rgb + readback.applied_ambient_rgb
    )
    assert session.lighting_readback is not readback and session.lighting_work is not costs
    with pytest.raises(FrozenInstanceError):
        setattr(readback, "source_ambient_rgb", (0.0,) * 3)
    with pytest.raises(FrozenInstanceError):
        costs.refresh_returns = 0
    object.__setattr__(source, "seed", 19)
    object.__setattr__(context, "seed", 19)
    object.__setattr__(readback.context, "seed", 19)
    object.__setattr__(costs, "read_attempts", 0)
    assert session.lighting_context == pick_cube_lighting_context()
    assert session.lighting_readback.source_ambient_rgb == (1.0, 0.5, 0.25)
    assert environment.scene.reads == 3 and session.lighting_work == work()
    session.step(ACTION)
    session.episode()
    assert environment.scene.reads == 6 and session.lighting_work == work(reads=6)


@pytest.mark.parametrize(
    "bad",
    [
        np.ones(3, dtype=np.float64),
        np.ones(3, dtype=np.int32),
        np.ones(3, dtype=np.bool_),
        np.ones((1, 3), dtype=np.float32),
        np.ones(4, dtype=np.float32),
        np.array(0.5, dtype=np.float32),
        [1.0, 0.5, 0.25],
        np.array([0.0, 0.5, 0.25], dtype=np.float32),
        np.array([-0.5, 0.5, 0.25], dtype=np.float32),
        np.array([1.01, 0.5, 0.25], dtype=np.float32),
        np.array([float("nan"), 0.5, 0.25], dtype=np.float32),
        np.array([float("inf"), 0.5, 0.25], dtype=np.float32),
        np.ones(3, dtype=np.float32).view(ForeignArray),
    ],
)
def test_should_refuse_malformed_native_source_before_setter_or_refresh(boundary, bad):
    environment, _ = boundary
    environment.scene.value = bad
    with pytest.raises(ValueError) as caught:
        open_lighting()
    assert getattr(caught.value, "lighting_work") == work(
        reads=1, returned=0, writes=0, refreshes=0
    )
    assert getattr(caught.value, "lighting_readback") is None
    assert environment.events == ["reset", "read", "close"]


@pytest.mark.parametrize("units", [1, 3])
def test_should_refuse_unrepresentable_exact_half_before_write_and_preserve_request(
    boundary, units
):
    environment, _ = boundary
    environment.scene.value[0] = np.float32(units * 2**-149)
    with pytest.raises(ValueError) as caught:
        open_lighting()
    assert getattr(caught.value, "lighting_work") == work(reads=1, writes=0, refreshes=0)
    assert getattr(caught.value, "lighting_source_ambient_rgb")[0] == float(
        np.float32(units * 2**-149)
    )
    assert getattr(caught.value, "lighting_requested_ambient_rgb")[0] != units * 2**-150
    assert getattr(caught.value, "lighting_readback") is None
    assert environment.scene.writes == [] and environment.refreshes == 0


@pytest.mark.parametrize("damage", ["unchanged", "wrong_target", "wrong_dtype"])
def test_should_not_claim_applied_readback_before_exact_setter_verification(boundary, damage):
    environment, _ = boundary
    scene = environment.scene

    def change():
        scene.value = (
            np.array([1.0, 0.5, 0.25], dtype=np.float32)
            if damage == "unchanged"
            else np.array([0.4, 0.25, 0.125], dtype=np.float32)
            if damage == "wrong_target"
            else np.array([0.5, 0.25, 0.125], dtype=np.float64)
        )

    scene.after_write = change
    with pytest.raises(ValueError) as caught:
        open_lighting()
    assert getattr(caught.value, "lighting_work") == work(
        reads=2, returned=1 if damage == "wrong_dtype" else 2, refreshes=0
    )
    assert getattr(caught.value, "lighting_readback") is None
    assert getattr(caught.value, "lighting_source_ambient_rgb") == (1.0, 0.5, 0.25)
    assert getattr(caught.value, "lighting_requested_ambient_rgb") == (0.5, 0.25, 0.125)
    assert environment.close_calls == 1 and environment.refreshes == 0


@pytest.mark.parametrize("read", [1, 2, 3])
def test_should_preserve_read_failure_payment_and_only_previously_verified_applied_light(
    boundary, read
):
    environment, _ = boundary
    environment.scene.read_error_at = read
    with pytest.raises(RuntimeError, match="ambient read failed") as caught:
        open_lighting()
    assert getattr(caught.value, "lighting_work") == work(
        reads=read, returned=read - 1, writes=int(read > 1), refreshes=int(read == 3)
    )
    assert (getattr(caught.value, "lighting_readback") is not None) is (read == 3)
    assert environment.close_calls == 1 and environment.commands == []


@pytest.mark.parametrize("failure", ["write", "refresh", "encode"])
def test_should_keep_original_exception_and_paid_refresh_even_if_encoding_fails(boundary, failure):
    environment, _ = boundary
    if failure == "write":
        environment.scene.write_error = RuntimeError("setter failed")
    elif failure == "refresh":
        environment.refresh_error = RuntimeError("refresh failed")
    else:
        environment.refreshed_packet["sensor_data"]["base_camera"]["rgb"] = np.zeros(
            (128, 128, 3), dtype=np.uint8
        )
    with pytest.raises((RuntimeError, ValueError)) as caught:
        open_lighting()
    expected = dict(
        write=work(reads=1, written=0, refreshes=0),
        refresh=work(reads=2, refreshed=0),
        encode=work(),
    )
    assert getattr(caught.value, "lighting_work") == expected[failure]
    assert (getattr(caught.value, "lighting_readback") is not None) is (failure != "write")
    assert environment.close_calls == 1 and environment.commands == []


def test_should_not_access_lighting_if_reset_fails_before_binding(boundary):
    environment, _ = boundary
    environment.reset_error = RuntimeError("reset failed")
    with pytest.raises(RuntimeError, match="reset failed") as caught:
        open_lighting()
    assert not hasattr(caught.value, "lighting_work")
    assert environment.events == ["reset", "close"]


def test_should_preserve_original_lighting_failure_if_cleanup_also_throws(boundary):
    environment, _ = boundary
    environment.scene.read_error_at = 1
    environment.close_error = RuntimeError("cleanup also failed")
    with pytest.raises(RuntimeError, match="ambient read failed") as caught:
        open_lighting()
    assert getattr(caught.value, "lighting_work") == work(
        reads=1, returned=0, writes=0, refreshes=0
    )
    assert environment.close_calls == 1
    assert "cleanup also failed" in str(getattr(caught.value, "lighting_cleanup_error"))


@pytest.mark.parametrize(
    "damage", ["ambient", "scene", "base", "removed", "context", "helper_context"]
)
def test_should_permanently_refuse_unobserved_changes_before_control(boundary, damage):
    from types import SimpleNamespace

    environment, _ = boundary
    session = open_lighting()
    scene = environment.scene
    if damage == "ambient":
        scene.value[0] = 0.75
    elif damage == "scene":
        environment.scene = Scene(environment.events)
    elif damage == "base":
        environment.unwrapped = SimpleNamespace(scene=scene)
    elif damage == "removed":
        session._lighting_context = None
    elif damage == "context":
        session._lighting_context = replace(pick_cube_lighting_context(), seed=19)
    else:
        session._lighting._context = replace(pick_cube_lighting_context(), seed=cast(Any, 17.0))
    with pytest.raises(ValueError):
        session.step(ACTION)
    assert environment.commands == [] and scene.reads == (4 if damage == "ambient" else 3)
    environment.unwrapped = environment
    environment.scene = scene
    scene.value[:] = (0.5, 0.25, 0.125)
    session._lighting_context = session._lighting._context = pick_cube_lighting_context()
    before = len(environment.events)
    with pytest.raises(ValueError):
        session.step(ACTION)
    with pytest.raises(ValueError):
        session.episode()
    assert len(environment.events) == before


@pytest.mark.parametrize(
    "phase", ["source", "setter", "refresh", "before_control", "after_control"]
)
@pytest.mark.parametrize("damage", ["scene", "base", "context"])
def test_should_recheck_bindings_after_every_external_callback_and_keep_paid_controls(
    boundary, phase, damage
):
    from types import SimpleNamespace

    environment, _ = boundary
    scene = environment.scene
    session: Any = port.ManiSkillPickCubeEpisode.__new__(port.ManiSkillPickCubeEpisode)

    def corrupt():
        if damage == "scene":
            environment.scene = Scene(environment.events)
        elif damage == "base":
            environment.unwrapped = SimpleNamespace(scene=scene)
        else:
            session._lighting_context = None

    fail_read = dict(source=1, before_control=4, after_control=5).get(phase)
    if fail_read is not None:
        scene.on_read = lambda: corrupt() if scene.reads == fail_read else None
    elif phase == "setter":
        scene.after_write = corrupt
    else:
        environment.after_refresh = corrupt
    options: dict[str, Any] = dict(
        context=CONTEXT,
        seed=17,
        visual_observation=True,
        lighting_context=pick_cube_lighting_context(),
    )
    if phase in ("source", "setter", "refresh"):
        with pytest.raises(ValueError) as caught:
            session.__init__("lighting-17", **options)
        expected = dict(
            source=work(reads=1, returned=0, writes=0, refreshes=0),
            setter=work(reads=1, refreshes=0),
            refresh=work(reads=2),
        )[phase]
        assert getattr(caught.value, "lighting_work") == expected
        assert environment.close_calls == 1
        assert (getattr(caught.value, "lighting_readback") is not None) is (phase == "refresh")
    else:
        session.__init__("lighting-17", **options)
        with pytest.raises(ValueError):
            session.step(ACTION)
        assert fail_read is not None
        assert session.lighting_work == work(reads=fail_read, returned=fail_read - 1)
        assert session.work == port.ManiSkillWork(
            int(phase == "after_control"),
            int(phase == "after_control"),
            5 * int(phase == "after_control"),
            0,
        )
        scene.on_read = None
        environment.scene, environment.unwrapped = scene, environment
        session._lighting_context = pick_cube_lighting_context()
        before = len(environment.events)
        with pytest.raises(ValueError):
            session.step(ACTION)
        with pytest.raises(ValueError):
            session.episode()
        assert len(environment.events) == before
    assert len(environment.commands) == int(phase == "after_control")


@pytest.mark.parametrize("phase", ["read", "setter", "refresh"])
@pytest.mark.parametrize("damage", ["source", "target", "redirect"])
def test_should_refuse_self_consistent_helper_mutation_inside_external_calls(
    boundary, phase, damage
):
    from types import SimpleNamespace

    environment, _ = boundary
    scene = environment.scene
    session: Any = port.ManiSkillPickCubeEpisode.__new__(port.ManiSkillPickCubeEpisode)

    def corrupt():
        helper = session._lighting
        if damage == "source":
            helper._source = (0.5, 0.5, 0.5)
        elif damage == "target":
            helper._target = (0.25, 0.25, 0.25)
            scene.value[:] = helper._target
        else:
            other = Scene(environment.events)
            other.value[:] = (0.5, 0.25, 0.125)
            helper._scene = other
            helper._base = SimpleNamespace(scene=other)

    if phase == "read":
        scene.on_read = lambda: corrupt() if scene.reads == 4 else None
    elif phase == "setter":
        scene.after_write = corrupt
    else:
        environment.after_refresh = corrupt
    options: dict[str, Any] = dict(
        context=CONTEXT,
        seed=17,
        visual_observation=True,
        lighting_context=pick_cube_lighting_context(),
    )
    if phase == "read":
        session.__init__("lighting-17", **options)
        with pytest.raises(ValueError):
            session.step(ACTION)
        assert scene.reads == 4 and session.lighting_work.read_returns == 3
        assert environment.commands == []
        scene.on_read = None
        session._lighting._source = (1.0, 0.5, 0.25)
        session._lighting._target = (0.5, 0.25, 0.125)
        session._lighting._scene, session._lighting._base = scene, environment
        scene.value[:] = (0.5, 0.25, 0.125)
        with pytest.raises(ValueError):
            session.step(ACTION)
        with pytest.raises(ValueError):
            session.episode()
    else:
        with pytest.raises(ValueError) as caught:
            session.__init__("lighting-17", **options)
        assert getattr(caught.value, "lighting_work") == (
            work(reads=1, refreshes=0) if phase == "setter" else work(reads=2)
        )
        assert environment.commands == [] and environment.close_calls == 1


def test_should_refuse_helper_redirection_inside_identity_accessor_before_native_read(
    boundary, monkeypatch
):
    from types import SimpleNamespace

    environment, _ = boundary
    session = open_lighting()
    scene = environment.scene
    redirected = Scene(environment.events)
    calls = []
    armed = [True]

    def unwrapped(owner):
        assert owner is environment
        calls.append(True)
        if armed[0]:
            session._lighting._scene = redirected
            session._lighting._base = SimpleNamespace(scene=redirected)
        return environment

    monkeypatch.setattr(Environment, "unwrapped", property(unwrapped), raising=False)
    with pytest.raises(ValueError):
        session.step(ACTION)
    assert calls and environment.commands == [] and scene.reads == 3 and redirected.reads == 0
    assert session.lighting_work == work()
    session._lighting._scene, session._lighting._base = scene, environment
    armed[0] = False
    before, before_calls = len(environment.events), len(calls)
    with pytest.raises(ValueError):
        session.step(ACTION)
    with pytest.raises(ValueError):
        session.episode()
    assert len(environment.events) == before and len(calls) == before_calls


def test_should_verify_light_on_episode_export_and_permanently_refuse_repair(boundary):
    environment, _ = boundary
    environment.done_at = 1
    session = open_lighting()
    session.step(ACTION)
    environment.scene.value[0] = 0.75
    with pytest.raises(ValueError):
        session.episode()
    environment.scene.value[0] = 0.5
    with pytest.raises(ValueError):
        session.episode()
    assert session.lighting_work == work(reads=6)
    assert session.work == port.ManiSkillWork(1, 1, 5, 0)


@pytest.mark.parametrize("phase", ["refresh", "step"])
def test_should_reject_changed_native_light_after_returned_observation_or_paid_control(
    boundary, phase
):
    environment, _ = boundary
    scene = environment.scene
    if phase == "refresh":
        environment.after_refresh = lambda: scene.value.fill(0.75)
        with pytest.raises(ValueError) as caught:
            open_lighting()
        assert getattr(caught.value, "lighting_work") == work()
        assert getattr(caught.value, "lighting_readback").applied_ambient_rgb == (0.5, 0.25, 0.125)
        assert environment.close_calls == 1 and environment.commands == []
    else:
        session = open_lighting()
        environment.after_step = lambda: scene.value.fill(0.75)
        with pytest.raises(ValueError):
            session.step(ACTION)
        assert session.lighting_work == work(reads=5)
        assert session.work == port.ManiSkillWork(1, 1, 5, 0)
        scene.value[:] = (0.5, 0.25, 0.125)
        with pytest.raises(ValueError):
            session.step(ACTION)
        with pytest.raises(ValueError):
            session.episode()


def test_should_keep_one_refresh_and_paid_fifty_controls_when_horizon_flag_is_missing(boundary):
    environment, _ = boundary
    environment.done_at = 51
    session = open_lighting()
    for _ in range(49):
        session.step(ACTION)
    with pytest.raises(ValueError, match="time-limit flag"):
        session.step(ACTION)
    assert session.work == port.ManiSkillWork(50, 50, 250, 0)
    assert session.lighting_work == work(reads=103)
    assert environment.refreshes == 1 and len(environment.scene.writes) == 1
    assert session.visual_work.observation_returns == 51
    with pytest.raises(ValueError):
        session.episode()
    with pytest.raises(ValueError):
        session.step(ACTION)


def test_should_not_allow_reapplying_the_intervention_or_renewing_its_work(boundary):
    environment, _ = boundary
    session = open_lighting()
    before = session.lighting_work
    with pytest.raises(ValueError, match="repeated"):
        session._lighting.apply_and_refresh()
    assert session.lighting_work == before and environment.scene.reads == 3


@pytest.mark.parametrize("visual", [False, True])
def test_should_preserve_default_factory_and_never_resolve_scene_or_refresh(monkeypatch, visual):
    from test_maniskill_continuous import FakeEnvironment

    class NoLighting:
        @property
        def scene(self):
            pytest.fail("default port resolved lighting scene")

        def get_obs(self):
            pytest.fail("default port requested observation refresh")

    class State(NoLighting, FakeEnvironment):
        pass

    class Visual(NoLighting, RGBEnvironment):
        pass

    environment: Any = Visual() if visual else State(steps=1)
    if visual:
        environment.done_at = 1
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    with port.ManiSkillPickCubeEpisode(
        "default-17", "pick-cube", visual_observation=visual
    ) as session:
        assert (
            session.lighting_context is session.lighting_readback is session.lighting_work is None
        )
        session.step(ACTION)
        assert session.episode().context == "pick-cube"
    assert calls == ([dict(visual_observation=True)] if visual else [{}])
    assert environment.reset_calls == [17] and environment.close_calls == 1
