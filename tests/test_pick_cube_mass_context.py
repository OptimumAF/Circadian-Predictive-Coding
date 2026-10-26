"""Fixed mass-only admission through fake IO; no physics or learning claims.

The fake cube records reads and writes independently of the port's accounting.
No simulator, learner, optimizer, teacher or prototype update is executed here.
"""

from dataclasses import FrozenInstanceError, asdict, replace

import numpy as np
import pytest

from src.core.pick_cube_camera_context import pick_cube_camera_context
from src.core.pick_cube_mass_context import (
    PickCubeMassContext,
    pick_cube_mass_context,
    require_mass_binding,
    validate_pick_cube_mass_context,
)
from src.infra import maniskill_continuous as port
from test_maniskill_continuous import FakeEnvironment, ForeignArray
from test_maniskill_rgb_observations import packet, public_feature_oracle


CONTEXT = "pick-cube-mass-x2-train-17"
ACTION = (0.125, -0.25, 0.5, -1.0)


class FakeCube:
    """One owned CPU array, with configurable observed setter/read failures."""

    def __init__(self, events, mass=0.125):
        self.events = events
        self.value = np.array([mass], dtype=np.float32)
        self.reads = 0
        self.writes = []
        self.read_error_at = None
        self.write_error = None
        self.after_write = None
        self.on_read = None

    @property
    def mass(self):
        self.reads += 1
        self.events.append("read")
        if self.reads == self.read_error_at:
            raise RuntimeError("mass read failed")
        if self.on_read is not None:
            self.on_read()
        return self.value

    def set_mass(self, value):
        self.events.append("write")
        self.writes.append(value)
        if self.write_error is not None:
            raise self.write_error
        self.value = np.array([value], dtype=np.float32)
        if self.after_write is not None:
            self.after_write()


class MassEnvironment(FakeEnvironment):
    def __init__(self, *, mass=0.125, visual=False, **kwargs):
        super().__init__(**kwargs)
        self.events = []
        self.cube = FakeCube(self.events, mass)
        self.reset_cube = None
        self.after_step = None
        self.visual = visual
        if visual:
            self.obs_mode = "rgb"
            from types import SimpleNamespace

            self.backend = SimpleNamespace(render_backend="sapien_cpu")
            self.reset_observation = packet()

    def reset(self, *, seed):
        self.events.append("reset")
        result = super().reset(seed=seed)
        if self.reset_cube is not None:
            self.cube = self.reset_cube
        return result

    def step(self, command):
        self.events.append("step")
        result = super().step(command)
        if self.after_step is not None:
            self.after_step()
        if self.visual:
            result = (packet(len(self.commands)), *result[1:])
        return result

    def close(self):
        self.events.append("close")
        super().close()


@pytest.fixture
def boundary(monkeypatch):
    environment = MassEnvironment()
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    yield environment, calls
    # Construction failures close themselves; successful sessions are closed
    # here so an assertion failure cannot leave the boundary double open.
    if calls and environment.close_calls == 0:
        environment.close()


def open_mass(**kwargs):
    options = dict(context=CONTEXT, seed=17, mass_context=pick_cube_mass_context())
    options.update(kwargs)
    return port.ManiSkillPickCubeEpisode("mass-train-17", **options)


def test_should_return_fresh_strict_immutable_mass_only_metadata():
    value = pick_cube_mass_context()
    assert asdict(value) == dict(
        context=CONTEXT,
        seed=17,
        mass_multiplier=2.0,
        semantics="mass_only_no_inertia_scaling",
    )
    owned = validate_pick_cube_mass_context(value)
    assert owned == value and owned is not value and pick_cube_mass_context() is not value
    with pytest.raises(FrozenInstanceError):
        value.mass_multiplier = 3.0
    object.__setattr__(value, "seed", 19)
    assert owned.seed == 17


@pytest.mark.parametrize(
    "changes",
    [
        dict(seed=True),
        dict(seed=17.0),
        dict(seed=np.int64(17)),
        dict(seed=19),
        dict(context="pick-cube"),
        dict(context=CONTEXT + " "),
        dict(context=None),
        dict(mass_multiplier=2),
        dict(mass_multiplier=np.float64(2.0)),
        dict(mass_multiplier=3.0),
        dict(mass_multiplier=float("nan")),
        dict(semantics="density_scaling"),
    ],
)
def test_should_refuse_changed_or_equal_wrong_type_metadata(changes):
    with pytest.raises(ValueError):
        validate_pick_cube_mass_context(PickCubeMassContext(**changes))


@pytest.mark.parametrize("damage", ["none", "mapping", "subclass", "mutated"])
def test_should_refuse_nonexact_or_corrupted_records(damage):
    value = pick_cube_mass_context()
    if damage == "none":
        value = None
    elif damage == "mapping":
        value = asdict(value)
    elif damage == "subclass":

        class Derived(PickCubeMassContext):
            pass

        value = Derived()
    else:
        object.__setattr__(value, "mass_multiplier", 2)
    with pytest.raises(ValueError):
        validate_pick_cube_mass_context(value)


@pytest.mark.parametrize(
    "changes",
    [
        dict(context="pick-cube"),
        dict(context=None),
        dict(seed=19),
        dict(seed=True),
        dict(seed=17.0),
        dict(seed=np.int64(17)),
        dict(role="inner_guard"),
        dict(role="probe_train"),
        dict(role="probe_test"),
        dict(role="final_test"),
    ],
)
def test_should_bind_only_declared_train17(changes):
    options = dict(context=CONTEXT, seed=17, role="train")
    options.update(changes)
    with pytest.raises(ValueError):
        require_mass_binding(pick_cube_mass_context(), **options)


def test_should_require_explicit_reserved_binding_and_own_accepted_input():
    assert require_mass_binding(None, context="pick-cube", seed=17, role="train") is None
    with pytest.raises(ValueError):
        require_mass_binding(None, context=CONTEXT, seed=17, role="train")
    source = pick_cube_mass_context()
    owned = require_mass_binding(source, context=CONTEXT, seed=17, role="train")
    object.__setattr__(source, "semantics", "changed")
    assert owned == pick_cube_mass_context() and owned is not source


@pytest.mark.parametrize(
    "changes",
    [
        dict(context="pick-cube"),
        dict(seed=19),
        dict(seed=True),
        dict(role="inner_guard"),
        dict(role="probe_train"),
        dict(role="probe_test"),
        dict(role="final_test"),
        dict(mass_context=None),
        dict(mass_context={}),
        dict(mass_context=PickCubeMassContext(mass_multiplier=3.0)),
        dict(camera_context=pick_cube_camera_context(), visual_observation=True),
        dict(visual_observation=1),
    ],
)
def test_should_refuse_invalid_binding_before_environment_creation(boundary, changes):
    environment, calls = boundary
    with pytest.raises(ValueError):
        open_mass(**changes)
    assert calls == [] and environment.events == []


@pytest.mark.parametrize("mass", [0.125, 0.5, 0.1])
def test_should_reset_then_double_observed_mass_and_preserve_actual_controls(boundary, mass):
    environment, calls = boundary
    environment.steps = 1
    stale = environment.cube
    reset_cube = FakeCube(environment.events, mass)
    environment.reset_cube = reset_cube
    session = open_mass()
    target = float(np.float32(float(np.float32(mass)) * 2.0))
    assert calls == [{}] and environment.reset_calls == [17]
    assert stale.reads == 0 and stale.writes == []
    assert environment.events == ["reset", "read", "write", "read", "read"]
    assert reset_cube.writes == [target]
    assert session.mass_readback.source_mass_kg == float(np.float32(mass))
    assert session.mass_readback.applied_mass_kg == target
    assert session.spec.action_units == port.ACTION_UNITS
    assert session.spec.lower_bounds == (-1.0,) * 4
    assert session.spec.upper_bounds == (1.0,) * 4
    initial = session.observation
    record = session.step(ACTION)
    episode = session.episode()
    np.testing.assert_array_equal(environment.commands[0], np.array([ACTION], dtype=np.float32))
    assert record.observation == initial and record.action == ACTION
    assert episode.context == CONTEXT and episode.role == "train"
    assert record.reward == 37.5 and record.terminated and not record.truncated
    assert asdict(session.mass_work) == dict(
        read_attempts=6, read_returns=6, write_attempts=1, write_returns=1
    )
    assert session.work.step_attempts == session.work.completed_steps == 1


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True)])
def test_should_preserve_independent_terminal_flags_without_renewing_mass(
    boundary, terminated, truncated
):
    environment, _ = boundary
    environment.terminated, environment.truncated = terminated, truncated
    session = open_mass()
    session.step(ACTION)
    final = session.step(ACTION)
    assert (final.terminated, final.truncated) == (terminated, truncated)
    assert len(session.episode().transitions) == 2
    with pytest.raises(ValueError):
        session.step(ACTION)
    assert environment.cube.reads == 8 and len(environment.cube.writes) == 1
    assert asdict(session.work) == dict(
        step_attempts=2,
        completed_steps=2,
        completed_control_physics_steps=10,
        possible_failed_physics_steps=0,
    )


def test_should_own_original_readback_and_work_without_implicit_native_reads(boundary):
    environment, _ = boundary
    environment.steps = 1
    source = pick_cube_mass_context()
    session = open_mass(mass_context=source)
    context, readback, work = session.mass_context, session.mass_readback, session.mass_work
    assert type(readback) is port.ManiSkillMassReadback
    assert type(work) is port.ManiSkillMassWork
    assert session.mass_readback is not readback and session.mass_work is not work
    with pytest.raises(FrozenInstanceError):
        readback.applied_mass_kg = 9.0
    with pytest.raises(FrozenInstanceError):
        work.read_attempts = 0
    object.__setattr__(source, "seed", 19)
    object.__setattr__(context, "seed", 19)
    object.__setattr__(readback.context, "mass_multiplier", 3.0)
    object.__setattr__(readback, "source_mass_kg", 9.0)
    object.__setattr__(work, "read_attempts", 0)
    assert environment.cube.reads == 3
    assert session.mass_context == pick_cube_mass_context()
    assert session.mass_readback.source_mass_kg == 0.125
    assert session.mass_work.read_attempts == 3
    session.step(ACTION)
    session.episode()
    assert environment.cube.reads == 6 and len(environment.cube.writes) == 1


@pytest.mark.parametrize(
    "bad",
    [
        np.array([0.125], dtype=np.float64),
        np.array([1], dtype=np.int32),
        np.array(0.125, dtype=np.float32),
        np.array([[0.125]], dtype=np.float32),
        np.array([0.125, 0.125], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([-0.125], dtype=np.float32),
        np.array([0.75], dtype=np.float32),
        np.array([1.25], dtype=np.float32),
        np.array([float("nan")], dtype=np.float32),
        np.array([float("inf")], dtype=np.float32),
        np.array([0.125], dtype=np.float32).view(ForeignArray),
    ],
)
def test_should_refuse_invalid_source_read_without_writing_and_close(boundary, monkeypatch, bad):
    environment, _ = boundary
    # Keep the array-subclass refusal local: no optional Torch import is needed.
    monkeypatch.setattr(port, "import_module", lambda _name: (_ for _ in ()).throw(ImportError()))
    environment.cube.value = bad
    with pytest.raises(ValueError):
        open_mass()
    assert environment.events == ["reset", "read", "close"]
    assert environment.cube.writes == [] and environment.close_calls == 1


@pytest.mark.parametrize("damage", ["unchanged", "wrong_target", "invalid_readback"])
def test_should_require_exact_immediate_float32_setter_readback(boundary, damage):
    environment, _ = boundary
    cube = environment.cube

    def damage_readback():
        cube.value = (
            np.array([0.125 if damage == "unchanged" else 0.3], dtype=np.float32)
            if damage != "invalid_readback"
            else np.array([0.25], dtype=np.float64)
        )

    cube.after_write = damage_readback
    with pytest.raises(ValueError):
        open_mass()
    assert environment.events == ["reset", "read", "write", "read", "close"]
    assert cube.writes == [0.25] and environment.commands == []


@pytest.mark.parametrize("read_error_at", [1, 2, 3])
def test_should_close_once_on_source_immediate_or_encoding_read_error(boundary, read_error_at):
    environment, _ = boundary
    environment.cube.read_error_at = read_error_at
    with pytest.raises(RuntimeError, match="mass read failed"):
        open_mass()
    assert environment.cube.reads == read_error_at
    assert len(environment.cube.writes) == int(read_error_at > 1)
    assert environment.commands == [] and environment.close_calls == 1


def test_should_preserve_failed_write_attempt_without_retry(boundary):
    environment, _ = boundary
    environment.cube.write_error = RuntimeError("mass write failed")
    with pytest.raises(RuntimeError, match="mass write failed"):
        open_mass()
    assert environment.events == ["reset", "read", "write", "close"]
    assert environment.cube.reads == 1 and environment.cube.writes == [0.25]
    assert environment.commands == [] and environment.close_calls == 1


def test_should_not_read_or_write_mass_if_explicit_reset_fails(boundary):
    environment, _ = boundary
    environment.reset_error = RuntimeError("reset failed")
    with pytest.raises(RuntimeError, match="reset failed"):
        open_mass()
    assert environment.events == ["reset", "close"]


@pytest.mark.parametrize(
    "damage", ["mass", "cube", "removed", "metadata", "seed", "context", "role"]
)
def test_should_permanently_refuse_unobserved_changes_before_control(boundary, damage):
    environment, _ = boundary
    session = open_mass()
    cube = environment.cube
    if damage == "mass":
        cube.value[0] = 0.5
    elif damage == "cube":
        environment.cube = FakeCube(environment.events, 0.25)
    elif damage == "removed":
        session._mass_context = None
    elif damage == "metadata":
        session._mass_context = replace(pick_cube_mass_context(), semantics="density_scaling")
    else:
        setattr(
            session, "_" + damage, dict(seed=19, context="pick-cube", role="probe_train")[damage]
        )
    with pytest.raises(ValueError):
        session.step(ACTION)
    assert environment.commands == [] and cube.writes == [0.25]
    assert cube.reads == (4 if damage == "mass" else 3)
    environment.cube = cube
    cube.value[0] = 0.25
    session._mass_context = pick_cube_mass_context()
    session._seed, session._context, session._role = 17, CONTEXT, "train"
    before = len(environment.events)
    with pytest.raises(ValueError):
        session.step(ACTION)
    with pytest.raises(ValueError):
        session.episode()
    assert len(environment.events) == before


def test_should_revalidate_mass_at_episode_export_and_not_publish_after_failure(boundary):
    environment, _ = boundary
    session = open_mass()
    session.step(ACTION)
    environment.cube.value[0] = 0.5
    with pytest.raises(ValueError):
        session.episode()
    environment.cube.value[0] = 0.25
    with pytest.raises(ValueError):
        session.episode()
    assert environment.cube.reads == 6 and len(environment.commands) == 1
    assert session.work.completed_steps == 1


@pytest.mark.parametrize("failure", ["native_step", "changed_after_step", "read_before_step"])
def test_should_retain_observed_work_and_refuse_retry_on_late_failure(boundary, failure):
    environment, _ = boundary
    session = open_mass()
    if failure == "native_step":
        environment.step_error = RuntimeError("native step failed")
    elif failure == "changed_after_step":
        environment.after_step = lambda: environment.cube.value.fill(0.5)
    else:
        environment.cube.read_error_at = 4
    with pytest.raises((ValueError, RuntimeError)):
        session.step(ACTION)
    assert environment.cube.reads == (5 if failure == "changed_after_step" else 4)
    assert len(environment.commands) == int(failure != "read_before_step")
    assert session.work.step_attempts == int(failure != "read_before_step")
    assert session.work.completed_steps == int(failure == "changed_after_step")
    assert session.mass_work.read_returns == (
        3 if failure == "read_before_step" else environment.cube.reads
    )
    before = len(environment.events)
    with pytest.raises(ValueError):
        session.step(ACTION)
    with pytest.raises(ValueError):
        session.episode()
    assert len(environment.events) == before


@pytest.mark.parametrize("visual", [False, True])
def test_should_leave_default_factory_and_mass_access_untouched(monkeypatch, visual):
    class DefaultEnvironment(MassEnvironment):
        @property
        def cube(self):
            pytest.fail("default episode accessed cube mass")

        @cube.setter
        def cube(self, _value):
            pass

    environment = DefaultEnvironment(visual=visual, steps=1)
    calls = []
    monkeypatch.setattr(
        port, "_make_environment", lambda **kwargs: calls.append(kwargs) or environment
    )
    with port.ManiSkillPickCubeEpisode(
        "default", "pick-cube", visual_observation=visual
    ) as session:
        assert session.mass_context is session.mass_readback is session.mass_work is None
        session.step(ACTION)
        episode = session.episode()
        assert episode.context == "pick-cube" and episode.role == "train"
    assert calls == ([dict(visual_observation=True)] if visual else [{}])
    assert environment.events == ["reset", "step", "close"]


def test_should_combine_mass_with_public_rgb_without_changing_recipe_or_action(monkeypatch):
    environment = MassEnvironment(visual=True, steps=1)
    calls = []
    monkeypatch.setattr(
        port, "_make_environment", lambda **kwargs: calls.append(kwargs) or environment
    )
    with open_mass(visual_observation=True) as session:
        assert session.observation == tuple(public_feature_oracle(environment.reset_observation))
        assert session.spec.observation_dim == 789
        original_recipe = session.observation_recipe
        session.step(ACTION)
        episode = session.episode()
        assert episode.spec.observation_dim == 789
        assert session.observation_recipe == original_recipe
        assert (
            session.visual_work.observation_attempts == session.visual_work.observation_returns == 2
        )
        assert session.mass_work.read_attempts == session.mass_work.read_returns == 6
    assert calls == [dict(visual_observation=True)]
    assert environment.cube.reads == 6 and environment.cube.writes == [0.25]


def test_should_refuse_adding_mass_binding_to_an_unmodified_default_episode(boundary):
    environment, _ = boundary
    session = port.ManiSkillPickCubeEpisode("default", "pick-cube", seed=17)
    session._context = CONTEXT
    session._mass_context = pick_cube_mass_context()
    with pytest.raises(ValueError):
        session.step(ACTION)
    assert environment.cube.reads == 0 and environment.cube.writes == []
    assert environment.commands == []
    with pytest.raises(ValueError):
        session.episode()


@pytest.mark.parametrize("phase", ["source", "before_control", "after_paid_control"])
@pytest.mark.parametrize("damage", ["cube", "metadata"])
def test_should_refuse_identity_changes_inside_native_mass_accessor(boundary, phase, damage):
    environment, _ = boundary
    session = port.ManiSkillPickCubeEpisode.__new__(port.ManiSkillPickCubeEpisode)
    cube = environment.cube
    fail_read = dict(source=1, before_control=4, after_paid_control=5)[phase]

    def corrupt_at_external_return():
        if cube.reads == fail_read:
            if damage == "cube":
                environment.cube = FakeCube(environment.events, 0.25)
            else:
                session._mass_context = None

    cube.on_read = corrupt_at_external_return
    arguments = dict(context=CONTEXT, seed=17, mass_context=pick_cube_mass_context())
    if phase == "source":
        with pytest.raises(ValueError):
            session.__init__("mass-train-17", **arguments)
    else:
        session.__init__("mass-train-17", **arguments)
        with pytest.raises(ValueError):
            session.step(ACTION)
    assert cube.reads == fail_read
    assert len(cube.writes) == int(phase != "source")
    assert session.mass_work.read_returns == fail_read - 1
    paid = int(phase == "after_paid_control")
    assert (
        len(environment.commands)
        == session.work.step_attempts
        == session.work.completed_steps
        == paid
    )
    if phase != "source":
        cube.on_read = None
        environment.cube = cube
        session._mass_context = pick_cube_mass_context()
        before = len(environment.events)
        with pytest.raises(ValueError):
            session.step(ACTION)
        with pytest.raises(ValueError):
            session.episode()
        assert len(environment.events) == before
    else:
        assert environment.close_calls == 1


@pytest.mark.parametrize("damage", ["cube", "metadata"])
def test_should_revalidate_identity_after_setter_before_accepting_readback(boundary, damage):
    environment, _ = boundary
    session = port.ManiSkillPickCubeEpisode.__new__(port.ManiSkillPickCubeEpisode)
    cube = environment.cube

    def corrupt_after_write():
        if damage == "cube":
            environment.cube = FakeCube(environment.events, 0.25)
        else:
            session._mass_context = None

    cube.after_write = corrupt_after_write
    with pytest.raises(ValueError):
        session.__init__("mass-train-17", CONTEXT, mass_context=pick_cube_mass_context())
    assert environment.events == ["reset", "read", "write", "close"]
    assert cube.writes == [0.25]
    assert session.mass_work == port.ManiSkillMassWork(1, 1, 1, 1)
