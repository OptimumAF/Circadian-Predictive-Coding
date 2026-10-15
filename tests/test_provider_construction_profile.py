"""Independent constructor enrollment controls, with no original fixtures.

The positive factory preserves one finite table across early clock dispatch and
delegate creation. Ten children record actual denied-attempt counts through an
authorized test-only abort recorder, then invoke the unchanged inherited abort.
Native frames are exercised by real events, not foreign frame simulations.
"""

from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import CodeType, FrameType, FunctionType, GetSetDescriptorType
from typing import NoReturn, cast

import pytest

from provider_code_path import pin_provider_code_path
from provider_construction_profile import OwnerConstructionRole, ProviderConstructionProfile


LABELS = (
    "factory",
    "clock_ctor",
    "clock_call",
    "budget_ctor",
    "budget_read",
    "budget_elapsed",
    "port_ctor",
    "port_call",
    "delegate",
)
POSITIVE_COUNTS = dict(zip(LABELS, (1, 1, 2, 1, 2, 1, 1, 1, 1)))
NEGATIVE_COUNTS = {
    "unenrolled": dict(zip(LABELS, (1, 0, 1, 0, 0, 0, 0, 0, 0))),
    "wrongcaller": {**dict.fromkeys(LABELS, 0), "clock_ctor": 1, "wrongcaller": 1},
    "duplicate": dict(zip(LABELS, (1, 2, 0, 0, 0, 0, 0, 0, 0))),
    "reentrant": dict(zip(LABELS, (1, 2, 0, 0, 0, 0, 0, 0, 0))),
    "failed-finally": dict(zip(LABELS, (1, 1, 0, 0, 0, 0, 0, 0, 0))),
    "foreignreceiver": dict(zip(LABELS, (1, 0, 1, 0, 0, 0, 0, 0, 0))),
    "posthookdictswap": dict(zip(LABELS, (1, 1, 1, 1, 1, 0, 0, 0, 0))),
    "entrylimit": dict(zip(LABELS, (1, 1, 2, 1, 2, 1, 0, 0, 0))),
    "capacityoverflow": dict(zip(LABELS, (1, 2, 0, 0, 0, 0, 0, 0, 0))),
    "pending-frame-locals-hook": dict(zip(LABELS, (1, 1, 0, 0, 0, 0, 0, 0, 0))),
}
NEGATIVE_BODIES = {
    "unenrolled": ["factory"],
    "wrongcaller": ["wrongcaller"],
    "duplicate": ["factory", "clock_ctor"],
    "reentrant": ["factory", "clock_ctor"],
    "failed-finally": ["factory", "clock_ctor"],
    "foreignreceiver": ["factory"],
    "posthookdictswap": ["factory", "clock_ctor", "budget_ctor", "budget_read"],
    "entrylimit": [
        "factory",
        "clock_ctor",
        "budget_ctor",
        "budget_read",
        "clock_call",
        "budget_elapsed",
        "budget_read",
    ],
    "capacityoverflow": ["factory", "clock_ctor"],
    "pending-frame-locals-hook": ["factory", "clock_ctor"],
}


class RecordingProfile(ProviderConstructionProfile):
    """Evidence only: retain the inherited hardexit and exact scalar receipt."""

    record_path: Path

    def _abort(
        self, reason: str, label: str = "profile", attempt: int = 0, limit: int = 0
    ) -> NoReturn:
        try:
            raw = json.dumps(dict(counts=self.counts)).encode("utf8")
            if len(raw) <= 4096:
                self.record_path.write_bytes(raw)
        finally:
            super()._abort(reason, label, attempt, limit)


def make_role(owner_type, caller, capacity=1, callback_path=None):
    namespace = owner_type.__dict__
    constructor = cast(FunctionType, namespace["__init__"])
    raw_call = cast(FunctionType, namespace["__call__"])
    return OwnerConstructionRole(
        owner_type,
        cast(GetSetDescriptorType, namespace["__dict__"]),
        constructor,
        constructor.__code__,
        caller,
        raw_call,
        raw_call.__code__,
        capacity,
        "value" if callback_path is not None else None,
        "value" if callback_path is not None else None,
        callback_path,
    )


def make_standins(mode="positive", path=None):
    bodies: list[str] = []
    hook_events: list[str] = []
    holders: list[ProviderConstructionProfile] = []
    pending_frames: list[FrameType] = []

    def record_setup(label):
        bodies.append(label)
        if path is not None:
            raw = json.dumps(dict(bodies=bodies)).encode("utf8")
            assert len(raw) <= 4096
            (path / "setup.json").write_bytes(raw)

    def mark_offending():
        if path is not None:
            (path / "offending_body").write_text("forbidden", encoding="utf8")

    def pending_probe():
        # Why: an unrelated deeper call proves retained ancestor locals reproof.
        mark_offending()

    class Clock:
        def __init__(self):
            if mode in ("duplicate", "reentrant", "capacityoverflow") and "clock_ctor" in bodies:
                mark_offending()
            record_setup("clock_ctor")
            if mode == "pending-frame-locals-hook":
                pending_probe()
            if mode == "reentrant":
                Clock.__init__(self)
            if mode == "failed-finally":
                try:
                    raise RuntimeError("stand-in constructor failure")
                finally:
                    if path is not None:
                        (path / "failed_setup").write_text("allowed setup unwind", encoding="utf8")
            self.value = 0.0
            self.reads = 0

        def __call__(self):
            if mode in ("unenrolled", "foreignreceiver", "posthookdictswap") or (
                mode == "entrylimit" and "clock_call" in bodies
            ):
                mark_offending()
            record_setup("clock_call")
            self.reads += 1
            return self.value

    class Port:
        def __init__(self, value):
            record_setup("port_ctor")
            self.value, self.calls, self.blocked = value, 0, False
            return None

        def __call__(self):
            record_setup("port_call")
            self.calls += 1
            return self.value()

    class Budget:
        def __init__(self, clock):
            record_setup("budget_ctor")
            self.clock = clock
            self.started = self.read_clock()

        def read_clock(self):
            record_setup("budget_read")
            return self.clock()

        def elapsed(self):
            record_setup("budget_elapsed")
            return self.read_clock() - self.started

    class ForeignReceiver:
        pass

    def factory():
        record_setup("factory")
        if mode == "unenrolled":
            Clock.__call__(object.__new__(Clock))
        if mode == "foreignreceiver":
            getattr(Clock, "__call__")(object.__new__(ForeignReceiver))
        clock = Clock()
        if mode == "capacityoverflow":
            Clock()
        if mode == "duplicate":
            Clock.__init__(clock)
        budget = Budget(clock)
        if mode == "entrylimit":
            budget.elapsed()

        def delegate():
            record_setup("delegate")
            return 7

        port = Port(delegate)
        assert holders[0].enrollment_counts == (1, 1)
        assert holders[0].counts["clock_call"] == 1
        assert port() == 7
        assert budget.elapsed() == 0.0
        return clock, port, budget, delegate

    def wrongcaller():
        record_setup("wrongcaller")
        return Clock()

    root = cast(FunctionType, factory)
    root_code = root.__code__
    caller = pin_provider_code_path(root, root_code, ())
    delegate_index, delegate_code = next(
        (index, value) for index, value in enumerate(root_code.co_consts) if type(value) is CodeType
    )
    callback_path = pin_provider_code_path(root, root_code, (delegate_index,))
    clock_role = make_role(Clock, caller)
    port_role = make_role(Port, caller, callback_path=callback_path)
    tokens = (
        root_code,
        clock_role.constructor_code,
        clock_role.call_code,
        Budget.__dict__["__init__"].__code__,
        Budget.__dict__["read_clock"].__code__,
        Budget.__dict__["elapsed"].__code__,
        port_role.constructor_code,
        port_role.call_code,
        delegate_code,
    )
    limits = (
        1,
        2 if mode in ("duplicate", "reentrant", "capacityoverflow") else 1,
        1 if mode == "entrylimit" else 2,
        1,
        2,
        1,
        1,
        1,
        1,
    )
    entries = tuple((token, label, limit) for token, label, limit in zip(tokens, LABELS, limits))
    if mode == "wrongcaller":
        entries += ((wrongcaller.__code__, "wrongcaller", 1),)

    def prior(frame, event, arg):
        if mode == "posthookdictswap" and event == "call" and frame.f_code is clock_role.call_code:
            hook_events.append("swap")
            object.__setattr__(frame.f_locals["self"], "__dict__", {"value": 0.0, "reads": 0})
        if mode == "pending-frame-locals-hook" and event == "call":
            if frame.f_code is clock_role.constructor_code:
                pending_frames.append(frame)
            elif frame.f_code is pending_probe.__code__:
                hook_events.append("pending_local_swap")
                pending_frames[0].f_locals["self"] = object.__new__(ForeignReceiver)

    return dict(
        Clock=Clock,
        Port=Port,
        Budget=Budget,
        root=root,
        caller=caller,
        callback_path=callback_path,
        factory=factory,
        wrongcaller=wrongcaller,
        roles=(clock_role, port_role),
        entries=entries,
        bodies=bodies,
        holders=holders,
        prior=prior,
        hook_events=hook_events,
    )


def assert_roles_refused(parts, receipt, roles, entries=None):
    with pytest.raises((TypeError, ValueError)):
        ProviderConstructionProfile(
            parts["entries"] if entries is None else entries, receipt, roles
        )
    assert parts["bodies"] == []
    assert not receipt.exists()


def test_should_refuse_malformed_role_rows_capacities_and_partial_delegate_fields(tmp_path):
    parts = make_standins()
    clock_role, port_role = parts["roles"]
    events: list[str] = []

    class ForeignRole(OwnerConstructionRole):
        def __getattribute__(self, name):
            events.append("row_attribute")
            raise AssertionError("foreign row getter dispatched")

    class ForeignTuple(tuple):
        def __len__(self):
            events.append("tuple_length")
            raise AssertionError("foreign tuple length dispatched")

    malformed: list[object] = [
        (),
        [clock_role],
        ForeignTuple((clock_role,)),
        (clock_role,) * 65,
        (object(),),
        (object.__new__(OwnerConstructionRole),),
        (object.__new__(ForeignRole),),
    ]
    for roles in malformed:
        assert_roles_refused(parts, tmp_path / "breach", roles)
    for capacity in (False, 0, -1, 65, 1.0):
        assert_roles_refused(
            parts, tmp_path / "breach", (replace(clock_role, max_instances=capacity),)
        )
    assert_roles_refused(parts, tmp_path / "breach", (clock_role, clock_role))
    assert_roles_refused(
        parts,
        tmp_path / "breach",
        (
            replace(clock_role, max_instances=33),
            replace(port_role, max_instances=32),
        ),
    )
    variants = (
        replace(port_role, callback_slot=None),
        replace(port_role, callback_argument=None),
        replace(port_role, callback_path=None),
        replace(port_role, callback_slot=""),
        replace(port_role, callback_slot="x" * 129),
        replace(port_role, callback_argument=""),
        replace(port_role, callback_path=object()),
    )
    for role in variants:
        assert_roles_refused(parts, tmp_path / "breach", (role,))
    assert events == []
    with pytest.raises(FrozenInstanceError):
        setattr(clock_role, "max_instances", 2)


def test_should_accept_capacity_boundaries_and_refuse_missing_or_conflicting_table_tokens(tmp_path):
    parts = make_standins()
    clock_role, port_role = parts["roles"]
    for roles in (
        (replace(clock_role, max_instances=64),),
        (replace(clock_role, max_instances=32), replace(port_role, max_instances=32)),
    ):
        guard = ProviderConstructionProfile(parts["entries"], tmp_path / "breach", roles)
        guard.require_current_enrollments()
        assert guard.enrollment_counts == tuple(0 for _ in roles)
    for token in (
        clock_role.constructor_code,
        clock_role.call_code,
        parts["caller"].token,
        parts["callback_path"].token,
    ):
        entries = tuple(row for row in parts["entries"] if row[0] is not token)
        assert_roles_refused(parts, tmp_path / "breach", parts["roles"], entries)
    assert_roles_refused(
        parts, tmp_path / "breach", (replace(port_role, callback_path=parts["caller"]),)
    )
    assert parts["bodies"] == [] and not (tmp_path / "breach").exists()


def test_should_refuse_wrong_native_type_code_path_descriptor_and_self_signature(tmp_path):
    parts = make_standins()
    role = parts["roles"][0]
    variants = (
        replace(role, owner_type=object()),
        replace(role, descriptor=object()),
        replace(role, descriptor=parts["roles"][1].descriptor),
        replace(role, constructor=len),
        replace(role, constructor_code=role.constructor_code.replace()),
        replace(role, raw_call=len),
        replace(role, call_code=role.call_code.replace()),
        replace(role, caller=object()),
    )
    for variant in variants:
        assert_roles_refused(parts, tmp_path / "breach", (variant,))

    class WrongSelf:
        def __init__(receiver):
            raise AssertionError("pure control must not dispatch")

        def __call__(self):
            raise AssertionError("pure control must not dispatch")

    wrong_role = make_role(WrongSelf, parts["caller"])
    entries = parts["entries"] + (
        (wrong_role.constructor_code, "wrong_ctor", 1),
        (wrong_role.call_code, "wrong_call", 1),
    )
    assert_roles_refused(parts, tmp_path / "breach", (wrong_role,), entries)


def test_should_refuse_class_hooks_inheritance_descriptors_and_generator_flags_without_dispatch(
    tmp_path,
):
    parts = make_standins()
    role = parts["roles"][0]
    events: list[str] = []

    def hook(*args):
        events.append("hook")
        raise AssertionError("class hook dispatched")

    for name in ("__new__", "__getattribute__", "__getattr__", "__setattr__", "__delattr__"):
        setattr(parts["Clock"], name, hook)
        assert_roles_refused(parts, tmp_path / "breach", (role,))
        delattr(parts["Clock"], name)
    setattr(parts["Port"], "value", property(hook))
    assert_roles_refused(parts, tmp_path / "breach", parts["roles"])
    delattr(parts["Port"], "value")

    class Parent:
        __slots__ = ()

    class Child(Parent):
        def __init__(self):
            raise AssertionError("pure control must not dispatch")

        def __call__(self):
            raise AssertionError("pure control must not dispatch")

    def generator_constructor(self):
        yield None

    async def coroutine_constructor(self):
        return None

    async def async_generator_constructor(self):
        yield None

    def forbidden_call(self):
        raise AssertionError("pure control must not dispatch")

    flagged_classes = tuple(
        type(name, (object,), {"__init__": constructor, "__call__": forbidden_call})
        for name, constructor in (
            ("GeneratorOwner", generator_constructor),
            ("CoroutineOwner", coroutine_constructor),
            ("AsyncGeneratorOwner", async_generator_constructor),
        )
    )
    for owner_type in (Child, *flagged_classes):
        invalid_role = make_role(owner_type, parts["caller"])
        entries = parts["entries"] + (
            (invalid_role.constructor_code, "invalid_ctor", 1),
            (invalid_role.call_code, "invalid_call", 1),
        )
        assert_roles_refused(parts, tmp_path / "breach", (invalid_role,), entries)
    property_class = type("PropertyOwner", (), {"__dict__": property(hook)})
    assert_roles_refused(parts, tmp_path / "breach", (replace(role, owner_type=property_class),))
    assert events == []


def test_should_preflight_all_class_keys_and_detect_foreign_row_or_metaclass_without_lookup(
    tmp_path,
):
    parts = make_standins()
    role = parts["roles"][0]
    events: list[str] = []

    class HostileMeta(type):
        def __getattribute__(self, name):
            events.append("metaclass_attribute")
            raise AssertionError("metaclass getter dispatched")

        def __eq__(self, other):
            events.append("metaclass_equality")
            raise AssertionError("metaclass equality dispatched")

        def __hash__(self):
            events.append("metaclass_hash")
            raise AssertionError("metaclass hash dispatched")

        def __repr__(self):
            events.append("metaclass_repr")
            raise AssertionError("metaclass repr dispatched")

    class ForeignOwner(metaclass=HostileMeta):
        pass

    assert_roles_refused(parts, tmp_path / "breach", (replace(role, owner_type=ForeignOwner),))
    assert events == []
    owner_type = parts["Clock"]
    initial_size = len(owner_type.__dict__)
    for index in range(512 - initial_size):
        setattr(owner_type, f"field{index}", None)
    guard = ProviderConstructionProfile(parts["entries"], tmp_path / "breach", (role,))
    guard.require_current_enrollments()
    setattr(owner_type, "overflow", None)
    assert_roles_refused(parts, tmp_path / "breach", (role,))
    with pytest.raises((TypeError, ValueError)):
        guard.require_current_enrollments()
    delattr(owner_type, "overflow")
    setattr(owner_type, "k" * 129, None)
    assert_roles_refused(parts, tmp_path / "breach", (role,))
    assert events == []


def test_should_preserve_one_table_across_early_clock_and_delegate_and_validate_restored_state(
    tmp_path,
):
    parts = make_standins()
    guard = ProviderConstructionProfile(parts["entries"], tmp_path / "breach", parts["roles"])
    parts["holders"].append(guard)
    table, counters, lock = guard._table, guard._counts, guard._lock
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    with guard.measure():
        clock, port, budget, delegate = parts["factory"]()
    assert all(state.pending is None for state in guard._states)
    assert [len(state.ready) for state in guard._states] == [1, 1]
    assert all(enrollment.frame is None for state in guard._states for enrollment in state.ready)
    assert guard._table is table and guard._counts is counters and guard._lock is lock
    assert tuple((row[1], row[2]) for row in table) == tuple(
        (row[1], row[2]) for row in parts["entries"]
    )
    assert guard.counts == POSITIVE_COUNTS
    assert parts["bodies"] == [
        "factory",
        "clock_ctor",
        "budget_ctor",
        "budget_read",
        "clock_call",
        "port_ctor",
        "port_call",
        "delegate",
        "budget_elapsed",
        "budget_read",
        "clock_call",
    ]
    assert clock.reads == 2 and budget.started == 0.0
    assert port.value is delegate and delegate.__code__ is parts["callback_path"].token
    assert guard.enrollment_counts == (1, 1)
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    guard.require_current_enrollments()
    assert not (tmp_path / "breach").exists()
    events: list[str] = []

    def hostile_hook(*args):
        events.append("hook")
        raise AssertionError("post-window validator dispatched hook")

    for name, value in (("extra", property(hostile_hook)), ("__getattribute__", hostile_hook)):
        setattr(parts["Clock"], name, value)
        with pytest.raises((TypeError, ValueError)):
            guard.require_current_enrollments()
        delattr(parts["Clock"], name)
        guard.require_current_enrollments()
    original_descriptor = parts["roles"][0].descriptor
    with pytest.raises(AttributeError, match="not writable"):
        setattr(parts["Clock"], "__dict__", property(hostile_hook))
    assert parts["Clock"].__dict__["__dict__"] is original_descriptor
    guard.require_current_enrollments()
    for function in (
        parts["roles"][0].constructor,
        parts["roles"][0].raw_call,
        parts["root"],
        delegate,
    ):
        original_code = function.__code__
        function.__code__ = original_code.replace()
        with pytest.raises((TypeError, ValueError)):
            guard.require_current_enrollments()
        function.__code__ = original_code
        guard.require_current_enrollments()
    clock_descriptor = parts["roles"][0].descriptor
    original_namespace = clock_descriptor.__get__(clock, parts["Clock"])
    object.__setattr__(clock, "__dict__", original_namespace.copy())
    with pytest.raises((TypeError, ValueError)):
        guard.require_current_enrollments()
    object.__setattr__(clock, "__dict__", original_namespace)
    guard.require_current_enrollments()
    other_class = type("OtherClock", (), {})
    object.__setattr__(clock, "__class__", other_class)
    with pytest.raises((TypeError, ValueError)):
        guard.require_current_enrollments()
    object.__setattr__(clock, "__class__", parts["Clock"])
    guard.require_current_enrollments()
    port_namespace = parts["roles"][1].descriptor.__get__(port, parts["Port"])
    foreign = FunctionType(delegate.__code__, delegate.__globals__, closure=delegate.__closure__)
    assert foreign.__code__ is delegate.__code__ and foreign is not delegate
    port_namespace["value"] = foreign
    with pytest.raises((TypeError, ValueError)):
        guard.require_current_enrollments()
    del port_namespace["value"]
    with pytest.raises((TypeError, ValueError)):
        guard.require_current_enrollments()
    port_namespace["value"] = delegate
    guard.require_current_enrollments()
    original_namespace[3] = None
    with pytest.raises((TypeError, ValueError)):
        guard.require_current_enrollments()
    del original_namespace[3]
    guard.require_current_enrollments()
    assert events == [] and guard.counts == POSITIVE_COUNTS and len(parts["bodies"]) == 11
    assert guard.enrollment_counts == (1, 1)
    with pytest.raises(RuntimeError, match="cannot renew"):
        with guard.measure():
            pytest.fail("spent enrollment allowance reused")
    (tmp_path / "work.json").write_text(
        json.dumps(
            dict(
                bodies=11,
                factory_calls=1,
                provider_calls=4,
                counts=guard.counts,
                enrollment_counts=list(guard.enrollment_counts),
                workers=0,
                joined=True,
                restored=True,
            )
        ),
        encoding="utf8",
    )


def run_construction_negative(mode, directory):
    path = Path(directory)
    parts = make_standins(mode, path)
    if mode in ("posthookdictswap", "pending-frame-locals-hook"):
        sys.setprofile(parts["prior"])
    guard = RecordingProfile(parts["entries"], path / "breach.json", parts["roles"])
    guard.record_path = path / "counts-sidecar.json"
    parts["holders"].append(guard)
    try:
        with guard.measure():
            if mode == "wrongcaller":
                parts["wrongcaller"]()
            else:
                parts["factory"]()
    except BaseException:
        (path / "caught").write_text("forbidden", encoding="utf8")
    (path / "continued").write_text("forbidden", encoding="utf8")


@pytest.mark.parametrize(
    "mode",
    (
        "unenrolled",
        "wrongcaller",
        "duplicate",
        "reentrant",
        "failed-finally",
        "foreignreceiver",
        "posthookdictswap",
        "entrylimit",
        "capacityoverflow",
        "pending-frame-locals-hook",
    ),
)
def test_should_charge_denied_entry_and_exit_before_offending_body_catch_or_continuation(
    mode, tmp_path
):
    args = [
        sys.executable,
        "-B",
        "-X",
        "utf8",
        "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from test_provider_construction_profile import run_construction_negative; "
        "run_construction_negative(sys.argv[2], sys.argv[3])",
        str(Path(__file__).resolve().parent),
        mode,
        str(tmp_path),
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, timeout=4)
    assert result.returncode == 73 and len(result.stdout) + len(result.stderr) <= 4096
    assert not any((tmp_path / name).exists() for name in ("offending_body", "caught", "continued"))
    setup_raw = (tmp_path / "setup.json").read_bytes()
    counts_raw = (tmp_path / "counts-sidecar.json").read_bytes()
    assert len(setup_raw) <= 4096 and len(counts_raw) <= 4096
    actual_bodies = json.loads(setup_raw)["bodies"]
    actual_counts = json.loads(counts_raw)["counts"]
    assert actual_bodies == NEGATIVE_BODIES[mode]
    assert actual_counts == NEGATIVE_COUNTS[mode]
    raw = (tmp_path / "breach.json").read_bytes()
    assert len(raw) <= 1024
    receipt = json.loads(raw)
    expected = (
        dict(reason="entry_limit", counter="clock_call", attempt=2, limit=1)
        if mode == "entrylimit"
        else dict(reason="measurement_failure", counter="profile", attempt=0, limit=0)
    )
    assert receipt == expected
    assert (tmp_path / "failed_setup").exists() is (mode == "failed-finally")
    (tmp_path / "child.json").write_text(
        json.dumps(
            dict(
                mode=mode,
                exit=result.returncode,
                setup_bodies=len(actual_bodies),
                setup_body_labels=actual_bodies,
                counts=actual_counts,
                offending_bodies=0,
                caught=0,
                continued=0,
                stdout_bytes=len(result.stdout),
                stderr_bytes=len(result.stderr),
                receipt_bytes=len(raw),
                receipt=receipt,
            )
        ),
        encoding="utf8",
    )
