"""Finite deferred admission controls; raw source, no original providers."""

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import CodeType, FunctionType, GetSetDescriptorType, ModuleType
from typing import NoReturn, cast

import pytest

from provider_construction_profile import ProviderConstructionProfile
from provider_deferred_construction import DeferredConstructionRole
from provider_source_classes import FixtureSourceClasses, LoadedSourceClasses, PreparedSourceClasses
from provider_source_loader import ApprovedSourceImport, FixtureSourceLoader, LoadedFixtureSource


LABELS = (
    "module",
    "clock_class",
    "port_class",
    "factory",
    "clock_ctor",
    "clock_call",
    "port_ctor",
    "port_call",
    "delegate",
)
ZERO = dict.fromkeys(LABELS, 0)
POSITIVE = dict(zip(LABELS, (1, 1, 1, 1, 1, 2, 1, 1, 1)))
MODULE_BODIES = ["module", "clock_class", "port_class"]
POSITIVE_BODIES = [
    *MODULE_BODIES,
    "factory",
    "clock_ctor",
    "clock_call",
    "port_ctor",
    "port_call",
    "delegate",
    "clock_call",
]
MODES = (
    "unadmitted-constructor",
    "unadmitted-provider",
    "unadmitted-factory",
    "wrongglobals-caller",
    "priorhook-witness-swap",
    "capacityoverflow",
)
EXPECTED_COUNTS = {
    "unadmitted-constructor": {
        **ZERO,
        "module": 1,
        "clock_class": 1,
        "port_class": 1,
        "clock_ctor": 1,
    },
    "unadmitted-provider": {
        **ZERO,
        "module": 1,
        "clock_class": 1,
        "port_class": 1,
        "clock_call": 1,
    },
    "unadmitted-factory": {**ZERO, "module": 1, "clock_class": 1, "port_class": 1, "factory": 1},
    "wrongglobals-caller": {**ZERO, "module": 1, "clock_class": 1, "port_class": 1, "factory": 1},
    "priorhook-witness-swap": {
        **ZERO,
        "module": 1,
        "clock_class": 1,
        "port_class": 1,
        "factory": 1,
    },
    "capacityoverflow": {
        **ZERO,
        "module": 1,
        "clock_class": 1,
        "port_class": 1,
        "factory": 1,
        "clock_ctor": 2,
    },
}
EXPECTED_BODIES = {
    mode: MODULE_BODIES + (["factory", "clock_ctor"] if mode == "capacityoverflow" else [])
    for mode in MODES
}
EXPECTED_EFFECTS = {
    mode: {label: EXPECTED_BODIES[mode].count(label) for label in LABELS} for mode in MODES
}
RECEIPT = dict(reason="measurement_failure", counter="profile", attempt=0, limit=0)
SOURCE = b"""import deferredeffects
deferredeffects.counts['module'] += 1
deferredeffects.bodies.append('module')
class Clock:
    def __init__(self):
        deferredeffects.counts['clock_ctor'] += 1
        deferredeffects.bodies.append('clock_ctor')
        self.value = 7
        self.reads = 0
    def __call__(self):
        deferredeffects.counts['clock_call'] += 1
        deferredeffects.bodies.append('clock_call')
        self.reads += 1
        return self.value
deferredeffects.counts['clock_class'] += 1
deferredeffects.bodies.append('clock_class')
class Port:
    def __init__(self, value):
        deferredeffects.counts['port_ctor'] += 1
        deferredeffects.bodies.append('port_ctor')
        self.value = value
        self.calls = 0
    def __call__(self):
        deferredeffects.counts['port_call'] += 1
        deferredeffects.bodies.append('port_call')
        self.calls += 1
        return self.value()
deferredeffects.counts['port_class'] += 1
deferredeffects.bodies.append('port_class')
def factory():
    deferredeffects.counts['factory'] += 1
    deferredeffects.bodies.append('factory')
    clock = Clock()
    if deferredeffects.mode == 'capacityoverflow':
        Clock()
    early = clock()
    def delegate():
        deferredeffects.counts['delegate'] += 1
        deferredeffects.bodies.append('delegate')
        return 7
    port = Port(delegate)
    return clock, port, early, delegate
"""
MODULE_DICT = cast(
    GetSetDescriptorType, type.__dict__["__dict__"].__get__(ModuleType, type)["__dict__"]
)


class RecordingProfile(ProviderConstructionProfile):
    """Snapshot only; retain the original inherited abort and hard exit."""

    record_path: Path
    effects: dict

    def _abort(
        self, reason: str, label: str = "profile", attempt: int = 0, limit: int = 0
    ) -> NoReturn:
        try:
            raw = json.dumps(
                dict(
                    counts=self.counts,
                    side_effects=self.effects["counts"].copy(),
                    body_labels=list(self.effects["bodies"]),
                    enrollment_counts=list(self.enrollment_counts),
                )
            ).encode("utf8")
            if len(raw) <= 4096:
                self.record_path.write_bytes(raw)
        finally:
            super()._abort(reason, label, attempt, limit)


def make_parts(mode="positive", source: bytes = SOURCE):
    dependency = ModuleType("deferredeffects")
    effects = MODULE_DICT.__get__(dependency, ModuleType)
    effects.update(counts=ZERO.copy(), bodies=[], mode=mode)
    loader = FixtureSourceLoader((ApprovedSourceImport("deferredeffects", dependency),))
    classes = FixtureSourceClasses(loader)
    prepared = classes.prepare(
        source,
        sha256(source).hexdigest(),
        "bounded_deferred_source",
        "deferred.py",
        ("factory",),
        class_roots=("Clock", "Port"),
    )
    root_code = prepared.source.root_codes[0][1]
    delegate_index, delegate_code = next(
        (index, value) for index, value in enumerate(root_code.co_consts) if type(value) is CodeType
    )
    clock, port = prepared.classes
    roles = (
        DeferredConstructionRole(classes, prepared, 0, "factory", (), 1),
        DeferredConstructionRole(
            classes,
            prepared,
            1,
            "factory",
            (),
            1,
            "value",
            "value",
            "factory",
            (delegate_index,),
        ),
    )
    tokens = (
        prepared.source.module_code,
        clock.class_code,
        port.class_code,
        root_code,
        clock.constructor_code,
        clock.call_code,
        port.constructor_code,
        port.call_code,
        delegate_code,
    )
    limits = (1, 1, 1, 1, 2 if mode == "capacityoverflow" else 1, 2, 1, 1, 1)
    entries = tuple((code, label, limit) for code, label, limit in zip(tokens, LABELS, limits))
    return dict(
        dependency=dependency,
        effects=effects,
        classes=classes,
        prepared=prepared,
        roles=roles,
        entries=entries,
        events=[],
        root_code=root_code,
        delegate_index=delegate_index,
        delegate_code=delegate_code,
    )


def require_zero(parts, receipt):
    assert parts["effects"]["counts"] == ZERO
    assert parts["effects"]["bodies"] == [] and parts["events"] == []
    assert not receipt.exists()


def refuse_profile(parts, receipt, roles, entries=None):
    with pytest.raises((TypeError, ValueError)):
        ProviderConstructionProfile(
            parts["entries"] if entries is None else entries, receipt, (), deferred_roles=roles
        )
    require_zero(parts, receipt)


def new_profile(parts, receipt):
    return ProviderConstructionProfile(parts["entries"], receipt, (), deferred_roles=parts["roles"])


def test_deferred_refuses_malformed_native_declarations(tmp_path):
    parts = make_parts()
    clock, port = parts["roles"]
    receipt = tmp_path / "breach.json"
    events = parts["events"]

    class ForeignRole(DeferredConstructionRole):
        def __getattribute__(self, name):
            events.append("role_getter")
            raise AssertionError

    class ForeignTuple(tuple):
        def __len__(self):
            events.append("tuple_length")
            raise AssertionError

        def __iter__(self):
            events.append("tuple_iteration")
            raise AssertionError

    class ForeignName(str):
        def __hash__(self):
            events.append("name_hash")
            raise AssertionError

        def __eq__(self, other):
            events.append("name_equality")
            raise AssertionError

    malformed: list[object] = [
        (),
        [],
        ForeignTuple((clock,)),
        (object(),),
        (object.__new__(DeferredConstructionRole),),
        (object.__new__(ForeignRole),),
        (clock,) * 65,
    ]
    for roles in malformed:
        refuse_profile(parts, receipt, roles)
    changes: tuple[dict[str, object], ...] = (
        dict(class_index=False),
        dict(class_index=-1),
        dict(class_index=2),
        dict(class_index=1.0),
        dict(caller_root=""),
        dict(caller_root="missing"),
        dict(caller_root=ForeignName("factory")),
        dict(caller_root="x" * 129),
        dict(caller_indexes=[]),
        dict(caller_indexes=ForeignTuple(())),
        dict(max_instances=False),
        dict(max_instances=0),
        dict(max_instances=65),
        dict(max_instances=1.0),
    )
    for change in changes:
        refuse_profile(parts, receipt, (replace(clock, **change),))
    callback_changes: tuple[dict[str, object], ...] = (
        dict(callback_slot=None),
        dict(callback_argument=None),
        dict(callback_root=None),
        dict(callback_indexes=None),
        dict(callback_slot=""),
        dict(callback_slot="x" * 129),
        dict(callback_argument=""),
        dict(callback_argument=ForeignName("value")),
        dict(callback_argument="missing"),
        dict(callback_slot="__class__"),
        dict(callback_slot="__dict__"),
        dict(callback_root=ForeignName("factory")),
        dict(callback_indexes=[]),
        dict(callback_indexes=ForeignTuple((parts["delegate_index"],))),
    )
    for change in callback_changes:
        refuse_profile(parts, receipt, (replace(port, **change),))
    require_zero(parts, receipt)


def test_deferred_requires_original_unexecuted_prepared_witnesses(tmp_path):
    parts = make_parts()
    receipt = tmp_path / "breach.json"
    role = parts["roles"][0]
    prepared = parts["prepared"]
    parts["classes"].require_unexecuted(prepared)

    class ForeignPrepared(PreparedSourceClasses):
        def __getattribute__(self, name):
            parts["events"].append("prepared_getter")
            raise AssertionError

    class ForeignClasses(FixtureSourceClasses):
        def __getattribute__(self, name):
            parts["events"].append("classes_getter")
            raise AssertionError

    for candidate in (
        object(),
        object.__new__(PreparedSourceClasses),
        object.__new__(ForeignPrepared),
        replace(prepared),
        replace(prepared, source=replace(prepared.source)),
        replace(prepared, classes=tuple(list(prepared.classes))),
    ):
        refuse_profile(parts, receipt, (replace(role, prepared=candidate),))
    other = make_parts()
    for candidate in (object(), object.__new__(ForeignClasses), other["classes"]):
        refuse_profile(parts, receipt, (replace(role, source_classes=candidate),))
    row = prepared.classes[0]
    mutations = (
        (prepared, "source", replace(prepared.source)),
        (prepared, "classes", tuple(list(prepared.classes))),
        (prepared.source, "module_code", prepared.source.module_code.replace()),
        (prepared.source, "root_codes", tuple(list(prepared.source.root_codes))),
        (prepared.source, "module_name", "bounded_foreign_deferred"),
        (row, "class_code", row.class_code.replace()),
        (row, "constructor_code", row.constructor_code.replace()),
        (row, "call_code", row.call_code.replace()),
    )
    for owner, field, replacement in mutations:
        original = getattr(owner, field)
        object.__setattr__(owner, field, replacement)
        try:
            refuse_profile(parts, receipt, parts["roles"])
            with pytest.raises((TypeError, ValueError)):
                parts["classes"].require_unexecuted(prepared)
        finally:
            object.__setattr__(owner, field, original)
    parts["classes"].require_unexecuted(prepared)
    require_zero(parts, receipt)
    require_zero(other, tmp_path / "other-breach.json")


def test_deferred_requires_original_table_tokens_and_finite_capacity(tmp_path):
    parts = make_parts()
    receipt = tmp_path / "breach.json"
    clock, port = parts["roles"]
    for token, _, _ in parts["entries"]:
        entries = tuple(row for row in parts["entries"] if row[0] is not token)
        refuse_profile(parts, receipt, parts["roles"], entries)
    for position, row in enumerate(parts["entries"]):
        candidate_entries = list(parts["entries"])
        candidate_entries[position] = (row[0].replace(), row[1], row[2])
        refuse_profile(parts, receipt, parts["roles"], tuple(candidate_entries))
    refuse_profile(parts, receipt, (clock, replace(clock)))
    refuse_profile(parts, receipt, (replace(clock, max_instances=64), port))
    for roles in (
        (replace(clock, max_instances=64),),
        (replace(clock, max_instances=32), replace(port, max_instances=32)),
    ):
        guard = ProviderConstructionProfile(parts["entries"], receipt, (), deferred_roles=roles)
        guard.require_current_enrollments()
        assert guard.enrollment_counts == tuple(0 for _ in roles)
        assert guard.counts == ZERO
    extras = tuple(
        (parts["prepared"].source.module_code.replace(co_name=f"unused{i}"), f"unused{i}", 0)
        for i in range(55)
    )
    largest = parts["entries"] + extras
    assert len(largest) == 64
    guard = ProviderConstructionProfile(largest, receipt, (), deferred_roles=parts["roles"])
    guard.require_current_enrollments()
    assert len(guard.counts) == 64 and all(value == 0 for value in guard.counts.values())
    overflow = largest + (
        (parts["prepared"].source.module_code.replace(co_name="overflow"), "overflow", 0),
    )
    refuse_profile(parts, receipt, parts["roles"], overflow)
    require_zero(parts, receipt)


def test_deferred_refuses_invalid_caller_and_callback_paths(tmp_path):
    parts = make_parts()
    receipt = tmp_path / "breach.json"
    clock, port = parts["roles"]
    invalid_paths = ((False,), (-1,), (512,), ("0",), (0,), (0,) * 17)
    for indexes in invalid_paths:
        refuse_profile(parts, receipt, (replace(clock, caller_indexes=indexes),))
        refuse_profile(parts, receipt, (replace(port, callback_indexes=indexes),))
    for change in (
        dict(callback_root="missing"),
        dict(callback_indexes=()),
        dict(callback_root="Clock"),
    ):
        refuse_profile(parts, receipt, (replace(port, **change),))
    # Caller and callback must not claim the same nested leaf token.
    refuse_profile(parts, receipt, (replace(port, caller_indexes=(parts["delegate_index"],)),))
    require_zero(parts, receipt)


def test_deferred_rechecks_retained_declarations_without_dispatch(tmp_path):
    parts = make_parts()
    receipt = tmp_path / "breach.json"
    guard = new_profile(parts, receipt)
    clock, port = parts["roles"]
    prepared = parts["prepared"]
    for owner, field, replacement in (
        (clock, "max_instances", 2),
        (clock, "caller_root", "changed"),
        (clock, "prepared", replace(prepared)),
        (port, "callback_argument", "changed"),
        (port, "callback_indexes", tuple(list(port.callback_indexes))),
        (prepared.source, "source_sha256", "0" * 64),
        (prepared.classes[0], "call_code", prepared.classes[0].call_code.replace()),
    ):
        original = getattr(owner, field)
        assert replacement is not original
        object.__setattr__(owner, field, replacement)
        try:
            with pytest.raises((TypeError, ValueError)):
                guard.require_current_enrollments()
        finally:
            object.__setattr__(owner, field, original)
        guard.require_current_enrollments()
    with pytest.raises((TypeError, ValueError)):
        guard.admit_source(object.__new__(LoadedSourceClasses))
    assert guard.counts == ZERO and guard.enrollment_counts == (0, 0)
    require_zero(parts, receipt)


def test_deferred_admits_source_before_early_clock_and_delegate(tmp_path):
    parts = make_parts()
    receipt = tmp_path / "breach.json"
    guard = new_profile(parts, receipt)
    table, counters, lock = guard._table, guard._counts, guard._lock
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    assert guard.enrollment_counts == (0, 0)

    class ForeignLoaded(LoadedSourceClasses):
        def __getattribute__(self, name):
            parts["events"].append("loaded_getter")
            raise AssertionError

    with guard.measure():
        with pytest.raises((TypeError, ValueError)):
            guard.admit_source(object.__new__(LoadedSourceClasses))
        assert guard.counts == ZERO and parts["effects"]["counts"] == ZERO
        loaded = parts["classes"].execute(parts["prepared"])
        assert parts["effects"]["bodies"] == MODULE_BODIES
        assert guard.enrollment_counts == (0, 0)
        assert all(module.frame is None for module in guard._deferred.modules)
        for candidate in (
            object(),
            object.__new__(LoadedSourceClasses),
            object.__new__(ForeignLoaded),
            replace(loaded),
        ):
            with pytest.raises((TypeError, ValueError)):
                guard.admit_source(candidate)
        original_source = loaded.source
        object.__setattr__(loaded, "source", replace(original_source))
        try:
            with pytest.raises((TypeError, ValueError)):
                guard.admit_source(loaded)
        finally:
            object.__setattr__(loaded, "source", original_source)
        guard.admit_source(loaded)
        with pytest.raises((TypeError, ValueError)):
            guard.admit_source(loaded)
        assert guard._table is table and guard._counts is counters and guard._lock is lock
        root = loaded.source.roots[0]
        assert root.__code__ is parts["root_code"] and root.__globals__ is loaded.source.namespace
        clock, port, early, delegate = root()
        # The first dispatch happened while the source factory was still running.
        assert early == 7 and clock.reads == 1
        assert guard.counts["clock_call"] == 1 and guard.enrollment_counts == (1, 1)
        assert port.value is delegate and delegate.__code__ is parts["delegate_code"]
        assert port() == 7
        assert clock() == 7
        guard.require_current_enrollments()
        parts["classes"].require_current(loaded)
    assert guard._table is table and guard._counts is counters and guard._lock is lock
    assert guard.counts == POSITIVE and parts["effects"]["counts"] == POSITIVE
    assert parts["effects"]["bodies"] == POSITIVE_BODIES
    assert clock.reads == 2 and port.calls == 1 and guard.enrollment_counts == (1, 1)
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    assert threading.active_count() == 1 and not receipt.exists()
    assert all(module.frame is None for module in guard._deferred.modules)
    assert all(state.pending is None for state in guard._states)
    assert [len(state.ready) for state in guard._states] == [1, 1]
    assert all(owner.frame is None for state in guard._states for owner in state.ready)
    guard.require_current_enrollments()
    with pytest.raises((TypeError, ValueError)):
        guard.admit_source(loaded)
    with pytest.raises((TypeError, ValueError)):
        parts["classes"].require_unexecuted(parts["prepared"])
    with pytest.raises((TypeError, ValueError)):
        new_profile(parts, tmp_path / "spent-breach.json")
    namespace = loaded.source.namespace
    binding = loaded.classes[0]
    original_class = namespace["Clock"]
    namespace["Clock"] = object()
    try:
        with pytest.raises((TypeError, ValueError)):
            guard.require_current_enrollments()
    finally:
        namespace["Clock"] = original_class
    original_code = binding.raw_call.__code__
    binding.raw_call.__code__ = original_code.replace()
    try:
        with pytest.raises((TypeError, ValueError)):
            guard.require_current_enrollments()
    finally:
        binding.raw_call.__code__ = original_code
    port_storage = loaded.classes[1].descriptor.__get__(port, loaded.classes[1].owner_type)
    foreign = FunctionType(delegate.__code__, delegate.__globals__)
    port_storage["value"] = foreign
    try:
        with pytest.raises((TypeError, ValueError)):
            guard.require_current_enrollments()
    finally:
        port_storage["value"] = delegate
    guard.require_current_enrollments()
    parts["classes"].require_current(loaded)
    assert guard.counts == POSITIVE and parts["effects"]["counts"] == POSITIVE
    assert parts["events"] == [] and parts["effects"]["bodies"] == POSITIVE_BODIES
    with pytest.raises(RuntimeError, match="cannot renew"):
        with guard.measure():
            raise AssertionError("guard unexpectedly renewed")
    (tmp_path / "work.json").write_text(
        json.dumps(
            dict(
                bodies=10,
                attempts=sum(guard.counts.values()),
                module_calls=1,
                factory_calls=1,
                provider_calls=4,
                counts=guard.counts,
                side_effects=parts["effects"]["counts"],
                body_labels=parts["effects"]["bodies"],
                enrollment_counts=list(guard.enrollment_counts),
                frames_cleared=True,
                workers=0,
                joined=True,
                restored=True,
            )
        ),
        encoding="utf8",
    )


def test_deferred_spends_failed_module_without_admission(tmp_path):
    marker = b"deferredeffects.bodies.append('module')\n"
    failed_source = SOURCE.replace(
        marker,
        marker + b"if deferredeffects.mode == 'failed-module':\n"
        b"    raise RuntimeError('planned deferred module failure')\n",
        1,
    )
    parts = make_parts("failed-module", source=failed_source)
    receipt = tmp_path / "breach.json"
    guard = new_profile(parts, receipt)
    table, counters, lock = guard._table, guard._counts, guard._lock
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    expected = {**ZERO, "module": 1}
    with guard.measure():
        with pytest.raises(RuntimeError, match="planned deferred module failure"):
            parts["classes"].execute(parts["prepared"])
        assert len(guard._deferred.modules) == 1
        observer = guard._deferred.modules[0]
        assert observer.phase == "failed" and observer.frame is None and observer.seen == []
        assert guard.enrollment_counts == (0, 0)
        # Matching prepared identity must refuse the failed phase before the
        # deliberately foreign source row reaches the class helper checker.
        native_fake = LoadedSourceClasses(
            parts["prepared"], cast(LoadedFixtureSource, object()), ()
        )
        with pytest.raises(ValueError, match="invalid deferred construction witness"):
            guard.admit_source(native_fake)
        with pytest.raises((TypeError, ValueError)):
            parts["classes"].require_unexecuted(parts["prepared"])
        with pytest.raises((TypeError, ValueError)):
            parts["classes"].execute(parts["prepared"])
        guard.require_current_enrollments()
        assert guard._table is table and guard._counts is counters and guard._lock is lock
        assert guard.counts == expected and parts["effects"]["counts"] == expected
        assert parts["effects"]["bodies"] == ["module"]
    assert guard._table is table and guard._counts is counters and guard._lock is lock
    assert guard.counts == expected and parts["effects"]["counts"] == expected
    assert parts["effects"]["bodies"] == ["module"] and parts["events"] == []
    assert observer.phase == "failed" and observer.frame is None
    assert all(state.pending is None and state.ready == [] for state in guard._states)
    assert guard.enrollment_counts == (0, 0) and not receipt.exists()
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    assert threading.active_count() == 1
    with pytest.raises(RuntimeError, match="cannot renew"):
        with guard.measure():
            raise AssertionError("guard unexpectedly renewed")
    (tmp_path / "failed-module-work.json").write_text(
        json.dumps(
            dict(
                bodies=1,
                attempts=sum(guard.counts.values()),
                module_calls=1,
                factory_calls=0,
                provider_calls=0,
                counts=guard.counts,
                side_effects=parts["effects"]["counts"],
                body_labels=parts["effects"]["bodies"],
                enrollment_counts=list(guard.enrollment_counts),
                frames_cleared=True,
                workers=0,
                joined=True,
                restored=True,
            )
        ),
        encoding="utf8",
    )


def run_deferred_negative(mode, directory):
    path = Path(directory)
    parts = make_parts(mode)
    holders: list[LoadedSourceClasses] = []

    def prior(frame, event, arg):
        if event == "call" and frame.f_code is parts["root_code"]:
            holders[0].source.namespace["Clock"] = object()

    if mode == "priorhook-witness-swap":
        sys.setprofile(prior)
    guard = RecordingProfile(
        parts["entries"], path / "breach.json", (), deferred_roles=parts["roles"]
    )
    guard.record_path = path / "counts-sidecar.json"
    guard.effects = parts["effects"]
    try:
        with guard.measure():
            loaded = parts["classes"].execute(parts["prepared"])
            holders.append(loaded)
            root = loaded.source.roots[0]
            if mode == "unadmitted-constructor":
                loaded.classes[0].owner_type()
            elif mode == "unadmitted-provider":
                receiver = object.__new__(loaded.classes[0].owner_type)
                loaded.classes[0].raw_call(receiver)
            elif mode == "unadmitted-factory":
                root()
            else:
                guard.admit_source(loaded)
                if mode == "wrongglobals-caller":
                    foreign = FunctionType(root.__code__, root.__globals__.copy())
                    foreign()
                else:
                    root()
    except BaseException:
        (path / "caught").write_text("forbidden", encoding="utf8")
    (path / "continued").write_text("forbidden", encoding="utf8")


@pytest.mark.parametrize("mode", MODES)
def test_deferred_charges_denied_attempt_before_body(mode, tmp_path):
    args = [
        sys.executable,
        "-B",
        "-X",
        "utf8",
        "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from test_provider_deferred_construction import run_deferred_negative; "
        "run_deferred_negative(sys.argv[2], sys.argv[3])",
        str(Path(__file__).resolve().parent),
        mode,
        str(tmp_path),
    ]
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    env.pop("PYTHONPATH", None)
    result = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, timeout=3)
    assert result.returncode == 73 and len(result.stdout) + len(result.stderr) <= 4096
    assert not any((tmp_path / name).exists() for name in ("caught", "continued"))
    snapshot_raw = (tmp_path / "counts-sidecar.json").read_bytes()
    assert len(snapshot_raw) <= 4096
    snapshot = json.loads(snapshot_raw)
    assert set(snapshot) == {"counts", "side_effects", "body_labels", "enrollment_counts"}
    assert snapshot["counts"] == EXPECTED_COUNTS[mode]
    assert snapshot["side_effects"] == EXPECTED_EFFECTS[mode]
    assert snapshot["body_labels"] == EXPECTED_BODIES[mode]
    assert snapshot["enrollment_counts"] == ([1, 0] if mode == "capacityoverflow" else [0, 0])
    # The attempted label is charged once more than its actual body effect.
    assert sum(snapshot["counts"].values()) == len(snapshot["body_labels"]) + 1
    raw = (tmp_path / "breach.json").read_bytes()
    assert len(raw) <= 1024 and json.loads(raw) == RECEIPT
    (tmp_path / "child.json").write_text(
        json.dumps(
            dict(
                mode=mode,
                exit=result.returncode,
                setup_bodies=len(snapshot["body_labels"]),
                setup_body_labels=snapshot["body_labels"],
                counts=snapshot["counts"],
                side_effects=snapshot["side_effects"],
                enrollment_counts=snapshot["enrollment_counts"],
                offending_bodies=0,
                caught=0,
                continued=0,
                stdout_bytes=len(result.stdout),
                stderr_bytes=len(result.stderr),
                receipt_bytes=len(raw),
                receipt=json.loads(raw),
            )
        ),
        encoding="utf8",
    )
