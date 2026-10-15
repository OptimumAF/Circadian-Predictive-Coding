"""Native value continuity controls; no provider bodies or original domain imports."""

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from types import CodeType, FunctionType, GetSetDescriptorType, ModuleType
from typing import Any, Iterator, NamedTuple, cast

import pytest

from provider_code_path import ProviderCodePath, pin_provider_code_path
from provider_value_witness import ProviderValueBinding, ProviderValueWitnesses


CALLBACK_EVENTS: list[str] = []
TYPE_DICT = type.__dict__["__dict__"]
MODULE_DICT = cast(GetSetDescriptorType, TYPE_DICT.__get__(ModuleType, type)["__dict__"])


def callback_root():
    def callback():
        CALLBACK_EVENTS.append("callback body")
        return 7

    return callback


class Parts(NamedTuple):
    module: ModuleType
    record_type: type[Any]
    descriptor: GetSetDescriptorType
    path: ProviderCodePath
    callback: FunctionType


class Audit:
    """Count actual public operations and explicit owned fixture construction."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.pin_attempts = self.pin_accepts = self.pin_refusals = 0
        self.require_attempts = self.require_accepts = self.require_refusals = 0
        self.declaration_attempts = self.declaration_accepts = self.declaration_refusals = 0
        self.namespace_fixtures = self.record_fixtures = self.row_fixture_allocations = 0
        self.observed_record_constructor_calls = 0
        self.events: list[str] = []
        self.owners: list[ProviderValueWitnesses] = []

    def declare(self, parts: Parts, **changes: Any) -> ProviderValueWitnesses:
        arguments: dict[str, Any] = dict(
            record_module=parts.module,
            record_attribute="Record",
            record_type=parts.record_type,
            descriptor=parts.descriptor,
            callback_path=parts.path,
        )
        arguments.update(changes)
        self.declaration_attempts += 1
        try:
            owner = ProviderValueWitnesses(**arguments)
        except (TypeError, ValueError):
            self.declaration_refusals += 1
            raise
        self.declaration_accepts += 1
        self.owners.append(owner)
        return owner

    def pin(self, owner: ProviderValueWitnesses, value: Any) -> ProviderValueBinding:
        self.pin_attempts += 1
        before_attempts, before_bindings = owner.attempts, owner.binding_count
        try:
            binding = owner.pin(value)
        except (TypeError, ValueError):
            self.pin_refusals += 1
            assert owner.attempts == before_attempts + 1
            assert owner.binding_count == before_bindings
            raise
        self.pin_accepts += 1
        assert owner.attempts == before_attempts + 1
        assert owner.binding_count == before_bindings + 1
        assert type(binding) is ProviderValueBinding and binding.value is value
        return binding

    def require(self, owner: ProviderValueWitnesses, binding: Any, candidate: Any) -> None:
        self.require_attempts += 1
        before = owner.attempts, owner.binding_count
        try:
            owner.require(binding, candidate)
        except (TypeError, ValueError):
            self.require_refusals += 1
            assert (owner.attempts, owner.binding_count) == before
            raise
        self.require_accepts += 1
        assert (owner.attempts, owner.binding_count) == before

    def record(self, parts: Parts, snapshots: Any, examples: Any, payload_bytes: Any) -> Any:
        # These are test setup calls, distinct from forbidden helper construction.
        self.record_fixtures += 1
        return parts.record_type(snapshots, examples, payload_bytes)

    def finish(self, directory: Path) -> None:
        assert not CALLBACK_EVENTS and not self.events
        helper_constructor_calls = self.observed_record_constructor_calls - self.record_fixtures
        assert helper_constructor_calls == 0
        assert sum(owner.attempts for owner in self.owners) == self.pin_attempts
        successful = sum(owner.binding_count for owner in self.owners)
        assert successful == self.pin_accepts
        assert self.pin_attempts == self.pin_accepts + self.pin_refusals
        assert self.require_attempts == self.require_accepts + self.require_refusals
        row = dict(
            test_name=self.name,
            pin_attempts=self.pin_attempts,
            pin_accepts=self.pin_accepts,
            pin_refusals=self.pin_refusals,
            require_attempts=self.require_attempts,
            require_accepts=self.require_accepts,
            require_refusals=self.require_refusals,
            declaration_attempts=self.declaration_attempts,
            declaration_accepts=self.declaration_accepts,
            declaration_refusals=self.declaration_refusals,
            successful_bindings=successful,
            binding_allocations=self.pin_accepts,
            namespace_fixtures=self.namespace_fixtures,
            record_fixtures=self.record_fixtures,
            row_fixture_allocations=self.row_fixture_allocations,
            record_constructor_setup_calls=self.observed_record_constructor_calls,
            callback_calls=len(CALLBACK_EVENTS),
            hostile_events=list(self.events),
            helper_constructor_calls=helper_constructor_calls,
            provider_calls=0,
            delegate_calls=0,
            source_compilations=0,
            workers=0,
        )
        (directory / "value-work.json").write_text(json.dumps(row), encoding="utf8")


def make_parts(audit: Audit) -> Parts:
    @dataclass(frozen=True)
    class Record:
        snapshots: int
        examples: int
        payload_bytes: int

        def __post_init__(self) -> None:
            audit.observed_record_constructor_calls += 1

    module = ModuleType("owned_value_record")
    namespace = cast(dict[str, Any], MODULE_DICT.__get__(module, ModuleType))
    namespace["Record"] = Record
    descriptor = cast(GetSetDescriptorType, TYPE_DICT.__get__(Record, type)["__dict__"])
    root = cast(FunctionType, callback_root)
    code = root.__code__
    index, token = next(
        (index, value) for index, value in enumerate(code.co_consts) if type(value) is CodeType
    )
    path = pin_provider_code_path(root, code, (index,))
    callback = FunctionType(token, root.__globals__)
    audit.namespace_fixtures += 2
    return Parts(module, Record, descriptor, path, callback)


@contextmanager
def mapping_change(mapping: dict[Any, Any], key: Any, value: Any) -> Iterator[None]:
    present = key in mapping
    original = mapping.get(key)
    mapping[key] = value
    try:
        yield
    finally:
        if present:
            mapping[key] = original
        else:
            del mapping[key]


@contextmanager
def class_change(kind: type[Any], key: str, value: Any) -> Iterator[None]:
    namespace = TYPE_DICT.__get__(kind, type)
    present = key in namespace
    original = namespace.get(key)
    type.__setattr__(kind, key, value)
    try:
        yield
    finally:
        if present:
            type.__setattr__(kind, key, original)
        else:
            type.__delattr__(kind, key)


@contextmanager
def native_row_change(row: Any, field: str, value: Any) -> Iterator[None]:
    original = object.__getattribute__(row, field)
    object.__setattr__(row, field, value)
    try:
        yield
    finally:
        object.__setattr__(row, field, original)


@contextmanager
def padded_namespace(parts: Parts, use_class: bool, size: int) -> Iterator[None]:
    namespace = (
        TYPE_DICT.__get__(parts.record_type, type)
        if use_class
        else MODULE_DICT.__get__(parts.module, ModuleType)
    )
    keys = [f"padding_{index}" for index in range(size - len(namespace))]
    try:
        for key in keys:
            if use_class:
                type.__setattr__(parts.record_type, key, None)
            else:
                namespace[key] = None
        assert len(namespace) == size
        yield
    finally:
        for key in reversed(keys):
            if use_class:
                type.__delattr__(parts.record_type, key)
            else:
                del namespace[key]


def test_value_witness_accepts_original_three_modes(tmp_path: Path) -> None:
    audit = Audit("test_value_witness_accepts_original_three_modes")
    parts = make_parts(audit)
    owner = audit.declare(parts)
    negative_zero = float("-0.0")
    values = (
        0,
        2**2048,
        float("8.0"),
        negative_zero,
        audit.record(parts, 0, 0, 0),
        audit.record(parts, 2, 3, 8),
        parts.callback,
    )
    modes = ("primitive", "primitive", "primitive", "primitive", "record", "record", "callback")
    for value, mode in zip(values, modes):
        binding = audit.pin(owner, value)
        assert binding.mode == mode and binding.value is value
        audit.require(owner, binding, value)
    assert owner.binding_count == 7
    audit.finish(tmp_path)


def test_value_witness_refuses_declarations_and_policy_mutation(tmp_path: Path) -> None:
    audit = Audit("test_value_witness_refuses_declarations_and_policy_mutation")
    parts = make_parts(audit)
    owner = audit.declare(parts)
    binding = audit.pin(owner, 0)
    audit.require(owner, binding, 0)
    malformed: list[dict[str, Any]] = [
        {"record_module": object()},
        {"record_attribute": 0},
        {"record_type": object()},
        {"descriptor": object()},
        {"callback_path": object()},
        {"record_attribute": ""},
        {"record_attribute": "x" * 129},
        {"record_attribute": "missing"},
        {"max_values": 0},
        {"max_values": 65},
        {"max_values": True},
        {"max_values": 1.5},
        {"descriptor": MODULE_DICT},
    ]
    for changes in malformed:
        with pytest.raises((TypeError, ValueError)):
            audit.declare(parts, **changes)
    namespace = cast(dict[str, Any], MODULE_DICT.__get__(parts.module, ModuleType))
    with mapping_change(namespace, "R" * 128, parts.record_type):
        audit.declare(parts, record_attribute="R" * 128)
    with mapping_change(namespace, "Record", object()):
        with pytest.raises((TypeError, ValueError)):
            audit.declare(parts)
    for key in (1, "x" * 129):
        with mapping_change(namespace, key, None):
            with pytest.raises((TypeError, ValueError)):
                audit.declare(parts)
    for use_class in (False, True):
        with padded_namespace(parts, use_class, 512):
            audit.declare(parts)
        with padded_namespace(parts, use_class, 513):
            with pytest.raises((TypeError, ValueError)):
                audit.declare(parts)
    forbidden = (
        "__call__",
        "__new__",
        "__getattribute__",
        "__getattr__",
        "__slots__",
        "snapshots",
        "examples",
        "payload_bytes",
    )
    for field in forbidden:
        with class_change(parts.record_type, field, object()):
            with pytest.raises((TypeError, ValueError)):
                audit.declare(parts)

    class Base:
        pass

    class Child(Base):
        pass

    masked = type(
        "Masked", (), {"__dict__": property(lambda self: audit.events.append("descriptor"))}
    )
    audit.namespace_fixtures += 3
    for kind in (Child, masked):
        with mapping_change(namespace, "Record", kind):
            descriptor = TYPE_DICT.__get__(kind, type).get("__dict__", parts.descriptor)
            with pytest.raises((TypeError, ValueError)):
                audit.declare(parts, record_type=kind, descriptor=descriptor)
    for mutate in (
        mapping_change(namespace, "extra", object()),
        class_change(parts.record_type, "extra", object()),
    ):
        with mutate:
            with pytest.raises((TypeError, ValueError)):
                audit.require(owner, binding, 0)
        audit.require(owner, binding, 0)
    with pytest.raises(AttributeError):
        MODULE_DICT.__set__(parts.module, namespace.copy())
    assert MODULE_DICT.__get__(parts.module, ModuleType) is namespace
    audit.require(owner, binding, 0)
    audit.finish(tmp_path)


def test_value_witness_refuses_hostile_types_rows_and_keys(tmp_path: Path) -> None:
    audit = Audit("test_value_witness_refuses_hostile_types_rows_and_keys")
    parts = make_parts(audit)
    owner = audit.declare(parts)
    binding = audit.pin(owner, 0)
    audit.require(owner, binding, 0)

    class HostileMeta(type):
        def __hash__(cls) -> int:
            audit.events.append("type hash")
            raise AssertionError("foreign type hash dispatched")

        def __eq__(cls, other: object) -> bool:
            audit.events.append("type equality")
            raise AssertionError("foreign type equality dispatched")

    class Hostile(metaclass=HostileMeta):
        def __getattribute__(self, name: str) -> Any:
            audit.events.append("attribute")
            raise AssertionError("foreign getter dispatched")

        def __call__(self, *args: Any) -> Any:
            audit.events.append("call")
            raise AssertionError("foreign call dispatched")

        def __eq__(self, other: object) -> bool:
            audit.events.append("equality")
            raise AssertionError("foreign equality dispatched")

        def __hash__(self) -> int:
            audit.events.append("hash")
            raise AssertionError("foreign hash dispatched")

        def __repr__(self) -> str:
            audit.events.append("repr")
            raise AssertionError("foreign repr dispatched")

    class IntSubclass(int):
        pass

    class FloatSubclass(float):
        pass

    hostile: Any = object.__new__(Hostile)
    values: tuple[Any, ...] = (
        True,
        IntSubclass(7),
        FloatSubclass(7),
        float("nan"),
        float("inf"),
        float("-inf"),
        hostile,
        len,
        cast(FunctionType, lambda: None),
        object(),
    )
    for value in values:
        with pytest.raises((TypeError, ValueError)):
            audit.pin(owner, value)

    class HostileBinding(ProviderValueBinding):
        def __getattribute__(self, name: str) -> Any:
            audit.events.append("row attribute")
            raise AssertionError("foreign row getter dispatched")

    subclass_row: Any = object.__new__(HostileBinding)
    incomplete: Any = object.__new__(ProviderValueBinding)
    object.__setattr__(incomplete, "mode", "primitive")
    audit.row_fixture_allocations += 3
    for row in (hostile, subclass_row, incomplete):
        with pytest.raises((TypeError, ValueError)):
            audit.require(owner, row, 0)

    class HostileKey(str):
        __hash__ = str.__hash__

        def __len__(self) -> int:
            audit.events.append("key length")
            raise AssertionError("foreign key length dispatched")

        def __eq__(self, other: object) -> bool:
            audit.events.append("key equality")
            raise AssertionError("foreign key equality dispatched")

        def __repr__(self) -> str:
            audit.events.append("key repr")
            raise AssertionError("foreign key repr dispatched")

    record = audit.record(parts, 0, 0, 0)
    for changes in (
        {"record_attribute": HostileKey("Record")},
        {"callback_path": hostile},
        {"descriptor": hostile},
    ):
        with pytest.raises((TypeError, ValueError)):
            audit.declare(parts, **changes)
    storage = parts.descriptor.__get__(record, parts.record_type)
    with mapping_change(storage, HostileKey("foreign_record_key"), 0):
        with pytest.raises((TypeError, ValueError)):
            audit.pin(owner, record)
    namespace = cast(dict[str, Any], MODULE_DICT.__get__(parts.module, ModuleType))
    with mapping_change(namespace, HostileKey("foreign_module_key"), 0):
        with pytest.raises((TypeError, ValueError)):
            audit.require(owner, binding, 0)
    audit.require(owner, binding, 0)
    with mapping_change(namespace, "Record", Hostile):
        with pytest.raises((TypeError, ValueError)):
            audit.declare(parts, record_type=Hostile)

    class HostileModule(ModuleType):
        def __getattribute__(self, name: str) -> Any:
            audit.events.append("module attribute")
            raise AssertionError("foreign module getter dispatched")

    foreign_module = cast(Any, ModuleType.__new__)(HostileModule, "foreign_module")
    audit.namespace_fixtures += 2
    with pytest.raises((TypeError, ValueError)):
        audit.declare(parts, record_module=foreign_module)
    with pytest.raises((TypeError, ValueError)):
        audit.declare(parts, max_values=IntSubclass(1))
    audit.finish(tmp_path)


def test_value_witness_refuses_record_invariants_and_identity_changes(tmp_path: Path) -> None:
    audit = Audit("test_value_witness_refuses_record_invariants_and_identity_changes")
    parts = make_parts(audit)
    owner = audit.declare(parts)
    record = audit.record(parts, 1, int("1000"), int("1000"))
    binding = audit.pin(owner, record)
    audit.require(owner, binding, record)

    class IntSubclass(int):
        pass

    invalid: tuple[tuple[Any, Any, Any], ...] = (
        (-1, 0, 0),
        (1, -1, 1),
        (1, 1, -1),
        (True, 1, 1),
        (1, True, 1),
        (1, 1, True),
        (IntSubclass(1), 1, 1),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 0),
    )
    for fields in invalid:
        value = audit.record(parts, *fields)
        with pytest.raises((TypeError, ValueError)):
            audit.pin(owner, value)
    malformed = audit.record(parts, 0, 0, 0)
    malformed_storage = parts.descriptor.__get__(malformed, parts.record_type)
    original_fields = malformed_storage.copy()
    shapes: tuple[dict[Any, Any], ...] = (
        {"snapshots": 0, "payload_bytes": 0},
        {"snapshots": 0, "examples": 0, "payload_bytes": 0, "extra": 0},
        {"snapshots": 0, "examples": 0, 1: 0},
        {"snapshots": 0, "examples": 0, "x" * 129: 0},
    )
    try:
        for shape in shapes:
            malformed_storage.clear()
            malformed_storage.update(shape)
            with pytest.raises((TypeError, ValueError)):
                audit.pin(owner, malformed)
    finally:
        malformed_storage.clear()
        malformed_storage.update(original_fields)
    storage = parts.descriptor.__get__(record, parts.record_type)
    replacement = storage.copy()
    try:
        parts.descriptor.__set__(record, replacement)
        assert parts.descriptor.__get__(record, parts.record_type) is replacement
        with pytest.raises((TypeError, ValueError)):
            audit.require(owner, binding, record)
    finally:
        parts.descriptor.__set__(record, storage)
    audit.require(owner, binding, record)
    equal_examples = int("1000")
    assert equal_examples == storage["examples"] and equal_examples is not storage["examples"]
    namespace = cast(dict[str, Any], MODULE_DICT.__get__(parts.module, ModuleType))
    foreign_type = type("ForeignRecord", (), {})
    audit.namespace_fixtures += 1
    for mutate in (
        mapping_change(storage, "examples", equal_examples),
        mapping_change(storage, "extra", 0),
        mapping_change(namespace, "Record", foreign_type),
        class_change(
            parts.record_type,
            "snapshots",
            property(lambda self: audit.events.append("field getter")),
        ),
    ):
        with mutate:
            with pytest.raises((TypeError, ValueError)):
                audit.require(owner, binding, record)
        audit.require(owner, binding, record)
    audit.finish(tmp_path)


def test_value_witness_refuses_candidate_and_binding_replacements(tmp_path: Path) -> None:
    audit = Audit("test_value_witness_refuses_candidate_and_binding_replacements")
    parts = make_parts(audit)
    owner = audit.declare(parts)
    primitive = int("1000")
    record = audit.record(parts, 0, 0, 0)
    values = (primitive, record, parts.callback)
    bindings = [audit.pin(owner, value) for value in values]
    for binding, value in zip(bindings, values):
        audit.require(owner, binding, value)
    primitive_clone = int("1000")
    assert primitive_clone == primitive and primitive_clone is not primitive
    candidates = (
        primitive_clone,
        audit.record(parts, 0, 0, 0),
        FunctionType(parts.callback.__code__, cast(FunctionType, parts.callback).__globals__),
    )
    for binding, candidate in zip(bindings, candidates):
        with pytest.raises((TypeError, ValueError)):
            audit.require(owner, binding, candidate)
    for binding, value in zip(bindings, values):
        clone = ProviderValueBinding(binding.mode, binding.value)
        audit.row_fixture_allocations += 1
        with pytest.raises((TypeError, ValueError)):
            audit.require(owner, clone, value)
        other_mode = "callback" if binding.mode != "callback" else "primitive"
        for field, replacement in (("mode", other_mode), ("value", object())):
            with native_row_change(binding, field, replacement):
                with pytest.raises((TypeError, ValueError)):
                    audit.require(owner, binding, value)
            audit.require(owner, binding, value)
    audit.finish(tmp_path)


def test_value_witness_reproves_callback_code_globals_and_path(tmp_path: Path) -> None:
    audit = Audit("test_value_witness_reproves_callback_code_globals_and_path")
    parts = make_parts(audit)
    owner = audit.declare(parts)
    original = audit.pin(owner, parts.callback)
    first_capture_clone = FunctionType(
        parts.callback.__code__, cast(FunctionType, parts.callback).__globals__
    )
    assert first_capture_clone is not parts.callback
    clone_binding = audit.pin(owner, first_capture_clone)
    audit.require(owner, original, parts.callback)
    audit.require(owner, clone_binding, first_capture_clone)
    leaf = parts.callback.__code__
    equal_leaf = leaf.replace()
    assert equal_leaf == leaf and equal_leaf is not leaf
    foreign_globals: dict[str, Any] = {}
    bad_candidates = (
        FunctionType(leaf, foreign_globals),
        FunctionType(equal_leaf, cast(FunctionType, parts.callback).__globals__),
        cast(FunctionType, lambda: None),
    )
    for candidate in bad_candidates:
        with pytest.raises((TypeError, ValueError)):
            audit.pin(owner, candidate)
    for candidate in bad_candidates[:2]:
        with pytest.raises((TypeError, ValueError)):
            audit.require(owner, original, candidate)
    root_code = parts.path.root_code
    equal_root = root_code.replace()
    indexes_clone = tuple(list(parts.path.indexes))
    assert indexes_clone == parts.path.indexes and indexes_clone is not parts.path.indexes
    changes = (
        native_row_change(parts.callback, "__code__", equal_leaf),
        native_row_change(parts.path.root, "__code__", equal_root),
        native_row_change(parts.path, "root", FunctionType(root_code, foreign_globals)),
        native_row_change(parts.path, "root_code", equal_root),
        native_row_change(parts.path, "indexes", indexes_clone),
        native_row_change(parts.path, "token", equal_leaf),
    )
    for mutate in changes:
        with mutate:
            with pytest.raises((TypeError, ValueError)):
                audit.require(owner, original, parts.callback)
        audit.require(owner, original, parts.callback)
    audit.finish(tmp_path)


def test_value_witness_charges_shared_capacity_without_refund(tmp_path: Path) -> None:
    audit = Audit("test_value_witness_charges_shared_capacity_without_refund")
    parts = make_parts(audit)
    owner = audit.declare(parts, max_values=64)
    audit.declare(parts, max_values=1)
    primitive = audit.pin(owner, 0)
    with pytest.raises((TypeError, ValueError)):
        audit.pin(owner, True)
    record = audit.record(parts, 0, 0, 0)
    record_binding = audit.pin(owner, record)
    callback = audit.pin(owner, parts.callback)
    for _ in range(60):
        audit.pin(owner, 0)
    assert owner.attempts == 64 and owner.binding_count == 63
    with pytest.raises((TypeError, ValueError)):
        audit.pin(owner, 0)
    assert owner.attempts == 65 and owner.binding_count == 63
    for _ in range(2):
        for binding, value in (
            (primitive, 0),
            (record_binding, record),
            (callback, parts.callback),
        ):
            audit.require(owner, binding, value)
    assert owner.attempts == 65 and owner.binding_count == 63
    for field in ("attempts", "binding_count"):
        with pytest.raises(AttributeError):
            object.__setattr__(owner, field, 0)
    assert owner.attempts == 65 and owner.binding_count == 63
    audit.finish(tmp_path)


def test_value_witness_requires_fresh_reproof_for_each_mode(tmp_path: Path) -> None:
    audit = Audit("test_value_witness_requires_fresh_reproof_for_each_mode")
    parts = make_parts(audit)
    owner = audit.declare(parts)
    values = (0, audit.record(parts, 0, 0, 0), parts.callback)
    bindings = [audit.pin(owner, value) for value in values]
    for binding, value in zip(bindings, values):
        audit.require(owner, binding, value)
    namespace = cast(dict[str, Any], MODULE_DICT.__get__(parts.module, ModuleType))
    changes = (
        mapping_change(namespace, "freshness_probe", object()),
        class_change(parts.record_type, "freshness_probe", object()),
        native_row_change(parts.path.root, "__code__", parts.path.root_code.replace()),
    )
    for mutate in changes:
        with mutate:
            for binding, value in zip(bindings, values):
                with pytest.raises((TypeError, ValueError)):
                    audit.require(owner, binding, value)
        for binding, value in zip(bindings, values):
            audit.require(owner, binding, value)
    audit.finish(tmp_path)
