"""Finite mixed-value controls with explicit setup accounting; no original fixtures."""

import builtins
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import FrameType, FunctionType, GetSetDescriptorType, ModuleType
from typing import Any, NoReturn, cast

import pytest

from provider_code_path import pin_provider_code_path
from provider_construction_profile import (
    ConstructionValuePolicy,
    OwnerConstructionRole,
    ProviderConstructionProfile,
)
from provider_deferred_construction import DeferredConstructionRole
from provider_source_classes import FixtureSourceClasses, LoadedSourceClasses
from provider_source_loader import ApprovedSourceImport, FixtureSourceLoader
from provider_value_witness import ProviderValueWitnesses

LABELS = (
    "module",
    "class_body",
    "factory",
    "constructor",
    "provider",
    "delegate",
    "native_callable",
)
ZERO = dict.fromkeys(LABELS, 0)
BREACH = "mixed-breach.json"
MODES = (
    "argument-swap-hook",
    "return-slot-clone",
    "record-field-hook",
    "primitive-delegate",
    "deferred-policy-hook",
    "capacityoverflow",
)
UNIT_NAMES = (
    "test_mixed_witness_current_is_fresh_without_charges",
    "test_mixed_refuses_native_policy_declarations",
    "test_mixed_refuses_hostile_policy_rows_and_keys",
    "test_mixed_reproves_policy_with_zero_owners",
    "test_mixed_enrolls_immediate_four_value_modes",
    "test_mixed_enrolls_deferred_four_value_modes",
    "test_mixed_retains_selected_policy_identity",
    "test_mixed_preflights_deferred_policy_before_source",
)
TYPE_DICT = type.__dict__["__dict__"]
MODULE_DICT = cast(GetSetDescriptorType, TYPE_DICT.__get__(ModuleType, type)["__dict__"])
SOURCE = b"""from mixedsupport import callable
import mixedsupport
mixedsupport.effects['counts']['module'] += 1
if mixedsupport.mode == 'deferred-policy-hook':
    mixedsupport.effects['offending'] += 1
mixedsupport.effects['bodies'].append('module')
class Port:
    def __init__(self, value):
        if mixedsupport.mode == 'argument-swap-hook':
            mixedsupport.effects['offending'] += 1
        if mixedsupport.mode == 'capacityoverflow' and mixedsupport.effects['counts']['constructor'] == 1:
            mixedsupport.effects['offending'] += 1
        mixedsupport.effects['counts']['constructor'] += 1
        mixedsupport.effects['bodies'].append('constructor')
        self.value = value
        self.calls = 0
        if mixedsupport.mode == 'return-slot-clone':
            self.value = mixedsupport.replacement
    def __call__(self):
        if mixedsupport.mode == 'return-slot-clone' or mixedsupport.mode == 'record-field-hook':
            mixedsupport.effects['offending'] += 1
        mixedsupport.effects['counts']['provider'] += 1
        mixedsupport.effects['bodies'].append('provider')
        self.calls += 1
        if mixedsupport.mode == 'primitive-delegate':
            return delegate()
        return self.value() if callable(self.value) else self.value
mixedsupport.effects['counts']['class_body'] += 1
mixedsupport.effects['bodies'].append('class_body')
def factory(value):
    mixedsupport.effects['counts']['factory'] += 1
    mixedsupport.effects['bodies'].append('factory')
    owner = Port(value)
    if mixedsupport.mode == 'capacityoverflow':
        Port(mixedsupport.replacement)
    return owner, owner()
def delegate():
    if mixedsupport.mode == 'primitive-delegate':
        mixedsupport.effects['offending'] += 1
    mixedsupport.effects['counts']['delegate'] += 1
    mixedsupport.effects['bodies'].append('delegate')
    return 7
"""


class Parts:
    """Owned counters and native rows."""

    def __init__(self, *, deferred: bool = False, mode: str = "positive", capacity: int = 4):
        self.deferred = deferred
        self.record_setup = {"class": 0, "constructor": 0}
        self.explicit_record_calls = 0
        setup = self.record_setup

        @dataclass(frozen=True)
        class Record:
            setup["class"] += 1
            snapshots: int
            examples: int
            payload_bytes: int

            def __post_init__(self):
                setup["constructor"] += 1

        self.record_type = Record
        self.record_module = ModuleType("mixed_record")
        self.record_ns = MODULE_DICT.__get__(self.record_module, ModuleType)
        self.record_ns["Record"] = Record
        self.record_reader = cast(GetSetDescriptorType, TYPE_DICT.__get__(Record, type)["__dict__"])
        self.initial_value: Any = 7
        replacement: Any = 8
        if mode in ("return-slot-clone", "record-field-hook"):
            self.initial_value = self.make_record(1, int("1000"), int("1000"))
        if mode == "return-slot-clone":
            replacement = self.make_record(
                1, self.initial_value.examples, self.initial_value.payload_bytes
            )
        elif mode == "capacityoverflow":
            replacement = self.make_record(1, 2, 3)
        self.effects: dict[str, Any] = {"counts": ZERO.copy(), "bodies": [], "offending": 0}
        self.support = ModuleType("mixedsupport")
        support = MODULE_DICT.__get__(self.support, ModuleType)
        support.update(
            callable=builtins.callable, effects=self.effects, mode=mode, replacement=replacement
        )
        self.support_ns = support
        self.classes = FixtureSourceClasses(
            FixtureSourceLoader((ApprovedSourceImport("mixedsupport", self.support),)),
            max_classes=1,
        )
        self.prepared = self.classes.prepare(
            SOURCE,
            sha256(SOURCE).hexdigest(),
            "mixed_source",
            "mixed_source.py",
            ("factory", "delegate"),
            class_roots=("Port",),
        )
        code = self.prepared.classes[0]
        tokens = (
            self.prepared.source.module_code,
            code.class_code,
            self.prepared.source.root_codes[0][1],
            code.constructor_code,
            code.call_code,
            self.prepared.source.root_codes[1][1],
            builtins.callable,
        )
        self.entries = tuple(zip(tokens, LABELS, (1, 1, 4, 4, 5, 2, 5)))
        self.loaded: Any = None
        self.role: Any
        self.factory: Any = None
        self.callback: Any = None
        self.owner_type: Any = None
        self.setup_bodies: list[str] = []
        if deferred:
            self.role = DeferredConstructionRole(
                self.classes,
                self.prepared,
                0,
                "factory",
                (),
                capacity,
                "value",
                "value",
                "delegate",
                (),
            )
        else:
            self.load()
            self.setup_bodies = list(self.effects["bodies"])
            self.effects["counts"] = ZERO.copy()
            self.effects["bodies"] = []
            row = cast(LoadedSourceClasses, self.loaded).classes[0]
            self.role = OwnerConstructionRole(
                row.owner_type,
                row.descriptor,
                row.constructor,
                code.constructor_code,
                pin_provider_code_path(
                    cast(FunctionType, self.factory), self.prepared.source.root_codes[0][1], ()
                ),
                row.raw_call,
                code.call_code,
                capacity,
                "value",
                "value",
                pin_provider_code_path(
                    cast(FunctionType, self.callback), self.prepared.source.root_codes[1][1], ()
                ),
            )
        self.policy = ConstructionValuePolicy(
            self.role, self.record_module, "Record", Record, self.record_reader
        )

    def load(self):
        self.loaded = self.classes.execute(self.prepared)
        self.factory, self.callback = self.loaded.source.roots
        self.owner_type = self.loaded.classes[0].owner_type

    def make_record(self, snapshots, examples, payload_bytes):
        self.explicit_record_calls += 1
        return self.record_type(snapshots, examples, payload_bytes)

    def profile(self, receipt: Path, policies=None, role=None, kind=ProviderConstructionProfile):
        selected = self.role if role is None else role
        return kind(
            self.entries,
            receipt,
            () if self.deferred else (selected,),
            deferred_roles=(selected,) if self.deferred else (),
            value_policies=(self.policy,) if policies is None else policies,
        )


class Audit:
    def __init__(self):
        self.values = dict.fromkeys(
            (
                "profile_attempts",
                "profile_accepts",
                "profile_refusals",
                "require_attempts",
                "require_accepts",
                "require_refusals",
                "witness_current_attempts",
                "witness_current_accepts",
                "witness_current_refusals",
                "pin_attempts",
                "pin_accepts",
                "pin_refusals",
            ),
            0,
        )
        self.hostile_events: list[str] = []

    def check(self, action, category="require", refused=False):
        self.values[category + "_attempts"] += 1
        if refused:
            with pytest.raises((TypeError, ValueError)):
                action()
            self.values[category + "_refusals"] += 1
            return None
        result = action()
        self.values[category + "_accepts"] += 1
        return result


def changed_check(audit: Audit, action, change, restore, category="require"):
    try:
        change()
        audit.check(action, category, refused=True)
    finally:
        restore()
    audit.check(action, category)


def write_unit(
    path: Path,
    name: str,
    kit: Parts,
    audit: Audit,
    guard=None,
    *,
    results=None,
    one_use=0,
    witness=None,
):
    counts = ZERO.copy() if guard is None else guard.counts
    labels = list(kit.effects["bodies"])
    record = dict(
        test_name=name,
        kind="deferred" if kit.deferred else "immediate",
        **audit.values,
        counts=counts,
        registered_attempts=sum(counts.values()),
        registered_bodies=len(labels),
        registered_body_labels=labels,
        fixture_setup_bodies=len(kit.setup_bodies),
        fixture_setup_body_labels=kit.setup_bodies,
        source_compilations=1,
        source_loads=int(kit.loaded is not None),
        record_class_setup_calls=kit.record_setup["class"],
        record_constructor_setup_calls=kit.record_setup["constructor"],
        record_constructor_helper_calls=kit.record_setup["constructor"] - kit.explicit_record_calls,
        callback_calls=kit.effects["counts"]["delegate"],
        hostile_events=audit.hostile_events,
        enrollment_counts=[] if guard is None else list(guard.enrollment_counts),
        runtime_identity_validated=guard is not None,
        results=[] if results is None else results,
        one_use_refusals=one_use,
        witness_attempts=0 if witness is None else witness.attempts,
        witness_binding_count=0 if witness is None else witness.binding_count,
        restored=True,
        receipt_absent=not (path / BREACH).exists(),
        workers=0,
        joined=True,
    )
    assert record["receipt_absent"]
    assert not audit.hostile_events
    (path / "mixed-unit-work.json").write_text(json.dumps(record), encoding="utf8")


def test_mixed_witness_current_is_fresh_without_charges(tmp_path):
    kit, audit = Parts(), Audit()
    path = pin_provider_code_path(kit.callback, kit.prepared.source.root_codes[1][1], ())
    witness = ProviderValueWitnesses(
        kit.record_module,
        "Record",
        kit.record_type,
        kit.record_reader,
        path,
        max_values=1,
    )
    current = witness.require_current
    audit.check(current, "witness_current")
    changed_check(
        audit,
        current,
        lambda: kit.record_ns.update(extra=object()),
        lambda: kit.record_ns.pop("extra"),
        "witness_current",
    )
    changed_check(
        audit,
        current,
        lambda: setattr(kit.record_type, "extra", object()),
        lambda: delattr(kit.record_type, "extra"),
        "witness_current",
    )
    original = kit.callback.__code__
    changed_check(
        audit,
        current,
        lambda: setattr(kit.callback, "__code__", original.replace()),
        lambda: setattr(kit.callback, "__code__", original),
        "witness_current",
    )
    audit.check(lambda: witness.pin(True), "pin", refused=True)
    audit.check(current, "witness_current")
    audit.check(current, "witness_current")
    audit.check(lambda: witness.pin(0), "pin", refused=True)
    audit.check(current, "witness_current")
    assert witness.attempts == 2 and witness.binding_count == 0
    write_unit(tmp_path, UNIT_NAMES[0], kit, audit, witness=witness)


def test_mixed_refuses_native_policy_declarations(tmp_path):
    kit, audit = Parts(), Audit()
    guard = audit.check(lambda: kit.profile(tmp_path / BREACH), "profile")
    policy = kit.policy
    bad: list[Any] = [[], (policy,) * 65, (object(),), (policy, policy)]
    for field, value in (
        ("role", object()),
        ("role", replace(kit.role)),
        ("record_module", object()),
        ("record_attribute", ""),
        ("record_attribute", "x" * 129),
        ("record_attribute", "missing"),
        ("record_type", object()),
        ("record_descriptor", MODULE_DICT),
    ):
        bad.append((replace(policy, **cast(dict[str, Any], {field: value})),))
    for policies in bad:
        audit.check(
            lambda policies=policies: kit.profile(tmp_path / BREACH, policies),
            "profile",
            refused=True,
        )
    for capacity in (0, 65, True):
        audit.check(
            lambda capacity=capacity: kit.profile(
                tmp_path / BREACH, role=replace(kit.role, max_instances=capacity)
            ),
            "profile",
            refused=True,
        )
    incomplete = replace(kit.role, callback_path=None)
    audit.check(
        lambda: kit.profile(tmp_path / BREACH, (replace(policy, role=incomplete),), incomplete),
        "profile",
        refused=True,
    )
    maximum = replace(kit.role, max_instances=64)
    audit.check(
        lambda: kit.profile(tmp_path / BREACH, (replace(policy, role=maximum),), maximum),
        "profile",
    )
    no_callback = replace(kit.role, callback_slot=None, callback_argument=None, callback_path=None)
    audit.check(
        lambda: kit.profile(tmp_path / BREACH, (replace(policy, role=no_callback),), no_callback),
        "profile",
        refused=True,
    )
    kit.record_ns[1] = object()
    try:
        audit.check(lambda: kit.profile(tmp_path / BREACH), "profile", refused=True)
    finally:
        del kit.record_ns[1]
    keys = ["bound_" + str(index) for index in range(512 - len(kit.record_ns))]
    try:
        kit.record_ns.update(dict.fromkeys(keys, None))
        audit.check(lambda: kit.profile(tmp_path / BREACH), "profile")
        kit.record_ns["over_bound"] = None
        audit.check(lambda: kit.profile(tmp_path / BREACH), "profile", refused=True)
    finally:
        kit.record_ns.pop("over_bound", None)
        for key in keys:
            del kit.record_ns[key]
    for name, value in (("x" * 129, None), ("examples", property(cast(Any, events_forbidden)))):
        try:
            setattr(kit.record_type, name, value)
            audit.check(lambda: kit.profile(tmp_path / BREACH), "profile", refused=True)
        finally:
            delattr(kit.record_type, name)
    audit.check(guard.require_current_enrollments)
    write_unit(tmp_path, UNIT_NAMES[1], kit, audit, guard)


def test_mixed_refuses_hostile_policy_rows_and_keys(tmp_path):
    kit, audit = Parts(), Audit()
    events = audit.hostile_events

    class Hostile:
        def __getattribute__(self, name):
            events.append("getattribute")
            raise AssertionError(name)

    class HostileKey(str):
        def __len__(self):
            events.append("len")
            raise AssertionError

        def __eq__(self, other):
            events.append("eq")
            raise AssertionError

        __hash__ = str.__hash__

    class HostileModule(ModuleType):
        def __getattribute__(self, name):
            events.append("module-getattribute")
            raise AssertionError(name)

    class HostileMeta(type):
        def __getattribute__(self, name):
            events.append("meta-getattribute")
            raise AssertionError(name)

    class ForeignRecord(metaclass=HostileMeta):
        pass

    class PolicySubclass(ConstructionValuePolicy):
        pass

    guard = audit.check(lambda: kit.profile(tmp_path / BREACH), "profile")
    foreign: Any = object.__new__(Hostile)
    incomplete: Any = object.__new__(ConstructionValuePolicy)
    subclass: Any = object.__new__(PolicySubclass)
    module: Any = cast(Any, ModuleType.__new__)(HostileModule, "hostile_module")
    key = HostileKey("unused")
    for policies in (
        (foreign,),
        (incomplete,),
        (subclass,),
        (replace(kit.policy, record_module=module),),
        (replace(kit.policy, record_type=ForeignRecord),),
        (replace(kit.policy, record_attribute=HostileKey("Record")),),
    ):
        audit.check(
            lambda policies=policies: kit.profile(tmp_path / BREACH, policies),
            "profile",
            refused=True,
        )
    kit.record_ns[key] = object()
    try:
        audit.check(lambda: kit.profile(tmp_path / BREACH), "profile", refused=True)
    finally:
        del kit.record_ns[key]
    audit.check(guard.require_current_enrollments)
    changed_check(
        audit,
        guard.require_current_enrollments,
        lambda: kit.record_ns.update({key: object()}),
        lambda: kit.record_ns.pop(key),
    )
    original = kit.policy.record_attribute
    changed_check(
        audit,
        guard.require_current_enrollments,
        lambda: object.__setattr__(kit.policy, "record_attribute", HostileKey("Record")),
        lambda: object.__setattr__(kit.policy, "record_attribute", original),
    )
    write_unit(tmp_path, UNIT_NAMES[2], kit, audit, guard)


def policy_mutations(kit: Parts):
    return (
        ("role", replace(kit.role)),
        ("record_module", ModuleType("foreign_record")),
        ("record_attribute", "".join(("Rec", "ord"))),
        ("record_type", type("Foreign", (), {})),
        ("record_descriptor", MODULE_DICT),
    )


def test_mixed_reproves_policy_with_zero_owners(tmp_path):
    kit, audit = Parts(), Audit()
    guard = audit.check(lambda: kit.profile(tmp_path / BREACH), "profile")
    current = guard.require_current_enrollments
    audit.check(current)
    for field, replacement in policy_mutations(kit):
        original = getattr(kit.policy, field)
        assert replacement is not original
        changed_check(
            audit,
            current,
            lambda field=field, replacement=replacement: object.__setattr__(
                kit.policy, field, replacement
            ),
            lambda field=field, original=original: object.__setattr__(kit.policy, field, original),
        )
    changed_check(
        audit,
        current,
        lambda: kit.record_ns.update(extra=object()),
        lambda: kit.record_ns.pop("extra"),
    )
    for name in ("extra", "__call__"):
        changed_check(
            audit,
            current,
            lambda name=name: setattr(kit.record_type, name, lambda *args: events_forbidden()),
            lambda name=name: delattr(kit.record_type, name),
        )
    policies = guard.value_policies
    changed_check(
        audit,
        current,
        lambda: setattr(guard, "value_policies", tuple(list(policies))),
        lambda: setattr(guard, "value_policies", policies),
    )
    for function in (kit.factory, kit.callback):
        original_code = function.__code__
        changed_check(
            audit,
            current,
            lambda function=function, original_code=original_code: setattr(
                function, "__code__", original_code.replace()
            ),
            lambda function=function, original_code=original_code: setattr(
                function, "__code__", original_code
            ),
        )
    assert guard.enrollment_counts == (0,)
    write_unit(tmp_path, UNIT_NAMES[3], kit, audit, guard)


def events_forbidden():
    raise AssertionError("foreign body dispatched")


def positive_window(tmp_path: Path, *, deferred: bool, name: str):
    kit, audit = Parts(deferred=deferred), Audit()
    guard = audit.check(lambda: kit.profile(tmp_path / BREACH), "profile")
    prior = (sys.getprofile(), threading.getprofile())
    current = guard.require_current_enrollments
    audit.check(current)
    record = kit.make_record(1, int("1000"), int("1000"))
    values: list[Any] = [7, 8.0, record]
    owners: list[Any] = []
    with guard.measure():
        if deferred:
            kit.load()
            guard.admit_source(kit.loaded)
        values.append(kit.callback)
        for value in values:
            owner, result = kit.factory(value)
            owners.append(owner)
            if value is kit.callback:
                assert result == 7
            else:
                assert result is value
        assert owners[-1]() == 7
        audit.check(current)
    assert (sys.getprofile(), threading.getprofile()) == prior
    audit.check(current)
    assert guard.enrollment_counts == (4,)
    assert kit.callback.__code__ is kit.prepared.source.root_codes[1][1]
    assert kit.callback.__globals__ is kit.loaded.source.namespace
    kit.classes.require_current_bindings(kit.loaded)
    record_clone = kit.make_record(1, record.examples, record.payload_bytes)
    record_dict = kit.record_reader.__get__(record, kit.record_type)
    callback_clone = FunctionType(kit.callback.__code__, kit.callback.__globals__)
    row = kit.loaded.classes[0]
    call_clone = FunctionType(row.raw_call.__code__, row.raw_call.__globals__)
    policies = guard.value_policies
    examples = record.examples
    mutations = [
        (lambda: setattr(owners[0], "value", 8), lambda: setattr(owners[0], "value", values[0])),
        (
            lambda: object.__setattr__(record, "examples", int("1000")),
            lambda: object.__setattr__(record, "examples", examples),
        ),
        (
            lambda: kit.record_reader.__set__(record, record_dict.copy()),
            lambda: kit.record_reader.__set__(record, record_dict),
        ),
        (
            lambda: setattr(owners[2], "value", record_clone),
            lambda: setattr(owners[2], "value", record),
        ),
        (
            lambda: setattr(owners[3], "value", callback_clone),
            lambda: setattr(owners[3], "value", kit.callback),
        ),
        (
            lambda: kit.record_ns.update(Record=object),
            lambda: kit.record_ns.update(Record=kit.record_type),
        ),
        (
            lambda: setattr(kit.owner_type, "__call__", call_clone),
            lambda: setattr(kit.owner_type, "__call__", row.raw_call),
        ),
        (
            lambda: setattr(guard, "value_policies", (replace(kit.policy),)),
            lambda: setattr(guard, "value_policies", policies),
        ),
    ]
    for change, restore in mutations:
        changed_check(audit, current, change, restore)
    if deferred:
        changed_check(
            audit,
            current,
            lambda: object.__setattr__(kit.policy, "role", replace(kit.role)),
            lambda: object.__setattr__(kit.policy, "role", kit.role),
        )
        changed_check(
            audit,
            current,
            lambda: kit.loaded.source.namespace.update(Port=object),
            lambda: kit.loaded.source.namespace.update(Port=kit.owner_type),
        )
    with pytest.raises(RuntimeError, match="^entry profile cannot renew its allowance$"):
        with guard.measure():
            pytest.fail("one-use window renewed")
    assert (sys.getprofile(), threading.getprofile()) == prior
    write_unit(
        tmp_path, name, kit, audit, guard, results=[7, 8.0, "record-original", 7, 7], one_use=1
    )


def test_mixed_enrolls_immediate_four_value_modes(tmp_path):
    positive_window(tmp_path, deferred=False, name=UNIT_NAMES[4])


def test_mixed_enrolls_deferred_four_value_modes(tmp_path):
    positive_window(tmp_path, deferred=True, name=UNIT_NAMES[5])


def test_mixed_retains_selected_policy_identity(tmp_path):
    kit, audit = Parts(), Audit()
    selected = replace(kit.policy)
    assert selected is not kit.policy and selected.role is kit.role
    policies = (selected,)
    guard = audit.check(lambda: kit.profile(tmp_path / BREACH, policies), "profile")
    current = guard.require_current_enrollments
    audit.check(current)
    changed_check(
        audit,
        current,
        lambda: setattr(guard, "value_policies", tuple(list(policies))),
        lambda: setattr(guard, "value_policies", policies),
    )
    changed_check(
        audit,
        current,
        lambda: setattr(guard, "value_policies", (replace(selected),)),
        lambda: setattr(guard, "value_policies", policies),
    )
    for field, replacement in policy_mutations(kit):
        original = getattr(selected, field)
        changed_check(
            audit,
            current,
            lambda field=field, replacement=replacement: object.__setattr__(
                selected, field, replacement
            ),
            lambda field=field, original=original: object.__setattr__(selected, field, original),
        )
    assert guard.enrollment_counts == (0,)
    write_unit(tmp_path, UNIT_NAMES[6], kit, audit, guard)


def test_mixed_preflights_deferred_policy_before_source(tmp_path):
    kit, audit = Parts(deferred=True), Audit()
    guard = audit.check(lambda: kit.profile(tmp_path / BREACH), "profile")
    current = guard.require_current_enrollments
    audit.check(current)
    policies = guard.value_policies
    changes = (
        (
            lambda: kit.record_ns.update(extra=object()),
            lambda: kit.record_ns.pop("extra"),
        ),
        (
            lambda: setattr(kit.record_type, "__call__", events_forbidden),
            lambda: delattr(kit.record_type, "__call__"),
        ),
        (
            lambda: object.__setattr__(kit.policy, "record_descriptor", MODULE_DICT),
            lambda: object.__setattr__(kit.policy, "record_descriptor", kit.record_reader),
        ),
        (
            lambda: setattr(guard, "value_policies", tuple(list(policies))),
            lambda: setattr(guard, "value_policies", policies),
        ),
        (
            lambda: object.__setattr__(kit.policy, "role", replace(kit.role)),
            lambda: object.__setattr__(kit.policy, "role", kit.role),
        ),
    )
    for change, restore in changes:
        changed_check(audit, current, change, restore)
    kit.classes.require_unexecuted(kit.prepared)
    with pytest.raises(ValueError):
        guard.admit_source(cast(Any, object()))
    kit.classes.require_unexecuted(kit.prepared)
    assert not kit.effects["bodies"]
    write_unit(tmp_path, UNIT_NAMES[7], kit, audit, guard)


class RecordingProfile(ProviderConstructionProfile):
    record_path: Path
    kit: Parts

    def _abort(
        self, reason: str, label: str = "profile", attempt: int = 0, limit: int = 0
    ) -> NoReturn:
        try:
            kit = self.kit
            snapshot = dict(
                counts=self.counts,
                side_effects=kit.effects["counts"].copy(),
                body_labels=list(kit.effects["bodies"]),
                enrollment_counts=list(self.enrollment_counts),
                offending_bodies=kit.effects["offending"],
                fixture_setup_bodies=len(kit.setup_bodies),
                fixture_setup_body_labels=kit.setup_bodies,
                record_class_setup_calls=kit.record_setup["class"],
                record_constructor_setup_calls=kit.record_setup["constructor"],
                record_constructor_helper_calls=kit.record_setup["constructor"]
                - kit.explicit_record_calls,
            )
            raw = json.dumps(snapshot).encode("utf8")
            assert len(raw) <= 4096
            self.record_path.write_bytes(raw)
        finally:
            super()._abort(reason, label, attempt, limit)


def run_mixed_negative(mode: str, directory: Path):
    assert mode in MODES
    deferred = mode in ("deferred-policy-hook", "capacityoverflow")
    kit = Parts(deferred=deferred, mode=mode, capacity=1)
    value = kit.initial_value
    new_examples = int("1000")
    fired = False

    def prior_hook(frame: FrameType, event: str, arg: Any):
        nonlocal fired
        if fired or event != "call":
            return
        if (
            mode == "argument-swap-hook"
            and frame.f_code is kit.prepared.classes[0].constructor_code
        ):
            fired = True
            frame.f_locals["value"] = 8
        elif mode == "record-field-hook" and frame.f_code is kit.prepared.classes[0].call_code:
            fired = True
            object.__setattr__(value, "examples", new_examples)
        elif mode == "deferred-policy-hook" and frame.f_code is kit.prepared.source.module_code:
            fired = True
            object.__setattr__(kit.policy, "record_descriptor", MODULE_DICT)

    original = sys.getprofile()
    sys.setprofile(prior_hook)
    try:
        guard = kit.profile(directory / BREACH, kind=RecordingProfile)
        guard.kit = kit
        guard.record_path = directory / "mixed-counts-sidecar.json"
        with guard.measure():
            if deferred:
                kit.load()
                guard.admit_source(kit.loaded)
            kit.factory(value)
        (directory / "continued").write_text("unexpected", encoding="ascii")
    except Exception:
        (directory / "caught").write_text("unexpected", encoding="ascii")
        raise
    finally:
        sys.setprofile(original)


@pytest.mark.parametrize("mode", MODES, ids=MODES)
def test_mixed_exits_before_foreign_dispatch(tmp_path, mode):
    prior = (sys.getprofile(), threading.getprofile())
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parent)
    program = "from pathlib import Path; from test_provider_mixed_construction import run_mixed_negative; import sys; run_mixed_negative(sys.argv[1], Path(sys.argv[2]))"
    child = subprocess.run(
        [sys.executable, "-B", "-c", program, mode, str(tmp_path)],
        env=environment,
        capture_output=True,
        timeout=3,
        check=False,
    )
    assert child.returncode == 73
    assert len(child.stdout) <= 4096 and len(child.stderr) <= 4096
    assert not child.stdout and not child.stderr
    for name in ("offending_body", "caught", "continued"):
        assert not (tmp_path / name).exists()
    receipt_raw = (tmp_path / BREACH).read_bytes()
    snapshot_raw = (tmp_path / "mixed-counts-sidecar.json").read_bytes()
    assert len(receipt_raw) <= 1024 and len(snapshot_raw) <= 4096
    receipt, snapshot = json.loads(receipt_raw), json.loads(snapshot_raw)
    assert receipt == dict(reason="measurement_failure", counter="profile", attempt=0, limit=0)
    assert snapshot["offending_bodies"] == 0
    assert (sys.getprofile(), threading.getprofile()) == prior
    record = dict(
        mode=mode,
        exit=child.returncode,
        registered_attempts=sum(snapshot["counts"].values()),
        registered_bodies=len(snapshot["body_labels"]),
        stdout_bytes=len(child.stdout),
        stderr_bytes=len(child.stderr),
        receipt_bytes=len(receipt_raw),
        snapshot=snapshot,
        receipt=receipt,
        caught=0,
        continued=0,
        parent_restored=True,
        workers=0,
    )
    (tmp_path / "mixed-child.json").write_text(json.dumps(record), encoding="utf8")
