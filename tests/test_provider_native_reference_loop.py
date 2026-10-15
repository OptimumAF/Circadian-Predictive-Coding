"""Public native-reference controls; no provider or callback body dispatch."""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import threading
from types import CodeType, FunctionType, GetSetDescriptorType, ModuleType
from typing import Any, cast

import pytest

from provider_code_path import pin_provider_code_path, require_provider_code_path
from provider_value_witness import ProviderValueBinding, ProviderValueWitnesses


ERROR = "invalid retained provider value witness"
TYPE_DICT = type.__dict__["__dict__"]
MODULE_DICT = cast(GetSetDescriptorType, TYPE_DICT.__get__(ModuleType, type)["__dict__"])
TINY_SOURCE = """def root():
    events.append(root_marker)
    def callback():
        events.append('callback')
        return 7
    return callback
"""


class Fixture:
    """Count explicitly retained native setup, separately from helper dispatch."""

    def __init__(self):
        self.events: list[str] = []
        self.observed_record_calls = 0
        self.explicit_record_calls = 0
        self.record_class_calls = 0
        self.code_fixtures = 3
        self.function_fixtures = 2
        self.tuple_fixtures = 1
        self.hostile_fixtures = 0
        self.int_fixtures = 0
        self.class_fixtures = 1
        owner = self

        @dataclass(frozen=True)
        class Record:
            owner.record_class_calls += 1
            snapshots: int
            examples: int
            payload_bytes: int

            def __post_init__(self):
                owner.observed_record_calls += 1

        self.record_type = Record
        self.module = ModuleType("native_loop_record")
        namespace = cast(dict[str, Any], MODULE_DICT.__get__(self.module, ModuleType))
        namespace["Record"] = Record
        self.descriptor = cast(GetSetDescriptorType, TYPE_DICT.__get__(Record, type)["__dict__"])
        module_code = compile(TINY_SOURCE, "native_loop_fixture.py", "exec", dont_inherit=True)
        root_code = cast(CodeType, module_code.co_consts[0])
        assert type(root_code) is CodeType
        assert root_code.co_name == "root"
        assert len(root_code.co_consts) <= 512
        callback_code: CodeType | None = None
        callback_index = -1
        for index, constant in enumerate(root_code.co_consts):
            if type(constant) is CodeType and constant.co_name == "callback":
                assert callback_code is None
                callback_code = constant
                callback_index = index
        assert callback_code is not None and 0 <= callback_index <= 511
        assert type(callback_code) is CodeType and callback_code.co_name == "callback"
        self.globals = {"events": self.events, "root_marker": "root"}
        self.root = FunctionType(root_code, self.globals)
        self.callback = FunctionType(callback_code, self.globals)
        self.path = pin_provider_code_path(self.root, root_code, (callback_index,))
        assert self.path.root is self.root and self.path.root_code is root_code
        assert self.path.indexes == (callback_index,) and self.path.token is callback_code
        require_provider_code_path(self.path)
        self.witness = ProviderValueWitnesses(
            self.module, "Record", Record, self.descriptor, self.path, max_values=1
        )
        self.before = [self.witness.attempts, self.witness.binding_count]
        assert self.before == [0, 0]
        self.requires: list[dict[str, Any]] = []
        self.prior = (sys.getprofile(), threading.getprofile())

    def pin(self, value: object) -> ProviderValueBinding:
        row = self.witness.pin(value)
        assert type(row) is ProviderValueBinding and row.value is value
        assert (self.witness.attempts, self.witness.binding_count) == (1, 1)
        return row

    def require(
        self,
        row: ProviderValueBinding,
        candidate: object,
        mutation: list[str],
        *,
        refused: bool = False,
    ):
        before = [self.witness.attempts, self.witness.binding_count]
        result: dict[str, Any]
        if refused:
            with pytest.raises(ValueError, match="^" + ERROR + "$") as fault:
                self.witness.require(row, candidate)
            assert type(fault.value) is ValueError and str(fault.value) == ERROR
            result = dict(
                mutation=mutation,
                result="refused",
                error_type="ValueError",
                error_message=str(fault.value),
                charges_before=before,
                charges_after=[self.witness.attempts, self.witness.binding_count],
            )
        else:
            self.witness.require(row, candidate)
            result = dict(
                mutation=mutation,
                result="accepted",
                error_type=None,
                error_message=None,
                charges_before=before,
                charges_after=[self.witness.attempts, self.witness.binding_count],
            )
        assert result["charges_after"] == before == [1, 1]
        self.requires.append(result)

    def finish(
        self,
        path: Path,
        name: str,
        binding: ProviderValueBinding,
        candidate: object,
        field_names: list[str],
    ):
        assert not self.events
        assert self.observed_record_calls == self.explicit_record_calls
        assert binding.value is candidate
        assert sys.getprofile() is self.prior[0]
        assert threading.getprofile() is self.prior[1]
        record = dict(
            test_name=name,
            declarations=1,
            declaration_accepts=1,
            declaration_refusals=0,
            pin_attempts=self.witness.attempts,
            pin_accepts=self.witness.binding_count,
            pin_refusals=self.witness.attempts - self.witness.binding_count,
            successful_bindings=self.witness.binding_count,
            mode=binding.mode,
            field_names=field_names,
            checks=self.requires,
            require_attempts=len(self.requires),
            require_accepts=sum(row["result"] == "accepted" for row in self.requires),
            require_refusals=sum(row["result"] == "refused" for row in self.requires),
            charges_before_pin=self.before,
            charges_after_checks=[self.witness.attempts, self.witness.binding_count],
            setup=dict(
                source_compilations=1,
                module_executions=0,
                code_fixtures=self.code_fixtures,
                function_fixtures=self.function_fixtures,
                path_rows=1,
                binding_rows=self.witness.binding_count,
                tuple_fixtures=self.tuple_fixtures,
                module_fixtures=1,
                namespace_fixtures=2 + self.explicit_record_calls,
                class_fixtures=self.class_fixtures,
                hostile_fixtures=self.hostile_fixtures,
                int_fixtures=self.int_fixtures,
                record_class_calls=self.record_class_calls,
                record_constructor_calls=self.explicit_record_calls,
            ),
            helper_record_constructor_calls=self.observed_record_calls - self.explicit_record_calls,
            callback_calls=0,
            root_calls=0,
            provider_calls=0,
            factory_calls=0,
            delegate_calls=0,
            registered_attempts=0,
            hostile_events=list(self.events),
            source_bodies=[],
            path_indexes=list(self.path.indexes),
            restored=True,
            restored_hooks=True,
            workers=0,
        )
        (path / "loop-work.json").write_text(json.dumps(record), encoding="utf8")


def test_native_reference_loop_preserves_issued_field_order(tmp_path):
    fixture = Fixture()
    value = 7
    binding = fixture.pin(value)
    fixture.require(binding, value, [])
    events = fixture.events

    class Hostile:
        def __getattribute__(self, name):
            events.append("getattribute")
            raise AssertionError(name)

        def __eq__(self, other):
            events.append("eq")
            raise AssertionError

        def __hash__(self):
            events.append("hash")
            raise AssertionError

        def __repr__(self):
            events.append("repr")
            raise AssertionError

    mode: Any = object.__new__(Hostile)
    foreign_value: Any = object.__new__(Hostile)
    fixture.hostile_fixtures = 2
    fixture.class_fixtures += 1
    original_mode, original_value = binding.mode, binding.value
    try:
        object.__setattr__(binding, "mode", mode)
        object.__setattr__(binding, "value", foreign_value)
        fixture.require(binding, value, ["mode", "value"], refused=True)
        object.__setattr__(binding, "mode", original_mode)
        fixture.require(binding, value, ["value"], refused=True)
    finally:
        object.__setattr__(binding, "mode", original_mode)
        object.__setattr__(binding, "value", original_value)
    fixture.require(binding, value, [])
    fixture.finish(
        tmp_path,
        "test_native_reference_loop_preserves_issued_field_order",
        binding,
        value,
        ["mode", "value"],
    )


def test_native_reference_loop_preserves_record_field_identity(tmp_path):
    fixture = Fixture()
    originals = (int("1000"), int("2000"), int("3000"))
    replacements = (int("1000"), int("2000"), int("3000"))
    fixture.int_fixtures = 6
    fixture.explicit_record_calls += 1
    value = fixture.record_type(*originals)
    storage = fixture.descriptor.__get__(value, fixture.record_type)
    binding = fixture.pin(value)
    fixture.require(binding, value, [])
    names = ["snapshots", "examples", "payload_bytes"]
    for name, original, replacement in zip(names, originals, replacements):
        assert type(original) is int and type(replacement) is int
        assert original >= 1000 and replacement == original and replacement is not original
        try:
            object.__setattr__(value, name, replacement)
            assert fixture.descriptor.__get__(value, fixture.record_type) is storage
            assert (
                value.snapshots > 0
                and value.examples >= value.snapshots
                and value.payload_bytes > 0
            )
            fixture.require(binding, value, [name], refused=True)
        finally:
            object.__setattr__(value, name, original)
    fixture.require(binding, value, [])
    assert all(storage[name] is original for name, original in zip(names, originals))
    fixture.finish(
        tmp_path,
        "test_native_reference_loop_preserves_record_field_identity",
        binding,
        value,
        names,
    )


def test_native_reference_loop_preserves_path_field_identity(tmp_path):
    fixture = Fixture()
    value = 7
    binding = fixture.pin(value)
    fixture.require(binding, value, [])
    original_root, original_indexes, original_token = (
        fixture.path.root,
        fixture.path.indexes,
        fixture.path.token,
    )
    root_clone = FunctionType(original_root.__code__, original_root.__globals__)
    indexes_clone = tuple(list(original_indexes))
    token_clone = original_token.replace()
    fixture.function_fixtures += 1
    fixture.tuple_fixtures += 1
    fixture.code_fixtures += 1
    assert root_clone is not original_root and root_clone.__code__ is original_root.__code__
    assert root_clone.__globals__ is original_root.__globals__
    assert indexes_clone == original_indexes and indexes_clone is not original_indexes
    assert len(original_indexes) == 1 and type(original_token) is CodeType
    assert token_clone == original_token and token_clone is not original_token
    for name, original, replacement in (
        ("root", original_root, root_clone),
        ("indexes", original_indexes, indexes_clone),
        ("token", original_token, token_clone),
    ):
        try:
            object.__setattr__(fixture.path, name, replacement)
            fixture.require(binding, value, [name], refused=True)
        finally:
            object.__setattr__(fixture.path, name, original)
    fixture.require(binding, value, [])
    assert (
        fixture.path.root is original_root
        and fixture.path.indexes is original_indexes
        and fixture.path.token is original_token
    )
    fixture.finish(
        tmp_path,
        "test_native_reference_loop_preserves_path_field_identity",
        binding,
        value,
        ["root", "root_code", "indexes", "token"],
    )
