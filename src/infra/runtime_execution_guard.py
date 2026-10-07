"""Global Python mutation denial with persistent poison, not runtime admission.

Prepare the complete actual GC function/code catalog before the hold. During an
explicit lifetime, observe every delivered CALL/PY_START/INSTRUCTION event and deny
creation, binding writes, unsupported native calls and instrumentation replacement.
Callbacks retain primitive identities only. Native/internal execution, source/build
correspondence and full observer-lifetime integration still require separate proof.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import dis
import gc
import sys
import threading
import types
from typing import Any, Literal, NoReturn

_WRITE_OPS = frozenset(
    {
        "MAKE_FUNCTION",
        "SET_FUNCTION_ATTRIBUTE",
        "STORE_ATTR",
        "DELETE_ATTR",
        "STORE_GLOBAL",
        "DELETE_GLOBAL",
        "STORE_NAME",
        "DELETE_NAME",
        "STORE_DEREF",
        "DELETE_DEREF",
        "STORE_SUBSCR",
        "DELETE_SUBSCR",
    }
)
_AUDIT_DENIALS = frozenset(
    {
        "import",
        "exec",
        "compile",
        "code.__new__",
        "function.__new__",
        "ctypes.dlopen",
        "ctypes.dlsym",
        "ctypes.call_function",
        "sys.settrace",
        "sys.setprofile",
        "sys.monitoring.register_callback",
    }
)


@dataclass(frozen=True)
class RuntimeExecutionCatalog:
    """Whole actual prepared identities; this is not approved-source provenance."""

    functions_by_id: Mapping[int, types.FunctionType]
    codes_by_id: Mapping[int, types.CodeType]
    instructions_by_id: Mapping[int, Mapping[int, str]]


def capture_runtime_execution_catalog() -> RuntimeExecutionCatalog:
    functions = tuple(value for value in gc.get_objects() if type(value) is types.FunctionType)
    codes: dict[int, types.CodeType] = {}
    pending = [function.__code__ for function in functions]
    while pending:
        code = pending.pop()
        if id(code) in codes:
            continue
        codes[id(code)] = code
        pending.extend(value for value in code.co_consts if type(value) is types.CodeType)
    instructions = {
        identity: types.MappingProxyType(
            {row.offset: row.opname for row in dis.get_instructions(code)}
        )
        for identity, code in codes.items()
    }
    return RuntimeExecutionCatalog(
        types.MappingProxyType({id(function): function for function in functions}),
        types.MappingProxyType(codes),
        types.MappingProxyType(instructions),
    )


def _exit_offsets(code: types.CodeType) -> frozenset[int]:
    rows = tuple(dis.get_instructions(code))
    by_offset = {row.offset: index for index, row in enumerate(rows)}
    with_regions = []
    for entry in dis.Bytecode(code).exception_entries:  # type: ignore[attr-defined]
        index = by_offset[entry.target]
        if rows[index].opname == "PUSH_EXC_INFO" and rows[index + 1].opname == "WITH_EXCEPT_START":
            with_regions.append((entry.start, entry.end, rows[index + 1].offset))
    # Actual3.14 normal exit is CALL3. The exception table distinguishes its
    # position outside the with body from an early manual CALL3 inside it.
    return frozenset(
        row.offset
        for index, row in enumerate(rows)
        if (
            row.opname == "WITH_EXCEPT_START"
            and any(row.offset == target for _, _, target in with_regions)
        )
        or (
            row.opname == "CALL"
            and row.arg == 3
            and not any(start <= row.offset < end for start, end, _ in with_regions)
            and index >= 3
            and all(
                item.opname == "LOAD_CONST" and item.argval is None
                for item in rows[index - 3 : index]
            )
        )
    )


class RuntimeExecutionGuard:
    """Single-use global enforcement with no reset and no admission authority."""

    def __init__(
        self, catalog: RuntimeExecutionCatalog, owner_code: types.CodeType, *, tool_id: int = 3
    ):
        if type(catalog) is not RuntimeExecutionCatalog or type(owner_code) is not types.CodeType:
            raise ValueError("runtime guard requires its actual catalog and owning code")
        if type(tool_id) is not int or not 0 <= tool_id < 6:
            raise ValueError("runtime guard requires an exact tool ID in0..5")
        self.catalog, self.owner_code, self.tool_id = catalog, owner_code, tool_id
        self.events: list[list[Any]] = []
        self.denials: list[dict[str, Any]] = []
        self.cleanup: list[dict[str, Any]] = []
        self.poison: str | None = None
        self.active = False
        self.used = False
        self._monitor: Any = getattr(sys, "monitoring", None)
        self._name = "p67-runtime-mutation-guard"
        self._exit_offsets = _exit_offsets(owner_code)
        self._callbacks = (self._on_call, self._on_start, self._on_instruction)
        self._audit_callback = self._audit
        self._support_codes = frozenset(
            value.__code__
            for value in type(self).__dict__.values()
            if type(value) is types.FunctionType
        )
        self._mask = 0
        self._audit_installed = False

    def install_audit(self) -> None:
        if self.active or self.used or self._audit_installed:
            raise RuntimeError("runtime guard audit must be installed once before use")
        sys.addaudithook(self._audit_callback)
        self._audit_installed = True

    def tool_state(self) -> list[dict[str, Any]]:
        return [
            {
                "tool_id": tool,
                "name": self._monitor.get_tool(tool),
                "global_events": self._monitor.get_events(tool),
            }
            for tool in range(6)
        ]

    def _preflight(self) -> None:
        if (
            sys.implementation.name != "cpython"
            or sys.version_info[:2] != (3, 14)
            or self._monitor is None
        ):
            raise RuntimeError("runtime guard requires measured CPython3.14 monitoring")
        if not hasattr(self._monitor, "clear_tool_id"):
            raise RuntimeError("runtime guard cannot prove callback cleanup")
        if sys.gettrace() is not None or sys.getprofile() is not None:
            raise RuntimeError("runtime guard rejects preexisting tracing/profile")
        if threading.active_count() != 1:
            raise RuntimeError("runtime guard rejects other live Python threads")
        if any(row["name"] is not None or row["global_events"] for row in self.tool_state()):
            raise RuntimeError("runtime guard rejects foreign monitoring tools")
        current = {id(value) for value in gc.get_objects() if type(value) is types.FunctionType}
        if current != set(self.catalog.functions_by_id):
            raise RuntimeError("runtime guard prepared whole GC function membership drifted")

    def __enter__(self) -> RuntimeExecutionGuard:
        if self.used or self.poison is not None or not self._audit_installed:
            raise RuntimeError("runtime guard is single-use and cannot reset a denial")
        if sys._getframe(1).f_code is not self.owner_code or not self._exit_offsets:
            raise RuntimeError("runtime guard requires the prepared owning with boundary")
        self._preflight()
        events = self._monitor.events
        self._mask = events.CALL | events.PY_START | events.INSTRUCTION
        self._monitor.use_tool_id(self.tool_id, self._name)
        try:
            for event, callback in zip(
                (events.CALL, events.PY_START, events.INSTRUCTION), self._callbacks
            ):
                if self._monitor.register_callback(self.tool_id, event, callback) is not None:
                    raise RuntimeError("runtime guard encountered a retained foreign callback")
            self.used, self.active = True, True
            self._monitor.set_events(self.tool_id, self._mask)
        except BaseException:
            self.active = False
            self._monitor.clear_tool_id(self.tool_id)
            self._monitor.free_tool_id(self.tool_id)
            raise
        return self

    def __exit__(self, error_type, error, traceback) -> Literal[False]:
        frame = sys._getframe(1)
        if frame.f_code is not self.owner_code or frame.f_lasti not in self._exit_offsets:
            raise RuntimeError("runtime guard release is outside its owning with boundary")
        self.active = False
        self._monitor.set_events(self.tool_id, 0)
        self._monitor.clear_tool_id(self.tool_id)
        self._monitor.free_tool_id(self.tool_id)
        self._monitor.use_tool_id(self.tool_id, self._name + "-cleanup-check")
        try:
            for event in (
                self._monitor.events.CALL,
                self._monitor.events.PY_START,
                self._monitor.events.INSTRUCTION,
            ):
                previous = self._monitor.register_callback(self.tool_id, event, None)
                self.cleanup.append(
                    {
                        "tool_id": self.tool_id,
                        "event": event,
                        "prior_callback_was_none": previous is None,
                    }
                )
                if previous is not None:
                    raise RuntimeError("runtime guard callback survived actual release")
        finally:
            self._monitor.free_tool_id(self.tool_id)
        return False

    def _deny(self, reason: str, code=None, offset=None, operation=None) -> NoReturn:
        if self.poison is None:
            self.poison = reason
        self.denials.append(
            {
                "reason": reason,
                "first_poison": self.poison,
                "code_id": id(code) if code is not None else None,
                "offset": offset,
                "callable_id": id(operation) if operation is not None else None,
                "actual_tool_state_before_raise": self.tool_state(),
            }
        )
        raise RuntimeError("runtime mutation guard denied before execution: " + reason)

    def _check_state(self) -> None:
        for row in self.tool_state():
            expected = (self._name, self._mask) if row["tool_id"] == self.tool_id else (None, 0)
            if (row["name"], row["global_events"]) != expected:
                self._deny("monitoring coverage/tool drift")
        if sys.gettrace() is not None or sys.getprofile() is not None:
            self._deny("tracing/profile coverage drift")

    def _on_call(self, code, offset, operation, argument) -> None:
        if not self.active:
            return
        self.events.append(
            [
                "CALL",
                id(code),
                offset,
                id(operation),
                type(operation).__name__,
                id(argument),
                type(argument).__name__,
            ]
        )
        if code in self._support_codes:
            return
        self._check_state()
        function = operation.__func__ if type(operation) is types.MethodType else operation
        if type(function) is types.FunctionType and function.__code__ in self._support_codes:
            if (
                function is RuntimeExecutionGuard.__exit__
                and (operation.__self__ if type(operation) is types.MethodType else argument)
                is self
                and code is self.owner_code
                and offset in self._exit_offsets
            ):
                return
            self._deny("untrusted guard method/release call", code, offset, operation)
        if self.poison is not None:
            self._deny("prior denial prevents subsequent call", code, offset, operation)
        if type(function) is not types.FunctionType:
            self._deny("unsupported native/opaque callable", code, offset, operation)
        if self.catalog.functions_by_id.get(id(function)) is not function:
            self._deny("unknown actual Python callable", code, offset, operation)
        if self.catalog.codes_by_id.get(id(function.__code__)) is not function.__code__:
            self._deny("unknown actual Python code", code, offset, operation)

    def _on_start(self, code, offset) -> None:
        if not self.active:
            return
        self.events.append(["PY_START", id(code), offset])
        if code in self._support_codes:
            return
        self._check_state()
        if self.poison is not None:
            self._deny("prior denial prevents subsequent body", code, offset)
        if type(code) is not types.CodeType or self.catalog.codes_by_id.get(id(code)) is not code:
            self._deny("unknown actual entered code", code, offset)

    def _on_instruction(self, code, offset) -> None:
        if not self.active:
            return
        self.events.append(["INSTRUCTION", id(code), offset])
        if code in self._support_codes:
            return
        self._check_state()
        if type(code) is not types.CodeType or self.catalog.codes_by_id.get(id(code)) is not code:
            self._deny("unknown actual instruction code", code, offset)
        operation = self.catalog.instructions_by_id[id(code)].get(offset)
        if operation is None:
            self._deny("unobserved instruction offset", code, offset)
        if operation in _WRITE_OPS:
            self._deny("mutation instruction " + operation, code, offset)

    def _audit(self, event: str, arguments: tuple[Any, ...]) -> None:
        if not self.active:
            return
        if event in _AUDIT_DENIALS or event == "object.__setattr__":
            self._deny("mutation audit " + event)

    def require_complete_admission(self) -> None:
        # Why this: event visibility and a whole GC catalog do not prove implicit
        # native execution, private storage, approved source or full lifetime use.
        raise RuntimeError(
            "complete runtime admission unavailable: native/internal/source/build/private/membership/lifetime correspondence remains unproved"
        )
