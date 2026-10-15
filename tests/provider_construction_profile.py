"""Cooperative CPython3.14 constructor enrollment with unchanged entry limits.

Retained native role/caller/code witnesses produce ready counts or exit73.
Why: owners can be called before their factory returns; share the original guard.
Source/allocation birth, globals/defaults/closures, imports, overlap, suppressed
hook work and concurrent atomicity are not proved. No callback is dispatched.
"""

from dataclasses import dataclass, field
import opcode
from pathlib import Path
import sys
import threading
from types import (
    CodeType,
    FrameType,
    FunctionType,
    GetSetDescriptorType,
    MappingProxyType,
    MemberDescriptorType,
    ModuleType,
)
from typing import Callable

from provider_code_path import (
    ProviderCodePath,
    require_provider_callback_code,
    require_provider_code_path,
)
from provider_deferred_construction import DeferredAdmission, DeferredConstructionRole
from provider_source_classes import LoadedSourceClasses
from provider_entry_profile import ProviderEntryProfile
from provider_owner_namespace_profile import OwnerNamespaceBinding, require_current_namespaces
from provider_runtime_binding import RuntimeBinding, require_current_bindings
from provider_value_witness import ProviderValueBinding, ProviderValueWitnesses

_TYPE_DICT = type.__dict__["__dict__"]
_GETSET_GET = _TYPE_DICT.__get__(GetSetDescriptorType, type)["__get__"]
_MEMBER_GET = _TYPE_DICT.__get__(MemberDescriptorType, type)["__get__"]
_SET_PROFILE = sys.setprofile
_GET_PROFILE = sys.getprofile
_MISSING = object()
_ERROR = "invalid witness"
_NORMAL_RETURNS = tuple(
    value
    for name in ("RETURN_VALUE", "RETURN_CONST")
    if type(value := opcode.opmap.get(name)) is int
)


def _read(reader, instance, kind):
    if type(reader) is GetSetDescriptorType:
        return _GETSET_GET(reader, instance, kind)
    if type(reader) is MemberDescriptorType:
        return _MEMBER_GET(reader, instance, kind)
    raise TypeError(_ERROR)


def _readers(kind, names):
    namespace = _GETSET_GET(_TYPE_DICT, kind, type)
    return tuple(namespace[name] for name in names)


_TYPE_BASES = _readers(type, ("__bases__",))[0]
_FUNCTION_CODE, _FUNCTION_GLOBALS = _readers(FunctionType, ("__code__", "__globals__"))
_CODE_NAMES, _CODE_BYTES, _CODE_FLAGS, _CODE_ARGS = _readers(
    CodeType, ("co_varnames", "co_code", "co_flags", "co_argcount")
)
_FRAME_CODE, _FRAME_GLOBALS, _FRAME_PARENT, _FRAME_LOCALS, _FRAME_LASTI = _readers(
    FrameType, ("f_code", "f_globals", "f_back", "f_locals", "f_lasti")
)
_DESCRIPTOR_OWNER, _DESCRIPTOR_NAME = _readers(GetSetDescriptorType, ("__objclass__", "__name__"))
_PATH_FIELDS = _readers(ProviderCodePath, ("root", "root_code", "indexes", "token"))


@dataclass(frozen=True, slots=True)
class OwnerConstructionRole:
    owner_type: type
    descriptor: GetSetDescriptorType
    constructor: FunctionType
    constructor_code: CodeType
    caller: ProviderCodePath
    raw_call: FunctionType
    call_code: CodeType
    max_instances: int
    callback_slot: str | None = None
    callback_argument: str | None = None
    callback_path: ProviderCodePath | None = None


_ROLE_FIELDS = _readers(
    OwnerConstructionRole,
    (
        "owner_type",
        "descriptor",
        "constructor",
        "constructor_code",
        "caller",
        "raw_call",
        "call_code",
        "max_instances",
        "callback_slot",
        "callback_argument",
        "callback_path",
    ),
)


def _role_values(role):
    if type(role) is not OwnerConstructionRole:
        raise TypeError(_ERROR)
    try:
        return tuple(_read(reader, role, OwnerConstructionRole) for reader in _ROLE_FIELDS)
    except AttributeError:
        raise ValueError(_ERROR) from None


def _path_values(path):
    require_provider_code_path(path)
    return tuple(_read(reader, path, ProviderCodePath) for reader in _PATH_FIELDS)


def _require_keys(namespace):
    if len(namespace) > 512:
        raise ValueError(_ERROR)
    for name in namespace:
        if type(name) is not str or len(name) > 128:
            raise ValueError(_ERROR)


def _class_dict(owner_type):
    if type(owner_type) is not type:
        raise TypeError(_ERROR)
    bases = _read(_TYPE_BASES, owner_type, type)
    if type(bases) is not tuple or len(bases) != 1 or bases[0] is not object:
        raise ValueError(_ERROR)
    namespace = _GETSET_GET(_TYPE_DICT, owner_type, type)
    if type(namespace) is not MappingProxyType:
        raise TypeError(_ERROR)
    _require_keys(namespace)
    return namespace


def _globals(function):
    if type(function) is not FunctionType:
        raise TypeError(_ERROR)
    namespace = _read(_FUNCTION_GLOBALS, function, FunctionType)
    if type(namespace) is not dict:
        raise TypeError(_ERROR)
    return namespace


def _check_function(function, code):
    if type(function) is not FunctionType or type(code) is not CodeType:
        raise TypeError(_ERROR)
    if _read(_FUNCTION_CODE, function, FunctionType) is not code:
        raise ValueError(_ERROR)
    names = _read(_CODE_NAMES, code, CodeType)
    if (
        type(names) is not tuple
        or len(names) > 512
        or any(type(name) is not str or len(name) > 128 for name in names)
    ):
        raise ValueError(_ERROR)
    raw = _read(_CODE_BYTES, code, CodeType)
    if type(raw) is not bytes or len(raw) > 65536:
        raise ValueError(_ERROR)
    if _read(_CODE_FLAGS, code, CodeType) & (0x20 | 0x80 | 0x200):
        raise ValueError(_ERROR)
    if _read(_CODE_ARGS, code, CodeType) < 1 or not names or names[0] != "self":
        raise ValueError(_ERROR)
    return _globals(function)


@dataclass(frozen=True, slots=True)
class ConstructionValuePolicy:
    role: OwnerConstructionRole | DeferredConstructionRole
    record_module: ModuleType
    record_attribute: str
    record_type: type
    record_descriptor: GetSetDescriptorType


_POLICY_FIELDS = _readers(
    ConstructionValuePolicy,
    ("role", "record_module", "record_attribute", "record_type", "record_descriptor"),
)
_DEFERRED_FIELDS = _readers(
    DeferredConstructionRole,
    (
        "source_classes",
        "prepared",
        "class_index",
        "caller_root",
        "caller_indexes",
        "max_instances",
        "callback_slot",
        "callback_argument",
        "callback_root",
        "callback_indexes",
    ),
)
_MODULE_DICT = _readers(ModuleType, ("__dict__",))[0]
_ATTACHMENT_NAMESPACE = _readers(OwnerNamespaceBinding, ("namespace",))[0]
_VALUE_FIELDS = _readers(ProviderValueBinding, ("mode", "value"))
_FORBIDDEN_RECORD_ATTRIBUTES = (
    "__call__",
    "__new__",
    "__getattribute__",
    "__getattr__",
    "__slots__",
    "snapshots",
    "examples",
    "payload_bytes",
)


def _policy_values(policy):
    if type(policy) is not ConstructionValuePolicy:
        raise TypeError(_ERROR)
    try:
        return tuple(_read(reader, policy, ConstructionValuePolicy) for reader in _POLICY_FIELDS)
    except AttributeError:
        raise ValueError(_ERROR) from None


def _record_metadata(values):
    _, module, attribute, kind, descriptor = values
    if (
        type(module) is not ModuleType
        or type(attribute) is not str
        or type(kind) is not type
        or type(descriptor) is not GetSetDescriptorType
    ):
        raise TypeError(_ERROR)
    if not 1 <= len(attribute) <= 128:
        raise ValueError(_ERROR)
    namespace = _read(_MODULE_DICT, module, ModuleType)
    if type(namespace) is not dict:
        raise TypeError(_ERROR)
    _require_keys(namespace)
    class_namespace = _class_dict(kind)
    bases = _read(_TYPE_BASES, kind, type)
    name = _read(_DESCRIPTOR_NAME, descriptor, GetSetDescriptorType)
    if type(name) is not str:
        raise TypeError(_ERROR)
    if (
        namespace.get(attribute, _MISSING) is not kind
        or _read(_DESCRIPTOR_OWNER, descriptor, GetSetDescriptorType) is not kind
        or name != "__dict__"
        or class_namespace.get("__dict__", _MISSING) is not descriptor
        or any(name in class_namespace for name in _FORBIDDEN_RECORD_ATTRIBUTES)
    ):
        raise ValueError(_ERROR)
    return namespace, class_namespace, bases


@dataclass(slots=True)
class _ValuePolicy:
    row: ConstructionValuePolicy
    values: tuple
    module_namespace: dict
    bases: tuple
    module_items: tuple[tuple[str, object], ...]
    class_items: tuple[tuple[str, object], ...]


def _prepare_policy(policy, roles, deferred_roles):
    values = _policy_values(policy)
    role = values[0]
    if not any(role is original for original in (*roles, *deferred_roles)):
        raise ValueError(_ERROR)
    if type(role) is OwnerConstructionRole:
        role_values = _role_values(role)
        complete = all(value is not None for value in role_values[8:11])
    elif type(role) is DeferredConstructionRole:
        try:
            role_values = tuple(
                _read(reader, role, DeferredConstructionRole) for reader in _DEFERRED_FIELDS
            )
        except AttributeError:
            raise ValueError(_ERROR) from None
        complete = all(value is not None for value in role_values[6:10])
    else:
        raise TypeError(_ERROR)
    if not complete:
        raise ValueError(_ERROR)
    namespace, class_namespace, bases = _record_metadata(values)
    return _ValuePolicy(
        policy,
        values,
        namespace,
        bases,
        tuple(namespace.items()),
        tuple(class_namespace.items()),
    )


def _require_policy(policy):
    values = _policy_values(policy.row)
    if any(value is not original for value, original in zip(values, policy.values)):
        raise ValueError(_ERROR)
    namespace, class_namespace, bases = _record_metadata(values)
    if namespace is not policy.module_namespace or bases is not policy.bases:
        raise ValueError(_ERROR)
    for current, saved in (
        (namespace, policy.module_items),
        (class_namespace, policy.class_items),
    ):
        if len(current) != len(saved) or any(
            current.get(name, _MISSING) is not original for name, original in saved
        ):
            raise ValueError(_ERROR)


def _attach_policy(state, policy):
    # Deferred callback authority is resolved only at runtime role creation.
    # Reprove pre-load metadata on both sides; never refresh record authority.
    _require_policy(policy)
    _, module, attribute, kind, descriptor = policy.values
    witnesses = ProviderValueWitnesses(
        module,
        attribute,
        kind,
        descriptor,
        state.values[10],
        max_values=state.values[7],
    )
    _require_policy(policy)
    witnesses.require_current()
    state.value_witnesses = witnesses


def _require_selected_value(state, enrollment, candidate):
    witnesses = state.value_witnesses
    binding = enrollment.value_binding
    if witnesses is None or binding is None:
        raise ValueError(_ERROR)
    witnesses.require(binding, candidate)
    mode, value = tuple(_read(reader, binding, ProviderValueBinding) for reader in _VALUE_FIELDS)
    if mode is not enrollment.mode or value is not enrollment.argument:
        raise ValueError(_ERROR)


@dataclass(slots=True)
class _Owner:
    owner: object
    frame: FrameType | None
    argument: object
    globals: dict | None
    attachment: OwnerNamespaceBinding
    binding: RuntimeBinding | None
    value_binding: ProviderValueBinding | None = None
    mode: str | None = None


@dataclass(slots=True)
class _State:
    role: OwnerConstructionRole
    values: tuple
    class_items: tuple[tuple[str, object], ...]
    init_globals: dict
    call_globals: dict
    caller: tuple
    caller_globals: dict
    delegate: tuple | None
    delegate_globals: dict | None
    ready: list[_Owner] = field(default_factory=list)
    pending: _Owner | None = None
    value_witnesses: ProviderValueWitnesses | None = None


def _prepare_role(role):
    values = _role_values(role)
    (
        owner_type,
        descriptor,
        constructor,
        constructor_code,
        caller,
        raw_call,
        call_code,
        cap,
        slot,
        argument,
        callback_path,
    ) = values
    if type(cap) is not int or not 1 <= cap <= 64:
        raise ValueError(_ERROR)
    namespace = _class_dict(owner_type)
    if type(descriptor) is not GetSetDescriptorType:
        raise TypeError(_ERROR)
    if (
        _read(_DESCRIPTOR_OWNER, descriptor, GetSetDescriptorType) is not owner_type
        or _read(_DESCRIPTOR_NAME, descriptor, GetSetDescriptorType) != "__dict__"
        or namespace.get("__dict__", _MISSING) is not descriptor
        or namespace.get("__init__", _MISSING) is not constructor
        or namespace.get("__call__", _MISSING) is not raw_call
    ):
        raise ValueError(_ERROR)
    if any(
        name in namespace
        for name in (
            "__new__",
            "__getattribute__",
            "__getattr__",
            "__setattr__",
            "__delattr__",
        )
    ):
        raise ValueError(_ERROR)
    init_globals = _check_function(constructor, constructor_code)
    call_globals = _check_function(raw_call, call_code)
    caller = _path_values(caller)
    caller_globals = _globals(caller[0])
    delegate = delegate_globals = None
    if slot is None and argument is None and callback_path is None:
        pass
    elif (
        type(slot) is str
        and 0 < len(slot) <= 128
        and type(argument) is str
        and 0 < len(argument) <= 128
        and type(callback_path) is ProviderCodePath
    ):
        if slot in namespace:
            raise ValueError(_ERROR)
        delegate = _path_values(callback_path)
        delegate_globals = _globals(delegate[0])
    else:
        raise ValueError(_ERROR)
    return _State(
        role,
        values,
        tuple(namespace.items()),
        init_globals,
        call_globals,
        caller,
        caller_globals,
        delegate,
        delegate_globals,
    )


def _require_role(state):
    values = _role_values(state.role)
    if any(current is not original for current, original in zip(values, state.values)):
        raise ValueError(_ERROR)
    (owner_type, _, constructor, constructor_code, caller, raw_call, call_code, _, _, _, path) = (
        values
    )
    namespace = _class_dict(owner_type)
    if len(namespace) != len(state.class_items) or any(
        namespace.get(name, _MISSING) is not value for name, value in state.class_items
    ):
        raise ValueError(_ERROR)
    if _check_function(constructor, constructor_code) is not state.init_globals:
        raise ValueError(_ERROR)
    if _check_function(raw_call, call_code) is not state.call_globals:
        raise ValueError(_ERROR)
    current_caller = _path_values(caller)
    if any(current is not original for current, original in zip(current_caller, state.caller)):
        raise ValueError(_ERROR)
    if _globals(current_caller[0]) is not state.caller_globals:
        raise ValueError(_ERROR)
    if state.delegate is not None:
        current_delegate = _path_values(path)
        if (
            any(
                current is not original
                for current, original in zip(current_delegate, state.delegate)
            )
            or _globals(current_delegate[0]) is not state.delegate_globals
        ):
            raise ValueError(_ERROR)


def _frame_read(frame, reader):
    if type(frame) is not FrameType:
        raise TypeError(_ERROR)
    return _read(reader, frame, FrameType)


def _frame_locals(frame):
    namespace = _frame_read(frame, _FRAME_LOCALS)
    _require_keys(namespace)
    return namespace


def _require_frame(frame, code, globals_dict):
    if (
        _frame_read(frame, _FRAME_CODE) is not code
        or _frame_read(frame, _FRAME_GLOBALS) is not globals_dict
    ):
        raise ValueError(_ERROR)


def _check_pending(state, enrollment):
    _require_frame(enrollment.frame, state.values[3], state.init_globals)
    namespace = _frame_locals(enrollment.frame)
    if namespace.get("self", _MISSING) is not enrollment.owner:
        raise ValueError(_ERROR)
    if state.delegate is not None:
        argument = namespace.get(state.values[9], _MISSING)
        if argument is not enrollment.argument:
            raise ValueError(_ERROR)
        if state.value_witnesses is not None:
            _require_selected_value(state, enrollment, argument)


def _check_owner(state, enrollment, ready):
    require_current_namespaces((enrollment.attachment,))
    mixed = state.value_witnesses is not None
    if mixed:
        _require_selected_value(state, enrollment, enrollment.argument)
        if ready:
            namespace = _read(_ATTACHMENT_NAMESPACE, enrollment.attachment, OwnerNamespaceBinding)
            _require_selected_value(
                state,
                enrollment,
                namespace.get(state.values[8], _MISSING),
            )
    if state.delegate is not None and (not mixed or enrollment.mode == "callback"):
        require_provider_callback_code(state.values[10], enrollment.argument)
        if (
            _globals(enrollment.argument) is not enrollment.globals
            or enrollment.globals is not state.delegate_globals
        ):
            raise ValueError(_ERROR)
        if ready:
            require_current_bindings((enrollment.binding,))
    if not ready:
        _check_pending(state, enrollment)


class ProviderConstructionProfile(ProviderEntryProfile):
    value_policies: tuple[ConstructionValuePolicy, ...]
    _original_value_policies: tuple[ConstructionValuePolicy, ...]
    _value_policy_states: tuple[_ValuePolicy, ...]
    _declared_deferred_roles: tuple[DeferredConstructionRole, ...]
    _roles: tuple[OwnerConstructionRole, ...]
    _states: tuple[_State, ...]
    _deferred: DeferredAdmission
    _immediate_roles: tuple[OwnerConstructionRole, ...]
    _deferred_states: list[_State | None]
    _original_runtime: tuple[tuple[tuple[object, str, int, str], ...], dict[str, int], object]

    def __init__(
        self,
        entries: tuple[tuple[Callable | CodeType, str, int], ...],
        receipt: Path,
        roles: tuple[OwnerConstructionRole, ...],
        *,
        deferred_roles: tuple[DeferredConstructionRole, ...] = (),
        value_policies: tuple[ConstructionValuePolicy, ...] = (),
    ) -> None:
        if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 14):
            raise ValueError(_ERROR)
        if not _NORMAL_RETURNS:
            raise ValueError(_ERROR)
        if type(roles) is not tuple or len(roles) > 64:
            raise ValueError(_ERROR)
        self._deferred = DeferredAdmission(deferred_roles)
        if not 0 < len(roles) + len(self._deferred.plans) <= 64:
            raise ValueError(_ERROR)
        self._immediate_roles = roles
        self._deferred_states: list[_State | None] = [None] * len(self._deferred.plans)
        self._roles = roles
        self._states = tuple(_prepare_role(role) for role in roles)
        if type(value_policies) is not tuple or len(value_policies) > 64:
            raise ValueError(_ERROR)
        self.value_policies = self._original_value_policies = value_policies
        self._declared_deferred_roles = deferred_roles
        self._value_policy_states = tuple(
            _prepare_policy(policy, roles, deferred_roles) for policy in value_policies
        )
        for index, policy in enumerate(self._value_policy_states):
            if any(
                policy.values[0] is prior.values[0] for prior in self._value_policy_states[:index]
            ):
                raise ValueError(_ERROR)
        for state in self._states:
            self._attach_value_policy(state, state.role)
        if sum(state.values[7] for state in self._states) + sum(self._deferred.capacities) > 64:
            raise ValueError(_ERROR)
        role_tokens = (
            tuple(token for state in self._states for token in (state.values[3], state.values[6]))
            + self._deferred.role_tokens
        )
        if any(
            token is prior
            for index, token in enumerate(role_tokens)
            for prior in role_tokens[:index]
        ):
            raise ValueError(_ERROR)
        caller_tokens = (
            tuple(state.caller[3] for state in self._states) + self._deferred.caller_tokens
        )
        callback_tokens = (
            tuple(state.delegate[3] for state in self._states if state.delegate is not None)
            + self._deferred.callback_tokens
        )
        if any(
            token is other
            for index, token in enumerate(callback_tokens)
            for other in role_tokens + caller_tokens + callback_tokens[:index]
        ):
            raise ValueError(_ERROR)
        super().__init__(entries, receipt)
        if any(
            not any(token is original for original, _, _, _ in self._table)
            for token in role_tokens
            + caller_tokens
            + callback_tokens
            + self._deferred.required_tokens
        ):
            raise ValueError(_ERROR)
        self._original_runtime = (self._table, self._counts, self._lock)
        self.require_current_enrollments()

    def _attach_value_policy(self, state, original_role):
        for policy in self._value_policy_states:
            if policy.values[0] is original_role:
                _attach_policy(state, policy)
                return

    def admit_source(self, loaded: LoadedSourceClasses) -> None:
        if (
            not self._active
            or threading.get_ident() != self._main
            or _GET_PROFILE() is not self._callback
            or self._pending_state() is not None
            or self._deferred.pending_module() is not None
        ):
            raise ValueError(_ERROR)
        self.require_current_enrollments()
        batch = self._deferred.prepare_admission(loaded)
        additions = tuple(_prepare_role(OwnerConstructionRole(*args)) for args in batch.arguments)
        slots = list(self._deferred_states)
        for index, state in zip(batch.indexes, additions):
            self._attach_value_policy(state, self._declared_deferred_roles[index])
            if slots[index] is not None:
                raise ValueError(_ERROR)
            slots[index] = state
        deferred_states = tuple(state for state in slots if state is not None)
        states = self._states[: len(self._immediate_roles)] + deferred_states
        roles = self._immediate_roles + tuple(state.role for state in deferred_states)
        self.require_current_enrollments()
        batch.source.current.require()
        batch.source.current.method(loaded)
        # All roles exist before one module permission changes; no allowance renews.
        self._deferred_states, self._states, self._roles = slots, states, roles
        self._deferred.commit(batch)
        self.require_current_enrollments()

    @property
    def enrollment_counts(self) -> tuple[int, ...]:
        return tuple(
            len(state.ready) for state in self._states[: len(self._immediate_roles)]
        ) + tuple(0 if state is None else len(state.ready) for state in self._deferred_states)

    def require_current_enrollments(self) -> None:
        if (
            type(self.value_policies) is not tuple
            or self.value_policies is not self._original_value_policies
        ):
            raise ValueError(_ERROR)
        for row, policy in zip(self.value_policies, self._value_policy_states):
            if row is not policy.row:
                raise ValueError(_ERROR)
            _require_policy(policy)
        for state in self._states:
            if state.value_witnesses is not None:
                state.value_witnesses.require_current()
        if any(
            current is not original
            for current, original in zip(
                (self._table, self._counts, self._lock), self._original_runtime
            )
        ):
            raise ValueError(_ERROR)
        self._deferred.require_current()
        if len(self._roles) != len(self._states) or any(
            role is not state.role for role, state in zip(self._roles, self._states)
        ):
            raise ValueError(_ERROR)
        pending_count = 0
        owners: list[object] = []
        for state in self._states:
            _require_role(state)
            if len(state.ready) + (state.pending is not None) > state.values[7]:
                raise ValueError(_ERROR)
            rows = (*state.ready, state.pending) if state.pending is not None else state.ready
            pending_count += state.pending is not None
            for enrollment in rows:
                if any(enrollment.owner is owner for owner in owners):
                    raise ValueError("duplicate enrollment")
                owners.append(enrollment.owner)
                ready = enrollment is not state.pending
                if ready and enrollment.frame is not None:
                    raise ValueError("ready frame retained")
                if not ready and type(enrollment.frame) is not FrameType:
                    raise TypeError("pending frame is not native")
                _check_owner(state, enrollment, ready)
        if pending_count > 1 or len(owners) > 64:
            raise ValueError(_ERROR)

    def _pending_state(self):
        return next((state for state in self._states if state.pending is not None), None)

    def _check_caller(self, state, frame):
        parent = _frame_read(frame, _FRAME_PARENT)
        _require_frame(parent, state.caller[3], state.caller_globals)

    def _admit(self, state, frame):
        if (
            self._pending_state() is not None
            or self._deferred.pending_module() is not None
            or threading.get_ident() != self._main
        ):
            raise ValueError(_ERROR)
        _require_frame(frame, state.values[3], state.init_globals)
        self._check_caller(state, frame)
        namespace = _frame_locals(frame)
        owner = namespace.get("self", _MISSING)
        if type(owner) is not state.values[0]:
            raise TypeError(_ERROR)
        if len(state.ready) >= state.values[7] or any(
            owner is enrollment.owner
            for candidate in self._states
            for enrollment in candidate.ready
        ):
            raise ValueError(_ERROR)
        storage = _GETSET_GET(state.values[1], owner, state.values[0])
        witness = OwnerNamespaceBinding(owner, state.values[0], state.values[1], storage)
        require_current_namespaces((witness,))
        if storage:
            raise ValueError(_ERROR)
        argument = globals_dict = binding = value_binding = mode = None
        if state.delegate is not None:
            argument = namespace.get(state.values[9], _MISSING)
            if state.value_witnesses is not None:
                value_binding = state.value_witnesses.pin(argument)
                mode = _read(_VALUE_FIELDS[0], value_binding, ProviderValueBinding)
            if state.value_witnesses is None or mode == "callback":
                require_provider_callback_code(state.values[10], argument)
                globals_dict = _globals(argument)
                if globals_dict is not state.delegate_globals:
                    raise ValueError(_ERROR)
                binding = RuntimeBinding(storage, state.values[8], argument, state.delegate[3])
        state.pending = _Owner(
            owner,
            frame,
            argument,
            globals_dict,
            witness,
            binding,
            value_binding,
            mode,
        )
        _check_pending(state, state.pending)

    def _check_return(self, state, frame, arg):
        enrollment = state.pending
        if enrollment is None or frame is not enrollment.frame or arg is not None:
            raise ValueError(_ERROR)
        _check_pending(state, enrollment)
        self._check_caller(state, frame)
        lasti = _frame_read(frame, _FRAME_LASTI)
        raw = _read(_CODE_BYTES, state.values[3], CodeType)
        if type(lasti) is not int or lasti < 0 or lasti >= len(raw) or lasti % 2:
            raise ValueError(_ERROR)
        if raw[lasti] not in _NORMAL_RETURNS:
            raise ValueError(_ERROR)
        _check_owner(state, enrollment, True)

    def _ready_owner(self, state, frame):
        _require_frame(frame, state.values[6], state.call_globals)
        owner = _frame_locals(frame).get("self", _MISSING)
        for enrollment in state.ready:
            if owner is enrollment.owner:
                _check_owner(state, enrollment, True)
                return enrollment
        raise ValueError(_ERROR)

    def _observe(self, frame, event, arg, after_hook):
        self._deferred.observe(
            frame,
            event,
            arg,
            after_hook,
            threading.get_ident() == self._main,
            self._pending_state() is not None,
            _NORMAL_RETURNS,
        )
        code = _frame_read(frame, _FRAME_CODE)
        for state in self._states:
            if code is state.values[3]:
                if event == "call":
                    if not after_hook:
                        self._admit(state, frame)
                    else:
                        if state.pending is None or state.pending.frame is not frame:
                            raise ValueError(_ERROR)
                        _check_pending(state, state.pending)
                        self._check_caller(state, frame)
                elif event == "return":
                    self._check_return(state, frame, arg)
                    if after_hook:
                        enrollment = state.pending
                        if enrollment is None:
                            raise ValueError(_ERROR)
                        enrollment.frame = None
                        state.ready.append(enrollment)
                        state.pending = None
                return
            if event == "call" and code is state.values[6]:
                if self._pending_state() is not None or self._deferred.pending_module() is not None:
                    raise ValueError(_ERROR)
                self._ready_owner(state, frame)
                return
            if event == "call" and state.delegate is not None and code is state.delegate[3]:
                if self._pending_state() is not None or self._deferred.pending_module() is not None:
                    raise ValueError(_ERROR)
                parent = _frame_read(frame, _FRAME_PARENT)
                enrollment = self._ready_owner(state, parent)
                if state.value_witnesses is not None and enrollment.mode != "callback":
                    raise ValueError(_ERROR)
                _require_frame(frame, state.delegate[3], enrollment.globals)
                return

    def _profile(self, frame: FrameType, event: str, arg: object) -> None:
        if not self._active:
            super()._profile(frame, event, arg)
            return
        try:
            code = _frame_read(frame, _FRAME_CODE)
            # Why: admission must follow original counting and precede the prior hook.
            if (event == "c_call" and arg is _SET_PROFILE) or (
                event == "call" and any(code is token for token in self._profile_setters)
            ):
                self._abort("profile_mutation")
            token = code if event == "call" else arg
            for original, label, limit, kind in self._table:
                if event == kind and token is original:
                    with self._lock:
                        self._counts[label] += 1
                        attempt = self._counts[label]
                    if attempt > limit:
                        self._abort("entry_limit", label, attempt, limit)
                    break
            self.require_current_enrollments()
            self._observe(frame, event, arg, False)
            self.require_current_enrollments()
            prior = self._prior_main if threading.get_ident() == self._main else self._prior_thread
            if prior is not None:
                try:
                    prior(frame, event, arg)
                except BaseException:
                    self._abort("prior_profile_failure")
            self.require_current_enrollments()
            self._observe(frame, event, arg, True)
            self.require_current_enrollments()
        except BaseException:
            self._abort("measurement_failure")
