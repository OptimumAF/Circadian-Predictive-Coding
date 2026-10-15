"""Cooperative retained callback checks for disposable profile controls.

Inputs: exact plain namespaces, original callback objects and independent raw
code/C tokens. Outputs: refusal or unchanged original profile counts/exit73.
This retains originals and aliases without wrapping dispatch. It does not attest
who borrowed a namespace, discover delegates/defaults, freeze concurrent writers,
or qualify original cases, native work or profiling-suppressed prior-hook work.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import (
    BuiltinFunctionType,
    CodeType,
    FrameType,
    FunctionType,
    MappingProxyType,
    MethodType,
)
from typing import Callable, Iterator, cast

from provider_entry_profile import ProviderEntryProfile

# Why native slots: object.__getattribute__ can still resolve a foreign property.
_TYPE_DICT = type.__dict__["__dict__"]
_MISSING = object()


@dataclass(frozen=True, slots=True)
class RuntimeBinding:
    """Retained originals; expected tokens must come from independent witnesses."""

    namespace: dict[str, object]
    slot: str
    callback: object
    token: CodeType | BuiltinFunctionType
    owner_type: type | None = None
    raw_call: FunctionType | None = None


_BINDING_FIELDS = tuple(
    _TYPE_DICT.__get__(RuntimeBinding, type)[name]
    for name in ("namespace", "slot", "callback", "token", "owner_type", "raw_call")
)


def _binding_values(binding: RuntimeBinding) -> tuple:
    if type(binding) is not RuntimeBinding:
        raise TypeError("runtime binding requires an exact retained row")
    try:
        return tuple(field.__get__(binding, RuntimeBinding) for field in _BINDING_FIELDS)
    except AttributeError:
        raise ValueError("runtime binding retained row is incomplete") from None


def _require_namespace(namespace: object) -> None:
    if type(namespace) is not dict:
        raise TypeError("runtime binding requires an exact plain namespace")
    if len(namespace) > 512:
        raise ValueError("runtime binding namespace exceeds512 fields")
    if any(type(key) is not str or len(key) > 128 for key in namespace):
        raise ValueError("runtime binding namespace requires bounded exact string keys")


def _require_function_token(function: FunctionType, token: object) -> None:
    if type(token) is not CodeType:
        raise TypeError("Python runtime binding requires an original raw code token")
    if function.__code__ is not token:
        raise ValueError("runtime binding original function code changed")


def _require_instance_call(
    callback: object, token: object, owner_type: object, raw_call: object
) -> None:
    if type(owner_type) is not type or type(callback) is not owner_type:
        raise TypeError("runtime binding requires its exact original class and metaclass")
    if type(raw_call) is not FunctionType:
        raise TypeError("runtime binding requires an original raw call function")
    namespace = _TYPE_DICT.__get__(owner_type, type)
    if type(namespace) is not MappingProxyType:
        raise TypeError("runtime binding requires a native class namespace")
    # Why bounded native keys: lookup must not hash or compare a foreign key.
    if len(namespace) > 512 or any(type(key) is not str or len(key) > 128 for key in namespace):
        raise ValueError("runtime binding class namespace requires bounded exact keys")
    if namespace.get("__call__", _MISSING) is not raw_call:
        raise ValueError("runtime binding requires its unchanged own raw call function")
    _require_function_token(raw_call, token)


def _require_callback(
    callback: object, token: object, owner_type: object, raw_call: object
) -> None:
    kind = type(callback)
    if kind is FunctionType or kind is MethodType or kind is BuiltinFunctionType:
        if owner_type is not None or raw_call is not None:
            raise ValueError("raw callback bindings cannot carry instance call witnesses")
        if kind is FunctionType:
            _require_function_token(cast(FunctionType, callback), token)
        elif kind is MethodType:
            # The retained method itself holds its immutable function/receiver pair.
            function = cast(MethodType, callback).__func__
            if type(function) is not FunctionType:
                raise TypeError("runtime binding method requires a raw Python function")
            _require_function_token(function, token)
        elif type(token) is not BuiltinFunctionType or callback is not token:
            raise ValueError("runtime binding original C callable changed")
        return
    _require_instance_call(callback, token, owner_type, raw_call)


def require_current_bindings(bindings: tuple[RuntimeBinding, ...]) -> None:
    """Check actual supplied slots against retained objects without dispatching them."""
    if type(bindings) is not tuple:
        raise TypeError("runtime bindings require an exact tuple")
    if not 0 < len(bindings) <= 64:
        raise ValueError("runtime bindings require one to64 rows")
    for binding in bindings:
        namespace, slot, callback, token, owner_type, raw_call = _binding_values(binding)
        _require_namespace(namespace)
        if type(slot) is not str or not 0 < len(slot) <= 128:
            raise ValueError("runtime binding slot requires a nonempty bounded exact string")
        if namespace.get(slot, _MISSING) is not callback:
            raise ValueError("runtime binding current callback slot changed or is missing")
        _require_callback(callback, token, owner_type, raw_call)


class ProviderRuntimeBindingProfile(ProviderEntryProfile):
    """Original finite entry measurement with cooperative pre/post dispatch checks."""

    def __init__(
        self,
        entries: tuple[tuple[Callable | CodeType, str, int], ...],
        receipt: Path,
        bindings: tuple[RuntimeBinding, ...],
    ) -> None:
        require_current_bindings(bindings)
        self._bindings = bindings
        super().__init__(entries, receipt)
        for binding in bindings:
            token = _binding_values(binding)[3]
            if not any(token is original for original, _, _, _ in self._table):
                raise ValueError("runtime binding token has no registered original entry")
        require_current_bindings(bindings)

    @contextmanager
    def measure(self) -> Iterator[None]:
        require_current_bindings(self._bindings)
        with super().measure():
            yield

    def _profile(self, frame: FrameType, event: str, arg: object) -> None:
        if not self._active:
            super()._profile(frame, event, arg)
            return
        try:
            require_current_bindings(self._bindings)
            super()._profile(frame, event, arg)
            # A trusted observer can mutate a slot while profiling is suppressed.
            require_current_bindings(self._bindings)
        except BaseException:
            self._abort("callback_binding_changed")
