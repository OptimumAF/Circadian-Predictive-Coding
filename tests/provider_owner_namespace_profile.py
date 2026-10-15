"""Cooperative attachment checks for retained provider owner dictionaries.

Inputs: retained owner/class/native descriptor/dictionary witnesses and explicit
callback rows. Outputs: refusal or unchanged original profile counts/exit73.
The native reader bypasses instance hooks but does not authorize them or attest
caller roots, field descriptors, original case activation or atomic concurrency.
No delegates, defaults, closures or live graphs are discovered or copied.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FrameType, GetSetDescriptorType, MappingProxyType
from typing import Callable, Iterator

from provider_runtime_binding import ProviderRuntimeBindingProfile, RuntimeBinding

# Why captured native operations: an ordinary owner.__dict__ lookup can run hooks.
_TYPE_DICT = type.__dict__["__dict__"]
_GETSET_GET = _TYPE_DICT.__get__(GetSetDescriptorType, type)["__get__"]
_RUNTIME_NAMESPACE = _TYPE_DICT.__get__(RuntimeBinding, type)["namespace"]
_MISSING = object()


@dataclass(frozen=True, slots=True)
class OwnerNamespaceBinding:
    """Independent original witnesses; replacing storage cannot renew attachment."""

    owner: object
    owner_type: type
    descriptor: GetSetDescriptorType
    namespace: dict[str, object]


_OWNER_FIELDS = tuple(
    _TYPE_DICT.__get__(OwnerNamespaceBinding, type)[name]
    for name in ("owner", "owner_type", "descriptor", "namespace")
)


def _owner_values(binding: OwnerNamespaceBinding) -> tuple:
    if type(binding) is not OwnerNamespaceBinding:
        raise TypeError("owner namespace requires an exact retained row")
    try:
        return tuple(field.__get__(binding, OwnerNamespaceBinding) for field in _OWNER_FIELDS)
    except AttributeError:
        raise ValueError("owner namespace retained row is incomplete") from None


def _require_dictionary(namespace: object) -> None:
    if type(namespace) is not dict:
        raise TypeError("owner namespace requires an exact retained dictionary")
    if len(namespace) > 512:
        raise ValueError("owner namespace dictionary exceeds512 fields")
    if any(type(key) is not str or len(key) > 128 for key in namespace):
        raise ValueError("owner namespace dictionary requires bounded exact string keys")


def _require_native_descriptor(owner_type: object, descriptor: object) -> None:
    if type(descriptor) is not GetSetDescriptorType:
        raise TypeError("owner namespace requires its original native dictionary descriptor")
    if descriptor.__objclass__ is not owner_type:
        raise ValueError("owner namespace descriptor belongs to a different class")
    name = descriptor.__name__
    if type(name) is not str or name != "__dict__":
        raise ValueError("owner namespace descriptor must name the original dictionary")
    namespace = _TYPE_DICT.__get__(owner_type, type)
    if type(namespace) is not MappingProxyType:
        raise TypeError("owner namespace requires a native class namespace")
    # Preflight every key before lookup so foreign key equality cannot dispatch.
    if len(namespace) > 512 or any(type(key) is not str or len(key) > 128 for key in namespace):
        raise ValueError("owner namespace class requires bounded exact string keys")
    if namespace.get("__dict__", _MISSING) is not descriptor:
        raise ValueError("owner namespace requires its unchanged own dictionary descriptor")


def require_current_namespaces(namespaces: tuple[OwnerNamespaceBinding, ...]) -> None:
    """Validate current native attachment without resolving instance attributes."""
    if type(namespaces) is not tuple:
        raise TypeError("owner namespaces require an exact tuple")
    if not 0 < len(namespaces) <= 64:
        raise ValueError("owner namespaces require one to64 rows")
    for binding in namespaces:
        owner, owner_type, descriptor, namespace = _owner_values(binding)
        if type(owner_type) is not type or type(owner) is not owner_type:
            raise TypeError("owner namespace requires its exact original class and metaclass")
        _require_dictionary(namespace)
        _require_native_descriptor(owner_type, descriptor)
        current = _GETSET_GET(descriptor, owner, owner_type)
        if current is not namespace:
            raise ValueError("owner namespace current dictionary attachment changed")
        _require_dictionary(current)


def _require_callback_coverage(
    bindings: tuple[RuntimeBinding, ...], namespaces: tuple[OwnerNamespaceBinding, ...]
) -> None:
    if type(bindings) is not tuple:
        raise TypeError("callback namespace coverage requires an exact tuple")
    if not 0 < len(bindings) <= 64:
        raise ValueError("callback namespace coverage requires one to64 rows")
    retained = tuple(_owner_values(binding)[3] for binding in namespaces)
    for binding in bindings:
        if type(binding) is not RuntimeBinding:
            raise TypeError("callback namespace coverage requires exact retained rows")
        try:
            namespace = _RUNTIME_NAMESPACE.__get__(binding, RuntimeBinding)
        except AttributeError:
            raise ValueError("callback namespace retained row is incomplete") from None
        if not any(namespace is original for original in retained):
            raise ValueError("callback namespace is not covered by a retained owner")


class ProviderOwnerNamespaceProfile(ProviderRuntimeBindingProfile):
    """Compose owner attachment with unchanged callback and finite entry checks."""

    def __init__(
        self,
        entries: tuple[tuple[Callable | CodeType, str, int], ...],
        receipt: Path,
        bindings: tuple[RuntimeBinding, ...],
        namespaces: tuple[OwnerNamespaceBinding, ...],
    ) -> None:
        require_current_namespaces(namespaces)
        _require_callback_coverage(bindings, namespaces)
        self._namespaces = namespaces
        super().__init__(entries, receipt, bindings)
        self._require_owned_bindings()

    def _require_owned_bindings(self) -> None:
        require_current_namespaces(self._namespaces)
        # Recheck coverage too: a replaced row namespace may still pass callbacks.
        _require_callback_coverage(self._bindings, self._namespaces)

    @contextmanager
    def measure(self) -> Iterator[None]:
        self._require_owned_bindings()
        with super().measure():
            yield

    def _profile(self, frame: FrameType, event: str, arg: object) -> None:
        if not self._active:
            super()._profile(frame, event, arg)
            return
        try:
            self._require_owned_bindings()
            super()._profile(frame, event, arg)
            self._require_owned_bindings()
        except BaseException:
            self._abort("owner_namespace_changed")
