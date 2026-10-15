"""Read-only continuity checks for independently retained fixture roots.

Inputs are exact native wrapper/module/function/code/globals witnesses. Success
returns None; inconsistent or foreign witnesses raise TypeError/ValueError.
Why native readers: wrapper attribute hooks must not participate in evidence.
This neither unwraps nor invokes fixtures, imports original modules, discovers
roots, proves source/birth/defaults/closures, freezes globals contents, nor adds
or renews a profile. Compose checks inside the existing one-use measurement.
"""

from dataclasses import dataclass
from types import CodeType, FunctionType, MemberDescriptorType

from provider_owner_namespace_profile import OwnerNamespaceBinding, require_current_namespaces

_TYPE_DICT = type.__dict__["__dict__"]
_MEMBER_GET = _TYPE_DICT.__get__(MemberDescriptorType, type)["__get__"]
_FUNCTION_CODE = _TYPE_DICT.__get__(FunctionType, type)["__code__"]
_FUNCTION_GLOBALS = _TYPE_DICT.__get__(FunctionType, type)["__globals__"]
_OWNER_FIELDS = tuple(
    _TYPE_DICT.__get__(OwnerNamespaceBinding, type)[name] for name in ("owner", "namespace")
)
_MISSING = object()


@dataclass(frozen=True, slots=True)
class FixtureRootBinding:
    """Original witnesses supplied independently of the current wrapper aliases."""

    namespace: dict[str, object]
    slot: str
    wrapper: OwnerNamespaceBinding
    raw_function: FunctionType
    raw_code: CodeType
    function_globals: dict[str, object]


_BINDING_FIELDS = tuple(
    _TYPE_DICT.__get__(FixtureRootBinding, type)[name]
    for name in ("namespace", "slot", "wrapper", "raw_function", "raw_code", "function_globals")
)


def _binding_values(binding: FixtureRootBinding) -> tuple:
    if type(binding) is not FixtureRootBinding:
        raise TypeError("fixture root requires an exact retained binding")
    try:
        return tuple(_MEMBER_GET(reader, binding, FixtureRootBinding) for reader in _BINDING_FIELDS)
    except AttributeError:
        raise ValueError("fixture root retained binding is incomplete") from None


def _require_dictionary(namespace: object) -> None:
    if type(namespace) is not dict:
        raise TypeError("fixture root requires an exact plain dictionary")
    if len(namespace) > 512 or any(type(name) is not str or len(name) > 128 for name in namespace):
        raise ValueError("fixture root dictionary requires <=512 bounded exact string keys")


def require_fixture_root_binding(binding: FixtureRootBinding) -> None:
    """Check module slot, native wrapper aliases and raw function witnesses."""
    namespace, slot, wrapper, raw_function, raw_code, function_globals = _binding_values(binding)
    _require_dictionary(namespace)
    _require_dictionary(function_globals)
    if type(slot) is not str or not 0 < len(slot) <= 128:
        raise ValueError("fixture root slot requires a nonempty bounded exact string")
    if type(wrapper) is not OwnerNamespaceBinding:
        raise TypeError("fixture root requires an exact native owner binding")
    require_current_namespaces((wrapper,))
    owner, storage = tuple(
        _MEMBER_GET(reader, wrapper, OwnerNamespaceBinding) for reader in _OWNER_FIELDS
    )
    if namespace.get(slot, _MISSING) is not owner:
        raise ValueError("fixture root module slot no longer holds its original wrapper")
    if type(raw_function) is not FunctionType or type(raw_code) is not CodeType:
        raise TypeError("fixture root requires an exact raw function and original code")
    if (
        storage.get("_fixture_function", _MISSING) is not raw_function
        or storage.get("__wrapped__", _MISSING) is not raw_function
    ):
        raise ValueError("fixture root original raw function aliases changed or are missing")
    if _FUNCTION_CODE.__get__(raw_function, FunctionType) is not raw_code:
        raise ValueError("fixture root original raw function code changed")
    if (
        function_globals is not namespace
        or _FUNCTION_GLOBALS.__get__(raw_function, FunctionType) is not function_globals
    ):
        raise ValueError("fixture root original globals must be its module namespace")
    require_current_namespaces((wrapper,))
