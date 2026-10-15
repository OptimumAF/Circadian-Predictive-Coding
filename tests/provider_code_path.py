"""Retain explicit original provider code paths before factory work.

Inputs are an independently retained original root code and bounded native
constant indexes. Outputs are strong original code witnesses or refusal.
The caller establishes root/source authority and the pre-factory capture point.
This proves code continuity only, not callback birth, closures/defaults/globals,
original import safety, complete coverage, profiling or concurrent atomicity.
"""

from dataclasses import dataclass
from types import CodeType, FunctionType

# Why eager native readers: foreign row properties cannot supply witness fields.
_TYPE_DICT = type.__dict__["__dict__"]
_FUNCTION_CODE = _TYPE_DICT.__get__(FunctionType, type)["__code__"]
_CODE_CONSTANTS = _TYPE_DICT.__get__(CodeType, type)["co_consts"]


@dataclass(frozen=True, slots=True)
class ProviderCodePath:
    """Strong originals; immutable code constants retain every intermediate node."""

    root: FunctionType
    root_code: CodeType
    indexes: tuple[int, ...]
    token: CodeType


_WITNESS_FIELDS = tuple(
    _TYPE_DICT.__get__(ProviderCodePath, type)[name]
    for name in ("root", "root_code", "indexes", "token")
)


def _witness_values(witness: ProviderCodePath) -> tuple:
    if type(witness) is not ProviderCodePath:
        raise TypeError("provider code path requires an exact retained witness")
    try:
        return tuple(field.__get__(witness, ProviderCodePath) for field in _WITNESS_FIELDS)
    except AttributeError:
        raise ValueError("provider code path retained witness is incomplete") from None


def _require_inputs(root: object, root_code: object, indexes: object) -> None:
    if type(root) is not FunctionType or type(root_code) is not CodeType:
        raise TypeError("provider code path requires a raw function and original raw code")
    if type(indexes) is not tuple:
        raise TypeError("provider code path indexes require an exact tuple")
    if len(indexes) > 16:
        raise ValueError("provider code path exceeds16 constant edges")
    for index in indexes:
        if type(index) is not int or not 0 <= index <= 511:
            raise ValueError("provider code path index must be an exact integer in0..511")


def _require_constants(node: CodeType) -> tuple[object, ...]:
    constants = _CODE_CONSTANTS.__get__(node, CodeType)
    if type(constants) is not tuple:
        raise TypeError("provider code path requires exact native constant tuples")
    if len(constants) > 512:
        raise ValueError("provider code path node exceeds512 constants")
    return constants


def _resolve_token(root_code: CodeType, indexes: tuple[int, ...]) -> CodeType:
    node = root_code
    for index in indexes:
        constants = _require_constants(node)
        if index >= len(constants):
            raise ValueError("provider code path index exceeds this node's constants")
        selected = constants[index]
        if type(selected) is not CodeType:
            raise TypeError("provider code path selected constant is not raw code")
        node = selected
    # The leaf is visited too, including an empty path's original root.
    _require_constants(node)
    return node


def _checked_token(root: FunctionType, root_code: CodeType, indexes: tuple[int, ...]) -> CodeType:
    _require_inputs(root, root_code, indexes)
    if _FUNCTION_CODE.__get__(root, FunctionType) is not root_code:
        raise ValueError("provider code path original root code changed")
    token = _resolve_token(root_code, indexes)
    if _FUNCTION_CODE.__get__(root, FunctionType) is not root_code:
        raise ValueError("provider code path root changed during selection")
    return token


def pin_provider_code_path(
    root: FunctionType, root_code: CodeType, indexes: tuple[int, ...]
) -> ProviderCodePath:
    """Pin a supplied original code path; never derive or renew root authority."""
    token = _checked_token(root, root_code, indexes)
    return ProviderCodePath(root, root_code, indexes, token)


def require_provider_code_path(witness: ProviderCodePath) -> None:
    """Reprove an original root and selected leaf using native identity only."""
    root, root_code, indexes, token = _witness_values(witness)
    if type(token) is not CodeType:
        raise TypeError("provider code path requires a retained original raw leaf")
    if _checked_token(root, root_code, indexes) is not token:
        raise ValueError("provider code path selected leaf identity changed")


def require_provider_callback_code(witness: ProviderCodePath, callback: FunctionType) -> None:
    """Check candidate code without calling it or proving its object provenance."""
    require_provider_code_path(witness)
    if type(callback) is not FunctionType:
        raise TypeError("provider code path callback requires an exact raw Python function")
    token = _witness_values(witness)[3]
    if _FUNCTION_CODE.__get__(callback, FunctionType) is not token:
        raise ValueError("provider callback code is not the retained original leaf")
    require_provider_code_path(witness)
