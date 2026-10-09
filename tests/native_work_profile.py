"""Bounded original call measurement; no method replacement, graphs or IO.

Inputs are raw defining classes/method names and finite counter maps. The
returned table retains code objects only, never frames or native payloads.
"""

from types import CodeType, FrameType, FunctionType


def bind_original_calls(
    entries: tuple[tuple[type, str, str], ...],
) -> tuple[tuple[CodeType, str], ...]:
    """Bind at most four distinct raw Python methods before measurement."""
    if not 0 < len(entries) <= 4:
        raise ValueError("original call measurement requires one to four methods")
    table: list[tuple[CodeType, str]] = []
    for owner, method, key in entries:
        original = type.__getattribute__(owner, "__dict__")[method]
        if type(original) is not FunctionType:
            raise TypeError("original call measurement requires a raw Python function")
        code = original.__code__
        if any(code is prior for prior, _ in table):
            raise ValueError("original call measurement requires distinct code identities")
        table.append((code, key))
    return tuple(table)


def count_original_calls(table, work, limits, frame: FrameType, event: str) -> None:
    """Charge each original entry before its body, including recursive entries."""
    if event != "call":
        return
    # Why identity: equal foreign code objects must not acquire original authority.
    for code, key in table:
        if frame.f_code is code:
            work[key] += 1
            assert work[key] <= limits[key], f"original call cap exceeded: {key}"
            return
