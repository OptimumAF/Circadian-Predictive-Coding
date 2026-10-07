"""Read finite unambiguous JSON and preserve exclusive confirmation artifacts.

Inputs are local files/text and explicit source pins. Outputs are decoded
objects or byte identities. This boundary owns no training, scores or scope
selection; scientific request/fact validation belongs to app.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, Iterator


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"confirmation JSON duplicate key: {name}")
        result[name] = value
    return result


def _nonfinite(token: str) -> Any:
    raise ValueError(f"confirmation JSON nonfinite token: {token}")


def _finite(value: Any) -> None:
    if type(value) is float and not isfinite(value):
        raise ValueError("confirmation JSON nonfinite number")
    if type(value) is dict:
        for item in value.values():
            _finite(item)
    elif type(value) is list:
        for item in value:
            _finite(item)


def parse_json(payload: str) -> dict[str, Any]:
    try:
        value = json.loads(payload, parse_constant=_nonfinite, object_pairs_hook=_object)
        _finite(value)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("confirmation JSON malformed or too deeply nested") from error
    if type(value) is not dict:
        raise ValueError("confirmation JSON must be an object")
    return value


def file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return parse_json(path.read_text(encoding="utf-8"))


def encoded_digest(value: Any) -> str:
    return sha256(_encode_json(value).encode("utf-8")).hexdigest()


def _encode_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_exclusive(path: Path, value: Any) -> None:
    # Why this: encode before opening; invalid values must not claim a path.
    encoded = _encode_json(value)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(encoded)


@contextmanager
def claim_artifacts(paths: dict[str, Path]) -> Iterator[None]:
    """Prevent competing writers from marking each other's request failed."""
    acquired = False
    try:
        with paths["claim"].open("x", encoding="ascii") as claim:
            acquired = True
            claim.write("p67_confirmation_train_claim_v1\n")
        occupied = [str(path) for name, path in paths.items() if name != "claim" and path.exists()]
        if occupied:
            raise FileExistsError(f"confirmation output already exists: {occupied}")
        yield
    finally:
        if acquired:
            paths["claim"].unlink(missing_ok=True)


def verify_source_files(root: Path, expected: dict[str, str]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or file_digest(path) != digest:
            raise ValueError(f"confirmation frozen source changed or missing: {name}")
        observed[name] = digest
    return observed
