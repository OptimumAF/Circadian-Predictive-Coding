"""Exact project-generated contextual source data for the installed CPU demo.

Owns bounded resource IO and lifetime, not training, restored runtime authority,
model selection, scientific admission or artifact substitution.
"""

from contextlib import contextmanager, nullcontext
from hashlib import sha256
from importlib import resources
from pathlib import Path

SOURCE_SHA = "addf37403d32682b680b74ea7a1bfc16ced2895cffe09b8cc5fdf701179be943"


def _require_source(path):
    with path.open("rb") as source:
        raw = source.read(512 * 1024 + 1)
    if len(raw) > 512 * 1024 or sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("bundled contextual source differs from the qualified example")


@contextmanager
def example_context_source():
    """Keep extracted resources alive through execution and output publication."""
    local = Path(__file__).resolve().parents[2] / "examples/context_head_source/source.json"
    origin = (
        nullcontext(local)
        if local.is_file()
        else resources.as_file(
            resources.files("src.infra.context_head_data").joinpath("source.json")
        )
    )
    with origin as path:
        _require_source(path)  # Corrupt local evidence must never fall back to another source.
        try:
            yield path
        finally:
            _require_source(path)
