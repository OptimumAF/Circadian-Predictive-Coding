"""Locate packaged TRAIN-only examples; no model, runtime authority or selection."""

from contextlib import contextmanager, nullcontext
from importlib import resources
from pathlib import Path


@contextmanager
def example_affine_training():
    """Keep a local or extracted package resource alive during decoding."""
    local = Path(__file__).resolve().parents[2] / "examples/minigrid_affine_train/episodes.json"
    origin = (
        nullcontext(local)
        if local.is_file()
        else resources.as_file(
            resources.files("src.infra.minigrid_affine_data").joinpath("episodes.json")
        )
    )
    with origin as path:
        if path.stat().st_size > 512 * 1024:
            raise ValueError("bundled affine TRAIN resource exceeds512KiB")
        yield path
