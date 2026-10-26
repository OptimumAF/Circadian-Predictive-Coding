"""Complete DATA checkpoints during the original managed B-to-C continuation.

The reusable saver validates independent controls, captures under original owner
leases/admission, then delegates file save/load to the existing bounded store.
Failed storage never refunds capture charges. The fixed example observes paused
B and committed C, then the original live workflow expires its raw payloads.
Files remain caller-owned: no live restore, authority, new storage, or recovery
completion is supplied. Directory publication is not transactional; successful
files may remain if a later operation/report fails.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

from src.adapters.coupled_candidate_handoff import BINDING, FAMILIES, MODES
from src.adapters.coupled_composite_capture import capture_coupled_managed_composite
from src.adapters.coupled_composite_codec import (
    CoupledCompositeCheckpointCodec,
    _copy_capture_limits,
)
from src.adapters.coupled_managed_serving import (
    _case,
    _report,
    _retention_policy,
    describe_committed_serving_example,
)
from src.app.managed_experience import ManagedExperienceOwner
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.managed_composite_state import CompositeCaptureLimits, ManagedCompositeState
from src.core.managed_lifecycle_state import LifecycleCaptureLimits
from src.infra.checkpoint_files import BoundedCheckpointFileStore, _controls

CAPTURE_LIMITS = CompositeCaptureLimits(LifecycleCaptureLimits(64, 128), 8192, 64, 65536, 32)
WIRE_LIMITS = CodecLimits(262144, 65536, 4, 32)


def save_coupled_managed_checkpoint(
    owner: ManagedExperienceOwner,
    path: Path,
    *,
    binding: CodecBinding,
    capture_limits: CompositeCaptureLimits,
    wire_limits: CodecLimits,
) -> dict[str, Any]:
    """Capture once, save, and reload validated DATA under independent bounds.

    Controls and lexical path validation precede capture and file work. The
    original capture reserves its RAW copy charge; save/load failures propagate
    without restoring that allowance or modifying the live model.
    """
    if not isinstance(path, Path):
        raise ValueError("complete checkpoint path must be a Path")
    owned_capture = _copy_capture_limits(capture_limits)
    owned_binding, owned_wire = _controls(binding, wire_limits)
    codec = CoupledCompositeCheckpointCodec(capture_limits=owned_capture)
    store: BoundedCheckpointFileStore[ManagedCompositeState] = BoundedCheckpointFileStore(
        path, codec
    )
    captured = capture_coupled_managed_composite(owner, limits=owned_capture)
    digest = store.save(captured.state, binding=owned_binding, limits=owned_wire)
    store.load(binding=owned_binding, expected_sha256=digest, limits=owned_wire)
    receipt = dict(
        path=str(store.path),
        digest=digest,
        data_only=True,
        retained_payload_bytes=captured.state.retained_payload_bytes,
    )
    if len(json.dumps(receipt, allow_nan=False, sort_keys=True).encode("utf8")) > 65536:
        raise ValueError("complete checkpoint receipt exceeds output bound")
    return receipt


def describe_complete_checkpoint_example() -> dict[str, Any]:
    """Describe the six fixed controls, without paths, models, IO or threads."""
    description = describe_committed_serving_example()
    description.update(
        protocol_id="coupled_live_complete_checkpoints_engineering_v1",
        retention_policy=asdict(_retention_policy(committed=True, complete_checkpoints=True)),
        capture_limits=asdict(CAPTURE_LIMITS),
        wire_limits=asdict(WIRE_LIMITS),
        binding=asdict(BINDING),
        checkpoint_phases=["B", "C"],
        checkpoint_files=12,
        capture_calls=12,
        capture_raw_charges=[328, 512],
        copy_charges=[24, 168, 184, 328, 472, 800, 944, 968, 968, 984, 984, 1496, 1496],
        retained_raw_bytes=[24, 168, 184, 328, 328, 328, 472, 496, 496, 512, 512, 512, 0],
        data_only=True,
        limitations="Complete data files are caller-owned copies outside original expiry cleanup. Original live ownership, fixed policies and quotas continue; no restoration, recovered authority or process completion. New directory publication is not transactional; successful files may remain after later failure. Existing codec unordered/depth bounds and trusted-parent/single-writer/no-power-loss file limits persist.",
    )
    return description


def run_complete_checkpoint_example(directory: Path) -> dict[str, Any]:
    """Save B/C for each fixed control, preserving original continuation/expiry."""
    if not isinstance(directory, Path):
        raise ValueError("complete checkpoint example directory must be a Path")
    directory = Path(os.path.abspath(directory))
    try:
        directory.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"complete checkpoint example requires a new directory: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    for mode in MODES:
        for family in FAMILIES:

            def observe(owner: ManagedExperienceOwner, phase: str) -> dict[str, Any]:
                if phase not in ("B", "C"):
                    raise ValueError("complete checkpoint phase must be B or C")
                return save_coupled_managed_checkpoint(
                    owner,
                    directory / f"{mode}-{family}-{phase}.json",
                    binding=BINDING,
                    capture_limits=CAPTURE_LIMITS,
                    wire_limits=WIRE_LIMITS,
                )

            rows.append(_case(family, mode, committed=True, checkpoint_observer=observe))
    result = _report(rows, committed=True)
    result.update(
        protocol=describe_complete_checkpoint_example(),
        directory=str(directory),
        complete_checkpoints_verified=True,
        data_only=True,
    )
    return result
