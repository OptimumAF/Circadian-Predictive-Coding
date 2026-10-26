"""Copy independently bounded coupled composite DATA bytes into a new directory.

Use --describe for a preview with no filesystem/model work. Actual operation uses
the existing generic store's load/save/load protocol, never native models or live
authority. The source is read only. Directory publication is not transactional:
after save, a later verification/report failure may leave the valid owned file.
Trusted parents and one cooperating writer are required; no power-loss guarantee.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.adapters.coupled_composite_codec import CoupledCompositeCheckpointCodec
from src.core.checkpoint_codec import CodecBinding, CodecLimits, require_codec_digest
from src.core.managed_composite_state import CompositeCaptureLimits, ManagedCompositeState
from src.core.managed_lifecycle_state import LifecycleCaptureLimits
from src.infra.checkpoint_files import BoundedCheckpointFileStore, _controls

DEFAULT_CAPTURE_LIMITS = CompositeCaptureLimits(
    LifecycleCaptureLimits(64, 128), 8192, 64, 65536, 32
)
DEFAULT_WIRE_LIMITS = CodecLimits(262144, 65536, 4, 32)
MAX_OUTPUT_BYTES = 65536


def describe_example() -> dict[str, object]:
    """Describe fixed defaults without inspecting paths or constructing a codec."""
    return {
        "protocol_id": "coupled_composite_files_engineering_v1",
        "data_only": True,
        "capture_limits": asdict(DEFAULT_CAPTURE_LIMITS),
        "wire_limits": asdict(DEFAULT_WIRE_LIMITS),
        "operations": ["load", "save", "load"],
        "output_filename": "checkpoint.json",
        "directory_policy": "New dedicated directory; existing paths are refused before source decoding",
        "expectations": "Binding, content digest and all bounds are independent caller inputs",
        "limitations": "Data only; no live authority, restoration or recovery completion. Trusted parent, single writer, no power-loss guarantee. A post-save verification/report failure can leave the valid checkpoint.",
    }


def copy_checkpoint(
    source: Path,
    directory: Path,
    *,
    binding: CodecBinding,
    expected_sha256: str,
    capture_limits: CompositeCaptureLimits,
    wire_limits: CodecLimits,
) -> dict[str, object]:
    """Transport a complete detached graph using independently fixed controls."""
    if not isinstance(source, Path) or not isinstance(directory, Path):
        raise ValueError("source and directory must be Path values")
    directory = Path(os.path.abspath(directory))
    # lstat also sees dangling symlinks. The cooperating-writer/trusted-parent
    # contract does not turn this precheck into a filesystem race defense.
    try:
        directory.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"output directory must be a new dedicated path: {directory}")
    owned_binding, owned_wire = _controls(binding, wire_limits)
    require_codec_digest(expected_sha256)
    codec = CoupledCompositeCheckpointCodec(capture_limits=capture_limits)
    source_store: BoundedCheckpointFileStore[ManagedCompositeState] = BoundedCheckpointFileStore(
        source, codec
    )
    state = source_store.load(
        binding=owned_binding, expected_sha256=expected_sha256, limits=owned_wire
    )
    # A refused source has performed no output mkdir/save. No second encoder or
    # equivalence framework is needed: the codec's canonical bytes bind the hash.
    directory.mkdir(parents=True, exist_ok=False)
    output_store: BoundedCheckpointFileStore[ManagedCompositeState] = BoundedCheckpointFileStore(
        directory / "checkpoint.json", codec
    )
    digest = output_store.save(state, binding=owned_binding, limits=owned_wire)
    if digest != expected_sha256:
        raise ValueError("saved composite digest differs from independently expected digest")
    output_store.load(binding=owned_binding, expected_sha256=expected_sha256, limits=owned_wire)
    return {
        "source": str(source_store.path),
        "directory": str(directory),
        "checkpoint": str(output_store.path),
        "digest": digest,
        "data_only": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--describe", action="store_true", help="Preview without filesystem/model work"
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--directory", type=Path)
    for name in ("source-sha256", "policy-sha256", "expected-sha256"):
        parser.add_argument("--" + name)
    for prefix, values in (
        ("record", asdict(DEFAULT_CAPTURE_LIMITS.records)),
        (
            "capture",
            {
                key: value
                for key, value in asdict(DEFAULT_CAPTURE_LIMITS).items()
                if key != "records"
            },
        ),
        ("wire", asdict(DEFAULT_WIRE_LIMITS)),
    ):
        for name, default in values.items():
            parser.add_argument(f"--{prefix}-{name.replace('_', '-')}", type=int, default=default)
    return parser


def _print_report(report: dict[str, object]) -> None:
    encoded = json.dumps(report, allow_nan=False, sort_keys=True, indent=2)
    if len((encoded + "\n").replace("\n", os.linesep).encode("utf8")) > MAX_OUTPUT_BYTES:
        raise ValueError("composite file report exceeds bounded output size")
    print(encoded)


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    operation = (
        args.source,
        args.directory,
        args.source_sha256,
        args.policy_sha256,
        args.expected_sha256,
    )
    if args.describe and any(value is not None for value in operation):
        parser.error("--describe cannot be combined with source/directory/digest arguments")
    if not args.describe and any(value is None for value in operation):
        parser.error(
            "--source, --directory, --source-sha256, --policy-sha256 and --expected-sha256 are required unless --describe is selected"
        )
    capture_limits = CompositeCaptureLimits(
        LifecycleCaptureLimits(args.record_max_records, args.record_max_identifier_bytes),
        args.capture_max_nodes,
        args.capture_max_depth,
        args.capture_max_array_bytes,
        args.capture_max_dimension,
    )
    wire_limits = CodecLimits(
        args.wire_max_encoded_bytes,
        args.wire_max_array_bytes,
        args.wire_max_layers,
        args.wire_max_dimension,
    )
    if args.describe:
        report = describe_example()
        report.update(capture_limits=asdict(capture_limits), wire_limits=asdict(wire_limits))
    else:
        report = copy_checkpoint(
            args.source,
            args.directory,
            binding=CodecBinding(args.source_sha256, args.policy_sha256),
            expected_sha256=args.expected_sha256,
            capture_limits=capture_limits,
            wire_limits=wire_limits,
        )
    _print_report(report)


if __name__ == "__main__":
    main()
