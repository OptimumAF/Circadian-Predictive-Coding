"""Bounded local JSON input and exclusive offline inspector publication.

Inputs are a saved game or sensor report and an explicit new HTML destination. Outputs
derive only from that report; no training, environment interaction, fetching or
modification of the source or existing dashboards occurs here.
"""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any

from src.app.key_door_inspector import project_key_door_report

MAX_INPUT_BYTES = 8 * 1024**2
MAX_LEARNABILITY_INPUT_BYTES = 16 * 1024**2
MAX_OUTPUT_BYTES = 16 * 1024**2


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"nonfinite JSON value: {value}")


def _finite_float(value: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError("nonfinite JSON number")
    return result


def render_key_door_inspector(projection: dict[str, Any]) -> bytes:
    """Embed detached data safely; the browser uses textContent for source text."""
    encoded = json.dumps(projection, allow_nan=False, ensure_ascii=True, separators=(",", ":"))
    # JSON strings must never terminate the inert application/json script element.
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template_name = {
        "minigrid_inspector_v1": "minigrid_inspector.html",
        "minigrid_learnability_inspector_v1": "minigrid_inspector.html",
        "minigrid_saved_fifo_inspector_v1": "minigrid_inspector.html",
        "minigrid_heldout_fifo_inspector_v1": "minigrid_inspector.html",
        "minigrid_affine_inspector_v1": "minigrid_inspector.html",
        "minigrid_private_affine_inspector_v1": "minigrid_inspector.html",
        "sensor_inspector_v1": "sensor_inspector.html",
        "coherent_stream_inspector_v1": "coherent_stream_inspector.html",
        "taskspace_memory_inspector_v1": "coherent_stream_inspector.html",
        "ticket_workflow_inspector_v1": "coherent_stream_inspector.html",
        "clm_probe_inspector_v1": "coherent_stream_inspector.html",
        "continuous_comparison_inspector_v1": "coherent_stream_inspector.html",
        "public_history_inspector_v1": "public_history_inspector.html",
    }.get(str(projection.get("schema_id")), "key_door_inspector.html")
    template = Path(__file__).with_name(template_name).read_text(encoding="utf8")
    if template.count("__INSPECTOR_DATA__") != 1:
        raise ValueError("inspector template requires one data insertion point")
    payload = template.replace("__INSPECTOR_DATA__", encoded).encode("utf8")
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ValueError("inspector HTML exceeds its 16 MiB output limit")
    return payload


def _read_report(path: Path) -> Any:
    """Apply the same size, duplicate-key and finite-number rules to each input."""
    with path.open("rb") as handle:
        payload = handle.read(MAX_LEARNABILITY_INPUT_BYTES + 1)
    if len(payload) > MAX_LEARNABILITY_INPUT_BYTES:
        raise ValueError("inspector source exceeds its 16 MiB maximum input limit")
    try:
        report = json.loads(
            payload.decode("utf8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except json.JSONDecodeError as error:
        if len(payload) > MAX_INPUT_BYTES:
            raise ValueError("inspector source exceeds its 8 MiB input limit") from error
        raise
    if (
        type(report) is dict
        and report.get("role") == "final_test"
        and "benchmark_updates" in report
    ):
        if len(payload) > 4 * 1024**2:
            raise ValueError("final FIFO source exceeds its 4 MiB input limit")
    if len(payload) > MAX_INPUT_BYTES:
        from src.app.minigrid_learnability_inspector import PROTOCOL_IDS
        from src.app.minigrid_saved_fifo_inspector import PROTOCOL_ID as SAVED_FIFO_PROTOCOL_ID

        # Why this: explicit saved probe families can exceed8MiB when pretty printed.
        # Its compact app envelope still obeys the existing8MiB/500000-node bound.
        protocol = report.get("protocol") if type(report) is dict else None
        if type(protocol) is not dict or protocol.get("protocol_id") not in (
            *PROTOCOL_IDS,
            SAVED_FIFO_PROTOCOL_ID,
        ):
            raise ValueError("inspector source exceeds its 8 MiB input limit")
    return report


def write_key_door_inspector(
    source: str | Path, destination: str | Path, *, resource_artifact: str | Path | None = None
) -> Path:
    """Validate complete input before publishing one new file without overwriting."""
    source_path, output_path = Path(source), Path(destination)
    resource_path = None if resource_artifact is None else Path(resource_artifact)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("inspector output must differ from its source artifact")
    if output_path.exists():
        raise FileExistsError(f"inspector output already exists: {output_path}")
    if resource_path is not None and output_path.resolve() == resource_path.resolve():
        raise ValueError("inspector output must differ from its resource artifact")
    try:
        report = _read_report(source_path)
        if resource_path is None:
            projection = project_key_door_report(report, source_label=source_path.name)
        else:
            projection = project_key_door_report(
                report,
                source_label=source_path.name,
                resources=_read_report(resource_path),
                resource_label=resource_path.name,
            )
        rendered = render_key_door_inspector(projection)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("inspector source must be bounded UTF-8 JSON") from error
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(rendered)
    return output_path
