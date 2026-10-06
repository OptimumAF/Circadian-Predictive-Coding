"""Validate every provided V1 runtime field/child and representable identity join.

Inputs are full decoded JSON and immutable requested endpoint names. A successful
return validates structure only. Provenance, actual process completeness, omitted
unreferenced objects, live ownership, source version and execution require ports.
"""

from __future__ import annotations

from typing import Any

from src.core.runtime_native_schema import validate_runtime_native
from src.core.runtime_python_schema import validate_runtime_endpoints, validate_runtime_python
from src.core.runtime_record_schema import RuntimeSchemaBindings, require_runtime, runtime_object


def validate_runtime_payload(body: Any, entrypoints: tuple[str, ...]) -> None:
    row = runtime_object(
        body, {"schema_id", "python", "native", "entrypoints", "source_attestation"}, "body"
    )
    require_runtime(
        type(row["schema_id"]) is str and row["schema_id"] == "p67_observed_process_runtime_v1",
        "body",
        "schema differs",
    )
    require_runtime(
        row["source_attestation"] is False, "body", "cannot claim source-version attestation"
    )
    bindings = RuntimeSchemaBindings()
    modules = validate_runtime_python(row["python"], bindings)
    validate_runtime_native(row["native"], bindings)
    validate_runtime_endpoints(row["entrypoints"], entrypoints, bindings, modules)
