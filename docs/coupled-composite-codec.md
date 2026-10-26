# Coupled composite checkpoint bytes

`CoupledCompositeCheckpointCodec` encodes the complete detached graph returned by
[coupled managed capture](coupled-managed-capture.md). Decode returns a
`ManagedCompositeState` data graph; original live authority stays outside the API.
For bounded disk persistence and a runnable verify/copy command, see
[complete checkpoint files](coupled-composite-files.md). It reuses the existing
generic atomic file store and preserves the same data-only authority boundary.

```python
from src.adapters.coupled_composite_codec import CoupledCompositeCheckpointCodec
from src.core.checkpoint_codec import CodecLimits

# capture_limits/capture come from the existing managed capture.
# binding/expected_sha256 are independently retained producer expectations.
codec = CoupledCompositeCheckpointCodec(capture_limits=capture_limits)
wire_limits = CodecLimits(262144, 65536, 4, 32)
raw = codec.encode(capture.state, binding=binding, limits=wire_limits)
decoded_state = codec.decode(
    raw, binding=binding, expected_sha256=expected_sha256, limits=wire_limits
)
```

The producer retains the bytes' digest separately. Values inside received bytes
cannot choose the expected source/policy binding, content digest or capture
limits. Matching tags establish agreement with supplied expectations, without
establishing provenance or restoration rights.

## State and identity

The graph includes complete learner settings/native state/geometry, current and
previous serving bundles, pending checkpoints/promotions, candidates and inbox
history, consent, retention and resource observations. `AuthorityPath` values are
inert observations. Copied ticket/receipt records lack original handle identity.

Aliases among records, containers, arrays and generators survive; scalar values
are retained by value. Distinct graph objects remain distinct. Arrays retain dtype,
endianness, raw bytes, shape, C/F order and read-only flags, with detached storage.
Mappings retain insertion order, including FIFO cache order. Tuples, lists, sets,
frozen sets, deques with maxlen and complete PCG64 state are supported; no random
draws occur. Deterministic structural ordering refuses ambiguous ties between
distinct identity-based unordered members instead of merging identities. Capture
still supports that subset; broader serialization qualification stays unfinished.

## Bounds and failures

Encoding checks complete coupled data, then admits metadata and exact array frame
lengths before copying array bytes. Array-byte and dimension bounds use the
smaller capture/wire limit. Wire `max_layers` bounds native hidden layers;
capture `max_depth` independently bounds graph traversal.

Decode bounds bytes before parsing and checks the complete node/reference graph,
schemas, depth, reachability and aggregate array sizes before array allocation.
Only allowlisted data records are constructed. Normal data policy validation and
complete graph checks precede publication; canonical re-encoding must match the
received bytes. Malformed or excessive data raises without returning partial state.

Bytes and decoded graphs are caller-owned copies outside the original lifecycle's
logical retained-raw ledger. The codec does not call that ledger, renew quotas,
construct learners or install state into a runtime. It uses no disk IO, new
environment setting or dependency. Decoding grants no recovery permission.

## Validation and extension

```text
src/adapters/coupled_composite_codec.py   public API, independent controls, wire admission
src/adapters/coupled_composite_graph.py   explicit data validation and alias projection
src/adapters/coupled_composite_wire.py    raw graph checks and bounded materialization
tests/test_coupled_composite_codec.py     independent frames, roundtrips and refusals
```

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_coupled_composite_codec.py
.venv/Scripts/python.exe -m ruff check src/adapters/coupled_composite_codec.py src/adapters/coupled_composite_graph.py src/adapters/coupled_composite_wire.py tests/test_coupled_composite_codec.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/coupled_composite_codec.py src/adapters/coupled_composite_graph.py src/adapters/coupled_composite_wire.py tests/test_coupled_composite_codec.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/adapters/coupled_composite_codec.py src/adapters/coupled_composite_graph.py src/adapters/coupled_composite_wire.py tests/test_coupled_composite_codec.py
```

Tests: `tests/test_coupled_composite_codec.py`. Exact bounded commands, lint/format/
type checks and outcomes: `artifacts/runs/coupled-composite-codec-20261015/RESULTS.md`.
Extend with an explicit wire version/allowlisted schema and independent raw-wire,
alias and allocation-bound tests. Preserve the app/recovery authority boundary.
See [ADR-0302](adr/ADR-0302-coupled-composite-data-bytes.md).
