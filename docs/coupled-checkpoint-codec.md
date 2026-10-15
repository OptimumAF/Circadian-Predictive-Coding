# Complete coupled native snapshot bytes

Preview the fixed engineering example:

```powershell
python -m scripts.run_coupled_checkpoint_codec --describe
python -m scripts.run_coupled_checkpoint_codec > coupled-codec-example.json
```

The first command constructs no model. The full example uses seed 23, one CPU
thread and three families: PC, neutral CPC and active CPC. Each takes two warm
updates, round-trips complete native bytes, restores a compatible fresh model,
then compares one original/restored update exactly. It also checks checksum and
policy-binding refusal, detached decoded storage and exact PC/neutral tensors.
The fixed path costs 12 training calls and 48 energy evaluations; output is capped
at 2 MiB including printed platform line endings. Runs are local engineering
checks, not scientific comparisons. No new dependency or configuration is needed.

## Files and boundaries

```text
src/adapters/coupled_checkpoint_codec.py     bounded native byte schema
scripts/run_coupled_checkpoint_codec.py      runnable round-trip/continuation
tests/test_coupled_checkpoint_codec.py       independent schema/ownership checks
tests/test_coupled_checkpoint_codec_cli.py   actual example and description
```

The CLI depends on the codec adapter, which depends on existing core snapshot and
codec ports plus the shared adapter array-frame helpers. No core layer imports
the codec. The codec constructs no model, performs no training or IO and does
not restore runtime permissions, budgets, data roles, lifecycle or serving state.

## API and caller obligations

`CoupledCheckpointCodec` implements existing `CheckpointCodec[CoupledSnapshot]`:

```python
raw = codec.encode(snapshot, binding=binding, limits=limits)
restored_snapshot = codec.decode(
    raw, binding=binding, expected_sha256=known_digest, limits=limits
)
```

Pass an exact `CodecBinding` with independently known source/policy SHA256
strings and exact `CodecLimits` for encoded bytes, aggregate array bytes, hidden
layers and dimensions. The expected payload digest must also come from the
caller. Matching strings and hashes establish consistency with those inputs;
they do not establish provenance, ownership, consent or restart authority.

The CLI uses illustrative deterministic fixture-label digests, stated explicitly
in its output. They are not source seals. A production caller must provide its
own trusted binding/digest and fixed native/adapter configuration. Training rates,
inference policy, runtime cursors, cumulative allowances, lifecycle/retention and
single-owner authority are separate from native snapshot bytes.

## Complete native state

Format 1 covers every current `CoupledSnapshot` field: all hidden/output weights
and biases, the precision tuple and optional complete `CoupledWakeState`. CPC
stores every chemistry-policy scalar, chemical and plasticity vector, and wake
update count. Geometry is derived from parameters and encoded explicitly; the
scalar BCE output and all layer joins/shapes are checked.

Precisions and policy scalars must have native non-boolean numeric types and
their current native ranges. CPC chemistry must be finite and nonnegative;
gates must exactly match the existing native gate function under the stored
policy. Zero updates require zero chemistry and unit gates. Versions, dimensions
and counts require exact integers; counts are nonnegative and are not truncated.
Resource or standard JSON integer/parser limitations fail with `ValueError`;
the codec never changes global parser limits.

Canonical UTF-8 JSON holds exact typed fields and canonical base64 array frames.
Finite native float64 C/F contiguous arrays retain bytes, signed zero and storage
order; singleton axes use the unique canonical C tag. Unsupported strided arrays
are refused explicitly. Actual native snapshot outputs are contiguous. Decode
returns independent writable arrays; native restore copies them again.

Why this: existing legacy codecs do not cover the new coupled snapshot. Reusing
the bounded port and frame helpers provides complete native state transport with
a small explicit schema. A parameter-only or scalar summary would reset CPC
chemistry and could not reproduce continuation.

## Admission and failure

Encode derives topology, aggregate bytes and exact encoded-size bounds before
finite-array scans, copies or base64 work. Decode checks exact bytes and the
encoded cap before hashing/parsing, then validates every metadata/frame field,
all aggregate limits and canonical envelope before any frame payload decoding.
Missing/extra/duplicate keys, wrong versions/kinds, malformed numeric types,
topology/dtype/shape/order/base64, noncanonical JSON, wrong binding/digest,
nonfinite values, inconsistent CPC state or resource excess fail loudly.

The codec has no live model to mutate. Rejected bytes cannot publish state or
spend a runtime allowance. A valid decoded snapshot still needs the unchanged
native restore checks and the caller's policy/authority decisions. Complete
byte round-trip is a prerequisite for disk and process-crash recovery; it does
not prove durable writes, restart ownership, runtime cursor recovery or R3.7.

## Checks and extension

```powershell
python -m pytest -q tests/test_coupled_checkpoint_codec.py tests/test_coupled_checkpoint_codec_cli.py
python -m ruff check src/adapters/coupled_checkpoint_codec.py scripts/run_coupled_checkpoint_codec.py tests/test_coupled_checkpoint_codec.py tests/test_coupled_checkpoint_codec_cli.py
python -m ruff format --check src/adapters/coupled_checkpoint_codec.py scripts/run_coupled_checkpoint_codec.py tests/test_coupled_checkpoint_codec.py tests/test_coupled_checkpoint_codec_cli.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/coupled_checkpoint_codec.py scripts/run_coupled_checkpoint_codec.py tests/test_coupled_checkpoint_codec.py tests/test_coupled_checkpoint_codec_cli.py
```

Both Windows and Linux type-check receipts are retained with the scoped run.
For a new native state format, add an explicit versioned schema and full bounded
state tests; preserve existing fields and refuse unsupported formats. Compose
disk/actor recovery separately with existing authority and cumulative-budget
contracts. Do not make a checksum match grant permission to resume training.
