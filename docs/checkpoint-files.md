# Bounded checkpoint file transport

The new file store composes an existing `CheckpointCodec[State]`. It stores raw
codec bytes and validates a caller-known digest before decoding. Existing legacy
pickle stores and artifact-directory publication are unchanged.

```powershell
python -m scripts.run_coupled_checkpoint_files --describe
python -m scripts.run_coupled_checkpoint_files --directory artifacts/my-new-checkpoint-example
```

Description performs no operational IO or model work. The actual example
requires a new dedicated directory and refuses an existing one before constructing
models. It leaves three checkpoint files for review. PC, neutral CPC and active
CPC each warm twice, save/load checkpoint A, continue original/restored models
exactly, refuse an injected fsync failure while preserving A, then successfully
replace it with B. Stale digest, wrong binding and excessive encoded size refuse.
The fixed path costs 12 training calls and 48 native energy evaluations, CPU1;
JSON output is capped at 2 MiB including platform/final line endings.

## Structure and API

```text
src/infra/checkpoint_files.py                  generic bounded file store
scripts/run_coupled_checkpoint_files.py        actual native file example
tests/test_checkpoint_files.py                 independent IO/continuation controls
tests/test_coupled_checkpoint_files_cli.py      description/directory/example checks
```

Infra imports only the existing core codec port and standard library. Concrete
native codecs and models are selected at the outer CLI. No inward layer imports
file storage, and the store does not inspect model structure or construct models.

```python
store = BoundedCheckpointFileStore(path, codec)
digest = store.save(snapshot, binding=binding, limits=limits)
decoded = store.load(binding=binding, expected_sha256=digest, limits=limits)
```

`path` is a read-only, lexically absolute `Path`, bound at construction without
filesystem resolution. A nonempty filename is required. The constructor performs
no file IO. Supply an exact `CodecBinding` and `CodecLimits`; they are validated
and copied per operation. The encoded byte ceiling is captured before a trusted
codec callback can mutate its own argument copies. Codec semantics, array limits
and decoded-state ownership remain the injected codec's responsibility. The
[coupled codec](coupled-checkpoint-codec.md) returns complete detached snapshots.

The expected digest and binding must come from an independently known caller
context. Never treat values read from the same untrusted checkpoint as restart
authority. The CLI's deterministic fixture-label bindings illustrate consistency;
they are not source seals, consent, ownership or permission to resume training.

## Save and failure recovery

Encode and require exact bounded bytes before any filesystem mutation. Compute
the digest, inspect the target, create a temporary file beside it, require a full
exact integer write count, then flush, fsync and close before atomic replacement.
Static links, directories and other nonregular targets refuse. A second target
check precedes replacement. Empty bytes are valid transport when their codec
accepts them; short, boolean or otherwise invalid write counts refuse.

The store retains sole raw-descriptor ownership with `fdopen(closefd=False)`.
This matters because buffer construction can fail after closing an internal
stream. Stream close and one explicit raw close complete before publication.
Raw-close tracking is cleared before the attempt; an ambiguous close error is
never retried against a potentially recycled descriptor number.

Precommit failure or cancellation preserves the previous checkpoint (or leaves
an absent target absent) and cleans only owned temporary resources. Parent
directories created for the operation may remain. Cleanup failure adds a useful
note to the original exception, including an orphan temporary path when needed.
The target is never deleted as cleanup. Successful replace is the final fallible
publication step; no later cleanup can falsely report that success as failure.

## Load and limits

Validate controls/digest before IO. Inspect the named target with `lstat`, reject
nonregular/static links, open and check `fstat` identity/size, then read only the
admitted actual size plus one sentinel byte. Reject short, oversized, changed or
incompatible content before codec decode. Post-read identity/size checks and
close complete before digest comparison and decode. A very large allowed ceiling
does not cause an oversized read request for a small actual file.

On this Windows Python runtime, `lstat` and `fstat` report different `ctime`
values for the same unchanged replaced file. The cross-API check therefore uses
device, inode, size and modification time on Windows. Both observations of the
opened descriptor still compare all fields including `ctime`; other platforms
also retain `ctime` across inspection and open. Expected byte hashes remain
mandatory. This avoids rejecting valid files while preserving comparable change
checks under the documented trusted-parent and quiescence requirements.

Invalid controls, types, bounds, file kind/change or digest produce `ValueError`.
Filesystem faults retain useful `OSError` subclasses/context; codec errors
propagate. Neither path retries or restores a runtime allowance.

## Scope and extension

Calls require quiescent input, a trusted parent and one cooperating writer. This
is atomic visibility on the same filesystem, not a hostile-filesystem race
defense, lock/CAS, directory-fsync or power-loss guarantee. OS-ambiguous replacement
outcomes, concurrent writers, full runtime cursor/lifecycle/permission/budget
reconstruction and durable single-owner restart remain separate requirements.
Correct model file transport alone cannot close R3.7 or claim process-crash
recovery of the adaptation runtime.

```powershell
python -m pytest -q tests/test_checkpoint_files.py tests/test_coupled_checkpoint_files_cli.py
python -m ruff check src/infra/checkpoint_files.py scripts/run_coupled_checkpoint_files.py tests/test_checkpoint_files.py tests/test_coupled_checkpoint_files_cli.py
python -m ruff format --check src/infra/checkpoint_files.py scripts/run_coupled_checkpoint_files.py tests/test_checkpoint_files.py tests/test_coupled_checkpoint_files_cli.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/infra/checkpoint_files.py scripts/run_coupled_checkpoint_files.py tests/test_checkpoint_files.py tests/test_coupled_checkpoint_files_cli.py
```

Scoped Windows/Linux checks and exact IO/native receipts are retained in
artifacts/runs/coupled-checkpoint-files-20261015. Extend through explicit codec
composition and fault controls, preserving caller-owned authority/cumulative
budgets. Never route these bounded native bytes through generic object loading.
