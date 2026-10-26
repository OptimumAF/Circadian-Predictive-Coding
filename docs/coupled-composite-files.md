# Complete coupled checkpoint files

The file CLI composes the complete coupled data codec with the existing
`BoundedCheckpointFileStore`. It copies a verified checkpoint to a new dedicated
directory and reads it back before reporting success. It performs no training,
native model construction, capture, or runtime restoration.

```powershell
.venv/Scripts/python.exe -m scripts.run_coupled_composite_files --describe
.venv/Scripts/python.exe -m scripts.run_coupled_composite_files `
  --source C:/checkpoints/received.json --directory C:/checkpoints/verified-copy `
  --source-sha256 $knownSourceSha256 `
  --policy-sha256 $knownPolicySha256 `
  --expected-sha256 $knownCheckpointSha256
```

Set the three variables to independently known lowercase SHA256 strings.
Obtain expectations from the producer separately. Values inside received bytes
cannot supply the expected binding, digest or limits. Hash agreement does not
establish provenance, trust, live authority or recovery rights. The output JSON
contains paths, a digest and `data_only: true`; it does not dump retained payloads.

The default finite bounds match the six-case engineering example: capture
records64/identifier128/nodes8192/depth64/array65536/dimension32; wire
encoded262144/array65536/layers4/dimension32. `--help` lists explicit overrides.
The complete capture limits must match the producer's independently known limits;
wire caps may be stricter. An invalid control or refused input creates no output
directory. Any existing output path, including a broken symlink, is refused.

## Save from the live managed-serving demo

The optional live mode captures complete B state after handoff, continues the
original owner to C against the same matched native reference, commits C, and saves
a second complete graph. It then runs the existing serving expiry and cleanup.
It uses fixed PC/neutral/active families and fixed/guarded policies, with no final
evaluation or scientific comparison.

```powershell
.venv/Scripts/python.exe -m scripts.run_coupled_candidate_handoff --describe --complete-checkpoints
.venv/Scripts/python.exe -m scripts.run_coupled_candidate_handoff --run --complete-checkpoints --directory C:/checkpoints/new-live-demo
```

Run requires an unused directory; it creates twelve files. Describe performs no
IO/model/thread work and takes no directory. Other experiment controls cannot be
combined with this mode. Output includes each file's digest and data-only receipt.
The fixture binding labels are illustrative engineering inputs, not provenance.

Why this allowance: existing committed serving spends656 logical raw-copy bytes;
B and C full captures reserve328 and512 more. This mode declares1496 at creation.
The original copy budget remains the same object throughout; failed storage never
refunds capture, and cleanup clears retained raw bytes without refunding1496 spent.
Existing modes keep their640/656 allowances. This ledger excludes native parameters,
temporary Python allocations and OS RSS; it is not a physical memory bound.

For an already installed supported paused owner:

```python
from pathlib import Path
from src.adapters.coupled_live_checkpoints import save_coupled_managed_checkpoint

receipt = save_coupled_managed_checkpoint(
    owner, Path("complete.json"), binding=binding,
    capture_limits=capture_limits, wire_limits=wire_limits,
)
```

This calls original capture once, saves and reads back data through the existing
codec/store. It does not install decoded state. The original owner must already
have sufficient lifetime copy admission. Files and loaded caller copies survive
original lifecycle cleanup and remain the caller's responsibility. Successful
files may remain if a later demo phase fails. Original authority, retention,
consent, resource, policy and process-recovery restrictions remain intact.
Evidence: `artifacts/runs/coupled-live-checkpoints-20261015/RESULTS.md`.

## Copy an existing complete file

```python
from pathlib import Path
from scripts.run_coupled_composite_files import copy_checkpoint

report = copy_checkpoint(
    Path("received.json"), Path("new-copy"), binding=binding,
    expected_sha256=expected_sha256,
    capture_limits=capture_limits, wire_limits=wire_limits,
)
```

For an already captured state, use the store directly; no additional storage
adapter is required:

```python
from src.adapters.coupled_composite_codec import CoupledCompositeCheckpointCodec
from src.infra.checkpoint_files import BoundedCheckpointFileStore

codec = CoupledCompositeCheckpointCodec(capture_limits=capture_limits)
store = BoundedCheckpointFileStore("checkpoint.json", codec)
digest = store.save(capture.state, binding=binding, limits=wire_limits)
data = store.load(binding=binding, expected_sha256=digest, limits=wire_limits)
```

Why this: the existing generic store already bounds actual reads, verifies bytes
before semantic decoding, and stages/flushes/closes files before atomic replacement.
Sharing that implementation preserves its tested descriptor and failure behavior.

## Failure and authority limits

The source is unchanged. A failed store replacement preserves the previous regular
file and cleans only its own temporary file where cleanup succeeds. A new CLI
directory is not a transaction: after a successful save, later read-back or report
failure can leave a valid checkpoint for inspection. Files require a trusted
parent, quiescent input and one cooperating writer. There is no lock/CAS,
hostile-filesystem race defense or power-loss durability guarantee.

Loaded `ManagedCompositeState` is detached data. It carries no live owner,
callbacks, authority identity, resource grants, restore permission or recovery
completion. Disk and decoded copies are caller owned, outside the original
lifecycle retained-raw ledger. The codec's explicit unordered structural-tie
refusal and full-wire depth bounds still apply. See
[codec limitations](coupled-composite-codec.md).

## Tests and extension

`tests/test_coupled_composite_files.py` covers full graph copies, refusal ordering,
CLI controls and real file-fault injection. Generic file-store regression tests
remain in `tests/test_checkpoint_files.py`. Exact bounded checks and saved-graph
integration evidence: `artifacts/runs/coupled-composite-files-20261015/RESULTS.md`.
Extend explicit CLI controls and behavioral tests; preserve independent expected
values and reuse the generic store rather than adding a parallel file mechanism.
