# Original managed update observations

`src/core/native_update_origin.py` defines the typed reference record and
observer port. `src/app/native_update_origin.py` implements synchronous access
and releases its own references on close. Neither module stores replay rows,
certifies provenance, performs IO, or grants consent or restore permission.

Pass `native_observer=callback` to `ExperienceInbox.drain`,
`ActorShadowRuntime.train_ready`, `ResourceSharedRuntime.train_ready`, or
`ManagedExperienceOwner.train_ready`. Existing calls retain their defaults.
The callback receives `(stage, read)`; call `read()` synchronously on its original
thread. It exposes the original stored source, label, learner, actual detached
native input objects, learner version, original committed receipt, and current
original budget count. No additional payload copy or owner instance field is added.

| Stage | Meaning |
| --- | --- |
| `started` | Original pre-update checks passed; native work may follow. |
| `completed` | Original work count and receipt committed, before final resource checks. |
| `refused` | An admitted, detached invocation failed before `started`. |
| `uncertain` | Failure after `started`, with no committed receipt. |
| `committed_failure` | Failure with the original committed receipt retained. |

Admission refusal, unavailable pairs, sharing deferral, consent denial, copy
guard failure, or input-copy failure before the detached invocation produce no
callback. Invalid callbacks are refused by the inbox before its clock/payload
access. The original managed/sharing/runtime gates remain in force.

Why this: content hashes and matching payload values cannot identify a producer.
Observing the original update boundary is a prerequisite for a future persistent
replay ledger. `completed` alone is not final success: a resource stop or observer
fault can follow. An observer fault terminates the invocation; a failure callback
fault is grouped with the primary exception. Original receipts and spent work
are retained. No committed ID is retried.

The access handle works only during callbacks, expires on return, rejects other
threads, and releases its own observer/supplier references on final close.
Records returned to callbacks contain live references, not immutable payload
snapshots. Callbacks are trusted observational code: they must not mutate owners,
inputs or consent, block, or retain payloads without owned retention/accounting.
References deliberately retained by a callback remain that caller's responsibility;
closing the handle cannot revoke them. A fabricated record is not a certificate.

Persistent row lineage, retention/dedup/eviction/fork/checkpoint/promotion/restore
and erasure integration remain **R3.5b2e5b3**. The current port proves none of
those paths. No native replay training or recovery experiment ran for this increment.

## Pure reference example

This illustrates access lifetime with scalar references, without a model, budget,
runtime, sampler, worker, or native update. Real observations must come from the
original managed invocation.

```python
from src.app.native_update_origin import NativeUpdateAccess
from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from src.core.native_update_origin import NativeUpdateOrigin

source = Experience("s", "e", 0, "actor", "x", "train", ExperiencePermissions(True, True))
label = LabelArrival("label", "s", "e", 0, "actor", "y")
origin = NativeUpdateOrigin(object(), source, label, "x", "y", "candidate", None, 0)
handles = []

def observe(stage, read):
    assert stage == "started" and read() is origin
    handles.append(read)

access = NativeUpdateAccess(observe, lambda: origin)
access.notify("started")
access.close()
try:
    handles[0]()
except ValueError:
    print("expired original callback access")
else:
    raise AssertionError("closed access remained usable")
```

## Verification and extension

Run the fixed scoped command recorded in
`artifacts/runs/r35b2e5b3a-update-origin-20261008/command-006.json`; it includes
new fake controls, five selected existing fake inbox controls and 990 current
composite/resource/codec controls. Native-dependent inbox parity tests are not
selected. Use `python -B -m mypy --platform win32 --no-incremental --cache-dir nul`
and the corresponding `linux` command; use Ruff check and format check on changed
sources. The development log records exact outcomes, failures and scope counts.

Add a separate bounded replay-write/row contract next. Bind writes to this
original invocation, preserve original consent references and terminal failures,
and use weak references or owned retention. Do not reconstruct lineage by hashes
or turn the observation record into restore permission.

Final verification:command-014 reruns all16 origin controls after the test-only
supplier lint correction;commands015/016 are final Windows/Linux742file typing;
017wholelint/018sevenformat/019acceptance/020closure/021readback all pass.


Original replay-write observation (ADR0239):core/replay_write_origin defines
a bounded original model/input window;app/replay_write_origin composes it with
the original managed producer. Native copy ranges,new snapshot references and
final retained identities are observed without extra array copies/native fields.
Explicit original ContextVar token/thread/callback lifetime survives refused
close;all default storage/policy/RNG rules preserved. See docs/replay-write-origin.md.
Pure/current gates precede a separately declared tiny native storage-only parity
fixture. No dependency/env/config changes. This is not a persistent row ledger or
consent/restore certificate. Full b3/e5b/e5 retained variants/lineage/bytes/recovery
remain open. Next consume actual copy identity under bounded weak/owned-accounted
retention and original consent/terminal-outcome authority before broadening capture.
