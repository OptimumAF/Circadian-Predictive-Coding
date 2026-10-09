# Replay admission before composite capture

`capture_numpy_managed_composite(owner, limits=..., replay_origins=ledger)` now
checks replay before native projection or raw copying. Empty replay remains
compatible without a ledger. Nonempty current-candidate replay requires the exact
original `ManagedReplayOrigins`, enrolled after lifecycle installation and before
training. Copied replay in actors, forks, checkpoints or other retained holders
refuses until an original holder lineage witness is qualified. Matching contents
or hashes do not grant that permission.

## Structure and boundaries

```text
src/app/replay_capture_origins.py          mandatory original replay admission
src/adapters/numpy_replay_capture.py       bounded native holder/row inventory
src/app/managed_replay_origins.py          leased ledger validation without lock reentry
src/app/managed_composite_capture.py       checks before/after original capture callbacks
tests/test_replay_capture_origins.py       fake original ledger refusal/lease controls
tests/test_native_replay_capture_origins.py separately budgeted actual trained-row capture
```

The inventory port returns actual model and row references, including native
snapshot state inside original `CandidateCheckpointView` wrappers. It allocates
bounded reference tuples, without native operations, projection, array copies or
payload hashes. The app imports no adapter. Injected inventory/project/copy ports
are trusted observational code, not a security boundary against malicious Python.

Capture already leases original owner, holders, runtime, registry, time, sharing,
copy budget and sampler. Its ledger lease takes only the ledger's nonblocking gate;
it never calls public `origins()` or reacquires owner/runtime/resource locks. Reads
check original consent, receipts, age, weak identities and bounded payload integrity.
Checks expire on exit and reject foreign threads, reentry and exhausted allowances.
Repeated inventory checks protect against changes in trusted callbacks before
measurement/copy and after final validation. No portable row origin is emitted.

Each ledger capture attempt charges one invocation and 1024 metadata accounting
units from the same original lifetime allowance used for training. Failed attempts
never refund or renew those charges. There are at most eight inventory checks per
capture and the original write-window notification cap also bounds ledger checks.
The original payload copy budget and final resource checks remain responsible for
copy/work/wall/RSS admission. Accounting units are not a physical heap/RSS ceiling.
No new native fields, schemas, dependencies, environment variables or defaults.

Why this: the prior synthetic replay alias fixture supplied no producer evidence.
It now explicitly refuses before projection/copy; empty CPC and Backprop graph
alias tests remain positive. A separate real trained-row integration qualifies
current-candidate copy admission. This is a correctness restriction on replay
capture, not a weakened producer criterion or a retained-holder lineage grant.

## Pure missing-origin example

No owner, model, arrays, worker or native operation is created.

```python
from types import SimpleNamespace
from src.app.replay_capture_origins import lease_replay_capture

marker = object()
def inventory(groups, limits):
    return ((marker, (object(),)),)

try:
    with lease_replay_capture(None, (), SimpleNamespace(max_nodes=2), inventory, None):
        raise AssertionError("untracked replay reached copy")
except ValueError as error:
    assert "nonempty replay" in str(error)
print("untracked replay refused before copy")
```

## Verification and next work

Exact pytest, both-platform mypy, Ruff lint/format, guide, cache, source and resource
commands and their outcomes are in the development log and local stage receipts.
Native integration runs only after the initial gates and a separately declared
allowance. Failures and spent fixture budgets are preserved.

Next establish original retained-holder copy witnesses at actual creation and
qualify actor/fork/checkpoint/promotion/restore/erase paths, every native compound
variant and alias history. Full R3.5b2e5b3d/b3/e5b/e5, canonical bytes, live/disk/
model/coordinator-loss recovery and scientific gates remain unchecked.


Qualification status: native capture timed out in d1. The leased-consent source
correction is pending fresh regression/type/native qualification. d1 remains
unchecked; pre-timeout green evidence does not qualify current source.


## Leased lifecycle repair qualification

Capture uses the original lifecycle's leased retention accessor for both runtime
open checks and row declaration checks. The runtime check retains original
retired/stopped checks and requires original registry/lifecycle identity. Ordinary
public ledger operations retain the ordinary lifecycle consent path. Capture
already owns the time gate; using its leased accessor preserves elapsed retention.
An installed-lifecycle/fake-learner regression patches class ordinary elapsed to
fail immediately while holding that gate. It covers valid capture/public reads,
revocation/optout,logical and elapsed expiry,rewind,changed guards and cleanup
failure. Pre-open/guard refusals have no new capture invocation charge; admitted
source refusals preserve the spent invocation/metadata charge. No native defaults,
instance schemas,environment variables or dependencies change.

Why this:the first real-lifecycle regression showed an earlier reentry through
runtime._require_open,with an exact traceback. Both earlier source diagnosis and
native timeout remain preserved. Full retained-holder lineage and recovery remain
unfinished. See new repair-stage receipts for current qualification;earlier failed
stage evidence remains historical.


## Current qualification — mandatory replay admission accepted

R3.5b2e5b3d1 now passes1099unique controls,both757type targets,static/guide/source/
cache/resource gates. Native capture completes with original elapsed-retention
tripwire;missingledger and copiedactor rows refuse before projection/copy. Old
unvalidated/failed notes above are preserved historical handoffs,not current status.
Original55s timeout stays failed/countsunknown. Actual freshnative4wakes20copies;
ordinarypublic default behavior preserved. Original retained-holder copy issuance,
fullvariants/history/bytes/recovery/scientific gates remain unfinished. See latest
development log and repair-stage evidence;next d2 creates actual holder witnesses.
