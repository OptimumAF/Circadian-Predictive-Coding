# Original payload copy leases

The internal source coordinator can reserve retained payload bytes while the
original budget gate is held. The public `reserve()` and `snapshot()` interfaces
retain their behavior. Reservations are cumulative; failed copying never refunds
them. Parameters, Python overhead and RSS remain outside this payload allowance.

## Modules

```text
src/app/payload_copy_budget.py          original monotonic budget and leased reservation
src/app/managed_lifecycle_capture.py    original common source gates, then metadata observation
tests/test_payload_copy_lease.py        admission, expiration and original-owner controls
```

`_lease_lifecycle_sources` yields the installed lifecycle, retained holder
observations and an optional reservation callable. All original driver, owner,
registry, holder, time, copy and sharing gates remain held until exit. A missing
copy policy yields `None`; it does not manufacture a new allowance. The existing
metadata-only captures take no reservation and invoke no clock or model ports.

Why this: a composite capture must size and admit copies during the same source
interval. A second acquire of the public budget gate refuses; releasing source
gates before reservation permits a different epoch. The callable expires on exit
and rejects another thread or a replaced original gate/policy. It is an internal
coordination aid, not a portable restore token or an arbitrary-code security
boundary.

## Budget-only example

```python
from src.app.payload_copy_budget import PayloadCopyBudget
from src.core.payload_bytes import PayloadCopyLimits

budget = PayloadCopyBudget(PayloadCopyLimits(16))
try:
    with budget._lease() as reserve:
        reserve(8)
        raise RuntimeError("copy failed after admission")
except RuntimeError:
    pass
assert budget.snapshot(0).charged_bytes == 8
try:
    reserve(1)
except ValueError:
    pass
else:
    raise AssertionError("expired reservation remained usable")
budget.reserve(8)
assert budget.snapshot(0).charged_bytes == 16
```

## Verification

Use a fresh unused pytest basetemp for each invocation:

```powershell
python -B -m pytest tests/test_payload_copy_lease.py tests/test_managed_record_checkpoint_codec.py tests/test_lifecycle_checkpoint_codec.py -q -o addopts= -p no:cacheprovider --basetemp=<new-path>
python -B -m mypy --platform win32 --no-incremental --cache-dir nul
python -B -m mypy --platform linux --no-incremental --cache-dir nul
python -B -m ruff check src tests scripts
python -B -m ruff format --check src/app/payload_copy_budget.py src/app/managed_lifecycle_capture.py tests/test_payload_copy_lease.py
```

## Next composition work

Full native/inbox/actor/sharing capture remains unfinished. Inventory supported
source graphs and cross-component aliases first. Perform the original consent,
retention, elapsed-budget and quiescence checks and bounded sizing before native
snapshot/copy operations; reserve on the original budget inside this source
lease. Build metadata after reservation so it records consumed charges. Bound
parameter copying separately, since this policy covers retained payload arrays.
Do not stitch separate captures, renew authority or claim model restore from this
prerequisite alone.
