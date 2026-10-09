# Native state copy observation

`src/core/native_state_copy.py` observes the actual dictionary deepcopies in
Circadian `snapshot_state` and `restore_state`. The source must be the original
dictionary: `model.__dict__` for snapshot or `snapshot.state` for restore.
The two callbacks run before allocation and after the actual copier returns.
`lookup(original)` uses that copier's memo, including replay row and array aliases.
Each borrowed reader expires when its callback ends. Retained references require
the caller's own admission and accounting.

Why this: restore creates new replay row identities, so a constructor witness
cannot establish their origin. State and constructor observation use independent
context channels with the same bounded, thread/context checked implementation.
Neither channel renews the other's allowance. Unobserved copies retain the
existing `deepcopy(source)` path.

```python
from src.core.native_model_copy import ModelCopyLimits
from src.core.native_state_copy import copy_native_state, observe_state_copies

source = {"history": []}
source["alias"] = source["history"]
stages = []

def observe(stage, read, lookup):
    assert read().source is source
    if stage == "copied":
        target = read().target
        assert lookup(source) is target
        assert lookup(source["history"]) is target["history"] is target["alias"]
    stages.append(stage)

with observe_state_copies(source, observe, ModelCopyLimits(1, 2, 8)):
    copied = copy_native_state(source)
assert copied["history"] is not source["history"]
assert stages == ["before_copy", "copied"]
```

Run scalar controls with
`python -B -m pytest tests/test_native_model_copy.py tests/test_native_state_copy.py`.
Run separately budgeted native controls with
`python -B -m pytest tests/test_native_state_copy_integration.py`.

A `copied` callback observes a temporary restored dictionary **before validation
and publication**. Successful observation does not prove successful restoration.
The native method retains its original validation and final `__dict__` update.
Checkpoint outer copies, inbox/source/label/receipt copies, retained holder
enrollment, original ledger anchor transition and handoff still require their own
observed chain. This primitive grants no copied-holder capture or consent.
