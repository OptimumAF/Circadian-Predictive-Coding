# Original native model-copy observations

`src/core/native_model_copy.py` supplies an explicit synchronous window around
`CircadianLearner` constructor copies, including the constructor called by `fork`.
The NumPy adapter supplies the actual `deepcopy` memo. Native model and learner
fields, snapshots, training policy, and the unobserved copy path stay unchanged.

The observer receives `before_copy` before allocation and `copied` after copying.
`read()` returns the original source and actual target. `lookup(original)` reads
the copier's identity memo; it cannot select a row by content. Each saved reader
expires after its own callback, including while later callbacks run. Windows
refuse nesting, reuse, reentry, wrong sources, foreign threads/contexts, exhausted
attempt/notification/read limits, and further copies after an uncertain attempt.

Why this: consent must follow an actual copy at its creation boundary. A later
matching payload cannot establish which original row produced it.

```python
from src.core.native_model_copy import ModelCopyLimits, observe_model_copies

# Window creation itself copies nothing and grants no consent or capture access.
source = object()
scope = observe_model_copies(source, lambda stage, read, lookup: None,
                             ModelCopyLimits(1, 2, 8))
with scope as window:
    assert window._copies == 0
assert window._source is None
```

Observers are trusted local ports. Borrowed raw references retained by a caller
need separate owned retention accounting. Lookup limits bound observation, not
the size of a whole native model copy or Python heap. Before-copy callbacks must
eventually enforce the original lifecycle, raw-copy, metadata, work and time
budgets before a managed retained copy is authorized. This primitive has no
payload accounting, receipt or consent decisions, holder enrollment, checkpoint,
promotion, restore, persistence, cleanup, or scientific authority. Unwitnessed
retained holders continue to fail replay capture. Full d2 acceptance is open.
