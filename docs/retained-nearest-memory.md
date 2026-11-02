# Continued TRAIN memory within a fixed storage budget

The existing `run_sequence` API has an explicit fixed-input retention mode.
It keeps learning when arrivals exceed stored capacity, while counting every
arrived row as a paid exposure. The default remains append-only and refuses
overflow. This option supports only TRAIN, not reserved probes or final data.

```python
from src.adapters.continuous_stream_sequence import run_sequence
from src.core.replay_retention import ReplayRetentionPolicy

# warm and releases are your already-released distinct TRAIN episodes.
# Labels still arrive after two logical ticks; no automatic publication.
report = run_sequence(
    warm, releases, fixed_input=True,
    retention=ReplayRetentionPolicy("recent_fifo"), max_prototypes=128,
)
# Alternative declared before use:
# retention=ReplayRetentionPolicy("seeded_reservoir", 17)
```

Capacity is1..256, each packet1..256 rows, and the existing maximum3/5/6 releases
is unchanged. Smaller capacity requires explicit retention; retention requires
fixed inputs. This bounded runner keeps at most seven input packets/receipts,
not an unlimited success chain. The learner itself retains at most capacity
rows/IDs and fixed-size counters, including across many direct authorized updates.

`recent_fifo` evicts the oldest retained unique content. `seeded_reservoir` uses
the existing deterministic bottom-k content ranks and an explicit seed, without
process RNG or a seen-ID archive. A duplicate still present in memory is paid
again but keeps its earlier position. A discarded row may be offered again on
return. Feature/target conflicts are different content, and nearest-distance
ties still select the earliest retained row. Neither rule guarantees retaining
all contexts; the literal FIFO check explicitly forgets old content.

The distinct `NumpyRetainedNearestBCLearner`/`RetainedNearestSnapshot` inherit
ordinary prediction math. Snapshots own exact arrays, IDs, rule and all counters;
fork and restore preserve the full state. `training_examples` is cumulative
exposure, while `retained_rows`/`prototype_rows` report current storage. Every
exposure is either currently retained, discarded or a retained duplicate.
Counters stop at native63-bit bounds. Content is qualified from canonical
float64 feature/target bytes; no input standardization is fitted.

Reports separate numeric bank bytes, encoded hex-ID bytes64/row, digest payload
bytes32/row, selected incoming SHA payload including domain, discarded/duplicate
bytes, offered selection candidates and reservoir rank evaluations. Existing
prediction receipts retain distance-candidate/feature-visit costs. Validation
hashing/copies, temporary arrays, Python containers and RSS are additional.

Source and stable actor remain warm. Observers receive detached readouts and
owned metadata. Delayed labels, original receipts, budgets and source guards use
the existing actor/shadow runtime. Corrupt prior receipts stop further work.
Failures retain successful receipts, attempted rows and observed native state.
If an update commits but its return is lost, native counts can advance while
the successful runtime budget count remains0; `unreturned_append_possible`
is true, the inbox stops, and no successful charge or retry right is invented.

The exact type now has scoped known-TRAIN Pick/Push serving through the existing
frozen loops, with full-state/identity/callback closure and retained-row work.
Other optional modes, reserved probes and robot publication remain refused.
Final24 fake-native checks pass. After a preserved recording failure, a reviewed
qualification completed all eight fixed known-TRAIN readouts. FIFO succeeds on
Pick19; seeded reservoir17 succeeds on Pick17/Pick20. Both fail Push17 and neither
retains all four tasks. [Outcomes, full costs and review limits](reports/retained-memory-storage-study-20261101.md).
This is not native reliability, promotion or recovery acceptance. The ordinary
default memory demo remains intact.
[Commands, literal outcomes, costs, failures and precise review scope](reports/retained-nearest-memory-20261101.md).

```powershell
# Maintainers: do not rerun an occupied/SPENT qualification stage.
# New finite correctness work needs its own prospective scope and limits.
python -m pytest -q tests/test_retained_nearest_memory.py
python -m ruff check src/core/nearest_retention.py src/adapters/numpy_retained_nearest_bc.py src/adapters/continuous_stream_sequence.py tests/test_retained_nearest_memory.py
python -m ruff format --check src/core/nearest_retention.py src/adapters/numpy_retained_nearest_bc.py src/adapters/continuous_stream_sequence.py tests/test_retained_nearest_memory.py
python -m mypy --follow-imports=silent src/core/nearest_retention.py src/adapters/numpy_retained_nearest_bc.py src/adapters/continuous_stream_sequence.py tests/test_retained_nearest_memory.py
```

The recorder uses Windows process-memory APIs or standard POSIX getrusage.
Windows execution is qualified; POSIX is source/type-checked, Linux repeat unrun.
Tests can be run without an output-receipt environment
variable; the qualified wrapper additionally binds cumulative slot costs and
captures receipts. Scientific/native experiments require a separate frozen
protocol. Extend through the same learner/stream and preserve all older negative
results; do not select a favorable rule/seed from final evaluation.

The first-party [CPU/native memory demo](taskspace-memory-demo.md#explicit-bounded-retained-memory)
now exposes these same explicit retention options. CPU readout fidelity is scored
separately from simulator success; the append-only default remains strict.

One [fixed acquisition/return recipe comparison](reports/retention-rule-comparison-20261102.md)
reused the qualified FIFO128 result and ran the original reservoir17 rule once.
Final known-task success was4/5 versus3/5, while reservoir's new TRAIN fit was
worse and its old-return CPU scores stayed unchanged. The rule applied during
source preparation too, so learned starting banks differ. Only final native
endpoints were observed; native acquisition/recovery curves remain unmeasured.
All failed work, selected costs and CPU readouts are retained. These controls
keep their original behavior and defaults; no favorable seed/capacity selection
or circadian advantage follows from this one fixed contrast.
