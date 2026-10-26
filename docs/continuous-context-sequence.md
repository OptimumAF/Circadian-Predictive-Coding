# Cumulative continuous TRAIN contexts

`run_continuous_demo` keeps its original two-episode behavior by default. An
explicit observer or `additional_episodes` selects a bounded cumulative sequence:

```python
from src.adapters.continuous_stream_demo import run_continuous_demo

def observe(readout, checkpoint):
    # Read-only detached nearest learner. Report any paid native/probe work here.
    return {"phase": checkpoint["phase"]}

report = run_continuous_demo(
    warm_train_episode,
    next_train_episode,
    additional_episodes=(third_train_episode,),
    checkpoint_observer=observe,
)
if not report["complete"]:
    print(report["error"], report["work"])
```

## Boundaries and behavior

```text
src/adapters/continuous_stream_demo.py       existing public dispatch/default
src/adapters/continuous_stream_sequence.py   bounded sequence and detached readouts
tests/test_continuous_stream_sequence.py     arrivals, retention, isolation, failures
```

All episodes must be released TRAIN records with the same observation/action spec,
distinct identities and at most 256 total transition rows. `additional_episodes`
must be an exact tuple with zero to two entries. Native callers retain authority
for qualifying teachers. The adapter does not release labels or authorize studies.

One A-only nearest learner is fitted once. One existing actor/shadow runtime gets
one birth budget for one to three new packet updates. Packet `i` is observed at
`3*i`, polled without labels at `3*i+1`, and appended at `3*i+2`. Source and serving
actor keep their original state; the private candidate accumulates prototypes
without eviction or statistics refitting. Receipts and permanent charges are
checked before later work is admitted.

Checkpoints are `warm`, then `before_release` and `after_release` for each packet.
Metadata contains the current tick, packet index, current episode identity, known
episode identities, prototype rows and publication count. It contains no future
labels. Every callback gets a fresh fork; retained earlier readouts never acquire
later labels. Mutating a snapshot copy or metadata does not change the learner.
Mutating the readout itself refuses the sequence.

The callback returns a finite JSON dictionary of at most 256 KiB. The adapter owns
that result before checking native state. The default scope has at most seven
checkpoint records and nine tick events. It retains no unbounded chain of model snapshots.
Callbacks may retain their own copies; that external storage is outside the
adapter's memory report and must be budgeted by the caller.

Direct `run_sequence` callers may explicitly select `fixed_input=True` with
`maximum_releases=5` or `6` for already encoded inputs. Six releases give at most
13 checkpoint records and18 tick events, within the same256-row capacity.
Each invocation owns one prospective budget sized to its actual releases; it
does not renew an earlier session. Original default calls are unchanged.
[Complete B22 repair and retention](taskspace-memory-control.md).

Why this: the legacy report answers a single delayed-update question. Cumulative
contexts need several updates and native readouts under the same budget. A small
adapter composes the existing runtime rather than adding another runner.

## Failures and costs

Admission errors raise before fitting. Setup, execution and callback failures
return `complete=False`, preserve attempts and available paid output, and stop.
`retained_applied_on_failure` records observed retained receipts separately from
successfully returned receipts. `observed_native_on_failure` distinguishes native
publication counts from permanent budget charges: publication can precede an
exception that prevents a receipt. No retry, rollback or fabricated charge occurs.

`work` counts adapter fits, initializer attempts/returns, append attempts/returns,
validated updates, callback attempts/returns and additional validation
normalization rows. Callback computation belongs in its returned result. The
adapter performs no predictions of its own. Prototype byte counts describe named
numeric payloads, not whole-process memory or Python/snapshot/callback storage.

## Focused commands

```powershell
.venv/Scripts/python.exe -m pytest tests/test_continuous_stream_sequence.py tests/test_continuous_stream_demo.py::test_should_apply_one_complete_packet_only_at_tick_two_and_keep_actor_warm -q
.venv/Scripts/python.exe -m ruff check src/adapters/continuous_stream_demo.py src/adapters/continuous_stream_sequence.py tests/test_continuous_stream_sequence.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/continuous_stream_demo.py src/adapters/continuous_stream_sequence.py tests/test_continuous_stream_sequence.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent src/adapters/continuous_stream_demo.py src/adapters/continuous_stream_sequence.py tests/test_continuous_stream_sequence.py
```

Adding another context safely means supplying a predeclared released episode
within these bounds, then budgeting callback readouts. A different learner family
needs its own meaningful behavior requirement; this slice makes no BP/PC/CPC
comparison, generalization, promotion or release-readiness claim.

See [fixed protocol and receipts](../artifacts/runs/continuous-context-sequence-20261020/PROTOCOL.md)
and [native schedule](../artifacts/runs/continuous-context-sequence-20261020/NATIVE_PROTOCOL.md).

## Executed outcome and limits

The scoped sequence tests (25 including legacy), native-composition stubs (17)
and one late-state refusal case pass on Windows and pinned Linux in separate
groups. Final scoped statics and independent source/protocol/result review pass.

One fixed native RGB TRAIN17 sequence fitted A29, appended qualified mass29 at
tick2 and lighting29 at tick5, and ended with87 prototypes/two permanent charges.
Lighting changed from50controls/return0 to29/1; A and mass success were retained.
Mass already succeeded in27controls before its append and used30 afterward, so
that append demonstrated no benefit. Eight sampled joint6 upper violations in the
failed lighting run remain; all other sampled readouts had zero. The case used
282controls including teacher collection, not just optimization work.

This demonstrates exemplar-memory adaptation and retention at known contexts.
It establishes no neural/generalization/CPC advantage, deployment or physical
safety. Full camera/dynamics/constraint and release requirements remain open.
[Complete results, exact commands and costs](../artifacts/runs/continuous-context-sequence-20261020/RESULTS.md),
[independent acceptance](../artifacts/runs/continuous-context-sequence-20261020/REVIEW.md).
