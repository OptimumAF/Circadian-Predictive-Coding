# Local experience and arrival contracts

R3.2 connects declared training events to the existing native learner step.
It preserves each learner's own loss, tensor layout, model state and settings.
This is a trusted local software boundary; role tags, clock values and version
names do not establish physical provenance or release scientific final-test data.

## Modules and direction

```text
src/core/experience.py          immutable metadata, permissions and logical clock
src/app/experience_inbox.py     bounded delivery, eligibility and native updates
src/app/learner_step.py         existing budget boundary plus optional lifecycle hooks
tests/test_experience_contracts.py  schema, clock and permission rules
tests/test_experience_inbox.py      delivery, isolation, failures and native parity
```

App depends on core and the existing app budget/learner step. Core uses the
standard library and the existing native diagnostic contract. No module here
imports infra, reads files, scores evaluation roles or constructs datasets.

`Experience` carries episode/sample identity, actor version, source observation
tick, opaque features, role and permissions. Optional candidate IDs retain their
declared order; an action must name a candidate. Signed finite rewards are metadata,
not training targets. `LabelArrival` has a globally unique event ID within an inbox,
the matching episode/sample/actor/role and an independent arrival tick and target.
`AppliedExperience` records the original actor tag, caller's learner tag, observation,
label and application ticks, completed budget ordinal and native diagnostic.
Version names are identifiers, not parameter hashes or promotion certificates.

## Clock and delivery policy

`LogicalClock` advances explicitly through nonnegative integer ticks. Equal ticks
are legal; backwards movement, booleans, floats and invalid values are rejected.
An injected `EventClock` must also be monotonic at observed drains. Event ticks
are separate from the existing budget's wall-clock seconds.

Sources and labels may be delivered in either order, including labels before their
matching source. Both declared arrival times must be reached before learning.
Labels preceding their source observation or contradicting its actor version/role
are refused before registry changes. Each drain orders eligible pairs by label
arrival, source observation, episode ID and sample ID. Late transport delivery
applies at the current clock tick; it cannot reorder or replay completed work.
Thus a complete ready set has transport-order-independent ordering, while streaming
history also records when each delivery was actually processed. No global watermark
or retroactive replay is implied.

Episode/sample pairs are unique for the inbox's lifetime. Duplicate sample pairs,
label pairs and reused label event IDs are rejected, including after consumption.
Conflicting submissions cannot overwrite an earlier event. `max_experiences`
bounds the union of source and label identities, including orphan labels and
completed history. This is a record-count limit, not a byte or RSS guarantee.
Payload sizes and resource limits need later R3 composition. Completed records
retain their detached inputs until the bounded inbox is disposed.

## Permissions and ownership

Only `train` experiences with explicit `training=True` and train-role labels enter
the learning inbox. `inner_guard`, `outer_selection` and `final_test` can be
represented as metadata, but cannot grant training/replay permission. Their
registration is denied before copying their payloads or calling a learner.
Evaluation and replay flags are recorded; this module implements neither operation
and grants no global final-label release. Existing role seals and scientific
source/native guards remain independent and unchanged.

Accepted records are deep-copied at registration. Learners receive detached copies
again, so caller or learner input mutation cannot change the buffered event.
Payloads are trusted local Python objects: copying may execute their custom copy
methods. This is not an untrusted decoder, data-source sandbox or alias/provenance
attestation. Do not relabel evaluation records as training records.

## Completion and failure

The shared `update_learner` keeps its existing budget behavior. Optional `on_started`
runs after the pre-update budget check, immediately before native training;
`on_completed` runs after the native return and budget counter commit, before the
post-update resource check. Existing callers omit both and behave identically.
Hooks are synchronous trusted app bookkeeping; their exceptions propagate.

The inbox records completed identity before a late resource stop. That label never
becomes retryable, and its native diagnostic remains in `applied_updates` even when
`drain()` raises. A pre-update budget refusal performs no training and leaves the
pair pending. An unfinished native call that raises any exception, including the
budget exception type, stops the inbox because its model may be partly mutated.
Other unexpected update errors also stop it. Restore/disposal is an explicit
caller decision; no automatic rollback is claimed. Nested drains or registrations
during training are refused.

Why this: a completion-only hook cannot distinguish a pre-update budget refusal
from the learner raising the same exception after partial mutation. The start
hook makes that distinction without duplicating native/budget orchestration.

## Example

```python
from src.app.experience_inbox import ExperienceInbox
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock

clock = LogicalClock()
# learner implements NativeLearner; budget is the existing ToyBudgetSession.
inbox = ExperienceInbox(learner, clock=clock, budget=budget,
                        learner_version="candidate-0", max_experiences=8)
inbox.record_experience(Experience("sample-1", "episode-1", 1, "actor-0",
                                  features, "train", ExperiencePermissions(training=True)))
inbox.record_label(LabelArrival("label-1", "sample-1", "episode-1", 3,
                               "actor-0", targets))
clock.advance_to(2)
assert inbox.drain() == ()
clock.advance_to(3)
updates = inbox.drain()
```

Verify with `python -m pytest -q tests/test_experience_contracts.py tests/test_experience_inbox.py`.
The native integration fixtures use the same fixed seed/settings as existing port
tests and compare complete model state; they are not a performance experiment.
No environment variable, external service or dependency is added. Extend through
an outer native adapter, or separately compose R3.3 actor/shadow ownership and
R3.4 promotion; preserve the arrival/role/duplicate rules and complete failure tests.
