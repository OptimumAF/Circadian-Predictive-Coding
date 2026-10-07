# ADR-0154: Validate confirmation JSON in explicit stages

Date: 2026-09-30. Status: b2a/b2b verified; b2c scientific execution unfinished.

## Context

B1 verifies exact live trajectories, but saved JSON needs independent checks.
Schedule/combined/parent validators have reusable pure per-seed seams; their
whole-result gates fix three development seeds. Gating/replay/sleep validators
live in CLI adapters, which the new app layer must not import. Complete
checkpoint fingerprints also carry duplicate clocks/retention/selector views
and canonical state tags that need schema/type/link checks, not merely hashes.

## Decision

Split b2 into b2a exact envelope/role/checkpoint schema validation, b2b every
raw family cost/guard/supply/selector link, then b2c bounded artifact execution
and two full reserved runs. Preserve all parent criteria and frozen scope.
No partial stage is a scientific success or a production publication gate.

B2a is pure: it reads finite JSON/configuration metadata, derives seeded
initial tensor fingerprints with local NumPy RNG, and verifies canonical
state inventories/configuration/capacity/views. It constructs no dataset or
model and computes no score. Test first existing development sources and
synthetic complete envelope metadata without reserving any scientific seed.
Synthetic data are schema fixtures, never training/confirmation evidence.

B2b separately validates raw facts and derives costs, composing original
pure per-seed seams without changing frozen sources or accepting partial
scientific scope. Gating/replay/sleep app checks are independent of their CLI
adapters. B2c binds the exact saved scope/reference/source identities and
enforces resource/artifact lifecycle gates before any reserved execution.

## Alternatives

Importing CLI validators into app reverses the dependency direction. Editing
old validators changes scientific source pins. Treating a matching digest as
proof of a coherent state allows self-consistent metadata forgeries. One
monolithic validator mixes schemas, family rules and IO and is hard to audit.

## Consequences

After-trained tensor fingerprints cannot reconstruct unseen tensor values;
independent JSON checks prove schemas and declared links. The future worker
must also compare saved full facts and every live held checkpoint before
scoring, with source/reference identity and resource gates. Every raw cost,
rejection and family endpoint remains required by b2b/c and the original
parent. No final value or reserved model/source is opened during b2a fixtures.

## Evidence and next schema dependency

All six actual development fixture bodies (56 model cells, initial/A/B and
supplemental guard snapshots) pass independent validation with source/model/
train/score functions forbidden. **51 new/125 related tests**, zero skipped,
Ruff/mypy (375 files)/format/diff pass. Whole-scope order and all sixty reserved
metadata rows are separately tested with a delegation spy, without a model
or dataset; this is not a 560-cell training claim. Resealed state/view clocks,
seeded initial parameters/RNG, configuration, lineage, selector, role IDs,
canonical ambiguity/type/nesting and rollback forgeries are rejected.

Gating/sleep raw parameter hashes have different prefixes from the uniform
confirmation/replay hash. Before implementation, split b2b into b2b1 explicit
parameter contracts/links and b2b2 remaining work/controller/full state checks.
The unrun checkpoint schema retains its uniform field and adds an exact
three-contract map. Capture each original digest from live tensor bytes;
independently derive seeded initial hashes and bind available raw endpoints,
sleep/schedule witnesses and combined/parent wake/guard/epoch chains to held
boundaries. Keep old helpers/pins intact. Intermediate combined/parent records
are hashes; full state and cost checks remain b2b2, with source-bound exact
train reproduction in b2c. Fifty added/223 related tests pass, zero skipped,
including all six normal and four forced-rejection families with validation
source/model/train/score sentinels. Check only b2b1. B2a's envelope checks do not
claim complete raw cost/decision or post-trained tensor reconstruction. Twelve
historical bundles/twenty usage files and the exact saved scope still validate;
no scientific source, reserved data/model, score or experiment file changed.

## B2b2 all-family work and state links

Use closed existing dataclass annotations and explicit TypedDict schemas for
dictionary-based combined/parent records. Reject unsupported/open Any; this
reduces repeated field/type ladders without relaxing nested inventories.
Implement gating/replay/sleep app checks from inspected original rules and
reuse unchanged pure per-seed periodic seams. Bind all held state digests,
clock/lineage/selector/retention/exposure and raw capacity, trigger energy
tails, full supplemental boundaries and baseline applied-work traffic.

The full API requires every frozen scientific row before work checks and
returns derived typed cost records with family and joint caps. Development
fixture/forgery/rejection/source/model/train/score gates pass: 110 new/333
related tests in 56.53 s, zero skipped, including the nested legacy sleep
final-seal regression/repair. Static and historical identity gates pass.
B2b2/b2b are complete for pure validation, preserving b2c/parents. JSON cannot
reconstruct unseen tensors, chemical variance/means or physical aliases;
source-bound live capture and exact complete repetition remain required.
