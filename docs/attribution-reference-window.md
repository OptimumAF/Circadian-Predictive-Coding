# Fixed references on a finite TRAIN window

For an already-trained chooser, the separate
[source-bound contextual-window API](attribution-contextual-window.md) reuses
this same finite loop with the original counterfactual owner. Fixed references
and their command defaults below keep their existing behavior.

The existing affine command now exposes the fixed references:

```console
python -m scripts.run_minigrid_linear_fit_control --context-linear-reference periodic --describe
python -m scripts.run_minigrid_linear_fit_control --context-linear-reference periodic --source TRAIN.json --output reference-window.json
python -m scripts.run_minigrid_linear_fit_control --context-linear-reference drift --prequential-feedback --describe
python -m scripts.run_minigrid_linear_fit_control --context-linear-reference drift --prequential-feedback --source TRAIN.json --output drift-window.json
python -m scripts.run_minigrid_linear_fit_control --context-linear-reference drift --ready-feedback --describe
```

`TRAIN.json` is an ordinary TRAIN source report with the original A/B blue/red
episodes. The output must not exist. The first command is metadata-only; the
second performs bounded analytical fits under the existing90second/512MiB,
six-update ceiling. Other choices are `no_learning` and `budget_aware`; `drift`
requires explicit `--prequential-feedback` or `--ready-feedback`. The other three controls may also
enable either mode to pay diagnostic opportunities. These flags require
`--context-linear-reference` and is refused with unrelated command modes before
source access. All output paths must be fresh, including after failed runs.
Original policy defaults apply: period2 and reserve4, with no new tuning flags.
No endpoint scores, weights, native games or final/probe data are exported or
evaluated by this mode. Existing command modes keep their behavior.

## Consume proven-correct feedback without fitting

The ordinary reference API/command can opt into the existing correct-packet
acknowledgement path:

```console
python -m scripts.run_minigrid_linear_fit_control --context-linear-reference drift --ready-feedback --acknowledge-correct --describe
python -m scripts.run_minigrid_linear_fit_control --context-linear-reference drift --ready-feedback --acknowledge-correct --reference-retain-arrived --source TRAIN.json --output fresh-ack-window.json
```

Python callers pass `acknowledge_correct=True` to
`run_context_linear_stream` or `describe_context_linear_reference`, alongside
an explicit `ReferencePolicy` and `ready_feedback=True`. The option defaults
to False. Prequential feedback, fresh A4/readout modes and unrelated command
modes are refused before source/model work. The optional arrived bank and
Python shift diagnostics remain supported. All original policy settings,
six-update/seven-opportunity/90-second/512MiB limits and default outputs remain.

A wait on an already-correct packet obtains a **fresh independent proof** under
the existing candidate lease before acknowledgement. An earlier zero-mistake
diagnostic alone cannot authorize it. Each opportunity permits one fit OR
acknowledgement. There is no implicit drain, new credit, budget renewal,
promotion or final/probe access. Incorrect feedback can remain pending under
`no_learning` or a waiting policy. This option does not promise task success.

`processed_packets`, `acknowledged_experiences`, `pending_unapplied` and
`consumption_complete` describe fitted-or-acknowledged packets.
`training_complete`, `all_packets_fitted` and `unfitted_packets` retain the
separate meaning of actual fitting. Optimizer receipts and budget updates count
only fits. In bank mode, committed current/replay/native rows follow actual
fitted packet indices; acknowledged earlier packets are not counted as fits.
The prepared plan is still an upper-bound template, not executed exposure.

Ready diagnosis and fresh proof costs stay separate in `ready_feedback` and
`acknowledgement.proof`. The description includes both: at most12 diagnostic
forks/predictions and108 rows, or216 rows with bank templates. Fresh proofs alone
are bounded by6 calls/54 rows, or108 bank rows. Attempted catalogue row bounds
are conservative when a failed call does not reveal a skipped packet index;
actual committed rows remain exact and unknown failed work remains explicit.
Corrupted/replaced receipts, lost proofs and expired original budgets stop the
window and preserve the known prefix and failed/unknown work.

[Focused software evidence and limits](reports/affine-acknowledgement-cli-20261027.md)
cover real stream/runtime composition with literal learners on Windows/Linux.
No native adaptation benefit, recovery, speedup or CPC advantage follows from
those fixtures; earlier failed experiments remain spent. Native use still needs
its own frozen local protocol and cost allowance.

One [metered native public command](reports/affine-acknowledgement-native-20261027.md)
now passes: four native fits54 rows, two fresh return ACKs, six consumed/zero
pending, original source/actor/RNG/budgets preserved. Ready/proof work was eight
prediction calls126 rows, with every replay exposure charged separately. There
were no games, promotion, final/probe readouts, new control comparison or native
speedup claim. Installed-console qualification remains separate.

## Opt-in native MiniGrid shift diagnostics

The Python stream API can observe newly arrived TRAIN packets without changing
the reference decisions:

```python
from scripts.run_minigrid_linear_fit_control import run_context_linear_stream
from src.core.reference_controller import ReferencePolicy

report = run_context_linear_stream(
    original_train_episodes,
    reference_policy=ReferencePolicy("periodic"),
    retain_arrived=True,
    shift_diagnostics=True,
)
```

Use the same admitted original source episodes as the ordinary API and budget
the run before execution. This is a Python option, not a new command flag.
It defaults to False, requires an exact bool and the fixed reference lane, and
adds no budget, update, promotion or controller feedback. Existing90-second/
512MiB limits include the extra diagnostics. Drift still requires its existing
explicit prequential or ready-feedback option; the new diagnostics do not supply
drift's decision feedback.

Each of six packets is observed exactly once after its delay-one label arrival
and before the current reference decision/fit, including all six under
`no_learning`. A separate reader predicts the full296public input coordinates
on the detached current candidate. Only the newly arrived nine rows enter these
diagnostics; older rows in a joint fit template remain excluded.

The feature signal uses eight fixed values: initial/current image channel means
(object, color, state) and direction. Existing categorical metrics report
H(p)/log(7) and mean(1-p_true); the existing detector's reference/window/
persistence/thresholds and incident onset/continued/clear semantics stay fixed.
Context names, labels and outcome scores do not select features or thresholds.
All measurements here occur after labels arrive; this is not a pre-label alarm.

`report["shift_diagnostics"]` has six bounded scalar slots, candidate versions/
updates distinct from arrival indices, prediction and metric attempts/returns,
explicit-array and scorer numeric storage, time, failed and not_run records.
`report["stream"]["arrived_observation"]` records guarded callback completion
and full-template input costs separately. A failed read or changed owner stops
the window, retains known work/unknown unreturned work and cannot resume it.
Zero fits and pending packets remain explicit. No probe/final labels are passed.

32distinct focused checks qualify per platform, with final scoped statics and
self-review. These are literal learning/readout fixtures plus real pure
statistics, not native task-success or deployable-trigger evidence.
[Implementation, failures, costs and next protocol](reports/minigrid-shift-diagnostics-20261026.md).

For an explicitly budgeted longer finite schedule built with the existing
`AttributionStream`, the adapter accepts a keyword-only observation horizon:

```python
from src.adapters.minigrid_shift_diagnostics import MiniGridShiftDiagnostics

diagnostics = MiniGridShiftDiagnostics(max_arrivals=len(training_batches))
window = owner.run_reference_window(
    ReferencePolicy("periodic"), observe_arrived=diagnostics.observe,
)
shift_report = diagnostics.report()
```

`owner` must already own that exact finite TRAIN schedule with delay-one labels,
nine rows/296 input features and seven classes per arrival, and its original
budget must cover all stream and diagnostic work. `max_arrivals` must be an
exact integer in1..64 and is validated before reader construction. It fixes the
number of scalar report slots; attempts beyond it are refused before prediction.
The default remains six with the same report schema. This option changes the
adapter's capacity; the fixed-six `run_context_linear_stream` API and CLI retain
their original schedule. It grants no extra updates, replay, time or memory.

The detector still uses its original reference/window/persistence rules and
receives only newly arrived TRAIN rows. Choosing a longer horizon does not
calibrate thresholds or establish a useful trigger. Freeze any scientific
schedule before execution, with an uncontaminated reference and declared
context boundaries if that is the question. In the original six-arrival ABA,
the earliest possible default onset is6 while B occurs at3–4. The separate
[native attempt](reports/minigrid-shift-native-20261026.md) failed before
prediction and remains spent with unknown outcomes. The
[bounded-horizon checks](reports/minigrid-shift-horizon-20261027.md) use scripted
heads; they do not renew that experiment or establish native benefit.

The separate [long-phase native observation](reports/minigrid-shift-long-phase-20261027.md)
completed with a clean A reference and twelve A/B/A arrivals. Both no_learning
and periodic controls showed feature onset at6 and clear at returned-A9, with
no new return onset; stationary controls had no onsets. Periodic FIFO updates
first trained B at10, after A returned. This is scoped timing evidence, not
successful adaptation, a pre-label alarm or general deployability.

For successful updates on its fixed seven-opportunity A/B/A window, the defaults
give0/3/2updates and6/3/4pending packets respectively. These dispatch counts were
checked with scripted learners; this CLI increment runs no native learning study.
Top-level `status="complete"` means that the command completed its window;
inspect `stream.training_complete` and `stream.pending_unapplied` separately.

Failures retain the inner window's paid progress. A late time-limit stop can
leave a known receipt in `stream.applied_updates` while the outer final capture
refuses the incomplete loop counter, leaving `committed_receipts=null` and a
`receipt_capture_error`. The command stays failed/exit2; it never relaxes that
counter check or represents the update as free. [CLI qualification](reports/affine-reference-cli-20261025.md).

`AttributionStream.run_reference_window(policy)` uses the existing delayed
stream, FIFO inbox and owned shadow learner. Fixed no-learning, periodic and
budget-aware choices decide whether to execute one wake at each opportunity.
It is a TRAIN-only API, with no serving, promotion or final-evaluation authority.

## Optional prequential feedback and fixed drift

The Python stream API additionally supports:

```python
result = stream.run_reference_window(
    ReferencePolicy("drift"), prequential_feedback=True
)
```

This uses the existing fixed threshold (default0.1), without tuning or a new
controller. `prequential_feedback=True` also works for the other three controls
so a future comparison can give them the same diagnostic opportunities. The
command-line choices above retain their original defaults. The matching API is
`run_context_linear_stream(episodes, reference_policy=policy,
prequential_feedback=True)`. Drift without explicit feedback is refused in both
interfaces. `describe_context_linear_reference` accepts the same keyword.
With feedback enabled, describe lists six diagnostic forks, six prediction
calls and54rows as **upper bounds**, not measured work. Actual counts, attempts
and failures live in the returned nested `stream.prequential` report.

For each packet, the stream predicts from features before the tick's label
arrival or training. A separate native fork restores the current candidate's
complete state; neither labels nor task metadata reach its `predict` call.
When the original delayed labels arrive, the saved class predictions produce
ordinary TRAIN mistake counts. If the actor/candidate/update binding changed,
the historical error is recorded as stale and supplies no wake vote. It is
never silently relabeled as a current-policy prediction. Drift still trains
only the oldest ready FIFO packet; the triggering error need not describe that
oldest packet. This is a scheduling rule, not a replay selector.

`result["prequential"]` reports attempted/completed forks, prediction calls and
rows, original observation/arrival ticks, per-packet errors and eligibility,
peak pending rows, and explicit unreturned-work unknown flags. These diagnostics
consume the original lifetime wall/RSS budget and remain charged on failure.
They are separate from `score_forward_calls`, which remains zero because the
window does not perform final scoring. No model/optimizer changes or promotions
occur in diagnostic forks. The caller must budget their real physical work.

The option accepts hard-class ordinary TRAIN batches, at most31packets/4096total
rows. Complete episode provenance is accepted only when role/context/class,
row IDs, features and targets match the core-validated episode exactly. These
IDs and receipts are retained; no recurrent trajectory inference is implied by
matrix predictions. Only predicted class IDs and their scalar bindings
wait for labels; the helper is discarded at terminal/failure. Binary decisions
use threshold0.5; categorical decisions retain the original first-argmax tie
direction and probability validation. Drift without this explicit option is
refused. Full source/actor/candidate and diagnostic-state checks remain active.

[Software qualification](../artifacts/runs/affine-drift-feedback-20261026/RESULTS.md)
covers the actual stream with literal learners on Windows/Linux. It does not
establish native learning, better drift detection, task success or efficiency.
The [CLI integration qualification](../artifacts/runs/affine-drift-cli-20261026/RESULTS.md)
adds58cases/platform, including original defaults, actual command output and
failure handling. Its first episode-admission failure and the explicit expanded
metadata-matching contract are preserved; earlier192evidence is not rewritten.

The [one native drift/periodic observation](reports/affine-drift-native-20261026.md)
is complete: drift ends old2/2,new0/2 after2fits; periodic ends old1/2,new1/2 after
3fits. Both pay six diagnostic opportunities. Drift's latest-packet feedback
can become correct on returned A while earlier A4 packets remain queued; its
current fixed policy is preserved with that negative result. Neither condition
establishes complete acquisition/retention/recovery or an efficiency/CPC benefit.

## FIFO-aligned arrived-label diagnostics

`--ready-feedback` is an explicit alternative to `--prequential-feedback`;
the flags are mutually exclusive. It requires reference mode and is available
for all four fixed controls. The matching Python option is
`run_reference_window(policy, ready_feedback=True)` or
`run_context_linear_stream(episodes, reference_policy=policy, ready_feedback=True)`.
Without either mode, existing defaults remain unchanged and drift is refused.

Ready feedback diagnoses the **oldest eligible FIFO TRAIN packet**, after its
original label arrives, using the current candidate. Thus the packet being
scored is the one an immediate wake would consume. Only features enter the
independent diagnostic fork; labels are compared afterward. Original feature
and label times remain in the event, alongside `scored_at`, current candidate
version/update count and `feedback_source="oldest_ready"`. Logical clock changes
during fork/restore/prediction are refused before any wake. This supervised
arrived-label diagnostic is distinct from the preserved prequential baseline.

`result["ready_feedback"]` records original attempts/completions/rows/forks,
failure unknowns and bounded scalar events. Correct oldest packets remain
queued; this mode adds no acknowledgement, bypass or forced update to reach
later work. Exhausted update credit cannot be renewed by diagnostics. Every
prediction uses the original lifetime wall/RSS budget, including failed calls.
Ready admission bounds `number_of_packets * largest_packet_rows <= 4096` to
cover repeated diagnosis of an unconsumed large first packet. Only one temporary
predicted class vector is retained; the original finite window stays unchanged.

Describe reports at most6diagnostic calls/forks54rows for the fixed six9-row
affine recipe. Actual counts remain nested in `stream.ready_feedback`, separate
from final scoring (zero). [Scoped software results](../artifacts/runs/affine-ready-feedback-20261026/RESULTS.md)
cover actual stream/API/CLI and preserved prequential behavior on both platforms.
The [single native ready-feedback trace](reports/affine-ready-native-20261026.md)
processed all six FIFO packets: old/new goals2/0→0/2→2/0 at fixed ticks2/4/6.
It acquired the known new task and relearned the old one after forgetting;
joint retention failed. Six fits/54rows and six diagnostic calls/54rows were
charged. This establishes no generalization, CPC or efficiency advantage.

## Optional arrived TRAIN bank

```console
python -m scripts.run_minigrid_linear_fit_control --context-linear-reference drift --ready-feedback --reference-retain-arrived --describe
python -m scripts.run_minigrid_linear_fit_control --context-linear-reference drift --ready-feedback --reference-retain-arrived --source TRAIN.json --output joint-reference.json
```

The matching API is `run_context_linear_stream(episodes, reference_policy=policy,
retain_arrived=True, ready_feedback=True)`; describe accepts the same retention
option. Retention also works with prequential feedback or references that need
no diagnostic feedback. The existing A/B/A schedule and delay1 remain fixed.
The CLI option requires reference mode; other recipes and endpoint handoff stay
unsupported. Original flags without this option keep their prior contracts.

Why: the native current-only trace learns new and relearns old but forgets the
other task. Reuse the existing finite catalogue: one prior arrived TRAIN packet
per public cue, fitted jointly with the current packet **only on a selected
wake**. Its labels arrived before the current observation. Sources/roles/spans,
FIFO and original clocks stay intact; no extra observation, trajectory or
consolidation event is invented. A correct oldest packet still blocks later
work. No acknowledgement, skip, automatic tail drain or budget renewal occurs.

`catalogue.planned` contains the prepared six-packet plan,213840input-array bytes,
at most18bank rows and90potential fit exposures (54current plus36prior rows).
These are preparation/storage and plan counts, not actual fits. Preparation
precedes the existing training budget's timer; whole qualification body costs
include it. Numeric bank equivalents exclude Python/metadata overhead; source
and stream copies still contribute to measured process RSS.
`catalogue.committed_rows` separately records actual current/replay/native row
exposures from validated receipts. `attempted_rows_upper_bound` includes attempted
selected wakes, including failed ones. Lost or corrupt receipt capture leaves
committed rows null, never zero; earlier charges and failure bounds remain.
Within-wake replay is counted here, separately from ToyBudget's consolidation
replay counter. No new credit is granted by the flag.

The same joint FIFO packet is diagnosed and consumed by an immediate ready
wake. Prediction costs remain nested in `stream.ready_feedback` or
`stream.prequential`. Prequential diagnostic rows are bounded by90, ready by108
because an unconsumed18-row packet may be scored repeatedly. A no-learning
window consumes0fit/replay rows while still paying preparation and enabled
diagnostics. Final scorecalls and endpoint exports remain zero. Original source,
actor, caller state, native counters and six-update90s/512MiB owner stay bound.

[Scoped software evidence](../artifacts/runs/affine-reference-catalogue-20261026/RESULTS.md)
uses literal learners through actual composition/API/stream/CLI. It establishes
the composition and accounting, not native retention, controller benefit,
generalization, matched computation or release readiness. A new native outcome
needs a separately frozen small protocol; accepted controls need no refit.

The [single selected-replay native observation](reports/affine-ready-replay-native-20261026.md)
has since completed through direct production composition/stream with genuine A4
TRAIN data: old/new2/0→2/2→2/2 at logical ticks2/4/6, four fits/54rows including
18replay rows, two return packets still pending. Tick6reuses the identical full
tick4state's completed readout, so eight unique games ran; no independent return
observation or recovery after forgetting is claimed. This known-layout result
does not qualify the public A/B-only wrapper as A4, or prove controller/CPC/
generalization/efficiency benefit. All earlier negatives and spent runs remain.

```python
from src.app.coupled_attribution import AttributionStream
from src.core.reference_controller import ReferencePolicy

# learner, TRAIN batches and original budget are caller-owned and prospectively
# budgeted. The observer may retain these bounded detached checkpoints.
states = {}
stream = AttributionStream(
    learner, training, (), training_only=True,
    checkpoint_updates=tuple(range(len(training) + 1)),
    label_delay=1, budget=budget,
    observe_checkpoint=lambda n, state, batch: states.__setitem__(n, state),
)
result = stream.run_reference_window(ReferencePolicy("periodic", period=2))
print(result["window_complete"], result["training_complete"], result["pending_unapplied"])
```

Why this: always draining every packet cannot represent a no-learning baseline
or a reference with insufficient opportunities. A completed observation window
must not imply that training completed or that pending examples were discarded.

Admission requires a pristine ordinary TRAIN-only stream, empty heldout data,
finite original update credit and every checkpoint count0..n (therefore at most
31packets under the original32checkpoint bound). Rehearsal, renewal, capacity,
prearrival callbacks, probe owners and externally supplied/learned feedback are
unsupported on this path. Fixed drift requires an explicit internal prequential
or ready-feedback mode above. Checkpoint observers keep their existing copied-state contract and
consume the original wall/RSS budget. Original `run_until` and `finish` retain
their prior complete-training/evaluation behavior.

The window runs ticks0..n+delay−1 exactly once. Every experience and label keeps
its original causal clock and FIFO position; a selected wake calls the original
`train_ready(max_updates=1)`. Exhausted credit causes waiting. No examples are
acknowledged or skipped, and no extra drain is appended. The owner seals at the
end even when packets remain; it cannot resume, train again, or release final
evaluation. Source, actor, policy and original budget bindings are checked around
the trusted observer boundaries. This is not hostile-Python isolation.

Returned dictionaries and `stream.reference_report` are detached. They include
bounded decisions, attempted wakes, actual receipts, committed budget updates,
rows and pending work. A lost native return stops the owner and remains explicit
attempted/unknown work; ToyBudget's original committed-update counter may still
be zero. A late stop after a committed update retains its receipt and charge.
Failed work is never retried or represented as a successful free update.

For the analytical MiniGrid learner, the existing checkpoint evaluator offers a
separate `reference_window=True` mode for actual alternating-cue prefixes0..6,
nine rows per fit, including unfitted uniform heads. It reads A3/A4 TRAIN games
under the original zero-update/replay budget; other recipe admission rules stay
unchanged. Actual game success remains distinct from coefficient fitting and
window completion.

[Qualification report](reports/affine-reference-window-20261025.md):63distinct
software checks qualify per platform, and the native TRAIN windows executed.
That original native game qualification failed on duplicate episode IDs; its
allowance remains spent and its full comparison incomplete. Distinct frozen
observations194and196later exercised the repaired native path; their scoped
results above do not complete or retry that earlier failed comparison. No CPC,
generalization, efficiency or release claim follows.

Extend through the existing stream and role-separated readout. Add an explicit
protocol before any new model work; never turn waiting into implicit budget
renewal or use evaluation outcomes to choose updates.


## Readonly joint prefixes for reference controls

The existing Python checkpoint evaluator has an explicit opt-in for the
one-prior-TRAIN-packet/cue bank:

```python
from src.adapters.minigrid_policy_evaluation import evaluate_context_linear_stream_checkpoint

games = evaluate_context_linear_stream_checkpoint(
    cold_source, actual_owned_checkpoint,
    budget=original_readout_budget, check_boundary=original_guard,
    reference_window=True, reference_catalogue=True,
)
```

Why: no-learning, periodic and budget-aware controls can stop at unfitted or odd
prefixes. They need the same four A3/A4 blue/red TRAIN games as drift. Accepting
only the older fixed2/4/6joint checkpoints would hide valid baseline outcomes.
The new exact-boolean option requires `reference_window=True`; other readout
modes cannot combine. Old flags/defaults and their admission remain unchanged.

| Actual FIFO fits | Current row exposures | Prior replay exposures | Total native rows | Blue/red head fit rows |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 0 | unfitted/unfitted |
| 1 | 9 | 0 | 9 | 9/unfitted |
| 2 | 18 | 0 | 18 | 9/9 |
| 3 | 27 | 9 | 36 | 18/9 |
| 4 | 36 | 18 | 54 | 18/18 |
| 5 | 45 | 27 | 72 | 18/18 |
| 6 | 54 | 36 | 90 | 18/18 |

These are the exact alternating-cue six-packet recipe prefixes. Wrong counts,
row totals, missing/swapped heads, per-head row sizes, failed work and changed
budgets are refused. Geometry is admission only: the caller still needs complete
causal TRAIN source/receipt evidence for the supplied actual state. No weights
are inferred from counts and no unsupported source is relabeled as TRAIN.

`source_current_row_exposures` and `source_replay_row_exposures` describe the
owned fitted prefix, not all arrived packets or the planned catalogue. The new
report role is `knownTRAIN_A3A4_reference_catalogue_prefix`; episode IDs include
branch row at every count. Existing native loop, raw K7 firstargmax, seed17,
max16, complete state copying, cold-source preservation and original readout-only
0update/0replay150s/512MiB budget remain. Source/frozen-scorer/caller RNG/clock/
limit checks and failed-attempt cleanup remain; the mode adds no training,
promotion, final/probe access, controller selection or budget renewal.

[Scoped software evidence](../artifacts/runs/affine-reference-catalogue-readout-20261026/RESULTS.md)
uses literal coefficient records and scripted probabilities/environments through
the original loop. It qualifies admission, routing, preservation, accounting and
cleanup; synthetic task success is not learned/native game evidence. The
[single frozen native control observation](reports/affine-bank-controls-native-20261026.md)
subsequently completed: no-learning old/new0/0 throughout; periodic1/0→2/0→2/1;
budget-aware0/0→0/0→2/0, with0/3/2fits and6/3/4pending packets respectively.
All windows complete; none drain training. Original budgets/source/actor/RNG and
preceding evidence checks held. Historical drift/fixed-six results remain
descriptive across producer epochs; no paired efficiency or CPC claim follows.
Further native work requires its own prospective protocol. Preserve every failed
or negative run; this readout grants no authority to retry a spent comparison.
