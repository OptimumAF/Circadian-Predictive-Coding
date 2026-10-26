# ADR-0387: Explicit public nuisance-invariant MiniGrid control

## Context

Raw exact memory can execute inspected games after memorization, but new wall
colors/absolute directions create different keys. These fields are cosmetic/
redundant to egocentric relative actions in the declared task. Cue/key colors,
door state, goal and pixel positions must remain available. Existing raw control
results and failed native qualification must not be rewritten as a new baseline.

## Decision

Introduce a pure explicit projection/history encoder and a distinct encoding-bound
composition of the existing exact/FIFO native count learners. Validate discrete
public codes, canonicalize wall color/direction only, preserve causal history and
complete owned episode metadata. Project all input rows before any native commit.
Bind policy/snapshots to the encoding ID/native kind/cap; reject raw or relabeled
noncanonical keys rather than converting historical states.

Why: a separately named representation control tests a concrete observation
hypothesis without changing objectives, action choice, replay or existing raw
baseline semantics. Existing generic probe and delayed stream ports already
support the owned wrapper; another runner is unnecessary.

## Alternatives

- Change raw inputs/defaults globally: invalidates old state meanings/evidence.
- Repair chosen actions with task logic: changes the policy rather than its input.
- Infer wall/direction invariance for arbitrary games: unsupported by this task.
- Relabel a raw checkpoint: does not transform its stored observation keys.

## Consequences

New policy/state metadata and projection/copy computation add cost beyond the
same retained native table cap. Roundoff normalization is explicit, not nearest
neighbor action repair. This is an engineering representation invariant, not
scientific evidence of transfer/CPC advantage. At initial component acceptance, actual TRAIN/evaluator admission and a frozen
matched experiment remained open; the explicit follow-up below delivers them. No final data,
promotion, persistence codec, baseline retuning or prior-budget renewal is added.

## Saved A32 integration follow-up

The explicit count-family admission now reuses the existing delayed TRAIN arm;
a separate representation entry point reuses the frozen evaluator. A small A32
recipe reuses original raw endpoints and original TRAIN packets, training only
the fresh bound control. The IO CLI reads the original evidence once through its
existing bounded loader. Completed training and four fixed readouts remain sealed
across external callbacks; old budgets are not restored. New diagnostics/projection
costs are explicit. See the frozen protocol and observed results under
artifacts/runs/minigrid-invariant-pilot-20261017. This decision does not establish
transfer benefit; actual results, including failures, remain separately reported.

## Neural port follow-up

Use the same explicit representation for BP, PC, neutral CPC and active CPC by
composing existing TorchFeatureLearner with a distinct bound policy/snapshot.
Keep every native objective and complete optimizer/gradient/adaptive/RNG payload.
The shared TRAIN arm validates fixed recipe policy/clocks; a separate frozen
readout entry point reuses existing A/E execution with tensor-aware state equality.
Why: changing raw policy fields or relabeling historical native states would blur
the input contract and invalidate earlier evidence. No persistence codec or generic
orchestration layer is needed. Typed metadata is not historical authenticity.
Native equivalence/isolation tests are engineering acceptance, not learning benefit;
actual matched neural game comparison and opt-in CLI integration remain open.


## Matched neural recipe follow-up

Reuse the existing history-smoke CLI with an explicit comparison flag and a
separate raw-neural A/E admission using the same runner as the bound route.
Why: the legacy raw evaluator scores different slots/labels; using it would not
isolate representation. Eight genuine fresh arms share native settings and TRAIN
dose; full initial state equality is checked before games. All training precedes
readouts; prior paid receipts remain guarded and bounded. Metrics are signed and
negative results remain valid. No new framework, dependencies or state export.
Known A/inspected E cannot establish retention, context recovery, independent
generalization or circadian advantage. Exact protocol/results are under
artifacts/runs/minigrid-neural-representation-comparison-20261017.


## Learnability routing follow-up

Wire the accepted bound wrappers into the existing minigrid_learnability factory,
TRAIN-fit admission and disposable probe path, selected explicitly by CLI flag.
Why: generic probe ownership already exists; another runner is unnecessary.
Validate source/checkpoint identity and roles before work, keep original native
objectives and separate probe budgets. Completed bound guards check local evidence
while the parent checks all completed arms/probes in a flat loop; calling parent
from a published guard recurses and was rejected by actual engineering evidence.
Full native comparison remains a separate unexecuted claim; true synthetic
source0 probes demonstrate port correctness only. Preserve all timeout/SPENT data.

## Single-PC qualification follow-up

Select canonical PC explicitly in the existing learnability route before another
matched comparison. Why: failed raw/bound game evidence and the accepted diagnostic
ports make acquisition qualification the concrete learner dependency. Reuse original
seed/objective/dose/metrics and source0/128 disposable E probes. Singleton reports
omit cross-method claims. The frozen case completed but all acquisition premises
failed; trained argmaxes all choose the majority action. Preserve this negative
result and existing SPENT comparison. Next adapt existing TRAIN activation health
to bound snapshots; raw-health acceptance does not establish bound health. No tuning,
automatic repeat, new runner or weight reconstruction from scalar JSON.

## Bound activation-health follow-up

Extend existing native snapshot reader and TRAIN-only learnability diagnostics.
Why: raw-only admission blocked observation of the failed bound-PC acquisition
condition; scalar JSON lacks weights to reconstruct it. Bound copier and whole-batch
public projection precede native float32 tanh-prior reads, with existing health
cutoffs/SVD. Do not label these priors as label-relaxed PC states. Original probe
budgets and final isolation remain. Guard paid score values/identities before
publishing health; mirror observed work to private charges with exact scalar types.
Record returned hidden affines even when later publication refuses. Zero-update
checks qualify integration; trained-state measurements need a separate frozen
small protocol and do not inherit any previous GO or spent allowance.


## Existing-method singleton qualification follow-up

Generalize the existing one-method recipe with an exact native method selector,
keeping the PC alias/default descriptors and omitted-option call shapes. Why:
trained PC failed acquisition without dead/saturated priors, while bound BP under
this full128 TRAIN/two E16 condition is unmeasured and the five-method800-native
protocol is unexecuted. The factory already supplies all four heads; another
runner or objective is unnecessary. Select existing configs unchanged, declare
method-specific descriptor IDs/maps, and refuse selector/alias conflicts before
resources. All singleton reports omit cross-method claims. A separate fixed BP
comparison can test a simpler control before feature/dose/objective changes;
exposure matching does not imply matched computation or circadian advantage.


## Canonical single-task acquisition follow-up

Add an explicit A-only BP recipe to the existing learnability route, reusing the
raw-history A128 concept, existing delayed stream, six source records and separate
E16 forks. Keep old defaults and native objectives/settings. Why: the canonical
continual BP and PC runs failed acquisition, so another return metric would not
answer whether the task can be acquired. This condition changes A exposure from
64 to128 while keeping128 total benchmark updates; it is not a matched-context
interference test. Report A gains and exact both-cue flags, with B/C untrained
fits; omit the original forgetting/return/acquisition summaries in this mode.
The frozen protocol and observed outcomes remain separate from source correctness
under artifacts/runs/minigrid-bp-a-only-20261017. No expanded probe port or runner.


## Distinct probe task follow-up

The accepted canonical E assay exactly overlaps A inputs/actions. Add an opt-in
fixed F key branch at row2 through the existing source/teacher/probe ports, with
actual exposed TRAIN support reported before disposable updates. Why: cosmetic
wall/direction changes disappear under the declared encoding, so the old E assay
cannot answer new-task acquisition. Existing non-F geometry/defaults remain;
F configuration/role/cue admission is explicit. Require all nine canonical inputs
per probe episode unseen in actual TRAIN; keep label-conflict counts and logical
costs. Finite support distinction is not generalization or an acquisition result.
A new callback cannot reseal changed preceding evidence; validate first and retain
completed diagnostic costs separately after refusal. No new runner or dependency.

## Old fit on the F-adapting fork

Add optional zero-to-two TRAIN readouts to the existing fresh-task use-case and
reuse the original A pair in learnability. Why: unchanged source weights cannot
establish what F updates do to old-task fit. Score on the same disposable fork
after each original F checkpoint; never train A or add it to F novelty admission.
Keep the original combined entry ceiling, fully validate/detach before callbacks,
retain paid F checkpoints if monitoring fails, and charge returned monitor work
before guards. Empty defaults preserve original shapes. This adds predictions,
array storage and report/guard computation; expose those costs. A17/18 does not
meet joint acquisition, so old fit drift cannot establish acquired-task retention.


## Fixed A return after measured old-fit interference

Old A fit fell17/18 to14/18 during the accepted F16 assay. Add an explicit fixed
A16 return on the same disposable learner/optimizer, with a separate fresh budget
and original outer deadline. Why: another readout alone cannot measure adaptation
when an old context returns. Reuse existing scoring/training loops and A/F readouts,
with two fresh A TRAIN template copies repeated in the frozen alternating order;
sixteen retained templates would break the original storage ceiling. Original F
report/status/time remains paid evidence; nested return failure retains it and
all known costs. Return0 must match F16/A16 before updates. Templates detach before
fork, monitoring packets stay readonly, original/return controls remain bound.
Empty defaults preserve old API/report shapes. A is not jointly acquired, so
any observed return fit recovery is not acquired-skill or generalization proof.


## Matched A-only control curricula

Extend only the existing A-only selector/descriptor/CLI admission to all four
explicit bound heads. Why: BP's A128 experiment and PC-family A/B/C/A experiments
had different context exposure; comparison requires identical packet schedules.
Keep native objectives/configurations, seed, metrics, budgets and exact acquisition
gates unchanged; BP descriptors and false/default behavior retain exact values.
Reuse accepted BP evidence when these assumptions hold. PC-neutral full numerical
parity remains an acceptance gate; active-vs-control contrasts retain all fixed
points and failed/negative results. Shared-runtime cost observations remain
confounded by ordering/allocator/import reuse and do not imply equal compute.


## Known-TRAIN neural live composition

The learnability route optionally evaluates its genuine owned benchmark128 in
only two known TRAIN A games after disposable diagnostics finish. Why: raw fit
cannot establish successful target-free control, and the existing frozen neural
representation evaluator already supplies the required state/role/episode ports.
Its new exactbool known_a_only option preserves the original A/E default and
constructs no E layout for this readout. Native0..128 clock admission stays;
no JSON weight reconstruction or post-probe144/160 widening. A separate zero
update/replay20s budget retains the original whole deadline and paid-state guards.
Partial results/costs remain visible; no new runner, algorithm or final-data use.


## Bounded exact-memory control in distinct-F learnability

Reuse the existing invariant count learner as a separate explicit singleton in
the existing A128/F16/old-A/A16 recipe. Why: existing count A/E evidence uses a
canonical E input domain that aliases A; it cannot establish distinct F adaptation
or explain the neural assay's acquisition failures. Preserve exact neural enums/
defaults and original count64keys/155136B/uniform-unseen law. Typed count copier/
counter and backend-neutral equality retain the same paid state/role/clock/budget
boundaries without a Torch runtime. Report count updates/storage, not SGD/H8/
neural parameters or equal compute. No FIFO, new algorithm, cap change, final
selection or post-probe live clock widening. Finite lookup success, if observed,
will not close neural/RL/heldout generalization requirements.

## Same-fork live endpoint readout boundary

Why this: exact A/F teacher fit does not establish target-free game behavior.
The private disposable learner must be read at its actual F16/A16 endpoints,
without exporting/reconstructing arbitrary snapshots or changing original A/E
clock admission. A new optional inner app observer receives learner/phase/guard
only after complete paid reports; the outer adapter binds exact count16/32/144/
160 states, fixed A/F roles and separate zero-update budgets to the existing
frozen evaluator. Rejected callbacks restore paid evidence and stop later work.
Defaults, count law/caps and final exclusion persist. The first20s native readout
expired before a complete game; preserve this failure and profile duplicate
validation before any prospective new case. No wider clock permission, budget
renewal, native-objective change or biological/generalization claim is implied.
