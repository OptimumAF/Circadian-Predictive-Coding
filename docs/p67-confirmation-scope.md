# P6.7a confirmation scope and prospective gates

Date: 2026-09-30. This inventory reads configurations and saved development
evidence only. It constructs no confirmation dataset or model and opens no
outer/final value. Original P6.3/c9/P6.7/final criteria remain unfinished.

## Preserve every informative factor

Confirm all six predeclared families, including negative, null and inactive
development outcomes. Retain every existing arm, treatment setting, source
geometry, initialization offset, guard rule, memory privilege and planned
width reference. No winner is selected and no baseline is retuned. Gating
and replay intentionally share one reservation; other sets are disjoint.

| Family | Reserved seeds, in order | Cells/seed | Maximum updates/seed | Maximum guards/seed |
|---|---|---:|---:|---:|
| Gating | 101,103,107,109,113,127,131,137,139,149 | 3 | 72 | 0 |
| Replay | Same ten as gating | 8 | 228 | 0 |
| Sleep effects | 151,157,163,167,173,179,181,191,193,197 | 9 | 216 | 5 |
| Schedule | 199,211,223,227,229,233,239,241,251,257 | 11 | 338 | 21 |
| Combined/removals | 277,281,283,293,307,311,313,317,331,337 | 17 | 516 | 48 |
| Parent ranking | 359,367,373,379,383,389,397,401,409,419 | 8 | 192 | 18 |

These are **560 cells / 60 family-seed instances / 50 distinct source seeds**,
disjoint from the fifteen development seeds. Reuse of ten seeds in two
families is not twenty independent sources. Report each within-family pair;
do not pool different families' absolute scores or treat repeated seeds as
independent replications. All reservations are used in order, with no seed
dropping or favorable early stopping. Ten was reserved prospectively, and
is a modest informative replication; it cannot establish broad superiority.

All sources retain A/B 160/160, A train/inner/outer 72/24/24, B 36/12/12,
and unopened final counts **40/40**. B arrives only after all family A work.
The confirmation endpoint is the **independent final role**, after global
freeze; original development/final roles stay unopened. No confirmation
outer-selection value is used for training, guarding, selection or scoring.

## Original matrix coverage and limits

| Minimum-matrix row | Preserved confirmation evidence |
|---|---|
| Matched backprop/ordinary PC | Replay, sleep, schedule, combined and parent width-eight references; same within-width initialization and arrived rows. Gating alone has ordinary PC/neutral controls, not a new backprop arm. |
| Gating only | Original three gating cells; paired chemical-gating minus neutral. |
| Replay without structure | Original eight replay cells; three on/off pairs with common offered/applied IDs. |
| Periodic structure without replay | Sleep's A-boundary structure plus combined's periodic structure-only control; original distinct schedule/threshold scope retained. |
| Homeostasis/reset at fixed width | Three original sleep-effect pairs, with chemical reset conditional on gating. |
| Full/minus-one | All seventeen combined cells and twenty-two original pairs. Default inactive splits stay valid; difficulty removal also changes importance history. |
| Matched baseline replay | Replay's shared schedule, schedule's neutral-controller commits, combined's full-controller commits. These are different declared controller scopes. |
| Planned wider PC/backprop | Width 12,14,13 references as originally fixed, never retrospective final-width oracles. |
| Scheduled/random growth | All eight parent cells/twenty pairs; common planned counts, eligibility/guard settings and actual unequal accepted costs retained. |

The factors have different reserved source seeds and work/capacity/guard
costs. This is a staged matrix, with within-family attribution, not a single
equal-cost ranking. Multiclass/real data, longer streams, full resource
attribution and cross-device/version portability remain separate tasks.

## Exact local budget, before new training

| Family | Ten-seed wake updates | Maximum executed updates including rejected replay | Maximum guards | Retained replay/supply array bytes per seed before copies |
|---|---:|---:|---:|---:|
| Gating | 720 | 720 | 0 | 0 |
| Replay | 1,920 | 2,280 | 0 | 576 |
| Sleep | 2,160 | 2,160 | 50 | 0 |
| Schedule | 2,640 | 3,380 | 210 | 768 |
| Combined | 4,080 | 5,160 | 480 | 2,304 |
| Parent | 1,920 | 1,920 | 180 | 960 |
| Total | **13,440** | **15,620** | **920** | Scope-specific, excluding role/parameter/metadata/copy arrays |

Bounds scale each validated pilot's prospective ceiling, not its observed
work. Schedule counts include fifteen possible rejected adaptive proposals
and at most two commits per seed; combined counts include rejected replay.
Record every actual/rejected update, latent loop, row, guard and transient
parameter/width, not just restored model clocks.

One joint process gate has a **16,000 executed-update hard cap**, **600-second
child wall limit**, **512-MiB observed whole-worker RSS limit** sampled every
5 ms. This new joint scope retains six families and both checkpoints: saved
three-seed combined/parent JSON alone is 4,635,365/3,045,849 bytes, so a
single-family 256-MiB budget does not describe the aggregate allocation.
The joint RSS budget is a prospective upper limit, not a measured claim.
Include training, held copies, independent validation and serialization;
stdout/parent publication remains outside sampling. A brief peak can be
missed. Fail rather than increase a budget after an observed outcome.

First implement/validate and repeat one **unscored** joint gate. A separate
scored continuation must reproduce the entire saved all-family/all-seed
fact object and validate every held parameter/width/full circadian checkpoint
before the first final input/label, then recheck after scoring. Preserve both
RNGs, selector state, chemistry, memory, lineage and clocks. Gating/replay
pilot `_run_seed` functions already score outer roles and are unsuitable for
this gate: compose their existing training helpers with held copies instead
of modifying their pinned sources. Other frozen pilot validators accept only
three seeds; confirmation needs its own strict manifest/validator.

After the full gate, score all 560 cells on A after A and A/B after B final
roles: **1,680 forward calls / 67,200 examples**. Preserve the same two-task
primary mean/signed-forgetting formulas, optional retention and all named
pairs: gating 1, replay 3, sleep 3, schedule 9, combined 22, parent 20
(**58 per seed / 580 pairs**). Include every raw endpoint and cost.
Before any scored fixture/final release, freeze the separate scored source/
reference identities and P6.11's uncertainty/multiple-contrast analysis.
No seed sweep stops when rankings become favorable. Confirmation and final
tasks remain unchecked until actual gates and complete analysis pass.

## Read-only acceptance and next implementation

Implement a pure manifest that binds every original resolved configuration,
all reserved seeds/cells/pairs, known development result digests and exact
prospective budgets. An inspection adapter must verify both complete saved
development bundles/frozen source identities for every family, inventory
reservation reuse and reject prior confirmation rows, then save an exclusive
finite scope record. Tests must prove configuration inspection cannot create
any source/model or compute a new score and rejects changed scope/budgets.

The pure `src/app/continual_confirmation_manifest.py` binds the original
resolved configuration bytes, reservations, all cells/pairs, source/result
digests and prospective budgets. `scripts/inspect_p67_confirmation_scope.py`
verifies twelve complete scored-development bundles plus their prerequisite
readers and frozen source maps. It scans twenty P6.3 result files for actual
reserved seed rows, distinguishing declarations from use: only the fifteen
development seeds occur. All fifty reserved source seeds remain unused in
this catalog; no confirmation source/model is constructed or scored.

All **21 focused tests pass, zero skipped**, including raising dataset/model/
accuracy sentinels, exact additive work/cell/call counts, changed/partial
family and role/budget/source rejection, nested prior-usage detection and
exclusive byte-preserving outputs. Ruff/mypy (365 files) pass. Two real
read-only CLI records repeat byte for byte:

```powershell
.\.venv\Scripts\python.exe -m scripts.inspect_p67_confirmation_scope --output-file artifacts/runs/p67-confirmation-scope.json
.\.venv\Scripts\python.exe -m scripts.inspect_p67_confirmation_scope --output-file artifacts/runs/p67-confirmation-scope-repeat.json
```

Both scope-record SHA-256:
`622feead54f155521928341151c8496c23c02a54b355b1b3b1e0f68b76c5772f`.
Manifest SHA-256:
`8d1ed66b33bbc1bf298cc60604c3741afa22bb4b7e0f6636efe166a52672951b`.
App/adapter source SHA-256:
`390cc56625983f2038ea98f689c68e11e754c971a49a17fcd63a227bb35b49ec`,
`e3f96f4e284b56a2db5a21d2c636f9ca4361e4b7fbce5cd8ba8c1ca833b41a1e`.
These records preserve each canonical/repeat request/result/audit digest and
all selected sources. This inspection is configuration/evidence validation,
not a confirmation run or resource measurement. Clean fixture tests need
no ignored artifacts; the reader requires the saved development bundles.

P6.7a is complete for the frozen inventory contract. Next implement P6.7b's
separate joint unscored trainer/validator, composing existing training and
recording every actual cost/checkpoint/guard/memory/selector fact. Verify
all-family outer/final/arrival and failure/budget sentinels before executing
and repeating under the joint caps. Scored P6.7c still requires separately
frozen reference/source and uncertainty gates; earlier scientific sources
and results remain unchanged.
