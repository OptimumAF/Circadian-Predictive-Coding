# Original P6.3c9 growth-control acceptance

## Scope and decision

**P6.3c9 passes its original acceptance criteria.** This audit joins the
completed c9a implementation, c9b paired train-only controls, c9c separately
gated development scoring, and the complete independent confirmation.
P6.3c, P6.3 and P6.7 remain open for their broader original acceptance audits.

Why this: the missing control was already implemented and confirmed by the
later source-bound pipeline. Rechecking its actual records and original
requirements resolves the outstanding task without another experimental run.
No source, baseline, metric, seed, interval, resource limit or reader port changed.
The earlier contracts and their then-unfinished confirmation statements remain
historical records; this document records the later parent acceptance.

The audit uses whole accepted input bytes and the previous complete current
reader proofs. It does not claim a new official reader dispatch. The public
saved-payload training validator checks all **560 cells / 60 family-seed rows**;
both whole scored payloads and the complete original analysis are revalidated.
The C9 evidence view retains every reserved parent seed and comparison.

## Acceptance evidence

| Original requirement | Verified evidence |
|---|---|
| Small explicit source-selection control; original behavior preserved | `ParentControlledCircadianNetwork` delegates usage ranking and reuses the original eligibility, budget and topology path. The immutable settings, stable-ID cursor, separate PCG64 state, lineage, atomic restoration and future retry are covered by unchanged c9a fixtures. |
| Freeze sources, seed roles, counts, capacity, work and guards before training | The original c9b manifest/request/result/audit and repeat remain exact. Development seeds 347/349/353 and confirmation seeds 359/367/373/379/383/389/397/401/409/419 remain disjoint. Planned width thirteen and the five add epochs preceded all outcomes. |
| Deterministic paired train-only controls before separately frozen scores | Complete preflight results repeat. The development validator checks all 24 cells and their complete train facts against that reference. The original global held-checkpoint gate precedes the first outer score and checks state again afterward. All 60 development pairs remain. |
| Scheduled/random matrix row and independent confirmation | Ten reserved seeds, all eight arms, 80 cells and 20 contrasts per seed (200 pairs), with three original endpoints, six arm metrics and five paired metrics. Both whole scored payloads reproduce the unchanged analysis using all 116 simultaneous primary statements. |
| Preserve functionality and prior accepted evidence | 150 source pins, 79 accepted test pins, five C9 test files, 174 whole input pins, fourteen direct original train/scored/cost files, and the four accepted current publication/readback proofs remain exact before and after the audit. All 24 scientific-operation guards in the audit process remain zero. |

The five C9 test files pass **152 tests, zero skipped**, in 35.1824169 seconds
under the prospectively recorded 180-second command cap. These are existing
bounded temporary fixtures, including forced guard rejection and full-state
rollback/retry; they are not new persisted experimental observations.

## Matched controls and cost scope

The unchanged arms are `backprop_off`, `pc_off`, `neutral_off`, `usage_growth`,
`scheduled_growth`, `random_growth`, `backprop_13_off` and `pc_13_off`.
All twenty original pairs remain, including each growth arm against all five
references and both planned-width versus width-eight baseline pairs.

Within the three growth controls, initialization, data, wake schedule, neutral
chemistry, eligible-parent tier, planned counts, guard rule and memory supply
are matched. Only parent ordering differs. Selector seed is source seed+5001;
model initialization is source seed+1001. Global add epochs are 4/8/12/16/20;
epoch 24 requests zero because the original phase budget prohibits another add.
The inner accuracy guard has tolerance zero.

On every confirmation seed, each growth arm actually has:

- Width 8 initially, 11 after A and 13 after B; peak width 13.
- 33 initial and 53 final parameters, with the complete recorded width/work history.
- 24 wake updates, 1,296 wake presentations, 48 latent loops and 2,592 example iterations.
- Five committed splits and six accepted guarded attempts, including the final zero-add event.
- Twelve guard evaluations / 216 inner examples, zero replay updates.

Across all parent confirmation seeds there are **720 decisions: 180 accepted,
540 not-due skips and zero observed rollbacks**. Rollback semantics are verified
by the forced-rejection fixtures, not inferred from these accepted experiments.

The fixed references retain their different capacity trajectories. BP has no
latent relaxation; growth adds guard work that fixed references do not have.
Width-thirteen references are wider from initialization and have different
initial tensors and per-update work. They were planned before outcomes and are
not retrospective final-width oracles. Matching optimizer updates does not
establish equal FLOPs, wall time or RSS. The original whole-process resource
measurements and memory scopes remain in the complete cost records.

## Confirmation result

All **40 parent-family primary simultaneous intervals include zero**.
The intervals below are the original 95% model-based intervals using the
Bonferroni scope of **116 statements**, with ten seed observations per vector.
They have not been recomputed with a smaller comparison family.

| Original left-minus-right pair | Metric | Mean | Simultaneous interval |
|---|---|---:|---|
| Usage − scheduled | Final mean accuracy | 0.0225 | [−0.030589, 0.075589] |
| Usage − scheduled | Signed A forgetting | 0.0175 | [−0.089153, 0.124153] |
| Usage − random | Final mean accuracy | 0.0275 | [−0.027465, 0.082465] |
| Usage − random | Signed A forgetting | 0.0300 | [−0.090490, 0.150490] |
| Scheduled − random | Final mean accuracy | 0.0050 | [−0.025540, 0.035540] |
| Scheduled − random | Signed A forgetting | 0.0125 | [−0.037844, 0.062844] |

Values are accuracy fractions. Higher final mean and lower signed forgetting
are favorable; both A endpoints remain available to interpret forgetting.
The evidence establishes no superiority, equivalence or broad rejection.
No treatment or seed was selected. All mixed, negative and null outcomes remain
in the accepted complete development/confirmation/cost/activity/matrix bodies.

## Local audit artifacts and commands

These ignored local records are exclusive evidence artifacts. Their producing
commands have completed; occupied outputs prevent rerunning them.

| Record under `artifacts/runs/` | Bytes | SHA-256 |
|---|---:|---|
| `c9-entry-reconciliation.json` | 75,680 | `a5844841b560dd2c5d0df723301885550ba0a32b854d3bbed749ac5169622e06` |
| `c9-focused-correctness.json` | 58,939 | `106451e51d56c7fd19744de51b4c0c3b8b51943e7aa7d101fd1f51ba64773c3c` |
| `c9-original-acceptance-source.json` | 80,172 | `185f98f20b54f6edc10647cb0a44817ee2743342585624203b74eb794a18547a` |
| `c9-original-acceptance-audit.json` | 322,749 | `60d2250ba0d528319a000032f649d869feb9d1f5e263ad25c80158e0deb85b4d` |
| `c9-original-acceptance-operation.json` | 1,659 | `3a21a589a3e9c5e518690ba4afe360a5fe59a6a900f03ad76c6a9c2d73ecfa99` |

Observed commands, each using the repository virtual environment:

```powershell
.\.venv\Scripts\python.exe artifacts/runs/inspect-c9-entry.py
.\.venv\Scripts\python.exe artifacts/runs/check-c9-controls.py
.\.venv\Scripts\python.exe artifacts/runs/audit-c9-original-acceptance.py --freeze
.\.venv\Scripts\python.exe artifacts/runs/run-c9-original-acceptance-audit.py
```

The freeze completes in 48.3907928 seconds before the saved-input audit.
Audit terminal exit is zero in **107.5824730 worker / 107.8431105 parent seconds**,
under the prospective 180-second wall cap. No new experimental producer,
training, scoring, final-role source access or dependency installation occurs.

For fresh isolated fixture verification:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_controlled_parent_selection.py tests/test_continual_parent_factor_preflight.py tests/test_continual_parent_factor_development.py tests/test_p63_parent_factor_preflight_cli.py tests/test_p63_parent_factor_development_cli.py
```

No production source changed, so the accepted same-source Ruff, format and
mypy gates remain applicable. Full-suite and GPU/cross-version tests were not
rerun for this saved-input audit. The closing check records current document
bytes, preserved input/source pins and `git diff --check`.

## Exact next action

Audit **P6.3c's original complete minimum mechanism matrix** against the accepted
six-family development and independent confirmation, including all matched BP/PC,
replay, planned-width references, factor isolation, equal/unequal costs and the
prospective role/budget/selection requirements. Keep P6.3c, P6.3 and P6.7 unchecked
until their respective original criteria pass. If that inspection reveals a
gap, implement the smallest unblocked increment and record the preserved
criterion and precise next step. No new algorithm or sweep is needed first.
