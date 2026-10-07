# P6.3c9b paired parent-control train-only contract

## Prospectively frozen question and scope

Protocol `continual_parent_factor_train_only_v1`: isolate split-parent ranking
under identical scheduled add requests, with unchanged original eligibility,
function-preserving splits and independent inner guards. This is a no-score,
growth-only factor. It does not compare the whole circadian stack, select
settings, replace previous results, release finals or complete confirmation.

Reuse the arrived v14/c2 source: A/B generated source 160/160, A train/inner/
outer 72/24/24, B 36/12/12; twelve full-batch wakes per phase, backprop .12,
PC/circadian .05, two latent steps at .2. Generate B only after all eight A
models finish. Outer arrays and final inputs/labels stay sealed to training.

Use the next three primes above the prior reservation maximum 337:
**347/349/353**. Reserve the next ten primes **359/367/373/379/383/389/397/
401/409/419** for later independent confirmation; none is opened here.
Retain every development seed. Initialization is seed+1001 within width;
the separate selector PCG64 uses seed+5001 in all three parent cells.
Initial stable-ID cursor is zero. Existing reservations remain unused.

| Cell | Frozen treatment |
|---|---|
| `backprop_off`, `pc_off`, `neutral_off` | Width-eight no-sleep/no-replay; exact PC/neutral wake parity. |
| `usage_growth`, `scheduled_growth`, `random_growth` | Identical structure-only config and explicit add counts; only the c9a parent mode differs. |
| `backprop_13_off`, `pc_13_off` | Width-thirteen no-sleep/no-replay references, fixed before training at the maximum planned growth width. |

These are **8 cells/seed, 24 total**. Growth reuses c3's neutral structure-only
configuration: min plasticity one, difficulty/homeostasis/reset/replay off,
split threshold zero, prune threshold one, existing split-score mixes and
noise .02. Disable pruning and set its maximum to zero; retain eight labeled
memory rows. Preserve all other original chemical, threshold, cooldown,
change-fraction and phase settings. The chemical-preferred tier is therefore
all eligible parents in this bounded factor; usage keeps the original score
rank, scheduled cycles stable IDs, random permutes without replacement.
This is explicitly distinct from the built-in strict-threshold route and
from retrospective matching to any c8 outcome.

Force phase-local interval four at A/B epochs 4/8/12. Request one add at
global epochs **4/8/12/16/20**, zero at **24**, and no prunes. Why this: the
unchanged split budget is zero in the final prune-only phase; do not relax
that constraint to make the last event grow. Each cell has five add attempts,
initial/minimum width eight and absolute/transient ceiling **13**, with
4×width+1 parameters (33 initially, at most 53). Counts are planned attempts,
not promises of guard commits; publish any unequal applied counts.

## Guard composition and state evidence

Use the existing schedule decision and NumPy guarded telemetry functions,
complete core snapshot/restore, inner-guard accuracy rule and tolerance zero:
accept iff post accuracy + tolerance >= pre accuracy. The old orchestration
helper hardcodes `adaptation_policy=None` and exposes no proposed selector
state before rollback. A separate focused app composes these existing pieces
around explicit policy sleep, capturing the proposed state before guard
rejection. Do not patch pinned helpers or hide a policy in the model.
Restore complete state on pre/core/post exceptions or nonfinite guards;
the public invocation records failure and produces no completed result.

Each epoch records all widths, parameters/hashes, available shared FIFO IDs
and complete circadian state hashes after wake and after the decision. Bind
the before-sleep checkpoint to the separately observed wake snapshot: a red
forgery fixture showed that hash-shape checking alone missed a changed before
hash. This strengthens checkpoint evidence without changing training settings.
Growth decisions record schedule/count,
core telemetry without measured durations, before/proposed/applied full state
and parameter hashes, lineage, clocks and selector views, plus actual original
split scores for permitted parents. Preserve proposed IDs/RNG/cursor on
rejection separately from restored applied state. Independently rederive
usage ordering from scores, cyclic IDs/cursor, PCG64 draws/state hashes,
phase budgets, guard acceptance, committed lineage/width/clocks and work.
Retain complete after-A/after-B state hashes for later global scoring gates.

All four circadian buffers and a prediction-independent shared FIFO retain
identical eight arrived training rows/**192 array bytes**. Supply/order/content
checks run before and after sleep; no replay is executed or consumed. Persistent
labeled-array scope of these replay/supply buffers per seed is **960 bytes**,
excluding source/role arrays, metadata, parameters and temporary copies.
Memory availability is held equal within the parent
cells; baselines receive no replay and have no own replay store.

Implementation inspection found that the existing replay supply helper requires
unprioritized sampling, while the inherited c3 configuration retains its inert
prioritized setting. Preserve that frozen configuration: compare all eight
available row content hashes/order/bytes directly, with no replay sampler preview.
This strengthens the memory-content gate without changing a training treatment.

## Acceptance, resource budget and prospective validation fixtures

Exactly **576 wake optimizer updates**, zero replay/other updates, hard cap
**600**. Each cell sees 1,296 wake examples; PC/circadian has 48 latent loops
and 2,592 example iterations, backprop none. There are 72 epoch opportunities,
216 growth decisions and **54 guarded attempts/108 evaluations/1,944 guard
examples**. Proposed/rejected/committed splits and no-op sleeps remain distinct;
no split itself counts as an optimizer update. Local worker cap **120 seconds**,
observed whole-process RSS **256 MiB**, sampled every **5 ms** through training,
independent payload validation and serialization; stdout/parent writes are
outside it and short peaks can be missed.

Before scientific public runs, require frozen-manifest refusal before source
access, all seed/cell training with outer/final and B-arrival sentinels, exact
within-width initialization and PC/neutral continuation, actual policy parent
choices, supply/cap/work/guard and complete snapshot invariants. A controlled
guard fixture must reject a proposed split in every mode, observe actual core
execution, preserve proposed selector facts, restore full state, and reproduce
future retry without tuning scientific guard outcomes. Exercise pre/core/post
exceptions/nonfinite guards and forged count/parent/RNG/checkpoint/work/role
facts. Unit gates use only this frozen tiny matrix or c9a's bounded fixtures;
no sweeps or scoring. Run related tests and Ruff/mypy/format/diff gates.

Pin selected source, manifest and adapter bytes after implementation tests and
before the first public process. Write exclusive request/result/audit or failure
sidecars; duplicate outputs fail before work and leave bytes unchanged. Repeat
all 24 cells in a second fresh bounded process, require identical deterministic
result bytes, read every artifact and publish every parent/capacity/work row.
No null, negative, inactive or coincidental selection outcome changes settings.
C9b stays unchecked until all criteria pass. C9c must separately freeze scoring
against the complete saved train gate; original c9/matrix/confirmation stay open.

## Implementation identities before official public runs

After the app/CLI correctness fixtures, the resolved manifest SHA-256 is
`a7938028ed3c9279ef74a5f9a2550012927e4bb626b72672861ae64aa71497c9`.
The selected **31-source** map digest is
`280210ea8215b24e3d5238b39daa94725b2091553bfed8d5bb7e5a23977339f9`;
it extends c7's selected 26 sources with the three new app modules, c9a's
selector and c7's adapter. Existing schedule/guard/core functions are
already selected. This is not a full dependency-tree hash.
Adapter byte SHA-256:
`39bfd61bd50a1abf0a00308ca40028152270b09f99eb401b4176924dd4c6a8b1`.
Implementation-only temporary CLI test bundles preceded the added wake/before
checkpoint linkage; no durable scientific result used the earlier code.
The final hashes bind that corrected linkage before official execution.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p63_parent_factor_preflight --output-dir artifacts/runs/p63-parent-factor-preflight
.\.venv\Scripts\python.exe -m scripts.run_p63_parent_factor_preflight --output-dir artifacts/runs/p63-parent-factor-preflight-repeat
```

Both official processes completed after the 210-test/static gate, and exact
results repeat at SHA-256
`555fc2fe5dfd86981d87af2d0c15bd8ab925417d9ad748783d95cee95f5a1874`.
Every artifact and cell passed independent finite-JSON/source/request/audit
readback. See [all work/capacity/parent/resource rows](p63-parent-factor-preflight-results.md)
and the c9b development-log entry. C9c scoring and original matrix/confirmation
criteria remain open; this evidence selects no treatment.
