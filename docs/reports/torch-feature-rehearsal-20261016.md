# Native Torch fixed-feature rehearsal — 2026-10-16

This is an engineering demonstration of the existing native heads on supplied
CPU features. It is separate from the coherent NumPy objective and its accepted
learning/probe results. No encoder, image dataset, probe or tuning is involved.

Eight fixed arms use replay off/on × BP/PC/neutral CPC/active CPC, seed17, D2/H3,
six two-row A2/B2/A2 wakes, delayed labels, original native objectives and rates.
On-mode commits one arrived retained row at wakes2/4/6 using the existing shared
selector, separate permanent replay allowance and four-row/96-array-byte bound.
Final A/B scoring follows all training for an arm. Initial parameters match;
PC/neutral reports, parameters and energies match exactly in each mode.

## All outcomes

B gain is B BCE0−BCE4; A loss is A BCE4−BCE2; return gain is A BCE4−BCE6.
Positive gains are better; lower A loss is better. Raw checkpoint levels and
accuracy for all64 scored contexts are retained in the linked artifact CSV.

| Replay | Method | B gain | A loss | Return gain | A BCE6 | B BCE4 |
|---|---|---:|---:|---:|---:|---:|
| off | BP | −.000545 | −.001211 | −.002660 | .698031 | .692596 |
| off | PC | −.000048 | .000007 | −.003809 | .697093 | .692099 |
| off | neutral CPC | −.000048 | .000007 | −.003809 | .697093 | .692099 |
| off | active CPC | −.000048 | .000007 | −.003809 | .697093 | .692099 |
| on | BP | −.002599 | −.012597 | .003548 | .675896 | .694650 |
| on | PC | −.000480 | −.003245 | −.007256 | .693627 | .692531 |
| on | neutral CPC | −.000480 | −.003245 | −.007256 | .693627 | .692531 |
| on | active CPC | −.000480 | −.003245 | −.007256 | .693627 | .692531 |

**All eight B gains are negative; seven return gains are negative.** Only replay
BP improves on returning A by this fixed measure. This tiny fixed native case
does not demonstrate useful B acquisition or a circadian advantage. No horizon,
seed, threshold, baseline or metric was adjusted afterward. Off/on differs in
work and selection exposure, so observed differences are not a causal replay or
efficiency estimate. Equal memory caps do not mean equal occupancy or compute.

## Verified costs and limits

The sole run executed60 native calls/returns:48 wakes/12 replay,45 PC-family
train steps/15 BP SGD steps,108 row exposures,64 score forwards,12 receipted
replay events, one joined RSS sampler. All native clocks, source learners and
on-mode deployed actors pass readback. CPC successful-update ages are0/3/6/9
with replay,0/2/4/6 without it; chemistry/traffic advance on replay too. App
wake/replay counters remain separate. On-mode plans and retention match across
all methods; no final label enters selection/training.

Pilot8.861001s, worker8.9903265s, parent9.9399215s; observed peak process RSS
311820288 bytes, JSON195747 bytes. This uses the prospectively declared Torch
engineering cap768 MiB/60s/2 MiB, preserving prior scientific budgets. Final
state tensor payloads: BP68 parameter+68 momentum+68 gradient bytes; PC80 head
bytes; CPC5268 head bytes including local RNG/adaptive state. These exclude
Python metadata, snapshots/copies, event logs and library/runtime overhead;
process RSS is separately reported. They are not efficiency claims.

Sixty-one focused tests/183 passing phases and five-path Ruff/format/Windows+
Linux mypy pass. Native tests verify actual native parity, complete owned state,
post-native failure atomicity, corruption refusal and component continuation.
They do not establish whole-stream resume. Independent SOURCE/GO/ACTUAL and
original R2.4 criterion mapping are in the artifact review.

R2.4 remains unchecked for application runtime/buffer/RNG/budget/receipt resume
and trajectory scope; full repository/release/scientific requirements remain
open. Accepted R2.1–R2.3 and earlier repeated-update evidence are reused unchanged.

Artifacts: [results and commands](../../artifacts/runs/r24-torch-feature-rehearsal-20261016/RESULTS.md),
[all outcomes](../../artifacts/runs/r24-torch-feature-rehearsal-20261016/outcomes.csv),
[all levels](../../artifacts/runs/r24-torch-feature-rehearsal-20261016/checkpoint-levels.csv),
[review](../../artifacts/runs/r24-torch-feature-rehearsal-20261016/REVIEW.md).
