# Recurrent key/door control — 2026-10-16

Two new recurrent arms reuse the accepted eight feedforward arms. All data, seeds,
contexts, metrics and replay selections match; architecture, initialization,
native objective and computation differ. No baseline was retrained. This is a
small fixed-exposure result, not an equal-compute or scientific-advantage claim.

[New raw output](../../artifacts/runs/r42-recurrent-key-door-20261016/pilot.json),
[accepted original output](../../artifacts/runs/r42-key-door-20261016/pilot.json),
[commands/review/limits](../../artifacts/runs/r42-recurrent-key-door-20261016/RESULTS.md).

## All arms and frozen CE directions

Gains: B CE0−CE4, C CE0−CE6, Areturn CE6−CE8; Aloss CE6−CE2. Paired-cue means.

| Method | Replay | B gain | C gain | A loss | Areturn gain | Live success | Return | Live forwards | Applied steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bp | False | 0.01103795087 | -0.01854898945 | -0.0134420243 | 0.009112964868 | 0/40 | 0.0 | 88 | 48 |
| pc | False | 0.00797907638 | 0.001774857201 | 0.007351447556 | 0.008649505335 | 0/40 | 0.0 | 67 | 27 |
| neutral_cpc | False | 0.00797907638 | 0.001774857201 | 0.007351447556 | 0.008649505335 | 0/40 | 0.0 | 67 | 27 |
| gated_cpc | False | 0.007978631436 | 0.001773874226 | 0.007351173637 | 0.008648196779 | 0/40 | 0.0 | 67 | 27 |
| bp | True | 0.01469234942 | -0.04595247957 | -0.03810016739 | 0.01055307064 | 0/40 | 0.0 | 88 | 48 |
| pc | True | 0.007067920443 | -4.208944093e-05 | 0.006499495647 | 0.005248023872 | 0/40 | 0.0 | 67 | 27 |
| neutral_cpc | True | 0.007067920443 | -4.208944093e-05 | 0.006499495647 | 0.005248023872 | 0/40 | 0.0 | 67 | 27 |
| gated_cpc | True | 0.007067198796 | -4.434104902e-05 | 0.006499082546 | 0.005246379902 | 0/40 | 0.0 | 67 | 27 |
| recurrent_bc | False | 0.01557807966 | -0.02036055468 | -0.01777396133 | 0.01091118596 | 0/40 | 0.0 | 82 | 42 |
| recurrent_bc | True | 0.02212122823 | -0.05139326488 | -0.04653693646 | 0.01180247044 | 0/40 | 0.0 | 88 | 48 |

## New recurrent curves: every context/checkpoint/replay mode

| Replay | Wake | Context | Mean CE | Action accuracy | Success | Return |
|---|---:|---|---:|---:|---:|---:|
| False | 0 | A | 1.609807268 | 0.125 | 0/2 | 0.0 |
| False | 0 | B | 1.603017442 | 0.1875 | 0/2 | 0.0 |
| False | 0 | C | 1.619304238 | 0.0625 | 0/2 | 0.0 |
| False | 0 | D | 1.611868751 | 0.0625 | 0/2 | 0.0 |
| False | 2 | A | 1.594626854 | 0.375 | 0/2 | 0.0 |
| False | 2 | B | 1.602839443 | 0.125 | 0/2 | 0.0 |
| False | 2 | C | 1.630808549 | 0 | 0/2 | 0.0 |
| False | 2 | D | 1.597051141 | 0.4375 | 0/2 | 0.0 |
| False | 4 | A | 1.57792264 | 0.5625 | 0/2 | 0.0 |
| False | 4 | B | 1.587439362 | 0.1875 | 0/2 | 0.0 |
| False | 4 | C | 1.645227871 | 0 | 0/2 | 0.0 |
| False | 4 | D | 1.580924348 | 0.5625 | 0/2 | 0.0 |
| False | 6 | A | 1.576852892 | 0.5625 | 0/2 | 0.0 |
| False | 6 | B | 1.571734237 | 0.3125 | 0/2 | 0.0 |
| False | 6 | C | 1.639664792 | 0 | 0/2 | 0.0 |
| False | 6 | D | 1.580027919 | 0.6875 | 0/2 | 0.0 |
| False | 8 | A | 1.565941706 | 0.5625 | 0/2 | 0.0 |
| False | 8 | B | 1.560137528 | 0.375 | 0/2 | 0.0 |
| False | 8 | C | 1.641455409 | 0 | 0/2 | 0.0 |
| False | 8 | D | 1.569481342 | 0.625 | 0/2 | 0.0 |
| True | 0 | A | 1.609807268 | 0.125 | 0/2 | 0.0 |
| True | 0 | B | 1.603017442 | 0.1875 | 0/2 | 0.0 |
| True | 0 | C | 1.619304238 | 0.0625 | 0/2 | 0.0 |
| True | 0 | D | 1.611868751 | 0.0625 | 0/2 | 0.0 |
| True | 2 | A | 1.58064091 | 0.4375 | 0/2 | 0.0 |
| True | 2 | B | 1.603264569 | 0.125 | 0/2 | 0.0 |
| True | 2 | C | 1.641849459 | 0 | 0/2 | 0.0 |
| True | 2 | D | 1.583452696 | 0.5 | 0/2 | 0.0 |
| True | 4 | A | 1.54389355 | 0.5 | 0/2 | 0.0 |
| True | 4 | B | 1.580896214 | 0.1875 | 0/2 | 0.0 |
| True | 4 | C | 1.675288562 | 0 | 0/2 | 0.0 |
| True | 4 | D | 1.547942821 | 0.5 | 0/2 | 0.0 |
| True | 6 | A | 1.534103973 | 0.5 | 0/2 | 0.0 |
| True | 6 | B | 1.563133191 | 0.1875 | 0/2 | 0.0 |
| True | 6 | C | 1.670697503 | 0 | 0/2 | 0.0 |
| True | 6 | D | 1.538726655 | 0.5 | 0/2 | 0.0 |
| True | 8 | A | 1.522301503 | 0.5 | 0/2 | 0.0 |
| True | 8 | B | 1.553475464 | 0.25 | 0/2 | 0.0 |
| True | 8 | C | 1.663528954 | 0 | 0/2 | 0.0 |
| True | 8 | D | 1.52739003 | 0.5 | 0/2 | 0.0 |

All80 new recurrent live episodes end on illegal actions, zero success/return.
The accepted320 feedforward episodes also remain illegal/zero-success. CE/action
score changes do not establish acquisition, retention, context recovery or D
layout generalization. Neither architecture solves this small fixed protocol.

## Observed cost and limits

New recurrent work:20 native SGD/160training rows,80classifier forwards/640rows,
80fresh live sessions/170live forwards/90applied live steps. Total970 unrolled
timesteps and218 actual environment steps including128 teacher steps.181float32
parameters=724B; after an update gradients and momentum each724B; live activation
32B. Snapshot/session birth copies, temporary BPTT graph and metadata are excluded
from logical payload totals, included in process resource observations.

Pilot 12.6665067s; sampled process RSS 313,589,760B. One joined sampler, no violations; original90/120s/768MiB/8MiB caps pass.

Old head payload includes its own native auxiliary tensors, so do not compare
that aggregate as a pure parameter count. Off/on exposure differs; no causal
replay, equal-memory or timing speedup claim. Raw scores/costs/negatives remain.

Next: extend the existing isolated fresh-task probe to categorical complete
episodes with explicit role/class binding and unchanged binary scoring. Its
current scorer is binary-only. Verify it before one frozen train-only fresh-task
learnability case; probe updates must stay separate from deployed/benchmark state.
Full R4.2 and separately evaluated recurrent RL remain open.
