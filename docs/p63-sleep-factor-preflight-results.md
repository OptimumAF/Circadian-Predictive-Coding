# P6.3 guarded sleep-factor train-only preflight result

The [frozen contract](p63-sleep-factor-preflight.md) ran on development
seeds 67/71/73 with no outer-selection or final-test value read. The
result establishes train-only feasibility and an auditable guard/capacity
path. It contains no accuracy or retention comparison between arms.

Two separate public bounded processes saved exclusive request/result/audit
sets under ignored `artifacts/runs/p63-sleep-factor-preflight/` and
`artifacts/runs/p63-sleep-factor-preflight-repeat/`. Both complete result
JSON files are byte-identical, SHA-256
`e3140d5a0529dc3023b90b60801d8f9caf2e15dd8edb4a85fa875f76f60d624d`.
The request SHA-256 values are
`2edce528fbb161439959fe695fc918dc859febb48a39bce99305248dfe39a2d2`
and
`949f9a6a474484b832d584afc016ee0dc3d9aa595b02bbbb1966180077d7b227`.
Read-only revalidation matched the frozen manifest/source/adapter identities,
both requests and audit file hashes, all 27 arm facts, role counts/hashes,
guard decisions and work/capacity bounds. Neither run has a failure
sidecar.

| Train-only fact | Observed |
|---|---|
| Development cells | 3 seeds × 9 fixed arms = 27 |
| Wake optimizer work | 648 total; 24 updates/1,296 row presentations per arm/seed |
| PC/circadian latent work | 48 loops and 2,592 example-iterations per arm/seed |
| Replay | Zero memory, updates and exposure in every trained arm |
| A-boundary guard | 15 attempts, 30 A-inner accuracy evaluations; all accepted |
| Structural proposals/commits | One split and one immediate prune per seed; all three accepted |
| Structural capacity | Starts/ends width 8, 33 parameters; transient width 9, 37 parameters |
| Fixed and planned capacity | Width 8/33 parameters; planned width 12/49 parameters |
| Role counts per seed | A train/inner/outer 72/24/24; B 36/12/12; six hashes |
| Process RSS | First 38,404,096 → 42,614,784 peak bytes; repeat 38,600,704 → 42,672,128 peak bytes; 10 samples each, 5 ms interval |
| Worker wall time | 0.440 and 0.448 seconds, below 120-second limit |

Ordinary PC and neutral circadian sham have exact final parameter parity.
The gated sham and gated reset arms match before and immediately after
the A-boundary event; the reset changes chemical state without directly
changing parameters. Their minimum A plasticity factors are 0.906077,
0.868976, and 0.869441, respectively by seed, so the gate is active.
The accepted homeostasis event changes parameters at fixed width.

A guard-rollback regression test forces the structural post score below
its pre score. The same one-split/one-prune proposal remains in telemetry,
applied changes become empty, and the model returns to its pre-sleep
parameter hash and width. This test does not alter the public runs.

The A-inner guard accuracies before and after sleep were respectively
0.333333/0.333333, 0.958333/0.958333, and 1.000000/1.000000 on the
three seeds for every arm. The first seed's coarse 24-row guard is weak
evidence about generalization; acceptance only means no measured A-inner
accuracy drop at that one boundary. The 256-MiB RSS cap applies to the
whole worker, and sampled high-water RSS can miss a shorter peak between
samples. These are not per-arm memory measurements. The width-12 arms are
planned capacity references and have different compute per update.

Final-source inputs/labels and outer-selection inputs/labels were replaced
by raising sentinels in the three-seed regression and stayed unread. B
arrived after A wake and sleep. The result is not independent confirmation.
The next gate is a separately frozen outer-selection scorer with a global
train-only completion check and every arm/seed retained; confirmation
seeds and final roles remain unopened.
