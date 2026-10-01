# P6.3 matched schedule-factor train-only result

The [frozen preflight contract](p63-schedule-factor-preflight.md) completed all 33 model cells on development seeds 79/83/89, with 216 policy decisions, exact ordinary-PC/neutral parity and fixed capacity. Neither outer-selection nor final-source values were read. The ten reserved confirmation seeds remain unopened. This establishes the train path and work accounting; it contains no accuracy/retention comparison.

Two fresh bounded public processes saved exclusive request/result/audit sets under ignored `artifacts/runs/p63-schedule-factor-preflight/` and `artifacts/runs/p63-schedule-factor-preflight-repeat/`. Both deterministic results are byte-identical, SHA-256 `87931c5fc3fad5bf50d5b18d07901d04b8a8e827ab4fce4429c0f1644c813c41`. The request hashes are `131a4f62dd8929056a0541acd102fca00a3ff25cbd7ef4e5ec7cb0213d606ef7` and `4b5c2837c66493a714349b8dff83abb153a66d01ea6e5c639add01d7e1e53299`; neither directory has a failure sidecar. Independent readback verified the frozen manifest/20-source/adapter identities, both requests/audits, every role, opportunity, decision, parameter, work and capacity cell, and identical result bytes.

## Complete policy and work facts

Each policy has matched width-eight backprop, PC and neutral circadian heads. Each seed also has both planned width-12 no-replay references. All eleven heads receive 24 wake updates/1,296 train-row presentations. Ordinary PC and neutral circadian match tensor hashes after A and B and are checked after every wake and committed replay. Every width-eight head starts/ends/peaks at width eight/33 parameters; both references stay width 12/49 parameters.

| Seed | Policy | Attempted | Accepted | Rolled back | Replay updates per method | Executed policy replay, all three methods |
|---:|---|---:|---:|---:|---:|---:|
| 79 | periodic | 6 | 6 | 0 | 12 | 36 |
| 79 | adaptive | 0 | 0 | 0 | 0 | 0 |
| 79 | no sleep | 0 | 0 | 0 | 0 | 0 |
| 83 | periodic | 6 | 6 | 0 | 12 | 36 |
| 83 | adaptive | 0 | 0 | 0 | 0 | 0 |
| 83 | no sleep | 0 | 0 | 0 | 0 | 0 |
| 89 | periodic | 6 | 6 | 0 | 12 | 36 |
| 89 | adaptive | 0 | 0 | 0 | 0 | 0 |
| 89 | no sleep | 0 | 0 | 0 | 0 | 0 |

All six periodic boundaries (A/B epochs 4,8,12) committed identical selected IDs to the three methods, with no rollback. All three policies had the same available eight-row/192-byte FIFO supply at each of 24 opportunities per seed, including epochs with no sleep. Selection is the newest two retained IDs, checked against detached content and each neutral model's memory. There was no adaptive or no-sleep baseline replay. Wider references never replayed.

Total executed work was **900 optimizer updates**: 792 wake plus 108 applied replay and zero rejected replay executions. Each seed used 300 updates. There were 18 accepted guarded events and 36 inner-guard accuracy evaluations (648 evaluated guard examples), below the predeclared maximum 1,014 executed updates, hard cap 1,100 and 63-attempt ceiling. Every periodic PC/neutral head additionally performed 24 replay latent loops/24 example-iterations; all PC/neutral heads performed 48 wake loops/2,592 wake example-iterations. The complete result retains all 33 individual method facts, including zero replay and all planned references.

Retained labeled-array memory is scoped explicitly: 192 shared-supply bytes plus 192 in each of three neutral buffers per seed (768 persistent labeled-array bytes). Baselines consume private replay row copies from the shared supply and retain no independent arrays. Model parameters, metadata and temporary copies are outside this array-byte count. Whole-worker RSS includes those allocations and runtime overhead; this is not a per-arm memory result.

## Why the adaptive rule stayed inactive

Every adaptive head reached 15 eligible wake epochs with minimum spacing ten and a complete eight-energy window. Some windows passed the plateau requirement, but **none** reached the fixed chemical-variance threshold .02:

| Seed | Eligible epochs | Plateau passes | Chemical-variance range | Variance passes |
|---:|---:|---:|---:|---:|
| 79 | 15 | 7 | .000419–.001898 | 0 |
| 83 | 15 | 4 | .002278–.006535 | 0 |
| 89 | 15 | 7 | .001135–.003786 | 0 |

This is an inactive-trigger finding for the neutral replay-only head's actual wake history and chemistry. It does not establish a better accuracy/cost tradeoff, and the thresholds were not changed to generate activity. The wider/full model and minus-one requirements remain separate.

A test fixture supplies constant wake energy and sufficient chemical variance under the unchanged trigger settings, accepts its guard, and commits precisely at wake batches 10 and 20. A separate forced guard-rejection test observes every replay training execution, restores model parameters/clocks, applies no baseline replay and checks the rejected-execution cost ledger. These fixture results validate paths; they are not substituted for the official inactive adaptive observation. Real three-seed outer/final raising sentinels and a B-arrival check also pass.

The two workers completed in .662/.672 seconds, with sampled whole-worker RSS peaks 43,040,768/42,766,336 bytes and 23 samples each, under the 120-second/256-MiB caps. The runtime is NumPy CPU on Windows 11, Intel Core i7-12700K, Python 3.14.7 and NumPy 2.4.6. Sampling may miss a short peak. The related 50-test gate, Ruff, mypy (335 source files), new-file format and diff checks passed; the full CPU/CUDA suites, scored development, confirmation and large sweeps were skipped.

P6.3c5 is complete for its train-only acceptance criteria. P6.3c6 must prospectively bind this result hash and all 33 cells, freeze every within-method policy contrast, then compare the full unscored fact object globally before reading outer roles. Full-minus-one, independent confirmation and per-arm resource attribution remain open.
