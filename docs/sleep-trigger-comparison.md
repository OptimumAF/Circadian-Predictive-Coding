# Fixed v13 sleep-trigger timing comparison

## Question and scope (frozen before v13 training/outcomes)

Does the current adaptive sleep trigger schedule useful consolidation
under stationary train noise and a genuine A-to-B distribution change,
compared with periodic forced sleep and no sleep? This first comparison
isolates trigger timing with a fixed-width, replay-free component sleep
that performs the historical chemical reset. It cannot establish the
effect of the full structural/replay/guarded sleep stack; P4.8b retains
that question. No new plateau, drift, hysteresis, or cooldown rule is
selected from this result.

## Frozen data and evaluation roles

Protocol `sleep_trigger_timing_v13` uses NumPy, new seeds 41 and 43,
conditions `stationary_noise` and `axis_shift`, and all three arms
`periodic`, `adaptive`, `no_sleep`: twelve trials. Each phase has forty
balanced development rows and forty independently generated final rows.
The base class centers are ±1.2 on feature x with Gaussian noise 0.55.
Phase B retains feature x in the stationary condition and moves the
centers to feature y in the shift condition. Paired seeds draw the same
noise within a phase across conditions; phase A is identical for both
conditions. The existing stratified four-role splitter reserves 24
training, eight inner-guard, eight outer-selection, and forty unopened
final rows per phase. Neither decision role selects settings or controls
sleep. Final roles are released once per condition/seed only after every
trial passes train-only preflight.

Each training epoch presents all 24 training role rows with their
unchanged labels, adding a predeclared Gaussian feature jitter of 0.15.
The jitter RNG seed is `1000*seed + 1` for A or `1000*seed + 2` for B,
independent of arm and condition. Thus the stream has stationary noise
at each epoch without sampling from guard, selection, or final roles;
the B condition differs only in the phase distribution. Hash the exact
per-epoch effective training arrays and require all arms within a
condition/seed to share them.

## Frozen models, schedules, and work

Every arm starts from the same seed-matched 8-wide shallow circadian
model and runs sixteen full-batch updates per phase. Wake rate is 0.05,
with two latent inference iterations at rate 0.1. Capacity is fixed at
eight; split and prune budgets are zero. `components` sleep enables only
chemical reset with the historical factor 0.45. Replay, homeostasis,
structural change, adaptive budget scaling, and reward modulation are
off. This keeps the trainable parameter count and wake work equal; any
number of performed chemical resets is the treatment, and its actual
attempts/events must be reported rather than treated as matched work.

Each of 32 completed epochs is a decision opportunity. The periodic
arm attempts forced sleep at epochs 8, 16, 24, and 32. The adaptive arm
has no periodic interval and uses the unchanged core defaults:
`min_epochs_between_sleep=10`, `sleep_energy_window=8`,
`sleep_plateau_delta=1e-3`, and
`sleep_chemical_variance_threshold=0.02`. The no-sleep arm schedules no
call in the same component-capable model. Use the existing runner
`decide_sleep_attempt` and core `should_trigger_sleep`/`sleep_event`
semantics. There is no inner guard or rollback in this timing control.
Log every due/attempt/performed decision, actual core wake/sleep/replay
clocks, energy improvement and chemical variance available at each
decision, and unchanged weights across a chemical-reset-only event.

Preflight all twelve unscored trials for exact manifest, role/batch
hashes, common initialization, 32 opportunities, 32 wake updates, 768
training presentations, 64 inference loops, 1,536 example-inference
loops, zero replay, no structural changes, width eight, constant
parameter count, and coherent actual attempt/event clocks. A failed
train-only gate stops final release; it does not prompt schedule or
threshold changes. After the global gate, score saved post-A and post-B
states on common A/B final roles. Report A-after-A, A-after-B, and
B-after-B accuracy and binary cross entropy, signed accuracy forgetting
(A-after-A minus A-after-B), every seed/condition/arm, and paired
periodic/adaptive minus no-sleep differences. The 40-row final accuracy
resolution, possible weak learning, and event-count differences limit
causal interpretation. Save train-only and scored JSON locally to new
paths and repeat each byte for byte. Do not choose a winner or infer
benefit from a trigger decision alone.

## Observed fixed run

Both ignored train-only artifacts (`data/sleep-trigger-v13-train.json` and
its repeat) are byte-identical, SHA-256
`5b1ae508b074f54198518bc908b733fbae8eff6f17c15868be9df428369c279f`.
They contain twelve preflighted cells and no final-role hashes, labels,
or scores. The scored
artifacts (`data/sleep-trigger-v13-result.json` and its repeat) are also
byte-identical, SHA-256
`e8dc920c1ca6944d5f0c48cc87023aaed0548cd830ac07a6c05746fefa6a9821`.
The fixed manifest digest is
`b361d99e1354d3de3a4099859177678a77bbb162ac28a94422d0162024ebb183`.
Every trial used 32 decision opportunities, 32 updates, 768 training
presentations, 64 inference loops, 1,536 example-inference loops, zero
replay, width eight, and 33 trainable parameters. Periodic performed
four chemical-reset-only events at the predeclared epochs; adaptive and
no-sleep performed zero. No structural changes or weight updates
occurred inside sleep.

The unchanged adaptive rule had 23 spacing/window-eligible decision
opportunities per trial. The table separates the two remaining gates;
their conjunction was zero in all four cells. Its threshold was not
changed after this observation.

| Condition | Seed | Plateau windows ≤0.001 | Variance windows ≥0.02 | Maximum observed variance |
|---|---:|---:|---:|---:|
| stationary noise | 41 | 0/23 | 0/23 | .012593 |
| stationary noise | 43 | 0/23 | 0/23 | .013546 |
| axis shift | 41 | 7/23 | 0/23 | .006380 |
| axis shift | 43 | 0/23 | 0/23 | .007998 |

The complete final-role metrics follow. `A/A`, `A/B`, and `B/B` are
accuracy at the named training/evaluation boundaries. Forgetting is
signed `A/A − A/B`; lower BCE is better. The adaptive and no-sleep
rows are retained separately even where exactly equal.

| Condition | Seed | Arm | A/A | A/B | B/B | Forgetting | A/A BCE | A/B BCE | B/B BCE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| stationary noise | 41 | periodic | .925 | .925 | .975 | .000 | .586928 | .473558 | .435863 |
| stationary noise | 41 | adaptive | .925 | .925 | .975 | .000 | .589645 | .485031 | .449460 |
| stationary noise | 41 | no sleep | .925 | .925 | .975 | .000 | .589645 | .485031 | .449460 |
| stationary noise | 43 | periodic | .650 | .825 | .800 | -.175 | .671285 | .492881 | .504657 |
| stationary noise | 43 | adaptive | .625 | .825 | .800 | -.200 | .674670 | .506821 | .520320 |
| stationary noise | 43 | no sleep | .625 | .825 | .800 | -.200 | .674670 | .506821 | .520320 |
| axis shift | 41 | periodic | .925 | .625 | .975 | .300 | .586928 | .629860 | .460977 |
| axis shift | 41 | adaptive | .925 | .625 | .975 | .300 | .589645 | .627222 | .477811 |
| axis shift | 41 | no sleep | .925 | .625 | .975 | .300 | .589645 | .627222 | .477811 |
| axis shift | 43 | periodic | .650 | .700 | .900 | -.050 | .671285 | .640534 | .410772 |
| axis shift | 43 | adaptive | .625 | .700 | .925 | -.075 | .674670 | .647266 | .416222 |
| axis shift | 43 | no sleep | .625 | .700 | .925 | -.075 | .674670 | .647266 | .416222 |

The paired adaptive-minus-no-sleep delta is exactly zero for every
printed metric and all four seed/condition pairs. For periodic minus
no-sleep, the paired accuracy deltas `(A/A, A/B, B/B, forgetting)` are
`(0,0,0,0)` for stationary seed 41 and axis-shift seed 41;
`(+.025,0,0,+.025)` for stationary seed 43; and
`(+.025,0,-.025,+.025)` for axis-shift seed 43. The corresponding BCE
deltas `(A/A, A/B, B/B)` are:

| Condition | Seed | Periodic minus no-sleep BCE delta |
|---|---:|---|
| stationary noise | 41 | `(-.002716, -.011473, -.013597)` |
| stationary noise | 43 | `(-.003384, -.013941, -.015663)` |
| axis shift | 41 | `(-.002716, +.002638, -.016834)` |
| axis shift | 43 | `(-.003384, -.006731, -.005450)` |

Periodic changes the outcome under fixed capacity, but the small
accuracy differences and mixed BCE/shift outcome do not establish a
general benefit. The adaptive arm's equality to no-sleep is an actual
zero-event observation at unchanged defaults, not evidence that
adaptive timing is useless on other models or longer streams. Forty
final rows give .025 accuracy steps, there are only two seeds, and this
is a synthetic shift with chemistry-only sleep. P4.8b therefore keeps
the full structural/replay/guarded sleep and broader shift comparison
open. No threshold, seed, metric, or work budget was retuned, and no
new trigger heuristic was selected.
