# Fixed v14 full-stack sleep-trigger comparison

## Question and prospective boundary

Does the current adaptive trigger improve retention and adaptation against
periodic and no-sleep controls when the existing guarded component sleep can
reset chemistry, change structure, apply homeostasis, and replay observed
training rows? This protocol was frozen before v14 training or final-role
scoring. The v13 chemical-reset-only results do not select these settings.
P4.8b1 constructs and verifies the train-only replay opportunity schedule;
P4.8b2 runs and scores the models. A failed train-only gate stops final
release and requires a documented prospective amendment, not selection from
held-out results.

## Fixed source, roles, and methods

Protocol `continual_trigger_opportunities_v14` uses the existing arrived
four-role source and seeds **47 and 53**. A contains 160 balanced two-cluster
source rows with noise scale 0.8. B independently contains 160 source rows
with noise scale 1.0, rotation 40 degrees, translation `(0.9, -0.7)`, and a
fixed 0.5 development exposure fraction. Each source reserves 25% for its
unopened final role. The arrived splitter assigns 20% of exposed
development rows to the inner guard and 20% to outer selection; the rest
are train. B is generated only after all A wake epochs complete. Final
fields stay inaccessible until all arms and seeds pass global train-only
preflight. No setting is selected by outer labels.

Each seed has periodic, current-adaptive, and no-sleep arms. Each arm trains
backpropagation, ordinary predictive coding, and circadian predictive coding
on the same ordered arrived train rows for 12 full-batch wake epochs in A
and 12 in B. Initial shallow width is eight. Wake rates are the existing
continual defaults: backprop 0.12, ordinary and circadian PC 0.05. Both PC
methods use two latent inference iterations per wake update. Sleep interval
is phase-local four for periodic, with forced attempts; adaptive has no
periodic interval and uses the unchanged core readiness defaults
(`min_epochs_between_sleep=10`, energy window eight, plateau delta 0.001,
chemical variance threshold 0.02); no-sleep has no scheduled attempt.
All 24 completed wake epochs remain explicit decision opportunities.

## Fixed sleep and shared replay limits

The circadian model uses `components` sleep with chemical reset factor
0.45; enabled split and prune with at most one proposal of each per sleep;
enabled homeostasis with weight downscale factor 0.99; and enabled replay.
Other structural thresholds, eligibility, and adaptive budget defaults
remain the historical core defaults. The model's existing width bounds
are four through 32; across one arm/seed, no more than six applied splits
and six applied prunes are permitted. The runner records proposals,
applied changes, width, trainable parameter count, and guard rollbacks;
it stops before final release if a global cap is exceeded. Inner-guard
drop tolerance is zero.

One shared prediction-independent `recent_fifo` buffer observes only the
arrived train batch after each wake epoch, retains at most eight distinct
float64 labeled rows and 192 copied-array bytes, and selects the newest
two retained rows in retention order. Priority and class-balanced
sampling are disabled. The selection is a *potential* replay opportunity
at every wake epoch, independent of arm, model predictions, and guard
outcomes. The circadian core uses the same retention policy and caps.
Only after an accepted guarded sleep may the circadian model commit its
two replay updates and the two baselines receive detached copies of the
same selected rows. A rejected or skipped event yields zero baseline
replay. Replay learning rate is the existing 0.01, with two inference
iterations per ordinary/circadian PC replay update. Each accepted event
therefore adds two labeled-row presentations and two optimizer updates
per method, zero backprop inference iterations, and four PC inference
iterations per PC method. Replay exposure is reported as treatment work,
not assumed equal when event counts differ.

P4.8b1 checks this opportunity supply against the frozen v9 periodic
schedule under identical source and buffer settings. At phase-local
epochs 4, 8, and 12, train-role hashes, retention/order/selection IDs,
and method potential work must match exactly. V9 remains periodic-only
and keeps its historical identity. All v14 schedule records must be
repeatable byte for byte in new ignored local artifacts with no decision
or final values, model states, or scores.

## Train-only and final gates for P4.8b2

Before scoring, preflight all six arm/seed trials for manifest identity,
same-seed role and wake-row hashes, identical per-method initial state,
24 wake updates and ordered exposure per method, 24 trigger decisions,
typed sleep and guard events, exact retained/selected/applied replay IDs,
actual update/inference counts, structural IDs and global caps, model
clocks, and no final reads. This comparison does not require equal
*applied* replay work across arms: different accepted sleep counts are
the causal treatment. It requires exact same rows and per-event work
whenever a sleep is accepted. If the current adaptive rule never fires,
report that null without changing its thresholds, seeds, or stopping
point.

After the global seal, score saved post-A and post-B states on common
A/B final roles. Report each seed/arm/method's A-after-A, A-after-B, and
B-after-B accuracy and BCE, signed A forgetting, balanced score, sleep
attempts/acceptances/rollbacks, retained and applied replay work, wake
and inference work, width and parameter capacity, and paired differences.
Keep every null and negative result. Save and repeat local JSON; run focused
and full CPU tests, Ruff, mypy, formatting, and diff checks. No new trigger
heuristic is in scope for this fixed protocol.

## P4.8b1 observed train-only schedule

The two ignored local opportunity artifacts,
`data/trigger-replay-v14-opportunities-verified.json` and
`data/trigger-replay-v14-opportunities-verified-repeat.json`, are byte-identical
at SHA-256
`1c30e960aa2dee65a862434fb584e12eaa31bd14a71cb73b3eb919f9dfa5d168`.
The fixed manifest digest is
`22e90b3b3b5312ea52abc956f8ac997a457126bb07a9e03ef0f404079a5a87d0`.
Both seeds produced 24 opportunities, with 72 A and 36 B arrived train
rows per wake epoch. The FIFO retained eight distinct rows/192 bytes and
offered two detached rows at every epoch. All six phase-local periodic
subsets matched v9's train hash, retention, order, selected IDs, and
three method-work records exactly for each seed. Source sentinels raised
on any premature final read; no model trained or final role opened.
These schedule facts establish supply only, not applied replay or a
full-stack outcome. P4.8b2 remains open.
