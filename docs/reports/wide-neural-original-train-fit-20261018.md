# Original width-32 TRAIN quartet — finite full-fit negative

## Fixed question and outcomes

Can each existing neural learner fit all54 original TRAIN actions within128
updates at the one fixed H32 recipe? The four prescribed cases completed and
qualified. Every full-fit witness is **false**.

| Method | Correct /54 | Exact /6 | Mean CE | Full fit |
| --- | ---: | ---: | ---: | --- |
| bp | 51 | 3 | 0.186232669571 | False |
| pc | 30 | 0 | 1.121374491357 | False |
| neutral_cpc | 30 | 0 | 1.121374491357 | False |
| gated_cpc | 30 | 0 | 1.192282153468 | False |

| Fixed contrast at128 | CE difference | Accuracy difference | Exact-trace difference |
| --- | ---: | ---: | ---: |
| gated_cpc-minus-pc | 0.070907662111 | 0.000000000000 | 0.000000000000 |
| gated_cpc-minus-neutral_cpc | 0.070907662111 | 0.000000000000 | 0.000000000000 |
| pc-minus-bp | 0.935141821785 | -0.388888888889 | -0.500000000000 |
| neutral_cpc-minus-pc | 0.000000000000 | 0.000000000000 | 0.000000000000 |

CE lower and accuracy/exact higher are better. Active CPC has higher CE than PC
and neutral at equal accuracy and exact-trace rate. PC/neutral raw readouts match
exactly (maximum probability difference0); all four initial readouts match.
These results do not establish a circadian advantage, useful acquired retention/
recovery or general model incapacity. One seed/catalogue is exploratory. Original
H8, weighted/relaxation, linear and kernel negative/mixed results remain unchanged.
Equal width/data/seed/exposure is not matched native computation.

Each fixed0/128readout,24final per-cue episode values, raw432probabilityrows and
512native diagnostics/receipts are retained in the stage JSON. Common unweighted
CE clips true probabilities at1e-12; firstargmax, accuracy and exact metrics are
unchanged. Original input bounds report54distinct conflict-free canonical keys,
accuracy ceiling1/CEfloor0; accepted helper evidence is reused, not reconstructed
from unpersisted weights or keys. No run, width, dose, endpoint, metric or seed
selection, and no retry occurred.

## Costs

Each case:128nativeupdates/6912trainingrows,12predictioncalls/108rows. Total512/
27648, P48/432. BP128backwards/SGD; each PC family128parameterupdates and13824
nominal latent-row steps (41472total). These derive from unchanged code and
completed native counters, not an external primitive meter. Native returned PC
legacy energy is distinct from prior CE and is not compared to BP as one loss.

Each model's shared trainable arrays contain9735float32parameters/38940array-
equivalentbytes. Adaptive/optimizer/gradient/RNG/Python state, copies, checkpoint
readers and temporary arrays add memory. Each case retains16422912logicalbytes
of stream input copies, at most2fit checkpoints plus bounded receipts/readouts.
Logical parameter/copy bytes are not peak RSS or a speed claim.

| Method | Body seconds | Supervised actual seconds | External RSS bytes |
| --- | ---: | ---: | ---: |
| bp | 4.8261640 | 7.4645353 | 346484736 |
| pc | 3.3591177 | 5.8358296 | 263503872 |
| neutral_cpc | 4.4915953 | 7.0247318 | 264523776 |
| gated_cpc | 4.5269271 | 6.9452290 | 264626176 |

Parent measurement includes imports/thread setup and all validation, restores,
projection, copies and reporting. All cases use actual CPU1/interop1/GPU0 under
PID supervision, RSS768MiB/stage32MiB ceilings. No replay/probe/final/teacher/
physics/live-game work or original control fitting. Each body60/actual90/cleanup
100/outer105; clocks/charges never renew. Science28.8684313999/480 and
engineering1004.3378845999/1200seconds CLOSED; all earlier grants CLOSED.

## Verification and remaining work

All11commands exit0 with no resource problem. SOURCE/PROTOCOL and ACTUAL independent
GO cover original TRAIN source/action identity, stdout/output equality, all432
probability/CE/accuracy/exact/context/gap reductions,512delay-one receipts and
streamcopy equality, diagnostic IDs, costs/resources and all fixed contrasts.
Unchanged194engineering checks/statics are reused; production source did not
change. No repository-wide, clean-clone or release claim.
[Frozen protocol](../../artifacts/runs/wide-neural-real-fit-20261018/PROTOCOL.md),
[all data and commands](../../artifacts/runs/wide-neural-real-fit-20261018/RESULTS.md),
[independent review](../../artifacts/runs/wide-neural-real-fit-20261018/REVIEW.md).

R4.2 needs complete target-free games. The existing evaluator currently admits
older H8 policies/clocks, so H32 joint endpoints require a narrow explicit extension.
Keep both original A cue slots, including the imperfectly fitted one; do not select
successful traces. Reuse native readers/game execution/zero-update guards. No
additional capacity/dose/classifier or audit increment is justified by this result.

Extend the existing known-TRAIN A rollout evaluator and constructive-fit CLI with an explicit H32 quartet option at the genuine 128-update/6912-row checkpoint. Run the fixed blue/red A games on owned frozen readers under a separate zero-update budget, retaining complete actions/returns/success and physics/prediction costs. Preserve H8 defaults, both cue slots, original training clocks/charges, source/actor state and final/probe exclusion.
