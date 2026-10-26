# Recurrent learning and retention across A → B → C → A

The fixed conventional two-head recurrent learner fits both C teacher sequences
and executes both C games at the final checkpoint. It also executes A-red, but
fails A-blue and both B games. B-red's earlier exact teacher sequence is lost.
This is partial useful learning with a retention failure, not reliable recovery,
generalization, a CPC advantage or repository-wide readiness.

The sole case completed128updates/1152TRAIN rows through the original delayed
stream. Two raw148/H8/K7/seed17/.03/.9 native recurrent heads select from the first
actual PUBLIC cue; no environment cue/targets route the learner. Each head pays
64updates576rows. No replay/probes/final/promotion, teacher collection, historical
archive/model refit or favorable-run selection. All fitting precedes readouts
from owned0/32/64/96/128states. This A32/B32/C_train32/A32 case does not retry the
earlier incomplete130A128case or renew its expired clock.

| Updates | A correct /18 (exact cues /2) | B correct /18 (exact cues /2) | C correct /18 (exact cues /2) |
| ---: | ---: | ---: | ---: |
| 0 | 0 (0) | 3 (0) | 3 (0) |
| 32 | 10 (0) | 10 (0) | 10 (0) |
| 64 | 11 (0) | 15 (1) | 11 (0) |
| 96 | 14 (0) | 16 (0) | 18 (2) |
| 128 | 17 (1) | 14 (0) | 18 (2) |

These are TRAIN teacher metrics, not native success. C64→96 changes11/18→18/18;
both sequences remain exact after the returned-A phase. B-red changes9/9at64 to
8/9at96/128. B's mean CE improves.628591→.412390at96 while losing that exact
sequence, illustrating why mean fit alone does not establish task retention.
After returning to A, B CE worsens to.668000 and total correctness falls16→14/18.
A's CE improves monotonically1.964951→1.080624→.898878→.626638→.269412; it did
not demonstrate prior A forgetting and subsequent recovery. At32 neither native
A game succeeds, so initial acquired-A retention cannot be claimed.

| Native checkpoint | Context | Blue | Red | Successes | Mean return |
| ---: | --- | --- | --- | ---: | ---: |
| 32 | A | truncates16 | truncates16 | 0/2 | 0 |
| 128 | A | truncates16 | goal9 | 1/2 | .246875 |
| 128 | B | truncates16 | truncates16 | 0/2 | 0 |
| 128 | C_train | goal9 | goal9 | 2/2 | .49375 |

All eight episodes complete/close, all actions are actual native seven-class
first-argmax inputs, and107physics steps return. There are no native0/64/96
readouts. C teacher-sequence retention is measured; native C success is observed
only at128, so native C96→128retention is not independently measured. Final A-red
success is new observed competence, not recovery of an initially successful A32.

App104.0909988s, metered body107.8867847s, parent108.4873676s; actual-self peak
359,546,880B under frozen420s/440s/768MiB. The original400s TRAIN lifetime clock
continues throughout validation, with one200s zero-update/replay readout budget.
Full source/actor/learned parameters/gradients/momentum/receipts/checkpoints and
global NumPy/Torch RNG preservation pass at completion. This is an isolated
candidate, with no publication of its weights to the actor.

Costs include176native owner constructions,528randn calls/229504private initial
scalars,128backwards1152rows/128SGD steps,265forwards1529rows (128TRAIN +30score
+107live calls),20054native snapshot validations/20532optimizer constructions,
117445Tensor clones/30932042cloned scalars, and one94705B packaged TRAIN read.
No GPU/PC/LAPACK/teacher/final/probe work. The two-head2638parameter architecture
and per-parameter update/momentum schedule differ from old raw1319/history2503;
neither same updates nor this timing establishes matched computation/capacity.

[Protocol, exact commands, failures, scoped checks and self-review](../../artifacts/runs/recurrent-context-stream-20261025/RESULTS.md).
[Raw local result](../../artifacts/runs/recurrent-context-stream-20261025/native-result.json):
1487883bytes, SHA2560b76d534e37b4c13aee5bd61bfdd48477e7ff82d20b8ace65c9aeb7ca1725763.
Python3.14.7/NumPy2.4.6/Torch2.14.0+cpu. Saved-only reconciliation recomputes all
30CE/accuracy/exact metrics from270rows and checks377argmax decisions/107physical
transitions/cleanup/phase summaries/costs, with zero model or simulator work.
New review is single-agent self-review, not independent acceptance. No new
statistical reliability, heldout, nativeRL, circadian or release claim.

The next concrete hypothesis is bounded whole-episode TRAIN replay against this
saved no-replay control, using existing StreamRehearsal and complete episode
schema. It addresses the measured B loss without a new optimizer or model/seed
search. Freeze one policy/dose/counter/receipt/memory/cost protocol after focused
software checks. Additional optimization/exposure must be explicit; it is not
matched compute. Existing R2.1 generic disposable probes already admit148/K7 and
owned native forks; do not rebuild that infrastructure merely to extend tracking.
