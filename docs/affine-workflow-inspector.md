# Inspect affine learning and native games

Fresh A3→A4→A3 reports use the same command below. Their export includes all
three causal stages, twelve public-view game slots, acquisition/retention/return
summary and separate teacher/fit/replay/game costs. [A4 export evidence](reports/a4-affine-inspector-20261025.md)
distinguishes passing saved-data/software checks from pending browser interaction
and visual verification. This extension does not rerun training or games.

Export a saved train-and-play result with the existing inspector command:

```powershell
python -m scripts.build_key_door_inspector --source-artifact fresh-affine.json --output affine.html
```

Open the resulting standalone HTML file in a browser. The file is offline and
read-only: it runs no learner, simulator, training, probe or promotion. Output
must be new. No experiment archive is required when you supply your own result
from the [portable train-and-play command](portable-affine-workflow.md).

The page shows three distinct kinds of evidence:

- Recorded TRAIN accuracy and cross-entropy for A/B/C blue/red at 0/2/4/6 fits.
  Dots show recorded values; missing checkpoints are not interpolated. These
  teacher-forced scores do not establish autonomous task success.
- Actual known B/C game traces: the public 7×7 view, retained initial frame,
  seven probabilities, selected action, return, ending flags and closure.
  Playback includes the terminal observation without inventing another decision.
- Original causal receipts, arrival/replay exposures, bounded catalogue storage
  and training/readout budgets. Joint replay adds fitting exposure. Planned rows
  and evidenced committed work remain distinct.

Source buttons open the exact saved JSON for every observed metric, frame,
decision, receipt and cost. The complete original source is retained in the
page. A missing game slot has no invented source record. Failed receipt capture
does not turn paid updates into zero work; missing/false/null values stay visible.
Training and readout segment times/RSS are not whole-process totals. When the
source lacks total runtime, total RSS or a caller RNG proof, the page says so.

The importer checks bounded JSON, unique keys, finite numbers, role/counter/
score/receipt consistency and native observation/action/return chains before
publishing. It does not authenticate external history or restore saved weights.
TRAIN data remains TRAIN; probe/final relabeling is refused by this report route.
Legacy MiniGrid, probe and FIFO inspector formats retain their existing routes.

The delivered [recorded example](../artifacts/runs/affine-workflow-inspector-20261024/affine.html)
uses the accepted installed run: six fits, 108 row exposures (54 arrivals and
54 joint replay), four known games solved in nine steps each. It is not evidence
of unseen-layout generalization, recurrent RL, reliable recovery or a circadian
advantage. Source/procedure/result self-review, not independent acceptance.

Focused checks cover scalar consistency, roles, malformed inputs, ownership,
negative outcomes, partial native work and failed receipt readback. Desktop and
narrow browser checks cover actual playback, source dialogs and old routes.
[Exact commands, failures, costs and skipped checks](../artifacts/runs/affine-workflow-inspector-20261024/RESULTS.md).

Extension: add recorded formats through the existing application projection and
dispatcher. Reuse the native decoder and renderer; keep missing source pointers
absent and preserve the original scientific scope. No model or environment
dependency belongs in a recorded projection.

## Ordinary recurring reports

The same command now accepts the original A/B/C/G/A result produced by
`circadian-affine-game --example-affine-recurring`:

```powershell
python -m scripts.build_key_door_inspector --source-artifact new-recurring-run.json --output recurring-learning.html
```

The page shows eight original TRAIN sources at0/2/4/6/8/10fits, both new G
teacher records, ten delayed receipts and the four known G/A game traces.
Acquisition and return stages appear beside recorded accuracy and CE. Native
games are observed after the complete TRAIN sequence, including A's return;
there are no invented earlier game outcomes. Source buttons retain original
observations, probabilities, native effects, endings, costs and preservation flags.

The saved recipe uses252fit-row exposures:90arrivals (72distinct source rows)
and162joint old-row reuses. The54-row bank and36-row maximum fit are separate
from Python/native copies and process RSS. Protocol-plan totals are labeled as
plans; committed exposure is tied to actual receipts. Teacher collection,
diagnostics, original TRAIN/readout budgets and the workflow clock/RSS each keep
their scope. External interpreter/import costs are not in this source and remain
unknown in the viewer. Failed collection, paid-but-uncaptured updates, partial
games, false preservation flags and unrun slots remain explicit.

[Open the recorded installed-run example](../artifacts/runs/recurring-affine-inspector-20261024/recurring-learning.html).
Its four known games succeed in9steps each, return0.49375. Final TRAIN accuracy
is18/18 per context, while A/C confidence is worse than their first fitted
checkpoints. A is trained again on return; this does not demonstrate recovery
after actual forgetting, heldout reliability, equal computation or CPC benefit.
Old normal/private reports retain their separate modes and rules.

92focused cases qualify per platform, with scoped statics and single-agent
source/procedure/result self-review. A saved-only actual export and12browser
assertions at desktop/narrow sizes pass with0model or simulator calls. This is
recorded presentation, not new native replication or repository-wide release
acceptance. [Exact commands, costs, failures and skipped checks](../artifacts/runs/recurring-affine-inspector-20261024/RESULTS.md).
