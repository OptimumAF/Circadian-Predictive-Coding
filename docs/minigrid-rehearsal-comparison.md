# Saved MiniGrid replay comparison

The frozen four-method study is complete and independently accepted for the
descriptive evidence below. Every pair matches original sources and benchmark
probabilities exactly. Warm BP rehearsal reduces F gain from0.72393438 to0.24798432
and increases old-A CE loss from0.37882906 to0.60802032. PC/neutral improve both
measures relative to their own controls and match each other; active improves
less. Cold and warm curves are all retained. Old A remains unacquired in every
case; no acquired-retention/recovery or circadian advantage is established.

[All results, exact commands, costs and preserved failures](../artifacts/runs/old-a-rehearsal-comparison-20261018/RESULTS.md).
New science cost1,024native calls/32,256training rows/P376/3,384,40teacher episodes/
360physics steps;189.0714691/900parent seconds. Original no-replay controls were
reused without training again. The saved tool adds no native model work.

Compare complete original invariant MiniGrid joint128/F16/old-A/A16 reports
without constructing a learner or environment:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m scripts.compare_minigrid_rehearsal --control artifacts/runs/joint-catalogue-study-20261018/bp.txt --rehearsal artifacts/runs/old-a-rehearsal-comparison-20261018/native-bp.txt --method bp
```

The other declared methods are `pc`, `neutral_cpc` and `gated_cpc`. The script
reads files and calls a pure adapter. It refuses incomplete or mismatched saved
evidence and prints qualified descriptive curves, differences and separate costs.
Input files are never modified. The existing pilot is the sole training runner.

## What is compared

The pair must use the original raw296/K7 invariant H8/seed17 native recipe,
exact six TRAIN sources/catalogue inputs and four F source episodes, identical
roles/layouts/seeds/transitions and benchmark readout anchors at0/32/64/96/128.
The new protocol must differ precisely by the declared old-A replay intervention.
All F/old-A/A-return curves and paid work must be complete and bound to the
correct episode identities/roles. Replay events must contain the original two
A TRAIN episodes in order, with full history transitions and exact costs.

The reducer recomputes TRAIN CE/first argmax/exact from saved probabilities and
probe/return CE/accuracy/exact from saved losses and decisions. Probe probability
vectors were not saved; it cannot independently recover native probe argmax or
negative-log probabilities. This limitation applies to both intervention arms.

For each source checkpoint, output includes every curve and replay-minus-control
differences in F CE gain, old-A CE loss during F and A-return CE gain. Positive
gain differences and negative old-A-loss differences are desirable. Acquisition
premises remain explicit; unacquired old A cannot support retained acquired skill
or recovery claims. Invocation time and RSS are separate observations.

Replay adds64native calls and576old-A rows per complete public case. Methods
within an intervention have matched sources/doses; intervention arms have
different exposure and compute. This cannot isolate replay structure from extra
old-A exposure, establish statistical robustness, or prove circadian advantage.
Final data remain absent from training, scoring, selection and this comparison.

## Structure and verification

```text
src/adapters/minigrid_rehearsal_comparison.py  pure saved arithmetic/qualification
scripts/compare_minigrid_rehearsal.py         argparse, JSON IO and error messages
tests/test_minigrid_rehearsal_comparison.py   real saved controls/fabricated replay DTOs
tests/test_minigrid_rehearsal_comparison_cli.py public saved-data path/refusals
```

Final focused111checks pass, including45new saved/CLI cases and66reducer
regressions. Four-path Ruff/format and two-source Windows/Linux mypy pass;
independent source/protocol/actual review GO. Initial source identity/content
gaps, typing errors and storage-amendment source correction are preserved in
the linked results. Original32MiB aggregate storage was prospectively increased
to96MiB after measuring required reports at56.7MB; per-report/time/model/RSS limits
were unchanged. No whole-repository/release claim follows from scoped checks.

The fake replay DTOs test accounting and refusal behavior; they are explicitly
not observed scientific evidence. Actual cases, protocol, resource receipts,
failures, observed outcomes and review are kept in the
[bounded study directory](../artifacts/runs/old-a-rehearsal-comparison-20261018/PROTOCOL.md).
Extend another recipe only with a separate declared protocol and behavioral
tests. Reuse accepted controls when sources and assumptions remain unchanged.
