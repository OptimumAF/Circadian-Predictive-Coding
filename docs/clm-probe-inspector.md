# Inspect saved CLM probe learning

The existing offline inspector accepts detached workflow32 CLM `saved-review.json`
files, including incomplete reviews with no native result. It copies recorded
values into source-linked tables; it loads no model, computes no new scores and
cannot train, promote or select a probe. This route and the CLI import no NumPy,
Torch or Transformers. No dependencies or environment variables were added.

```powershell
python -m scripts.build_key_door_inspector --source-artifact artifacts/runs/clm-workflow-fresh-native-20261025/saved-review.json --output clm-learning.html
python -m scripts.build_key_door_inspector --source-artifact artifacts/runs/clm-workflow-state-only-native-20261025/saved-review.json --output clm-unobserved.html
```

For an installed package without the research tree, this **illustrative schema
example is not experimental evidence**. Save it as `unknown-native.json` and run
`python -I -m scripts.build_key_door_inspector --source-artifact unknown-native.json --output unknown-native.html`:

```json
{
  "complete": false, "native_complete": null,
  "model_work": 0, "additional_scoring": 0, "effects": 0,
  "prohibited_imports": [],
  "read_errors": {"actual.json": "illustrative missing native input"},
  "execution": {"error": "illustrative preflight failure"},
  "training": null, "source_work": null, "probe": null, "probe_scores": null
}
```

The source archive includes this guide. The package retains its declared NumPy
dependency for other functionality; this export route does not import it.

Use only the accepted detached review when a case's original raw-input review
allowance is spent. These commands do not reopen the original actual/execution/
ledger files or reconstruct weights. The output is one standalone HTML file;
open it manually in a browser. A fresh destination is required: existing files
and source inputs are never overwritten. The shared loader retains its8MiB
input/16MiBoutput/depth64/500000node bounds and rejects duplicate keys,
nonfinite numbers and non-JSON data. Separate resource files are refused for
this self-contained format; other inspector routes retain their behavior.

The native-captured header copies `native_complete`. A null native result is
unknown, including a failed preflight; saved-review completion is a separate row.
Null, missing, false and zero remain distinct. A completed capture does not prove
successful learning. Missing native endpoints produce no invented case rows.

The case tables retain both markers, suffix queries and all twelve weighted old
TRAIN rows before/after the disposable probe, including wrong decisions and
negative margins. Old sealed/staged rows repeat inputs; raw10/12 accuracy can hide
two lost ready decisions. Metrics/calibration bins are copied per group, with
their original directions. The three- and five-candidate catalogs are separate;
no pooled calibration, generalization, deployment retention or context-return
recovery is inferred. Source preservation establishes isolation, not successful
retention in an adapting deployed model. The adapter's `after_return` phase name
does not establish an executed return-learning phase.

Click any recorded value to inspect its original JSON pointer. Full source,
work, clocks, scope metadata, native timing/RSS and separate reader timing/RSS
remain available. Source claims are not authenticated and metric consistency is
not recomputed. Zero saved-review work does not mean zero native computation.
Equal exposure is not equal compute; absent costs remain unknown.

The app API is `project_clm_probe_review(report, source_label=...)`, also routed
by `project_key_door_report`. Only detached workflow32 envelopes with explicit
zero reader work and bounded review fields are accepted, not arbitrary native
model reports. Success requires both fixed marker groups and old TRAIN endpoints.
Malformed case roles, catalogs, probabilities, metrics and work types fail before
publication. Partial or absent native observations remain viewable.

```text
src/app/clm_probe_inspector_projection.py  bounded source-cell projection
src/app/key_door_inspector.py             existing shared format dispatcher
src/infra/key_door_inspector_files.py     existing file admission and template map
src/infra/coherent_stream_inspector.html  existing table and source-dialog renderer
scripts/build_key_door_inspector.py       existing --source-artifact CLI
tests/test_clm_probe_inspector.py         pure metadata/export/dependency fixtures
```

Legacy game identity and legal-action imports now occur only in their existing
game-validation functions. Their numeric logic is unchanged; the CLM export path
uses standard-library presentation code only. Add future review fields as copied
source cells with bounded types; never connect this page to model operations or
use its observations for training/probe selection.

[Implementation checks, failures and remaining UI gate](reports/clm-probe-inspector-20261025.md).
The two generated local pages are in artifacts/runs/clm-probe-inspector-20261025/.
Browser automation rejected local file URLs; no workaround or automated
interactive acceptance was performed. Manual source-dialog/visual verification
is still required before issue154's browser criterion can be completed.
