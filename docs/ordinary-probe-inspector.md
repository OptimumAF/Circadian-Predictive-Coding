# Inspect saved ordinary learning probes

Use the existing offline inspector command from the repository root:

```powershell
$env:TEMP='D:\Dev\_Temp'; $env:TMP='D:\Dev\_Temp'
.venv/Scripts/python.exe -B -m scripts.build_key_door_inspector --source-artifact artifacts/runs/ordinary-linear-probe-native-20261030/native.json --resource-artifact artifacts/runs/ordinary-linear-probe-native-20261030/work.json --output D:/Dev/_Temp/ordinary-probe.html
```

Choose a **new** destination; existing output files are refused. Open the HTML
in a browser. It needs no server, network, model, numerical library or experiment.
The optional resource file is displayed as a separate, unverified association.
The input is bounded to 8 MiB and HTML to 16 MiB by the existing writer.

Supported sources are explicitly requested ordinary reserved-probe reports from
native BP, PC, neutral CPC, active CPC, or contextual linear-fit minimum-change
workflows, including shared-head `linear_context_return_ticket_workflow_v1`.
For the accepted local shared-head case, use
`artifacts/runs/shared-linear-once-comparison-20261031/joint.json` (or
`current_only.json`) as the source; its separate `work.json` remains an unverified
resource association. Original 18-row-return reports also remain supported.
Older report families keep their existing views. This does not add an
experiment runner or make the current source available in the older #325 wheel.

The page shows three saved task phases, old/fresh success totals, recorded
acquisition/retention/recovery, full fresh/old CE and class-accuracy tables,
four prediction-loss charts, source/private/return costs, saved actions and raw
scores, and availability of all five method families. Absent controls remain
visible as not provided. Charts use updates 0, 4, 8 and 16 within each phase,
with a common scale and gaps at absent or null readings. Narrow pages redraw
their charts when resized; tables scroll within their own regions.

Click a metric or chart point to see its exact saved JSON pointer and value.
Chart points also support Enter and Space. Object-valued cells open the original
record instead of filling a table with serialized JSON. Missing, explicit null,
false and zero remain distinct. The viewer validates report shape and fixed
roles/tasks before reading outcomes, but does not authenticate the producer or
recompute summaries, scientific metrics, preservation claims or resource totals.

Why this: successful task endpoints can conceal worsening prediction loss.
The accepted #328 example retains old success 4→4→4 and improves fresh success
1→2→2, while return CE worsens. Its initial failure was a near tie and recovery
was not observed. The page keeps these limitations next to the saved evidence.
Private learning does not update or promote the serving actor. Known TRAIN
templates, auxiliary exposures and unequal work do not establish generalization,
matched computation or circadian benefit.

Shared-head rehearsal reports now separate direct `train_batch` ports, logical
phase counts and auxiliary LS applications. A zero direct private counter does
not mean no fitting. The accepted once-return example records 32 auxiliary
applications, 144 current input rows and 288 additional row charges; its receipts
show 15 fitted rows in each fresh update and 12 in each return update. Original
reports retain 18 fitted return rows and 384 additional charges. These scopes are
not added into a newly inferred total. In `current_only`, extra rows repeat the
current input; retained TRAIN use remains separately recorded as zero.

Receipt rows preserve failed/running work, observed applications and unknown
applications. Click the original work object for packet/snapshot/SGD fields, and
the auxiliary budget for its original limits, admissions and clocks. Missing
recipe flags remain unknown; null, false and zero remain distinct. The wide
receipt table scrolls locally on small screens. These are recorded claims, not
source authentication, independent model replication or evidence of new-task
acquisition: both fresh tasks already passed before learning in the accepted
once-return case. Other unsupported cost categories remain unavailable.

[Executed checks, failures, artifacts and remaining scope](reports/reserved-probe-inspector-20261030.md).
[Shared-head source and browser qualification](reports/shared-linear-probe-inspector-20261031.md).
