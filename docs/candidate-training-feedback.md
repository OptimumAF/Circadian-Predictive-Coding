# Candidate TRAIN feedback for the existing drift schedule

```powershell
$env:PYTHONPATH = (Get-Location).Path
.venv\Scripts\python.exe -m scripts.run_bp_ticket_serving --source artifacts/runs/context-head-control-20261023/actual.json --context-head-serving-live --context-live-reference drift --context-candidate-feedback --output local-candidate-feedback.json
```

Output must not exist. The flag explicitly selects a development variant of drift.
Default drift still uses deployed-actor feedback. Existing fixed controls and the
fitted controller are unchanged; the latter was fitted with different feedback
semantics and cannot be transferred through this flag.

## Causal diagnostic

Before action selection, the existing schedule reads the oldest pending ordinary
TRAIN packet only if its labels have already arrived. It obtains a detached
candidate snapshot, reconstructs a disposable readout and returns scalar errors.
No final/probe/guard/live task outcome becomes controller feedback. No diagnostic
runs before label arrival or after the queue drains.

`candidate_feedback.records` preserves original label observation/arrival clocks,
the current diagnostic time, base/candidate version, wake/native update counters,
rows, bytes, return/completion and errors. This is a post-label TRAIN diagnostic;
its scalar feedback becomes visible at the current tick. It is not a deployed
actor's pre-label prediction. Original actor observations remain separately paid.

Every attempted forward and repeated read after deferral costs work. Failed or
invalid output, source/data/budget changes and mutated detached/original candidate
state stop before another update. The callback receives no live candidate handle.
Original runtime, budgets, gate, publication thresholds and rollback remain intact.

## Scope and extension

The sole frozen development case completed two updates over nine rows. Three
candidate reads over15rows exposed old-task regression; the second update was
deferred once and its repeated diagnostic was charged. The original guard accepted
publication at tick3, and all six common-window workflows succeeded. Rollback
restored the source and its original5/6 ordinary task result. No controls were rerun.

This adds15full-class prediction rows and three readout constructions versus the
saved periodic case, with identical learned states. Full case2.30410s, sampled
peak49,135,616B; timings are unpaired and do not establish efficiency.

The feature addresses the observed82 gap: a candidate can regress on an old packet
while the deployed actor still reports it clean. This does not establish task
generalization, cost efficiency, concurrent inference or CPC benefit. Compare
outcomes and diagnostic cost separately; a simpler periodic control may remain
preferable. Do not reinterpret this variant as the unchanged drift baseline.

Broader distinct-stream/noise/recurrence and long deployment criteria remain open.
Extend through the existing schedule and stream recipes with a separate frozen
protocol, preserving actor feedback as default and keeping final/probe data sealed.
The existing `build_stream` deliberately binds six-row/D7 native packets; the
contextual packets are three/six-row/D9. An explicit contextual data adapter is
needed before reusing recurrence/noise recipes; preserve old recipe shapes and
derive any corruption from named TRAIN rows rather than native row offsets.
Evidence and exact commands live in `artifacts/runs/candidate-training-feedback-20261023/`
and [issue83](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/83).
