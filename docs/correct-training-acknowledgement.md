# Correct TRAIN acknowledgements

An explicit opt-in lets candidate feedback consume an already-correct arrived
TRAIN packet without running an optimizer update. It addresses the observed
FIFO stall in issue85, where the controller repeatedly read a correct packet
while later packets remained queued and publication waited for a complete drain.

```powershell
$env:PYTHONPATH = (Get-Location).Path
.venv\Scripts\python.exe -m scripts.run_bp_ticket_serving --source artifacts/runs/context-head-control-20261023/actual.json --context-head-serving-live --context-live-reference drift --context-candidate-feedback --context-acknowledge-correct --context-stream-recipe context_recurring_clean --output local-correct-training.json
```

Output must not exist. Budget this model invocation separately before running.
Other controllers, implicit candidate feedback and nonboolean API flags are
refused. Existing commands and default `wait` behavior retain pending labels.
This is a conventional development behavior, not a new comparative result.

## Receipt semantics

`ActorShadowRuntime.acknowledge_ready` holds the candidate's exclusive lease
through a trusted read-only verifier and inbox commit. It validates current
actor/base/candidate version and update count before verification, then rechecks
versions, revision and open/unmanaged status before commit. The inbox admits only
the oldest ready, train-authorized source/label pair. Original observation and
label clocks remain; future labels, duplicates, wrong FIFO packet, stale owner,
failed proof, nonboolean return and changed event clock are refused. Payloads
are detached for verification and retained under the original identity capacity.

The categorical adapter checks the authoritative source/target payload against
the fixed TRAIN packet, reads the current detached candidate, validates complete
probabilities and refuses changed detached/original candidate state. Every
attempt, returned row and failed prefix remains charged. A zero-error result
creates a scalar `AcknowledgedExperience`; a nonzero result stays pending for
the normal drift decision. No old scalar diagnostic alone authorizes consumption.
One packet can settle per tick.

Acknowledgements prove correctness at that point in the stream. Later updates
can cause regression, so the original clean new/old guards still decide whether
the final shadow may be published. The new drained condition requires every
packet to have an actual update or acknowledgement, all original labels observed,
and an idle sharing gate. Native completions must equal actual optimizer updates.
No optimizer call, update charge, invented applied receipt or budget renewal is
used for acknowledgement. Diagnostic forwards and copies still cost resources.

`acknowledged_experiences` and `receipt_progress` report consumed receipts
separately from `applied_updates`, `training_work`, budget updates and admission
counters. `training_complete` retains its original meaning of completing the
planned number of optimizer updates; receipt completion may be true while that
field is false. Guard rejection, queued work and task failures remain explicit.

## Supported scope

The single clean engineering application completed two updates/nine TRAIN rows
and four acknowledgements. Original guards accepted at tick8; later ordinary
legacy tasks succeeded, common window5/6 includes the intentional denial, and
rollback restored the source. Software checks and scoped independent review
accepted functionality. The outer meter mistakenly retained larger inherited
ceilings; measured work stayed below the declared single-case limits, but full
prospective enforcement conformity remains unfulfilled. The executed meter and
failure are preserved; a future-only correction does not retroactively fix it.

Only unmanaged ordinary TRAIN owners with no consolidation allowance are
supported. Probe owners, managed lifecycle owners and sleep-capable owners are
refused. Existing cursor, handoff and erasure formats cannot represent this new
history, so they refuse acknowledged owners before payload copying. Persistence
support is unfinished; no codec or capture extension is implied. Default owners
keep their original field schema because acknowledgement history is created only
after a successful explicit consumption. Local callbacks are trusted; this API
does not contain arbitrary code or attest physical provenance.

Evidence/protocol: `artifacts/runs/correct-training-acknowledgement-20261023/` and
[issue86](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/86).
Extend through existing receipt and controller ports with separate consumed/
native accounting; freeze any new scientific protocol before further model work.
