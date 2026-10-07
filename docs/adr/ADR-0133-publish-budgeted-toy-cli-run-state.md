# ADR-0133: Publish a budgeted toy CLI run state

## Context

P5.5c1 stops the toy app runner at checked update, sleep, and final-role
boundaries. Its `ExperimentResult` is available only after full scoring.
The root CLI already writes completed JSON exclusively, but a limit or
training error previously left no external record. The trusted checkpoint
file is replaceable at each checked cursor and must not be mistaken for a
completed report.

## Decision

Add opt-in `--max-training-updates` and `--max-wall-seconds` to baseline
mode with a required `--run-state` path. A separate `--checkpoint` path is
optional on a fresh run and required for successful resume. Claim the
run-state path exclusively before training, keep a cooperative sidecar
lock for the whole invocation, and atomically replace its JSON only when
the exact prior bytes still match. Fresh checkpoint names are reserved
exclusively before the replaceable trusted store uses them.

The versioned state binds the initial complete resolved config, its runner
digest, exact artifact paths, and each attempt's explicit inputs and budget.
It records `running`, `incomplete` with an update/time reason, `error` with
an exception type, or `completed` after final scoring and publication.
Observed committed wake updates and durable checkpointed updates are
separate, since a checkpoint write can fail after an update completes.
The checkpoint identity includes a byte SHA-256, checked cursor, and
durable update count. A new invocation may raise its total-update limit,
but resume requires the same scientific request, artifact paths, and
unchanged trusted checkpoint bytes; the runner performs its existing
config/data/role validation before restoration. Without a verified
checkpoint, the state is explicitly non-resumable. Incomplete/error runs
produce no completed result.

## Alternatives

- Write a partial `ExperimentResult`: rejected because final-test scores
  would be absent or invented.
- Add execution flags to `ExperimentConfig`: rejected because a resource
  policy would change scientific and checkpoint identity.
- Infer error work only from the last checkpoint: rejected because a
  successful model update can precede a failed checkpoint save.
- Automatically resume a `running` state after a process death: rejected
  because the lock and checkpoint may represent live or unverified work.

## Consequences

The default unbudgeted CLI call, stdout, existing result schema, and fixed
v14 protocol remain unchanged. The CLI uses exit code 3 for an ordinary
budget stop; it re-raises training errors after recording `error`. The
state is local coordination, not authentication: checkpoint pickle input
must be trusted. A process killed without exception can leave `running`
and a stale lock; manual inspection is required before recovery. The wall
limit remains a boundary check, and replay/capacity/memory bounds remain
P5.5d. No baseline, seed, metric, or scientific result was changed.
