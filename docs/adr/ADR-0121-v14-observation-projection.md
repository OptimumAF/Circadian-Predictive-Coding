# ADR-0121: Derive observed streams from a completed v14 bundle

## Context

P5.2 asks for per-epoch metrics, sleep and topology history, replay,
validation decisions, and final results in inspectable formats. The
fixed v14 JSON already stores most of these facts, but its wake training
function discards method return values. Editing the v14 payload would
invalidate its fixed byte identity and could blur the global final-role
seal.

## Decision

Project the existing observed facts into deterministic JSONL files and
a final-row CSV beside an exclusively written P5.1 bundle. Verify the
completed source first; write a projection manifest last with source
and output hashes/counts. Verification re-derives exact bytes, so
rewriting a file and its recorded hash is insufficient. Mark missing
wake metrics explicitly. Preserve both raw seed-level JSON files and
reserve a new observation identity for prospective wake instrumentation.

## Alternatives

- Add guessed epoch values from guard/final scores: those roles and
  measurement times differ from wake updates.
- Mutate v14 train-only/scored JSON: this changes a frozen experiment
  protocol and its known hashes.
- Save only CSV: this loses typed event, role, replay, and topology facts.

## Consequences

The projection is an auditable view of a verified local source, not an
independent replay of training or a signed artifact. It cannot recover
training diagnostics never recorded in v14. P5.2b must capture those
prospectively, and P5.3 must add recoverable or atomic writes.
