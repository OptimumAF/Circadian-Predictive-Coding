# ADR-0277: Corrupt an owned native candidate before bounded reload

## Context

Native update completion validates a full B response, saves B and reloads it
before committing completion. R3.7f demonstrates actual worker interruption;
actual parent-owned checkpoint corruption still needs an integrated control.

## Decision

Extend the existing completion demo/CLI with explicit `--corrupt-checkpoint`.
After verifying the healthy response and saved bytes, overwrite only the final
byte of the owned B file, flush/fsync, and invoke the unchanged bounded store
with the ORIGINAL expected digest. Accept only its exact mismatch as the intended
control; leave `actual_b` null and retain damaged bytes separately for inspection.
Preserve original authority and spent costs, stable actor A and no retry. Keep
the registered child alive until terminal checks finish, then stop/reap normally.

## Alternatives

- A synthetic exception after writing valid B would not exercise actual reload.
- Changing the expected digest to the damaged file could authorize corrupt state.
- New recovery abstractions are unnecessary: existing coordinator and store fail
  safely when the trusted validation callback propagates the real refusal.

## Consequences

Default healthy completion remains available. The explicit finite fault example
demonstrates corrupted-checkpoint refusal across PC and both CPC controls, with
full A preservation and no completion/refund/retry. No new dependencies, inner
authority changes, algorithm changes or scientific claim. Hostile filesystems,
full managed components, actor publication and coordinator-loss recovery remain
separate unfinished requirements.
