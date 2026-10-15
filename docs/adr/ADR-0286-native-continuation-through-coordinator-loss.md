# ADR-0286: Continue complete native state through coordinator loss

## Context

R3.7o establishes physical coordinator transfer with metadata-only callbacks.
The native codec/store and explicit completion already exist, but their
composition with a replacement coordinator has not demonstrated full B restore
and subsequent learning under the original allowance.

## Decision

Add a small adapter session around the existing recovery coordinator. Reserve
copy/update/attempt costs before native IO or model work. Export the actual
acknowledged completion proof; bind it to independently retained authority and
complete fixed reference checkpoints. Retain completed B in the physical
supervisor while the old pair is alive, commit explicit transfer before trusted
adoption, charge full B restore, then complete B → C under original limits.

Use a fixed PC/neutral/active three-family CLI. Restore the complete supported
CoupledSnapshot, including CPC policy, chemistry, gates and wake history. Keep
actor A untouched and compare exact full bytes, energies and predictions with
uninterrupted references. Record native `_updated_parameters` separately from
the unused raw scaling kernel.

Why this: composing existing authority and checkpoint boundaries demonstrates a
useful native continuation without broadening generic floor rules or treating
arbitrary disk state as restart authority. Constructor report advancement must
be retained explicitly before validating the next admitted completion.

## Alternatives

Reopening a database as its own trusted floor loses independent completion
evidence. Reconstructing only weights loses CPC state. Resetting the start or
allowance permits excess work. Adding a general managed-runtime restart system
would exceed this bounded native integration.

## Consequences

No dependency, learner objective, seed, metric or baseline changes. Existing
core, SQLite, bounded transport and physical transfer contracts remain intact.
Failure is permanent; terminal reconciliation cannot overwrite an unsupported
newer state. Focused DI tests and one bounded physical run validate this leaf.
Full managed recovery, supervisor loss, power-loss durability and scientific
benefit remain open. Replacement interruption is the next concrete control.
