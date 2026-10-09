# ADR-0219: Persist terminal facts without renewing authority

## Context

Local coordinator sequencing left failures and the freshest clock/RSS observations
in memory. A new instance could not distinguish a failed lane from a viable one
using journal state alone. Unknown commit acknowledgment also needs explicit
handling without refunding an actual committed admission or retrying work.

## Decision

Add a reporting port and exact metadata-only transitions for monotone observations
and stopped history. Persist reports around callbacks and terminal status on
failure or close. Preserve original source/policy/component bindings, ownership,
caps and spent counters;never clear uncertainty or invent native completion.

Allow measured time/RSS overshoot only as terminal evidence. Why this: retain
negative measurements without changing the original allowance. All dispatch
still refuses terminal records, and other work counters keep their caps.

When terminal persistence is unconfirmed, preserve the original error and attach
an independent bounded witness. Provide explicit fresh-connection terminal-only
reconciliation of exact known pre/post/stop states. It stops work and preserves
actual charges;it does not grant restart admission or synthesize host facts.

## Alternatives

- A memory flag does not persist stopped history across coordinator instances.
- Clamping RSS or moving the original start would discard negative evidence.
- Retrying native work after an unknown commit could renew spent resources.
- Accepting arbitrary disk state would lose the independent authority boundary.

## Consequences

Original read/advance APIs and bounded storage schema remain;coordinator adapters
add reporting. Fake/local SQLite tests can prove stop/refund/commit-fault behavior.
No new dependency or environment option is required. Source/host authenticity,
publication leases,real worker crash/power loss,native codecs and coordinator-loss
recovery remain separate unchanged parent requirements. A trusted surviving
coordinator/private-file witness is necessary for terminal reconciliation.
