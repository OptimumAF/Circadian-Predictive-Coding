# ADR-0321: Replay complete demonstration packets in the existing buffer

## Context

R2.4's existing Torch route supports labeled row rehearsal and owner-local
continuation. Those arrays are not trajectories. A useful trajectory path must
retain actual transition order, actions, rewards, next states and episode ends,
without losing budgets, isolation or matched controls.

## Decision

Represent complete fixed-length binary-action demonstration episodes as validated
immutable records. Collect actual transitions in a bounded two-action line
environment. Add optional exact episode metadata to the existing attribution
batch; runtime experience/label IDs and copied observer data preserve identity.

Encode one complete packet into one existing shared-buffer row, including bounded
UTF8 identity/context fields and all transition values. Derive whole-episode
quotas by flooring native transition caps; count actual encoded bytes. Decode one
selected packet into one ordered ordinary native behavior-cloning batch. Charge
all transitions before that call, preserving attempted/returned/committed counts.
Continue using original retention/rotation, consolidation, clocks and resume.

Why this: composition preserves the established selection and authority path,
while the actual payload and counts prevent partial episodes or hidden exposure.
Native loss/optimizer/adaptive math stays unchanged; rewards/next states remain
evidence rather than an unreviewed RL objective.

## Alternatives

Attaching IDs to old independent rows would lose whole-episode selection. A new
trajectory runner/buffer would duplicate working orchestration. Padding UTF8
headers costs memory but gives a simple canonical bounded payload; optimize only
after measuring a real need. Arbitrary-length episodes, nonbinary actions,
recurrent learning and durable recovery need separately specified extensions.

## Consequences

Defaults remain compatible. Optional replay stores more bytes per native example,
reported explicitly and measured with total RSS/runtime. Owner-local continuation
carries encoded packets and native state under original budgets. The accepted
tiny comparison has negative accuracy/forgetting outcomes and establishes no
useful policy, circadian advantage or repository-wide readiness. Original bounded
R2.4 implementation acceptance does not close those scientific/release goals.
