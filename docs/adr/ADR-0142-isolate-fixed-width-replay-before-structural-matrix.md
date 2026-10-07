# ADR-0142: Isolate fixed-width replay before structural matrix expansion

## Context

The fixed v14 periodic arm applies replay and can prune capacity. Its
periodic-minus-no-sleep result cannot attribute a score change to replay.
P6.3 requires matched backprop/PC memory references and planned capacity
controls before expansion. The chemical-gating pilot had a null final
development result and did not exercise sleep or memory.

## Decision

Use a new development-only protocol with the same arrived-source geometry,
three fixed development seeds, and ten reserved confirmation seeds. Keep
width eight and disable all non-replay circadian sleep components. Give
backprop, ordinary PC, and neutral circadian separate replay-off/on arms
with identical initialization within width; add preplanned width-12
backprop/PC no-replay references. A shared prediction-independent FIFO
selects the same two newest retained IDs at six fixed boundaries per
seed. The circadian on/off pair keeps the same retention memory and
sleep-attempt schedule, changing only the replay enable switch. Accept
all replay-only opportunities without an inner guard in this feasibility
factor and score only outer selection. State replay's 12 additional
optimizer updates and the width controls' added capacity separately.

The pure two-task metric contract declares final mean accuracy and signed
forgetting as primary, with the full accuracy matrix and optional
zero-safe retention ratio. The existing balanced score is a compatibility
alias. The public adapter binds the manifest/source/adapter identities,
caps planned optimizer work and process time, and audits complete results.

## Alternatives considered

- Reuse v14 periodic-minus-no-sleep outcomes: rejected for replay/width
  confounding and already opened final roles.
- Match total optimizer updates by removing wake work from replay-on arms:
  rejected because it would change the A/B wake exposure and answer a
  different question. The replay treatment cost is reported explicitly.
- Choose width after observing a dynamic circadian run: rejected as a
  retrospective capacity oracle. Width 12 was fixed in advance.
- Use the inner guard at each replay-only opportunity: deferred to later
  full-mechanism/confirmation protocols because rollback would vary applied
  memory work across arms; this small factor declares all opportunities
  accepted and reports that narrower scope.

## Consequences

The replay switch is interpretable at fixed width within each method,
with exact shared IDs and explicit extra work. A neutral circadian/ordinary
PC parity check exposes hidden non-replay effects. The three-seed result
is exploratory and cannot rank methods or establish a circadian-specific
gain; structure, homeostasis/reset, schedule, full-minus-one, guard policy,
complete process cost, and independent confirmation remain open. The
historical v9-v14 protocols and artifacts retain their identities.
