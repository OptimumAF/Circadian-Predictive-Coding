# ADR-0450: Frozen BC execution in the existing robot smoke

## Context

The existing BC route fits TRAIN actions and exports an owned full snapshot.
Offline loss decreases through the fixed256 endpoint, but does not establish
closed-loop robot behavior. The original smoke already owns bounded real-port
execution, trace output and native failure accounting.

## Decision

Add an optional in-memory exact TRAIN BC learner to the existing smoke entry
point. Keep the CLI zero-action default. Fork the owner and infer without updates;
validate controller/spec/roles before the corresponding work. Compare full state
after predictions before controls, after controls, and after cleanup. Record
prediction/native work and any/final success separately from ordinary completion.

Why this: test the actual fixed policy through the accepted port without another
runner, checkpoint format or data-admission framework. Callers use the existing
owned snapshot API rather than introducing disk loading into the smoke.

## Alternatives

More TRAIN optimization alone cannot show task completion. Repeating the spent
held-out probe would spend evaluation exposure. A new simulator framework or
general model codec is not required for this runnable integration.

## Consequences

The outer CLI composes existing adapter and infra boundaries; core/learner/port
remain unchanged. Corrupt snapshots clear current state flags rather than
reusing earlier success. One reviewed TRAIN-context execution is negative and
does not qualify strong BC, generalization, adaptation or comparative benefit.
Broader R4.4 requirements and all earlier failures remain open/preserved.
