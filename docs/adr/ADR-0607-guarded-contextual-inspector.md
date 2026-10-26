# ADR-0607: Show guarded context learning through the existing source-cell viewer

## Context

The installed CPU learner now produces useful guarded task episodes. The
existing inspector dispatch recognizes only three older ticket formats; it
cannot show this report's arm/phase, actor reads, rejection and rollback.

## Decision

Add a bounded pure guarded-context projection and one explicit dispatch branch.
Reuse existing detached source cells, tables, JSON envelope, exclusive file
publication and inert source dialog. Change only the contextual report's title/
warning in the existing template; other schema behavior stays intact.

Why this: making delivered learning evidence inspectable needs neither another
renderer nor a model run. Scalar type/array bounds reject malformed presentation
data while missing/partial/null/false/zero remain distinct. Source flags and
recorded costs are shown, not authenticated or scientifically recomputed.

## Alternatives

Flattening into an older native report would lose phase/context/actor meaning.
A parallel viewer would duplicate rendering/security behavior. Re-running the
learner would spend model work without changing the display requirement.

## Consequences

Saved model states and original failures remain in detached source JSON. Visible
tables distinguish private candidate, served actor and rollback. The viewer
cannot train/select/promote or establish scientific benefit, safety, generality
or whole-repository readiness. Older routes keep their existing acceptance.
