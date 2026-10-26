# ADR-0353: Use the existing stream for fixed-A acquisition

## Context

The history-enabled live port works, but the A32 boundary has no native goals.
MiniGrid initial acquisition remains missing; retention and fresh-task probe
findings cannot supply that premise. Custom key-door has fixed-task acquisition
but is a different observation/action task.

## Decision

Narrowly admit exact predefined A geometry with its two original TRAIN packets.
Extend the existing history CLI with an explicit fixed-A acquisition option.
Reuse the guarded128 stream, existing raw checkpoint scorer at32/64/96 and live
evaluator at0/128. All five learners finish before any readout. Preserve original
paths, clocks, receipts, states and paid evidence; bound readout records at25.
Report both raw sequence fit and native goals, jointly and separately. Failed
or incomplete whole cases yield null inference fields while retaining paid data.

## Alternatives

Another trainer duplicates guarded runtime logic. Scoring/live testing every
checkpoint adds unnecessary physical work. Inferring skill from raw fit alone
does not establish autonomous acquisition.

## Consequences

The same original total128 dose supplies a fixed-task control, with all endpoints
and failures retained. Unequal A exposure relative to the four-phase stream
prevents causal interference conclusions. One frozen seed/dose is exploratory;
no automatic increase, final evaluation, tuning or scientific advantage follows.
Costs and the six-source collector's unused B/C episodes remain explicit.
