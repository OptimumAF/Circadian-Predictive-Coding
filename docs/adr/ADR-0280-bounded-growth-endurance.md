# ADR-0280: Stream bounded evidence for finite native endurance

## Context

The accepted growth example demonstrates two splits, one capped refusal and
continued native training. Its detailed per-request snapshots are useful for
small controls but unsuitable for many failed requests and cached actor reads.

## Decision

Run a frozen engineering protocol of128 original attempts per family, width8
ceiling,130requests including duplicate/exhaustion probes and64cached reads per
request. Use original runtime/admission and unchanged native policy. Verify full
state/serving invariants throughout; stream counters, bounded blocks and tiny
request samples instead of retaining per-request native payloads.

Cap the aggregate retained summary at16KiB and final cursors individually at16KiB;
retain complete initial/capped/final native states separately. Tests execute only
a tiny prefix. Run the single full local workload after prefix/static/source
gates and retain actual counts, resource observations and incomplete outcomes.

## Alternatives

- Accumulating detailed native snapshots grows diagnostic memory with requests.
- A tiny prefix alone cannot establish the fixed full workload's behavior.
- Unbounded loops or retries obscure original attempt/resource limits.

## Consequences

The finite endurance result is reviewable with bounded evidence and fixed native
work. It is not a scientific comparison or a general long-run/recovery proof.
Process RSS observations remain boundary samples. Existing source/actor/candidate
authority, algorithms, limits and negative research outcomes remain unchanged.
