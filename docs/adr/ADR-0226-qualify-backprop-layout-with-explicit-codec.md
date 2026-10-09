# ADR-0226 - Qualify Backprop native layout through an explicit codec

## Context

Current Backprop wirev1 preserves logical state and required aliases,including
strided inputs,but normalizes F arrays to C. Original scoped acceptance and
working native APIs must be preserved. Broad native storage recovery is missing.

## Decision

Add LayoutBackpropCheckpointCodec with explicit wirev2/kind backprop_layout_v2.
Reuse existing exact bounded Backprop schema helpers within the adapter layer
and shared C/F frame validation/encoding. Keep the original v1 adapter/source/
tests available. Refuse oldwire/unknown/corrupt/missing/noncontiguous v2 input.
Validate all bounded raw/finite/traffic payloads before detached NumPy arrays.
Native snapshotversion1,required alias identity and private array ownership are
preserved. Core typed port imports no adapters or infrastructure.

Why this:explicit selection preserves existing functionality and limits actual
schema duplication. Two callers justify sharing adapter implementation helpers;
they are not a new public interface. Fixed native continuation compares actual
preupdate BCE loss (the native model has no separate energy field),full state
and prediction bytes without selecting seeds/metrics/baselines.

## Alternatives

Replace existing v1 default:would alter working strided-input semantics. Generic
stride/object loader:would expand unsupported ownership and byte-budget scope.
Copy native/schema logic:would duplicate the supported state and alias checks.

## Consequences

Callers select the stronger codec explicitly;no implicit migration or live/disk
restore. Exact payload preflights do not prove process RSS,cumulative copy charging
or source/consent/singleowner authority. New component schemas need independent
complete native/inbox/lifecycle/actor/sharing validation and current gates before
composite recovery. Original F-to-C negative artifacts remain immutable.
