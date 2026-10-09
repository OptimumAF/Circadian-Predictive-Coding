# ADR-0227 - Encode complete supported NumPy inbox records

## Context

Native cursorv1/v2 records include complete live/unmatched/consumed/applied/erased
metadata and opaque payloads. Separate managers own consent,revocation,retention,
copy-budget and owner state. Generic/scalar serialization would omit payloads or
mistake observations for restore authority. Native training validation supports
real numeric widths/endian beyond the initial float64 annotation.

## Decision

Pin all supported fields/private state before coding. Add inner immutable expected
shape/dtype/record/string/candidate policy;explicit metadata schema,real numeric
payload frames and typed byte orchestration adapters. Preserve native cursorv1/v2
and every supported field. Metadata/role/permission checks precede payload access.
Bound actual-width aggregate/wire bytes;validate every raw payload before detached
NumPy materialization. Preserve C/F,dtype/byteorder/bytes,signedzero and independent
ownership;refuse unknown/foreign/shared/strided schemas explicitly.

Why this:separate responsibilities prevent generic opaque graph loading and keep
core independent of NumPy/IO adapters. Existing APIs/source remain intact. Initial
f8-only195green controls are insufficient;expand before acceptance. Version the
stronger complete numeric policy/frame wire to v2;retain unaccepted v1 evidence.

## Alternatives

Metadata-only cursor:drops complete payloads. Generic pickle/graph loading:unsafe
caller-controlled class/alias/callback behavior. Float64 normalization:loses valid
numeric dtype/byteorder and silently narrows current payload scope. Live manager
serialization:locks/callbacks/authority require independent explicit original ports.

## Consequences

Codec observations never grant model/budget/clock/actor/lifecycle authority or
renew consent/resource grants. Payload bounds do not establish totalRSS/cumulative
copy charging. Extended/other payload backends need separate supported codecs.
Current Windows longdouble isf8;Linux runtime/remote jobs remain untested. Full
lifecycle/consolidation/actor/sharing/Torch/native/model/coordinatorloss and original
parent criteria remain required before composite/disk/live restore or completion.
