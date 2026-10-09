# ADR-0235: Copy the complete owner graph after admission

## Context

Native snapshots, inbox cursors, actor bundles and metadata can share arrays and
records. Independent capture calls lose cross-component aliases or observe
different epochs. Existing retained-payload limits exclude parameters.

## Decision

Project complete source fields under original owner leases into explicit records.
Keep live authority references separate. Inject NumPy projection, full graph
preflight and copy ports inward. Read native state without calling snapshot
methods. Reserve original retained payload bytes before one graph copy, with
independent total-array/node/depth/dimension bounds. Preserve array writeability
alongside dtype and C/F layout. Record charges after admission; never refund.

## Alternatives

Per-component copies lose alias edges. A generic object loader could renew live
authority. Parameter bytes cannot silently become retained raw payload bytes.
Pickle is not a portable untrusted-data checkpoint format.

## Consequences

The existing metadata-only capture remains unchanged. Full source records and
original ports remain separately reviewable. Populated pending histories, sampler
concurrency/resource admission, retired holders and complete native variants need
further qualification; R3.5b2e5b and all full recovery parents remain unchecked.
This decision grants neither durable/model restore nor scientific admission.
