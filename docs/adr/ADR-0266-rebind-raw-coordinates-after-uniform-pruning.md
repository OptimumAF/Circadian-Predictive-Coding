# ADR-0266: Rebind raw coordinates after uniform-width pruning

## Context

Native pruning transfers effective tensors at one layer. Raw coupled state uses
one width across hidden layers, with width-dependent weight/bias/rate factors.
Copying raw values after pruning would change retained effective coordinates;
copying only native state would lose configuration and raw continuation semantics.

## Decision

Add a pure one-coordinate-per-layer operation within the supported uniform-width
domain. Select old effective tensors, then encode under the new config's factors.
Require a per-entry 8-epsilon relative/zero-absolute reconstruction bound, retaining
exact zeros and refusing critical underflow or unrepresentable candidates. Own the
complete raw/wake state and immediate coordinate metadata. Preserve precision,
policy and wake count; select last-layer chemicals/gates with the same coordinates.

Use a separate train/validation observed-damage gate, reusing native probe scoring.
Copy state/data/scores across callback boundaries, retain full source on rejection
or fault, and never let final-test data select a structural replacement. Create a
new raw learner for accepted geometry; existing strict restore remains unchanged.

## Alternatives

Arbitrary single-layer removal would require a separately derived nonuniform raw
config. Direct raw slicing would ignore changed factors. A loose absolute error
floor could hide lost subnormal coordinates. Treating encoding error as whole
function preservation would confuse coordinate representation with pruning damage.

## Consequences

Raw PC/CPC candidates can be validated and continued without losing chemical or
configuration state. The operation is non-preserving in general; future rates and
trajectories change. This increment supplies no runtime publication authority,
global damage bound, scientific benefit claim or generic split/merge operation.
