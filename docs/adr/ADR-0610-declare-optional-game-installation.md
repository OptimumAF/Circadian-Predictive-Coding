# ADR-0610: Declare the existing game environment as an optional extra

## Context

The installed FIFO36 describe command already works without optional imports.
Fresh game collection also has an existing runner. However, the wheel declares
only NumPy, and the missing-dependency message tells installed users to use a
requirements-minigrid.txt file that is present only in a checkout.

## Decision

Add a `minigrid` project extra containing exactly the five existing pinned
requirements. Preserve base dependencies, scripts, resources and runtime checks.
Give missing-dependency errors both a local-wheel-extra and checkout setup route.
Add package-facing commands and verify one offline wheel/isolated install with
describe outside the checkout, all optional imports/model work blocked.

Why this: the missing contract is installation metadata. A new game runner,
input resource or training experiment would duplicate working functionality.
The optional extra changes no numerical method, data role, seed or budget.

## Alternatives

Making game libraries base dependencies would expand the NumPy-only setup.
Requiring the research checkout would defeat installed usage. A second source
file for the same pins can drift, so focused tests compare the extra to the
existing requirements file and runtime's direct-package subset.

## Consequences

Users can install the local wheel's game dependencies explicitly. This task
does not download dependencies, prove clean resolver/platform coverage, rerun
gameplay or establish release readiness. Local MiniGrid metadata/license-file
disagreement remains recorded in the existing third-party notices. Original
failed studies, control results and broader parent requirements are unchanged.
