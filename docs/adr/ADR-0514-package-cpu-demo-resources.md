# ADR-0514: Package the CPU demo with its original numeric data

## Context

The offline and optional native commands work from the checkout. The accepted
isolated source copy reused installed NumPy; it did not establish ordinary
installation. No build configuration existed. Dataset lookup used a source path
that would be missing from a normal wheel.

## Decision

Use the installed setuptools backend and ordinary wheel/sdist packaging. Keep
the existing src/scripts package names and public module command. Runtime
dependencies cover the CPU demo's existing NumPy requirement; optional research
SDKs are separately prepared. Provide an additional console entrypoint.

Map the existing examples/taskspace_memory directory to the installed private
src.infra.taskspace_demo_data package. Include its original numeric JSON/notice
and the repository license. Resolve installed defaults with importlib.resources
and as_file; explicit --bundle always selects exactly the supplied path.

## Alternatives

Copying the JSON into a second source directory risks divergence. Searching the
working directory or research archives would make an installation depend on
unrelated state. A custom loader/build backend adds unnecessary machinery.

## Consequences

The source command and original example path remain usable, and the wheel owns
one copy of the data. The distribution is an alpha CPU example, not a release
qualification of every optional research command. Native/GPU dependencies, clean
Git-clone delivery and remaining roadmap/release criteria keep their own gates.
