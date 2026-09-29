# Atomic local artifact publication (P5.3a)

Completed P5.1 v14 bundles and P5.2 observation directories now use
the same local publication boundary. Every writer validates its
source and prepares the complete file bytes before calling it. The
boundary writes them under a hidden sibling path such as
`.run-id.pending.<unique>/`, checks the staged bytes, removes the
staging-only state file, and renames the directory to its public name
on the same volume. The manifest remains the last data file written.
Readers and report builders must consume only the public name after
its existing verifier succeeds.

An exclusive `.name.publish.lock` serializes cooperating publishers
for one target. An occupied public directory or existing lock is
refused; no writer overwrites a completed run or projection. The
lock is removed after normal success or caught failure. If a process
is killed without cleanup, a stale lock or hidden stage may remain
for later inspected recovery. Do not treat a hidden stage as a
completed result.

The hidden `publication-state.json` starts as `incomplete` and lists
files successfully written. A caught write error marks it `failed`;
KeyboardInterrupt/SystemExit mark it `canceled`. If the machine stops
before a state update, the `.pending.` path itself still identifies
an incomplete publication. Stages are retained for inspection rather
than recursively deleted. No status file appears in a published
directory; the existing P5.1/P5.2 manifest and verifier contracts are
unchanged. This covers visibility of completed directories. Full
crash durability of filesystem metadata and safe training continuation
remain separate from this boundary.

One bounded local measured run, `artifacts/runs/p53-atomic-a/`, and
both observation projections verified after publication. Its v14
training, scored, and wake diagnostic data hashes match the earlier
same-environment runs. Existing `p51-v14-schema-c/d` completed bundles
also still pass the unchanged P5.1 verifier. Fault-injection tests
cover incomplete/failed/canceled stages, public-path absence, and
retry from the same in-memory inputs after caught failures.

Why this: final-directory visibility is a clear success boundary for
artifact readers, while the hidden publication state gives the
[P5.3b checked resume route](v14-checked-resume.md) specific evidence
of interrupted artifact writes. Training has its own hidden run-state
cursor and trusted trial checkpoints; publication stages are not
training checkpoints.
