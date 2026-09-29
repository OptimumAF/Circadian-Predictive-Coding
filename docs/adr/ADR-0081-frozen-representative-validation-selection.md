# ADR-0081: Freeze representative matched heads before final access

## Context

P1.8o measured development feature cost and froze a larger request before any
new final-test access. The ordinary matched tuning runner built the final
CIFAR source while making its validation choice, and a hard worker timeout
could lose evidence from completed candidates. Neither behavior met P1.8p1.

## Decision

Add an opt-in `development_only_source` selection path that requires CIFAR and
`confirm_test=False`. The default runner remains available for earlier
protocols. The representative adapter restores the exact request, cost-probe,
archive, and pretrained-weight digests; checks a three-reading quiet CUDA
window; then runs the one declared seed and six candidates in a child with a
180-second cap. A source-construction trap rejects any `train=False` CIFAR
factory call. The development loader omits final IDs and labels and raises on
final iteration. These two boundaries keep final labels unavailable during
selection.

An optional attempt observer writes a flushed and synced JSONL event when a
candidate starts and when it completes or fails. A timeout retains completed
trials and identifies the in-flight candidate. A complete result is accepted
only when the journal agrees with all six attempt and trial rows. The adapter
checks shared role, feature, backbone, and initial-head hashes, exact frozen
candidate configs, equal trial counts, and independent outer-validation
choices. The existing first-declared-candidate tie rule remains fixed.
The manifest binds the selection digest to the three already declared
confirmation seeds and their fixed-data, wall-time, and isolated-memory
budgets. Failure artifacts retain available worker and journal evidence.

## Alternatives

- Construct the final source but defer its iterator. That would expose final
  labels before settings are frozen.
- Return a single result at child exit. A timeout would erase completed work.
- Adjust candidate rates or seeds after seeing the outer-validation scores.
  That would change the predeclared comparison.

## Consequences

The unchanged seed-179 request completed under quiet CUDA in 47.551 seconds,
well within 180 seconds. All six candidates trained on the same
16,384-example development role and shared guard/outer feature hashes;
there were 12 durable start/completion events, zero final-source
constructions, zero final iterations, and no final score. Outer accuracies
for candidates a/b were 0.8662109375/0.865234375 for backprop,
0.77734375/0.75732421875 for predictive coding, and
0.716796875/0.67236328125 for circadian predictive coding. Each head chose
candidate a; the circadian result remains lower and was not retuned.

The saved result and manifest are local ignored artifacts. Their selection,
typed-manifest, and outer-freeze digests were checked after persistence.
P1.8p2 must restore them and all request/source hashes before opening final
test, apply the predeclared quiet and scope caps, and retain every negative or
failed result. The subset and frozen-backbone inference limit remains.
