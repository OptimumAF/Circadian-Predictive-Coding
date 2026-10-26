# Bundled MiniGrid affine TRAIN examples

Project-generated synthetic demonstrations, MIT license (repository LICENSE).
Exactly six complete original TRAIN episodes: A blue/red, B blue/red and
C_train blue/red. Each contains nine transitions, raw 148-component public
observations and next observations, seven-class actions, rewards and terminal
flags. These are the original teacher trajectories, not learned-policy results.

Extracted without modifying training_sources from
artifacts/runs/joint-catalogue-study-20261018/bp.txt (original 2,346,812 bytes,
SHA256 4b29e1de792765fead9259ef1c80445b3ba083492a489f200a9fb2ac582dda28).
Only that TRAIN array is included; no fitted states, probes, final evaluation,
controller-validation records, budgets or old experiment authority are bundled.
The package is sufficient for normal first-party training; the historical source
is a provenance reference, not a runtime requirement.

The existing history encoder combines initial/current public observations and
the existing stream admits labels with delay one. Example order and catalogue
capacity are fixed before execution. Known B/C native games measure execution
on training contexts, not unseen layouts or a circadian advantage.
