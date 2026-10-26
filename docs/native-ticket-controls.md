# Native PC/CPC controls for full ticket tasks

This recipe runs the existing PC, neutral CPC or gated CPC learner through the
same TRAIN sequence as the saved qualified BP/new-context case. It changes no
native algorithm, parameter default, action, dataset or promotion policy.

```powershell
python -B -m scripts.run_native_ticket_controls --method pc --output fresh-pc.json
```

Requires existing CPU Torch. Output must be fresh. This is actual training;
the completed local cohort is governed by its
[fixed protocol](../artifacts/runs/native-ticket-family-controls-20261023/PROTOCOL.md).
Do not rerun or sweep automatically. Saved BP evidence is reused without refit.

The separate explicit [`--fresh-bp` demonstration](fresh-bp-ticket-demo.md) now
uses this same live recipe with fresh BP ownership. Original `--method` peer
choices and saved comparative evidence remain unchanged; BP is not admitted by
the peer method argument and the fresh mode is not another comparison.

## Boundaries

The script composes the original teacher builders, `AttributionStream`, native
learner adapter, frozen typed decision readout and sandbox task executor. One
live stream owns 96 joint invoice/parcel batches, eight legacy/password batches,
then eight invoice/parcel return batches. Each batch has six original TRAIN rows
and delay one. Native readouts occur at updates 96/104/112, locally 0/8/16.

Each endpoint observes the complete native state, a separate 12-row teacher
readout and the original fixed task slots: 20 full episodes total, including
stale revision recovery and refusal after commit revocation. Teacher/task reads
cannot update the source, seed, stable actor or training budget. Full native
state and caller RNG are checked. Failures preserve attempted/returned work,
the committed receipt prefix and unknown returns. No final, reserved probe,
replay, promotion or adaptive stopping is available.

The initial four parameter tensors and old training arrays match saved BP.
PC and neutral CPC must match every endpoint parameter tensor, raw teacher
logit, diagnostic, task trace and summary. CPC's extra state remains native;
this script adds no record reader or persistence framework.

Why this: an existing end-to-end BP success makes a small peer comparison more
useful than another checkpoint or provenance increment. The BP-informed fixed
dose is not equally optimized for peers. Equal exposure and geometry do not
mean equal objectives, computation or runtime lifecycle: BP warmup was restored
into a fresh local stream, whereas peers stay live throughout. Stable actors
and event-clock prefixes differ; neither is input to these learners.

## Interpretation and verification

The sole cohort completed336updates/2016rows. All three peers fail all ordinary,
stale and revocation slots at max6steps; PC-neutral evidence is exact. Teacher
accuracy stays4/12 despite lower loss. Revocation events are not reached, so
these failures do not establish permission refusal. Saved BP remains the working
demonstration at this BP-informed dose. Final Linux8checks and four statics pass;
Windows6passed before a fault-message assertion failed, preserved with all costs.
The final Linux tree confirms the correct stronger native clock refusal.

Inspect warmup ability before interpreting retention: zero forgetting when the
old task already fails is not retention. Initial new-context success is transfer,
not acquisition. Success after return is not recovery from forgetting unless an
old task was first lost. One synthetic seed cannot establish a general method
ranking, mechanism attribution, circadian advantage or release readiness.

See [results](../artifacts/runs/native-ticket-family-controls-20261023/RESULTS.md)
for observed task behavior, costs, preserved failures and limitations.

```text
scripts/run_native_ticket_controls.py   Existing-component fixed peer recipe
tests/test_native_ticket_controls.py   Tiny streams, parity and fault behavior
```

```powershell
python -m pytest -q tests/test_native_ticket_controls.py
python -m ruff check scripts/run_native_ticket_controls.py tests/test_native_ticket_controls.py
python -m ruff format --check scripts/run_native_ticket_controls.py tests/test_native_ticket_controls.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/run_native_ticket_controls.py tests/test_native_ticket_controls.py
```

Fixtures reduce warmup and each phase to one batch. They do not repeat the
scientific 112-update dose. Trusted synchronous source guards are not a hostile
process or concurrency boundary. Any new study needs its own useful question
and prospectively frozen small protocol; extending the dose to obtain a win is
not an automatic next step.
