# ADR-0165: Inventory resource fields with their original scope

Status: accepted and implemented; complete correctness and actual bounded
publication/repetition/readback acceptance pass.
Date: 2026-10-01.

## Context

The complete confirmation report and raw cost inspections preserve 560 arm
records, 1,680 checkpoint capacities and sixty large shared proof contexts.
Original train/scored audits record whole-process time and sampled RSS.
P6.10 asks for resource costs and outcomes against compute and memory. A
field must retain its unit, measurement scope and provenance before it can
support a comparison. Latent loop counts are derived from recorded work,
and rejected replay executes despite rollback. Shared FIFO supply and model
retention are distinct from process memory. Per-arm wall/RSS and isolated
sleep/guard durations were not measured.

## Decision

Implement P6.10a as a separate pure inventory over complete stored cost
inspection/report/audit inputs. Reconcile every original field, cell and
context; preserve raw source pointers, measured/derived/unmeasured status,
known units, recorded capacity history and unknowns with concrete next
actions. Keep whole-process observations in separate run rows. Count
rejected execution, shared supplies and retention before copies explicitly.
No per-arm attribution of shared time/RSS, cost composite or ranking.

The infrastructure boundary pins both complete original cost inspections,
both complete reports and all original inputs/audits/current sources, before
and after sequential reads. Recorded complete original reader validations
remain exact inputs. This consumer proves current saved-byte preservation
and independently rederived inventory consistency; it does not claim a new
complete scientific readback. P6.10b retains its separate unchanged official
complete-reader publication/readback requirement. This distinction avoids
repeating scientific reader work to answer a stored-field inventory question.

Freeze the complete new source/request closure before fabricated inventory
fixtures, then pass schema/arithmetic/corruption/artifact gates before local
publication/repetition/readback. Preserve every original scientific pin and
budget; the inventory has a separately declared derivative validation budget.
Any additional resource measurement requires its own prospective protocol.
P6.10 stays unchecked until its original acceptance is explicitly audited.

## Alternatives

- Dividing whole-process time or RSS by arm count invents measurements.
- Counting only committed replay hides execution spent on rollback.
- Summing retained bytes over successive checkpoints confuses occupancy with
  cumulative exposure.
- Treating formula-based loop counts as profiling overstates observation.
- Copying every large shared proof into each arm duplicates evidence without
  improving its source binding.
- Rewriting scientific validators or reopening final data changes scope.

## Consequences

The inventory is complete for the stored original measurement setting and
explicit about unavailable measurements. P6.10b can add outcome presentation
through a new module over the same verified evidence. Full resource/hypothesis,
stream and release acceptance remain separate. Tests use unscored development
or fabricated metadata; they do not establish reserved scientific authority.
All 99 new/576 related tests and static gates pass. Both actual publications
and both full saved-input reconstructions exit 0 within 120 seconds each,
with byte-identical full JSON/Markdown and all 24 scientific guards zero.
The resource document/log record exact sources, inputs, outputs and scope.
The inventory identifies 300 owned-retention fields absent from raw method
rows. Add P6.10b1 to recover those exact fields from the already saved complete
checkpoint proofs before outcome presentation. Missing projection metadata
does not itself require a new resource measurement or authorize allocation
of group bytes to arms. Original parent acceptance stays unchanged.
