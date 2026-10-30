# Complete episodes inside ordinary TRAIN packets

`scripts.context_catalog_replay.compile_context_packet(spec, batch)` attaches
one complete three-step legacy episode (`new-v1`) or two complete three-step
invoice/password episodes (`old-v1`) to the existing ordinary TRAIN packet.
It checks existing features and labels against authentic public-context sandbox
traces, copies the arrays, and preserves the original row IDs. Trace checking
itself executes sandbox transitions and must be counted as diagnostic work.
It does not recover previously retained memory or give new labels to a learner.

Pass these packets to the existing TRAIN-only `AttributionStream` with
`StreamRehearsal`, `EpisodeReplaySchema(9, 3, 6)` and `JointContextReplayPolicy`.
The whole original packet is one delayed wake update. The helper encodes each
complete episode separately in its existing bounded recent buffer, then uses
the original consolidation transaction and replay budget. Current rows repeated
in the joint solve are charged again. Episodes already wholly present in the
current packet are excluded by exact contiguous feature/target equality; no
model outcome participates in selection.

Why this: splitting the wake changes its timing, fit and cost. Combining two
independent tickets into a fictional six-step trajectory loses provenance.
Explicit ordered `AttributionBatch.packet_episodes` preserves both boundaries.
The default single-episode field remains supported. Packet metadata is admitted
only for ordinary TRAIN-only dynamic joint rehearsal, never probe/final streams,
fixed banks, renewal or capacity paths. Metadata and arrays must agree exactly;
original metadata stays pinned during owner-local pause/resume. No extra capture
format, callback authority, runner or checkpoint-history chain is introduced.

`current_only` still requires equal exposure in complete current packets: a
six-row current packet plus three retained rows is refused, because nine cannot
be tiled with complete six-row packets. This software increment is not a fair
new comparison, a deployed serving integration, an ACK consolidation proof or
evidence of a circadian advantage. #287 remains FAILED/SPENT.

Prospective #299 validation: pure ingress mismatches first; then bounded fixture
old6 → new3 → old6, both uninterrupted and paused after update2. Each full case
requires3 wakes/15 rows plus2 joint fits/18 rows =5 solves/33 fitted rows. A
replay-cap refusal executes2 wakes/9 rows, and prior-packet corruption executes
only1 wake/6 rows before refusal. One existing fixed-TRAIN/probe test (joint
control) costs4 solves/24 rows. Thus each complete platform cohort is17 solves/
105 fitted rows, with no task comparison cohort or source12-fit preparation.
Both platforms fit the shared40-solve/240-row engineering ceiling. Maximum3
complete cohorts/platform,120seconds and512MiB/process; caps are cumulative
over exclusive work receipts. Mismatch checks use0 solves. Preserve every failure.
Independent review is read-only with four bounded source/result reads, no model
work, imports, tests, writes or network. All scientific parents remain open.
