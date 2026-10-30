# R2.1-CTX.U — Earlier guarded serving and causal return

Issue271 scoped implementation, focused validation and one frozen local case
complete. Full repository goal and parents18/21/23/26 remain OPEN. Historical
plans/log stay frozen under AGENTS; issue comments carry live status/handoff.
Checkout0aae1d6 and inherited270 docs/test changes were preserved. No commits,
pushes, dependencies, global settings or other repositories changed.

## Delivered behavior

Two existing modules changed: `scripts/run_bp_ticket_serving.py` reuses its
two-window schedule and drain callback for explicit measured-refusal continuation;
`scripts/context_head_live.py` wires the exact paired learning/return route.
New `tests/test_context_learning_boundary.py` covers scalar state boundaries and
native literal-source integration. [Usage](../paired-learning-boundary.md).

The fixed learning boundary assesses four native receipts. Only an original
typed, validated guard refusal allows unchanged-actor continuation; bare None,
unknown callback outcomes, changed digests/revisions/decisions, changed actor or
candidate, and altered/replaced prior evidence refuse. Guard callbacks retain
original budget identity/limits/clock origin, training charges, receipts and full
sharing snapshot. Model/configuration/metadata/generation/version are pinned
before and after opportunities and completion; ordinary serving-cache changes
alone remain legal. At most two fixed guard records are retained, with the
original first record separately protected after a later publication.

Return observations12/13 and labels13/14 use their actual observed actor version.
The same six-update/60s/512MiB owner continues. If early accepted, keep the actor
and skip another publication. If refused, reassess as soon as all return receipts
drain; no artificial wait through the serving tail. The two named recipes,
original24worker slots, thresholds, data roles, seeds and default behavior stay
unchanged. No new runner, learner algorithm, probe conversion or dependency.

## Frozen native result

Stage `artifacts/runs/paired-boundary-train-20261029/`; original source277216B,
SHA256 `addf37403d32682b680b74ea7a1bfc16ced2895cffe09b8cc5fdf701179be943`.
Prior source12passes72rows16mistakes remain separate history per restored owner,
not a refit or restoration of old authority. Two D7/K6 heads/public D9 inputs,
96coefficients/768B. Original source owners and NumPy RNG are unchanged.

Current-only then joint, each six packets9,9,9,9,6,6: learning v1/v2 twice, then
old-v1/v2 return. Equal6native passes48rows,3passes24rows/head. Current new36old12
exposures versus joint new12old36. Old TRAIN rehearsal is paid even though the
separate replay-helper counter is0. No final evaluation or private probe data
enters training, promotion, selection or tuning.

| Observed outcome | Current-only | Joint rehearsal |
|---|---:|---:|
| After learning: new tasks |2/2|2/2|
| After learning: old tasks |2/4|4/4|
| Guard at tick10 |reject old retention|accept|
| Publication tick |16|10|
| After return candidate: old/new |4/4;2/2|4/4;2/2|
| Fixed late ordinary slots |7/8|8/8|
| All worker successes |16/24|19/24|
| Ordinary successes |14/18|16/18|
| Stale-once successes |2/3|3/3|
| Published readouts |6/6|6/6|
| Full source rollback/generation |yes/2|yes/2|

Primary direction was joint minus current successes over fixed late ordinary
ticks13,15,16,17,18,19,21,23: observed+1. The current tick13failure remains in
7/8. Full worker differences occur at tick11v1parcel,12stale v2password and
13ordinary v2password; all favor the earlier qualified joint publication.
Three expected permission refusals per arm at6/14/22 remain separate from
learning failures. Completion does not mean task success.

Current's first guard observes new utility1 and old10/12 versus actor old1;
original zero-old-drop policy correctly refuses. Its return repair passes at16.
Joint's first guard observes new1/old1 and publishes at10. Both retain original
min gain0.1/new minimum1/old drop0/latency0.25s/record4096B thresholds; measured
records841B. Current pays a second guard attempt; this asymmetry is part of cost.
Publication follows its tick's worker, so the new actor appears only afterward.

The actual joint deployed snapshot is early native16/108, heads10/66 and6/42;
current publishes native18/120, heads11/72 and7/48. They are not identical full
published states. Both final private candidates reach18/120 and diagnostic6/6.
Further candidate training never updates the early actor implicitly. Each has
one accepted publication to c0/generation1, then full rollback to original source
parameters at generation2. Native receipts apply at4,6,8,10,14,16; return receipts
retain qualified-12 for current and c0 for joint, without rebinding.

Readonly candidate diagnostics use this invocation's existing native returns4/6
through the frozen reader and original episode loop, separate zero-update30s
sessions. Their inference/copy/read/time/memory costs are paid; no extra fitting.

This is a descriptive conventional known-template result with local serving
usefulness. It establishes no held-out/generalizing/reliable/statistical or
circadian benefit, equal total computation, or repository-wide release readiness.
Changed publication/label timing means270's prior serving results are not a
matched new control. Its raw output was never opened; all old negatives remain.

## Commands, checks and cost

Every project command ran from physical `D:/Dev/Repositories/Circadian`,
TEMP/TMP=`D:/Dev/_Temp`, Python `-B`. Linux uses `/mnt/d/` equivalents and the
existing `/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python`.
No installed/global tool changes. Existing software observer was copied unchanged.

Final command, with `win-return-gates` or `linux-return-gates`:

```text
<python> -B artifacts/runs/paired-boundary-train-20261029/check.py <platform>-return-gates tests/test_context_learning_boundary.py tests/test_post_publication_learning.py::test_two_windows_keep_one_runtime_clock_budget_gate_envelope_and_truthful_versions tests/test_post_publication_learning.py::test_refused_publication_never_opens_second_window_or_predicts_new_data tests/test_post_publication_learning.py::test_unobserved_later_generation_refuses_before_fresh_label_or_native_use tests/test_post_publication_learning.py::test_generation_change_after_selection_refuses_before_second_admission tests/test_drained_ticket_publication.py::test_verified_publication_invalidates_feedback_without_rebinding_or_losing_old_evidence tests/test_drained_ticket_publication.py::test_guard_refusal_preserves_original_feedback_and_never_retries_at_tail
```

35PASS/platform,0skips. Windows1.9893932s/78,372,864B sampledRSS;
Linux5.629289461s/72,601,600B. Each17native attempts138rows includes the deliberate
lost-fifth-return fixture:5attempts42rows/4returns, retained actor generation1,
no falsely claimed rollback or refundable work. Two successful cases each6/48
verify native publication, causal return and rollback; other cases use scalar
fixtures to test state management. Earlier20/27/29/35checks all PASS; receipts
and JUnit files retain each invocation.

Original Windows five-invocation allowance now SPENT:58native attempts468rows,
60constructors120heads840matrix rows1094typed1334sandbox20teachers782packets,
8094context14534head copy visits1474656logical slots;8.7963991s,
peak79,114,240B. Linux two invocations34attempts276rows,36constructors72heads,
504matrix602typed746sandbox12teachers398packets,4814context8358head visits
863328slots;11.141951061s,peak72,601,600B. No forbidden imports or historical
scientific JSON reads. Observer `scripted_updates=0` does not instrument this
suite's LiteralLearner and is not a claim of zero scripted transitions; each
such runner's original budget/receipts bounds its scalar updates to at most2.

Final three-file statics PASS using `.venv/Scripts/python.exe -B -m`:
`ruff check`, `ruff format --check`, and
`mypy --explicit-package-bases --follow-imports=silent` (default and
`--platform=linux`) on both changed modules and the new test file.

Sole genuine command, EXIT0:

```text
.venv/Scripts/python.exe -B artifacts/runs/paired-boundary-train-20261029/native.py
```

12new passes96rows,14constructors28zero heads204matrix predictions300typed
requests348sandbox steps4teachers0traces;1991context3606head copy visits,
364224logical slots.513fixed-source reads return142,211,808B, attempted-byte
ceiling268,960,257B. Whole4.4653259s/71,352,320B sampledRSS,metadata0.0260776s,
output2,425,068B. All frozen180s/512MiB/4MiBoutput/16MiBstage and original
per-arm session caps held. Source history is separately carried, not new fitting.
Same sampler spans imports through capped writer and the sole raw metadata read;
sampling is not a hard OS peak. Scalar costs flush before output. Final tiny
receipts follow sampling.

`PROTOCOL.md`, `declared-slots.json` and `source-manifest.json` bind the final
source/test/producer/protocol before execution. `native-body-work.json`,
`native-final-work.json` and `native-summary.json` retain costs and scalar truth.
The raw native.json metadata read is consumed. Case SPENT: no rerun, raw reread,
rescoring, late salvage, restored capture or renewed budget.

## Failures, review and remaining delivery

No pytest or native case failed. Initial statics found the gate's required
keyword argument and generic/heterogeneous fixture annotations; repaired before
qualification. Two patches missed formatted anchors without changing source;
narrow edits replaced them. No failed work was erased.

Independent readonly source review found a real final-opportunity gap: pins
were checked before admission but not after the worker or at completion.
Lead added both checks and last-tick model/evidence mutation tests. A bounded
followup review found no further blocking issue, including immediate return-drain
publication and retained first evidence/latest actor pins. Both exceptions ended.
No reviewer ran tests, read raw scientific output, fit models or replicated the
experiment. Lead reviewed source, procedure and observed scalar results; native
result acceptance is self-review, not independent scientific acceptance.

During software development the initial repair assessment waited until tick24;
lead rejected that unnecessary delay before any scientific run. It could have
artificially increased the serving difference. The final implementation assesses
at the semantic return drain, retaining original policy and data. This rationale
and prospective scope changes are recorded in issue271 before execution.

Skipped: broad suite/CI, neural/physics/model-family sweeps, installed-wheel or
clean-clone delivery, and full native CLI encoder. Earlier wheel269 predates270
and271. No normal CLI flag exposes this new API yet. No scientific or release
parent is closed by scoped correctness or one conventional workflow case.

Exact next implementation action: expose `publish_after_learning` through the
existing context live CLI and source-free preview, refusing unsupported modes
before IO and retaining the original command behavior by default. Qualify its
bounded output with ordinary scalar/CLI fixtures; do not reopen this spent case
or reread its raw report merely to test serialization. This delivers the working
behavior through the normal command rather than adding another capture/provenance
or update-count increment. A new bounded owning issue must precede that work.
