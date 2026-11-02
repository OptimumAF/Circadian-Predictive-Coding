# Saved BP diagnosis: loss improves while routing remains unfitted

R2.1-TW.D / [#316](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/316).
Read only the prospectively fixed Linux BP fixture from #315. No model,
parameter reconstruction/arithmetic, encoder, teacher, forward, gradient, fitting
or simulator work. All original results, task definitions, directions and files
remain unchanged. This diagnosis narrows the gap; it does not prove its cause.

The model opens tickets and recognizes commit examples, but its routing decisions
are already wrong on supplied TRAIN monitors. Every checkpoint of both new
prototype contexts and both old TRAIN contexts has accuracy2/3. Cross-entropy
falls at the same time. The six closed-loop tasks remain0/6 in all three phases.
That is insufficient fitted task behavior, not demonstrated loss of plasticity,
old-task forgetting, an unobserved evaluation effect or a circadian advantage.

| Fixed metric | Before F | After F16 | After return16 |
|---|---:|---:|---:|
| New v1parcel CE |1.655321|1.408418|1.108025|
| New v2password CE |1.624289|1.382480|1.037488|
| Old v1 TRAIN CE |1.640727|1.398506|1.076149|
| Old v2 TRAIN CE |1.652161|1.488061|1.186120|
| Each context's row accuracy |2/3|2/3|2/3|
| Six actual tasks |0/6|0/6|0/6|

Intermediate checkpoints4/8 have the same2/3accuracy. Per-row diagnostics show
the required ready-step class is incorrect, while open/commit examples are
correct. New targets are route for v1parcel and inspect for v2password. Old TRAIN
v1 includes route/inspect, v2 inspect/defer. The observed successes are not full
three-step task successes; two common phases account for2/3of each trace's rows.

All44original updates returned in the fixture. Source12 pre-update CE reports
range1.789317→1.657197; source phases use different batches, so this is not a
global matched objective curve. F alternates the two new contexts: odd losses
1.655321→1.440058, even1.607997→1.397912. Return alternates old TRAIN contexts:
odd1.398506→1.126134, even1.477892→1.211943. These same-context improvements
show a changing fitted predictor, not sufficient convergence or task usefulness.
No new readouts or gradients were computed to make this statement.

## Direct control evidence

The first fixed task is v1invoice. All phases first select open in sealed state
and successfully move to ready. At the ready request
`ready invoice policy_v1 feedback applied please`, the required action is route:

| Phase | Chosen action/logit | Required route logit | Wrong-minus-target gap |
|---|---|---:|---:|
| Before F, source12 |open/.264310|-.096273|.360583|
| After F, private28 |commit/.729310|-.072278|.801588|
| After return, private44 |open/.955698|-.072700|1.028398|

The permitted wrong-phase choice is refused by the existing dispatcher, then
repeated until six-step `max_steps`. The feedback word changes to `wrong_phase`;
the model does not recover. `diagnosis.json` preserves all18episodes/108returned
steps, not just this example. All108selected IDs match the highest saved raw
logit and probability with the existing lexical tie-break; probability mass and
candidate masks check. No confidence cutoff or decoder remapping selected the
wrong action. Gaps are far larger than rounding-level ties. Do not weaken phase
semantics or mask the recorded errors to manufacture success.

## Observations, source inference and missing proof

Saved input diagnostics report F48rows6unique keys, old TRAIN12rows8keys,
joint60rows10keys. Equal-input label conflicts0, row-accuracy upper bound1.0,
normalized CE lower bound0.0. This excludes a recorded exact-alias contradiction
in those supplied arrays. It proves neither separability by this finite H3
network nor optimizer convergence, hidden-feature quality or an achievable
zero-loss parameter configuration.

Current source inspection confirms binary token-presence encoding, fixed D9
vocabulary, six sorted classes, raw allowed-logit softmax/argmax and ordinary
categorical CE/SGD. Teacher queries append a non-vocabulary `train`; feedback
tokens also fall outside that vocabulary. This supports alignment of the known
teacher/typed query families as a **source inference**. Raw source/probe feature
matrices were not serialized; no encoder was called here to assert exact vector
identity. Hidden activations/rank/saturation, post-update logits on every training
batch and a capacity/convergence witness are absent. No causal claim that H3,
dose, weighting, optimization or representation is the unique cause is justified.

## Reconcile existing controls before choosing more work

[#278](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/278#issuecomment-6496148973)
already implemented and qualified current-arrived-batch class-balanced BP, with
unchanged H3/seed17/dose/source guards. Its ONE12update72row qualification plus
four12-row TRAIN diagnostic reads (48rows total) ended ready0/4 at every checkpoint and all
ordinary tasks failed. Final TRAIN fit8/12 and CE1.693899 did not repair the
ordinary unweighted control's8/12/CE1.646444 result. Its scientific/raw-reader
budgets are spent/closed. Reuse this accepted negative result; no rerun or raw
reread was made. An initial next-action suggestion to qualify it was withdrawn
when the existing evidence was found, before any model work.

Existing original-source TRAIN-fit diagnostics already exist in
`scripts/ticket_workflow_native.py`; they use exactly12old TRAIN rows and no
future/probe/final data. Do not rebuild them. Continuous frozen readout fitting
also exists in `src/adapters/continuous_readout_fit.py` and its core, but its
clipped inverse-tanh continuous objective is not a categorical fit control.

ONE next implementation action: qualify an actual-owned, read-only H3 feature
port for a TRAIN-only frozen-feature categorical readout feasibility diagnostic,
reusing `src/core/categorical_linear_fit.py` for the solve. Keep the original
30feature-extractor parameters and24output coefficients/capacity unchanged; one
disposable fitted readout can test whether this frozen representation supports
the supplied routing decisions. Source/actor/optimizer/hidden weights never
change; no loader, promotion or new orchestration. Fit/evaluation use admitted
old TRAIN only, not the reserved probe/final labels. Verify whether an existing
equivalent suffices before adding an adapter. A failed least-squares readout is
still inconclusive about all possible nonlinear/CE optima; a successful one is
only a fixed TRAIN feasibility witness. Freeze a separate small protocol and
all feature/fit/readout work before numerical execution.

Why this: both uniform and class-balanced controls leave ready fit unresolved;
the new continuation also does not resolve it. A capacity/representation versus
readout-fit question is useful before another dose/seed/weight/capture variant.
The accepted working conventional LS serving control #314 remains available.
No circadian mechanism or favorable-run selection is needed to explain away
these negative results. Fullgoal/#21/#18/R2.1/reliability/science/release stay open.

## Execution and limits

Physical cwd `D:\Dev\Repositories\Circadian`; TEMP/TMP=`D:\Dev\_Temp`.
Frozen input `artifacts/runs/ordinary-native-reserved-probe-20261030/linux-bp.json.gz`
chosen before inspection for the established Linux producer, not its outcomes.
Its original pre-final-guard numerical case and costs are accepted315evidence;
later private-endpoint hardening did not remeasure pipeline overhead. No new
performance comparison or native acceptance is inferred here.
Final local stage7files379,655B, below2MiB, including full saved schema/extraction/
diagnosis and review. Scoped changed-document whitespace passed with Git LF/CRLF
advisories; inherited source/untracked/ignored work and frozen captures retained.

Three exact script invocations under
`artifacts/runs/ordinary-native-bp-diagnosis-20261030/`, using prefix:

```powershell
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B <script>
```

| Script | Result | Seconds | Peak RSS B |
|---|---|---:|---:|
| inspect.py |schema/first paid trace, EXIT0|.016918448|16,601,088|
| extract.py |fixed metrics/all18traces, EXIT0|.021213585|16,785,408|
| validate.py |13validation groups/108step checks, EXIT0|.029959776|18,821,120|

Total known process wall.068091809s, max18,821,120B, all under3calls/30s64MiB
each/2MiB stage.0project/NumPy/Torch imports, models, random draws, parameter
arithmetic, teachers, encoders, new forwards/gradients/fits/physics/probe updates,
final access or remote work. Initial verbose terminal output was truncated;
the complete JSON schema/extraction artifacts were saved, no raw evidence lost.
Final output was compact. No failed script or allowance amendment in this scope.

Lead source/procedure/evidence self-review completed. Finite independent
interpretation/source review accepted, no blocking findings:3read-only commands
.6747228s aggregate, all exit0, no writes/project imports/models/tests/encoders/
teachers/forwards/network/Git mutations/children. Reviewer independently checked
every per-row checkpoint decision against the original fixed gzip, all108saved
rankings/masks/masses, source interpretation and causal limitations. The sole
minor correction was #278's4x12=48diagnostic rows, now corrected. `review.txt`
preserves the handoff; exception ended. Production source unchanged. No new
pytest/lint/type/native/model or baseline cohort: unchanged code, explicit
zero-model scope; accepted315checks reused. No install/download/delete/commit/
push/global cache changes. Independent acceptance and a later readout-feasibility
experiment must not be inferred from these13metadata validations.
