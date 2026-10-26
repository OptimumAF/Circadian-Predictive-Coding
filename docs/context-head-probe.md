# Isolated further learning on contextual heads

The existing ticket probe recipe now supports the qualified conventional head
control from77:

```powershell
python -m scripts.probe_ticket_policy_learning --separate-context-heads --input artifacts/runs/context-head-control-20261023/actual.json --output head-probe.json
```

The input must match the qualified local source exactly; output must be new.
Original nativeBP/D7mode remains supported. This path reuses the existing generic
FreshTaskProbe, old-TRAIN return and endpoint observer. The small
`scripts/context_head_probe.py` helper prepares source/data/guards; it supplies no
new training runtime. Source weights are copied as data, not budget authority.

## What is measured

Warm(source12updates72rows) and fresh-zero controls both have96coefficients and
known publicv1/v2routing. Probe tasks are unused **known TRAIN** legacy templates,
not unseen or final data. Batches remain homogeneous by context. Each private arm
runs16alternating3rowprobe updates48rows, then16alternating6rowold-TRAINupdates
96rows on the **same disposable learner**. Fixed0/4/8/16 curves report full-class
CE/top1 and old-TRAIN monitors. After each phase, actual workflows read the old
four tasks plus two legacy tasks. Benchmark/deployed/source owners do not train.

Both16update budgets are born once at arm start. Each has30s512MiB; the return
deadline includes preceding probe time. Clocks and permanent charges are retained
on failure. A failed/altered/replaced first arm prevents admission of the second.

## Executed finding

| Control / endpoint | Old4 workflow success | Legacy2 workflow success |
|---|---:|---:|
|Warm after probe|3/4 (password lost)|2/2|
|Warm after old-TRAIN return|4/4 (password recovered)|2/2|
|Fresh after probe|1/4 (parcel transfer)|2/2|
|Fresh after old-TRAIN return|4/4 newly learned|1/2 (v1legacy lost)|

Warm v1legacy row fit grows2/3→3/3 by checkpoint4. Warm v2legacy was already3/3
before probing: inherited ability, not new gain; its coefficients never change.
Warm old v1fit drops6/6→5/6; returning old TRAIN restores6/6 by4 without losing
legacy endpoint success. Fresh legacy fits grow1/3→3/3 each by4. Its return learns
all old tasks while reducing v1legacy fit to2/3. Fresh old tasks were not initially
acquired, so their return success is not recovery. Password recovery in warm is
supported by qualified initial task success, observed loss and observed return.
There was no initial legacy workflow readout; initial legacy ability is measured
by the recorded full-class row scores, distinct from endpoint task success.

This demonstrates further same-context learning and a retention/recovery tradeoff
in a conventional control. It is not evidence of general plasticity, unseen-task
transfer, CPC advantage, statistical superiority or release readiness. Warm and
fresh have equal capacity/current dose but different prior training: warm44total
updates216rows versus fresh32/144 including source history. No all-in compute match.

## Costs and margin-report deviation

The sole two-arm case costs64new updates288rows18mistakes (warm1probe+1return;
fresh6+10), plus carried77source12/72/16mistakes.4composite learners8zerohead
initializations,2240context snapshot checks4816perceptron copy entries231168logical
coefficient slots,200routes,288full-class prediction rows,72typed decisions,
2teachers3packets174encodes96sandboxdispatches. Inputarrays6240B, model768B/arm,
four observed endpoint snapshots3072coefficient bytes. Python/copies/validation/
guard/IO overhead additional. Recipe17.68397s, meter18.20236s, sampledRSS52,510,720B.

The original protocol mistakenly promised margins from returned scores; generic
reports save CE/top1/class decisions but no complete score vectors. The raw case
is preserved unchanged. A separately frozen **read-only reference supplement**
finishes full-class target-minus-best-other logit margins, cross-checking every
original CE/top1/per-row decision and all72endpoint allowed-score rows. At4/8,
coefficient equality is inferred from an entirely zero-mistake suffix under the
pinned original unit rule; intermediate counter snapshots were not observed or
invented. The output labels this inference explicitly.

That supplement adds288reference rows13824affine terms, no training/model
construction/typed/environment calls;0.02623s27,324,416B boundaryRSS. Combined
full-class rows576remain below the original1000cap. It is a disclosed protocol
deviation, not a claim the original run captured margins or a new experiment.
Original clocks, source/actual bytes, metrics, dose, failed outcomes and costs
remain. Scripts/ledgers and exact commands are in
`artifacts/runs/context-head-probe-20261023/RESULTS.md`.

Windows89selected tests/Linux84pass,0skips; five redundant legacy scripted tests
are unselected on Linux after all passWindows. Three-file Ruff/format/fresh
default+Linux mypy pass. Source/procedure/result review is self-review, not
independent acceptance. Fullsuite/nativeexperiments/physics/math/packaging/
cleanclone/CI unrun. Live status:
[issue78](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/78).

## Next useful boundary

The [qualified-source serving slice](context-head-serving.md) now completes this
boundary without using these private probe weights. Mixed readonly guards and
NumPy typed serving reuse existing stream/promotion/rollback. Fixed2updates9rows
reject an old-row regression, then serve6/6known tasks versus5/6frozen, with all
costs and rollback preserved. The [live worker](context-head-live.md) now supports
public contexts too; its sole case observed zero typed-request/model-call overlap.
Next compare existing fixed controllers on this learnable policy, reusing the
periodic result. No new runtime or probe-to-deployment path is needed.
