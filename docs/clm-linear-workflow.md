# Direct-fit CLM control in the local ticket workflow

The existing workflow command accepts `--linear-fit-control`. It fits one
detached categorical classifier from the twelve fixed TRAIN transitions, then
uses that classifier through the existing typed predictor and local dispatcher.
The contrastive eight-update path remains the default.

With prepared local assets and a separately declared native execution allowance:

```powershell
python -m scripts.run_clm_ticket_workflow --model-dir C:\prepared\clm --head C:\prepared\head.pth --output fresh-linear.json --linear-fit-control --public-observation-train
```

The public-observation flag explicitly selects the teacher's existing live query
recipe, including its `please` suffix. Without it, the original TRAIN recipe
with a `train` suffix remains selected; the command does not silently change the
dataset. Both recipes encode six unique queries plus five action descriptions
in one eleven-sequence encoder call. The twelve supervised rows include repeated
sealed/staged inputs; they are not twelve distinct text patterns.
The existing loader still requires the released head file; those original heads
are preserved and do not compute the direct classifier's scores.

The direct path rejects `--reuse-action-features`, `--record-train-fit` and
`--learned-only` before loading or creating an output file when those CLI options
are combined. They describe the contrastive path. The direct classifier always
uses the fixed raw action declaration, performs one after-fit TRAIN diagnostic,
and observes only `linear_fit`. Unchanged and contrastive-fine-tuned episode
controls are explicitly unobserved. API options must be exact booleans, including
false options. Existing commands and default reports retain their behavior.

The model uses the [fixed direct fit](clm-linear-fit-control.md): one float64
minimum-norm least-squares solve, equal-weight one-hot labels, appended intercept,
fixed rcond1e-12 and softmax. It has no AdamW or OneCycle updates. No fake eight-step
training receipts or before-fit predictions are generated. The after-fit
diagnostic records original TRAIN probabilities/margins without model selection.
No final/probe labels, replay or promotion are introduced.

`CLMLinearDecisionPredictor` binds an exact TRAIN-fitted control to an existing
`FrozenCLMEncoder` and `FrozenActionFeatures`. A disposable `probe_train` fit is
refused for serving. Encoder identity and original catalog IDs, descriptions and
order must match. There are2..8 fixed classes at this typed boundary. Changing
the catalog requires a new admitted fit/provider, not silently masking classes.

Each accepted request encodes only the query once and performs one linear
readout. Action features are checked but not used in the class-score calculation;
there are no contrastive head forwards. State text construction matches the
existing provider. Ties use insertion order; reported confidence is the selected
probability minus the mean of the other probabilities, clipped to[0,1], not
calibrated confidence. Observed retained query tokens are reported; output-token
count is unknown. Original scanned/retained/padded token costs remain separate.

The provider requires an observed nonempty query tokenization. Exact encoder
counter deltas must agree with that returned token record; unexplained changes
are refused. Failed or lost returns keep their attempted work and permanently
stop that owner. Replacing/mutating the source or owned fitted model, raw actions,
encoder, limits, request or counters refuses further work. Only one last scalar
readout is retained. Raw action storage lasts for that exact frozen encoder,
catalog and fitted model; there is no refresh, time TTL or growing cache.

The composition preserves loaded heads, their modes/gradient ownership/storage,
scale, backbone, encoder declaration, caller RNG and original TRAIN features.
Prior fit/episode values and owners and reported work are checked around callback
boundaries, including final save. Failures retain trusted paid prefixes plus a
pending episode when its effects may be uncertain. A used report cannot restart
the run. Trusted local Python checks are not hostile-process containment.

The original three development scenarios are ordinary invoice, one stale parcel
revision, and invoice commit revocation. State transitions and external capability
checks are unchanged. Denial after revocation is external permission enforcement,
not evidence of intrinsic model refusal. `complete=True` means the bounded
capture completed, not that every task succeeded. Wrong choices remain wrong;
ground truth does not repair a permitted decision.

The workflow provider is born for18 requests and the encoder for19 forwards,
including initial feature production. Three episodes each allow six steps. The
standalone provider can declare1..36 requests at birth. New dictionaries or
providers do not renew an experiment allowance. The public command retains its
existing two-thread,540-second/22GiB sampled-RSS bounds; these are inner guards,
not permission to repeat native experiments or a guarantee of feasible latency.

Reports distinguish fit attempts/rows and logical array footprints, after-TRAIN
diagnostics, query encodings/readouts, permission outcomes and source checks.
Matrix validation, copies, scalar conversion and serialization cost work too;
logical counters are not all allocations, data comparisons or FLOPs. Whole-run
wall time/RSS include that overhead. Previously paid feature production is not
zero computation merely because the classifier has no native head forwards.

Structure:

```text
src/adapters/clm_linear_decision.py   fixed TRAIN control to probability port
scripts/clm_linear_workflow.py       existing teacher, fit and dispatcher composition
scripts/run_clm_ticket_workflow.py   default-preserving CLI selection and loader
tests/test_clm_linear_workflow.py    literal complete flow and refusal checks
```

The core fit and local workflow runtime are reused without changing their math
or effect rules. No dependencies, environment variables, new loader or stream
framework were added. Extend through the existing ports and freeze a new
protocol before interpreting native quality or comparative costs.

The [software report](reports/clm-linear-workflow-20261025.md) records Windows/
Linux checks, actual literal CLI JSON, failures, cost corrections and self-review.
The CLI fixture uses a literal loader and forces one thread while checking the
command's request for two; it is not a pretrained/native hardware qualification.
The software qualification alone establishes no native task acquisition,
retention, recovery, CPC advantage, independent acceptance, package installation
or release readiness.

A separately budgeted [native development result](reports/clm-linear-native-20261025.md)
now shows12/12 original TRAIN decisions, completed legacy and stale-password
tasks, and external invoice permission denial. It preserves the original reader
failure and its prospective engineering repair. This is known-task reproduction;
fresh-task retention, context return and comparative advantages remain open.

The provider also supports [explicit active-catalog binding](clm-linear-active-catalog.md)
for a full union model and a fixed declared tool subset. This read-only Python
option preserves the default workflow path. The separate explicit
[context workflow mode](clm-linear-context-workflow.md) now composes it with
the original delayed-label inbox and local workflow dispatcher.
