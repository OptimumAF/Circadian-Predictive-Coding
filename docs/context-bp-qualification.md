# Context-aware native BP joint-fit qualification

```powershell
python -B -m scripts.qualify_ticket_bp_fit --policy-context --output fresh-context-fit.json
```

This executes real native training. Use the existing CPU Torch environment and
the [frozen local protocol](../artifacts/runs/context-bp-qualification-20261023/PROTOCOL.md).
Output must be fresh. The default command retains its original D7/two-task recipe.
This mode creates a new D9 learner; old D7 checkpoints are not equivalent inputs.

## Fixed case

The existing qualifier and `AttributionStream` jointly train v1invoice/password
and v2invoice/parcel from the outset. Twelve TRAIN teacher rows include all three
workflow phases for each task. The actual sandbox verifies the teacher; strict
TRAIN/context groups, D9 schema, public-query alignment and nonconflicting labels
are checked before fitting. Policy context is public state, never a target label.

Existing BP D9/H3/K6, seed17, learning rate.03, momentum.9;96updates,1152row
exposures, delay1, fixed endpoints0/96. There is no early stopping or dose/seed/
width selection. One protected12row native teacher readout per endpoint reports
global CE/margins and permitted-action fit. All four tasks then run ordinary,
stale_once and revoke_commit episodes on frozen owned copies. Task truth comes
from actual effects; row accuracy is reported separately.

Source seed/local deployed actor, candidate/readout states, original budgets and
RNG remain guarded. Before further work, contextual inputs and paid initial
evidence must match their saved originals; altered evidence is restored and
refused. Failures retain paid prefixes, unknown work and original charges. The
new role/definition/markers keep it outside the strict old D7 source reader.

## Result: partial task learning; joint qualification unsuccessful

The sole fixed run completed all96updates. Global TRAIN CE1.788785→0.428979 and
permitted fit6/12→10/12. Both failed rows are ready-phase routing decisions:

| Task | Target | Selected at96 | Target margin | Ordinary / stale |
|---|---|---|---:|---|
|v1invoice|route|inspect|−0.885811|fail / fail|
|v1password|inspect|inspect|+0.875267|pass / pass|
|v2invoice|inspect|inspect|+0.882756|pass / pass|
|v2parcel|defer|inspect|−1.030482|fail / fail|

All initial ordinary/stale cases fail. At96 all four revoked commits are correctly
denied, which are refusals rather than completed workflows. All24episodes,
intermediate TRAIN receipts, margins, states and costs remain in the artifact.
The public context resolves the encoded rule conflict, but does not establish
that this fixed BP recipe can fit all four tasks. It does not prove the cause is
capacity, optimization, class frequency or plasticity. No CPC/controller,
generalization, matched-compute or release conclusion follows.

This differs from D7's earlier joint two-task fit: input geometry/initial tensors,
task set, batch size and row exposure changed. It is not an isolated causal
comparison of adding context. Both policies were available throughout, so it
does not measure chronological acquisition, forgetting or recovery.

## Costs and validation

D9 head54float32 parameters (216bytes), versus D7's48; full final parameter/
gradient/momentum payload648bytes. Actual96SGD/backward calls1152rows,
207native forwards (96TRAIN+2teacher+109typed),2048snapshot constructions/
170856logical tensor elements,210constructor calls/9450configured matrix slots,
133sandbox dispatches,2teachers/1packet. BP calls the shared PC-named initializer
then clones it; the constructor/matrix-slot count includes both owners and is
not a count of low-level random instructions. Global RNG is unchanged.

Recipe16.3131s/meter17.8011s; sampled peak331,730,944bytes, boundary333,189,120bytes.
Initial/final native payload216/648bytes; logical copy counts and RSS include
more than retained parameters. No replay/probe/final/controller updates occur.
[Commands, budgets, source pins, failures and full result](../artifacts/runs/context-bp-qualification-20261023/RESULTS.md).

```powershell
python -m pytest -q tests/test_context_bp_qualification.py tests/test_ticket_bp_qualification.py
python -m ruff check scripts/qualify_ticket_bp_fit.py tests/test_context_bp_qualification.py tests/test_ticket_bp_qualification.py
python -m ruff format --check scripts/qualify_ticket_bp_fit.py tests/test_context_bp_qualification.py tests/test_ticket_bp_qualification.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent scripts/qualify_ticket_bp_fit.py tests/test_context_bp_qualification.py tests/test_ticket_bp_qualification.py
```

21checks/platform and scoped default+Linux type checks pass. Tests execute tiny
native fixtures; keep their measured budgets separate from the scientific case.
Source/procedure/results are self-reviewed, not independent acceptance or full
repository readiness. Existing core/native mathematics/defaults are unchanged.

Next qualify the existing simpler perceptron on the same fixed contextual task
set through the local example, with its original eight passes and a separately
frozen budget. Share pure teacher selection through the existing teacher module
to avoid a script dependency cycle; do not extend BP dose or tune CPC. Different
parameter counts, algorithms and total computation must remain explicit.
