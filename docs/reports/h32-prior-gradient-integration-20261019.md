# H32 joint TRAIN prior-gradient integration

The existing learner runner can now diagnose original TRAIN cue-pair gradients
through the explicit H32 joint-catalogue route, for BP, PC, neutral CPC and active
CPC. It reuses the existing detached prior mean-CE diagnostic and fixed five
checkpoints. No objective, optimizer, seed, metric or deployed learner changes.

Preview the recipe without running a learner:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method pc --distinct-probe --old-task-readouts --return-to-a --joint-catalogue --wide-neural --prior-gradient-geometry --describe
```

Each method plans15diagnostic pairs over six original TRAIN episodes at
0/32/64/96/128. Additional work:30forward/backward calls,270rows for each,
30projections/639360arraybytes,75geometry reductions,zero diagnostic optimizer
steps or published updates. Python, native state, validation/copies and elapsed
time are additional. Benchmark/probe/return work remains separately charged.

The new combination excludes fixed64 controls, PC-update diagnostics, rehearsal,
extra F exposure and games. Existing H8/default routes and recurrent gradients
retain their behavior. Full H32 state and54-row clocks are checked. Original TRAIN
packets alone supply diagnostic labels; final/probe data is unavailable here.
The reader's completed results and costs remain sealed through later callbacks;
altered values, numeric types or equal-valued container replacements are refused.

## Verified scope and execution failure

Corrected259scripted checks pass, covering four methods × five points, public CLI,
roles/clocks, paid partial work and corruption refusals. Four bounded native H32
initial-state fixtures pass:4pairs/72rows per forward/backward/projection operation,
zero training/SGD, complete state and Python/NumPy/Torch RNG preserved. Scoped
statics and independent SOURCE/ACTUAL review accept this implementation scope.

The first check mistakenly included an old file's native learning tests. It passed
248cases but violated this increment's zero-training execution scope:196updates/
7524rows and102predictions/918rows are known. The repeat was interrupted on review;
its additional native prefix is unknown. Receipts and the original temporary BP
report are retained. **That original scope gate remains UNMET.** Corrective explicit
test selection and the separately authorized four gradient fixtures do not erase it.
Four pytest xunit2 property warnings remain; the observed properties were present.

This is implementation/engineering evidence, not an acquired-task diagnostic study,
causal interference, actual PC update geometry, circadian benefit or release proof.
No native comparative protocol was run. Full repository and scientific parents stay open.
[Exact argv, outcomes, resources and incident](../../artifacts/runs/h32-prior-gradient-integration-20261019/RESULTS.md).

Next freeze and review a small BP/PC H32 study, with the original recipes and all
five TRAIN points, before any new comparative execution. Preserve every negative
or incomplete run; do not tune either learner to force an advantage.
