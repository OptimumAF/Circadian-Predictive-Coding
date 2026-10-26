# Public-cue disposable probe games

Use the existing `ProbeRollouts(source_updates=2, public_cue=True)` with the
generic probe's `endpoint_observer=games.observe_qualified`. This explicit
mode plays two F `probe_test` games after the paid16-update fresh-task phase,
then two A `train` games after the paid16-update return phase. Predictions
receive actual observations only, with fresh recurrent sessions and all seven
actions available. There is no teacher fallback, fitting or promotion here.

The supplied raw148/H8/K7/seed17/.03/.9 learner must be the actual disposable
owner. Its full state must equal the supplied phase checkpoint, including
gradients, optimizer and failure fields. Warm source2 means one genuine
nine-row update per public-cue head; native counts must be18afterF and34after
return, heads9/17respectively and nine rows per native update. Logical probe
and return budgets remain separately16each. Other origins/configurations are
refused; the old count/H32 and benchmark interfaces keep their old rules.

Why this: generic isolated learning already supports this learner. The missing
piece was explicit native game admission at its paid phase endpoints, rather
than a second probe or game framework. `minigrid_policy_evaluation` supplies
this strict binding; the existing shared kernel handles sessions, first-argmax
actions, cleanup, full-state checks and recorded costs.

Each endpoint has its own original40-second zero-training/zero-replay budget.
The caller's outer source/actor/benchmark and whole-case guards still apply.
Qualified synchronous reads are checked before and after clocks. Prior paid
records are protected; a failed first endpoint refuses later public-cue work.
Valid unsuccessful games remain complete observations. Operational failure
stops later work and retains the partial prefix. Source/actor updates never
return from the disposable probe. Final D is unavailable throughout.

Thirty-five focused scripted cases passed perplatform (Windows34+1/Linux35),
including real generic-probe/observer/return composition with fake physics and
the old count/H32 routes. These fixtures are not native game success.

The ONE native observation is incomplete and spent. Warm2 and F16fits completed;
the first endpoint exceeded its40-second limit, so Areturn was correctly not
admitted. Fteacher accuracy reached5/9by predicting forward for every row—the
majority reference. Fblue failed at16steps; Fred outcome is unknown after partial
execution. Original source/actor/budgets and NumPy/TorchRNG were preserved; the
failed aggregate endpoint proof remains unverified. No retention, recovery,
generalization or CPC benefit is established.

See [results](reports/public-cue-probe-games-20261025.md),
[136](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/136) and the
fixed protocol/raw receipts in `artifacts/runs/public-cue-probe-games-20261025/`.
Do not renew the budget, rerun or reconstruct that cohort. Measured redundant
snapshot-validation optimizer setup is the next engineering task137; it cannot
complete the unrun games or justify a whole-game speedup claim without evidence.
