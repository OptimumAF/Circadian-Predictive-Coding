# Fitted readouts on frozen recurrent features

Run from the repository with the pinned Torch and MiniGrid optional runtime:

    python -m scripts.recurrent_readout_example --describe
    python -m scripts.recurrent_readout_example --output new-recurrent-readout.json

The output path must be new. Describe does not read data, initialize a learner,
fit or play. The example checks the native dependency before learning, then
uses bundled original A-blue/red TRAIN episodes through the existing delayed
TRAIN-only stream. Two float64 categorical LS fits use18rows total. The original
raw148/H8/K7/seed17 recurrent weights, gradients, momentum and native counters
remain frozen; readout fit counts are separate. This explicit control does not
change ordinary recurrent meanCE, weighted, BP, PC, CPC or other default routes.

Both fits finish before the existing evaluator reads two TRAIN scores and
plays two target-free A games, seed17/max16 with all seven first-argmax actions.
Prediction receives actual public frames, not labels, context names, a teacher
or an action sequence. Each session selects its cue once from the first frame.
It recomputes the actual bounded prefix through the existing recurrent reader
and applies the existing fixed readout to the last hidden row.

This simple implementation pays n(n+1)/2 native forward rows per n-step game:
45rows for9steps, at most136rows for16steps, plus the corresponding discarded
native output projections. The float64 readout is not silently cast into a
float32 native head. Prefix frame arrays occupy at most18944numeric bytes,
excluding tuple/object/native/guard/copy overhead. It retains one current state
and baseline, not an unlimited success history.

Use RecurrentReadoutLearner for explicit array TRAIN updates through a caller
that owns role/release, or train_episode for enforced complete K7 TRAIN records.
Snapshots/forks are detached; restore cannot change the frozen representation.
An unfitted selected head refuses prediction; live sessions require both cues.
Failed paid feature/fit work is terminal even if final capture becomes unavailable.
Failed live prediction is also terminal and preserves its last returned prefix.
Known returned feature costs remain charged even when a later mutation check
fails. Source/other-head/actor/RNG, prior state identities and values, inputs,
original budgets and preceding training evidence remain guarded.

Whole case120s/512MiB; TRAIN2fitattempts/60s, readout0updates/0replay/90s with one
birth across both games. Failure retains available paid prefixes and original
budgets; unknown proof stays unknown. No final/probe labels, promotion or
controller decision enters this example. No replay or native SGD is performed.
The saved JSON is a report, not a model or a resume/budget authority.

[Evidence and qualification scope](reports/recurrent-readout-policy-20261031.md),
[issue377](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/377).
Software checks demonstrate ownership/dispatch/sequence and failure behavior.
The sole separately frozen native case completed both known A tasks in9steps
each with return.49375, two TRAIN fits18rows and126native forward rows including
all recomputed prefixes. Both environments closed; source/actor/three RNG guards
passed. This is no reliability, generalization, retention/recovery, matched-compute,
CPC or release claim. All previous negative runs remain recorded.

## Isolated fresh-task and return probe

The source checkout also provides an explicit disposable diagnostic:

```powershell
python -m scripts.recurrent_readout_example --fresh-probe --describe
python -m scripts.recurrent_readout_example --fresh-probe --output new-readout-probe.json
```

`--fresh-probe` and `--recurring` are mutually exclusive; the original default
route is unchanged. The probe starts only after both original A games complete,
close and succeed. It keeps that trained reader and the original source, actor,
data, receipts, clocks and RNG unchanged. The existing generic probe owns a
separate fork, fits F blue/red once each, monitors old A without training on its
labels, then fits the two explicitly admitted old-A return templates once each.
F probe_test labels only score; final labels and promotion remain unavailable.

Both private phases have checkpoints0/1/2 and a distinct original two-update,
zero-replay60-second budget; a third original zero-update60-second budget covers
their games. All three budgets are born before F collection and never renewed.
F collection retains four complete nine-step teacher records. The existing
collector creates16 training copies; only its predetermined first two are fitted.
Complete packets stay bound outside the metadata-free raw148/K7 array bridge.
The learner's complete-episode port still accepts TRAIN only. Old296 affine
short-return and default16-update probe behavior are preserved.

After each private phase, the existing game loop reads F blue/red probe_test,
then A blue/red TRAIN, with all seven first-argmax actions and max16 steps. False
goals remain false. Scores on known teacher rows are separate from these games.
Every fit, monitor, prefix recomputation, collector copy, game and failed return
adds cost; equal update counts do not establish matched computation. The whole
first-party case keeps its existing120-second/512MiB ceiling and2MiB output cap.
Budgets created before a failed memory attachment remain in the failure receipt.

One separately frozen native case observed F2/2 and A0/2 after F fits, then A2/2
and F0/2 after explicit A-return fits. It demonstrates switching/refit recovery
with mutual forgetting, not simultaneous retention. The source stayed unchanged;
the pre-F native F baseline was unmeasured. Six fits54rows and all14closed owners
are charged. This is no reliability/generalization/CPC or deployed-benefit claim.
See [native evidence/costs](reports/recurrent-readout-probe-native-20261101.md).
See [software evidence and limitations](reports/recurrent-readout-probe-20261031.md)
and [issue394](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/394).
