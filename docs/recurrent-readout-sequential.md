# Sequential learning with frozen recurrent readouts

From the physical repository, using the pinned optional Torch/MiniGrid runtime:

    python -m scripts.recurrent_readout_example --recurring --describe
    python -m scripts.recurrent_readout_example --recurring --output new-sequential.json

Describe performs no source read, fit or game. Output must be a new path. The
ordinary command without --recurring retains the original two-fit A example.

The same module command works from an installed package; it does not need the
repository or its artifact directory. Install the local wheel and its existing
`native` and `minigrid` optional dependencies in your chosen environment. For an
offline installation, those dependencies must already be available. The wheel
includes the original TRAIN demonstrations and this guide under
`share/doc/circadian-predictive-coding/recurrent-readout-sequential.md`.
Use a new writable output path outside the package. A local development wheel
is a snapshot of its recorded source, not a clean-clone or release qualification.

The selected recipe uses the six bundled original TRAIN demonstrations, in
order A-blue/red, B-blue/red, C-blue/red, then A-blue/red as new chronological
arrivals. All episodes have nine steps; returning A receives distinct arrival
IDs. The existing delayed TRAIN-only stream makes eight categorical LS fits,
72row exposures (54unique TRAIN rows plus18repeated A exposures), with labeldelay1.
This is not independent new A data. Each fit replaces only the selected cue readout using
that current episode; there is no replay or retained-data fit. Frozen original
raw148/H8/K7/seed17 recurrent weights/gradients/momentum stay at0updates/0rows.
LSfit clocks are separate from native SGD. Ordinary meanCE/weighted controls and
their original128-update schedules are unchanged.

After2/4/6/8committed fits, a private reader uses the current candidate state
in the existing game loop. Fixed score/game rosters contain only tasks whose
labels have arrived: A, then A/B, then A/B/C, then A/B/C. Each context has blue
and red games, seed17/max16, all seven first-argmax actions from public frames.
No future-task scores, final/probe labels, teacher, promotion or outcome-based
selection. Native gameplay and same-row TRAIN exactness remain separate metrics.

The output records current-context acquisition at its first trained point;
old-task retention while another context is current; and A success on return.
Here acquisition means observed competence after that context's first fit pair;
there is no pre-fit B/C game control, so it is not an estimated learning gain.
`observed_recovery` on return requires actual initial A success, A failure at the
last pre-return point6, and success at8 for that cue. `observed_recovery_by_return`
allows an earlier observed loss at4or6; recovery already seen at6 is distinguished
from recovery on return. Neither is a causal attribution. A return success alone proves neither
recovery nor reliable adaptation. Known TRAIN geometry/seed observations do not
establish heldout transfer, generalization, superiority, matched compute or CPC
benefit. All tasks/points/cues, including failures, remain in the report.

One original8-fit60s TRAIN budget and one0-update/0-replay90s readout budget
cover the whole interleaved recipe, within120s/512MiB total. Readout starts once
after the first fit pair and is never renewed. Prior receipt values and all
containers/values of the four bounded readout records are checked before every
next fit and readout. A changed or equal-replaced success record refuses further
work; failed/unknown preservation stays incomplete. No unlimited success chain.

Inference retains the existing bounded prefix implementation: an n-step game
pays n(n+1)/2 native recurrent rows plus discarded original output projections,
then one float64 readout per actual action. This is not constant-cost recurrence.
Native SGD, replay, probe and teacher costs are0; fit/feature/readout/physics and
guard/copy/time/RSS overhead must be reported by each actual qualification.
Saved JSON is evidence, not a model, resume capability or budget authority.

The current caller reuses the existing paid-report guard to check every value
and container identity, with explicit outer root/list pins and unchanged check
frequency. A small fixed scalar comparison measured lower repeated guard cost;
this does not establish end-to-end latency. Afterfit8, the original TRAIN stream
is finalized and its report sealed before final readout-only games, without
renewing clocks or charges. A later game failure keeps TRAINcompletion separate.
See [guard-cost evidence](reports/recurrent-readout-guard-cost-20261031.md).

Implementation/validation and any executed result are tracked in
[issue378](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/378).
The earlier [known-A result](recurrent-readout-policy.md) remains accepted;
sequential functionality needs its own scoped gates/native qualification.

The sole Windows qualification exhausted its unchanged90s readout budget.
All8fits72exposures completed. Current A/B/C each solved both cues at its first
trained point; afterB, A failed both cues, and afterC, A/B failed all four games.
Both return-A games succeeded, followed by two B failures; C-blue then stopped
after1step/2predictions and C-red was unrun. Sixteen games completed, all17created
owners closed; overall preservation/outcomes remain unknown/null. This is a
FAILED/SPENT full protocol, not complete recovery or retention acceptance.
Final Linux phase-gate verification also remains open after runtime startup
failures. See the [full evidence](reports/recurrent-readout-sequential-20261031.md).

The corrected caller has since completed all18games under the same limits.
It solved each current context and regained A on return; all ten old-task
retention checks failed. All18owners closed, source/actor/three RNG checks passed,
and every previously recorded prefix group matched exactly. This qualifies
the current source command on Windows, while the original failed run above
remains FAILED/SPENT. Observed reacquisition is limited to these known TRAIN
tasks and repeated A data; it is not reliable or unseen-task recovery.
[Corrected-caller evidence](reports/recurrent-readout-caller-native-20261031.md).
