# Public-history acquisition: negative result

One frozen exploratory seed17 case completed using the existing paired learner,
delayed stream and native MiniGrid readout. Public first-frame inputs added1184
parameters (1319→2503), with copied raw initial predictions verified before any
training. Both conditions used64updates/128episode exposures/1152rows.

| Measure | Public history | Accepted raw paired control |
|---|---:|---:|
| CE improvement | 1.608734 | 1.655521 |
| Native success count change | +1 | +1 |
| Mean return change | −.253125 | −.253125 |
| Joint acquisition | false | false |

History-minus-raw CE improvement is−.046787. History matched17/18 teacher actions:
blue8/9 and native wrong-key failure/return−1; red9/9 and goal/return+.49375.
Endpoint action sequences matched the retained raw control. Mean return declined
despite gaining one goal. Every negative/cue result is preserved.

The simpler raw control has better CE improvement and fewer parameters in this
case. This establishes neither general superiority/equivalence, why learning
failed, lost plasticity, retention/recovery nor a circadian benefit. Capacity and
public input access differ. Historical raw endpoints are accepted separately
from their still-incomplete full-five-readout protocol; it was not relabeled.

181focused checks/scoped statics and independent SOURCE/ACTUAL review passed.
Native work:64SGD1152rows,64backwards1152rows,184forwards1272rows,48physics/
4episodes;20.1185parent seconds/peak414232576B RSS including setup/output.
Initial-law diagnostics cost4forwards36rows. No teacher, final, probe or replay.
Permanent MiniGrid training now1426updates16230rows; synthetic units separate.

[Frozen protocol, commands, failed typing attempt and full costs](../../artifacts/runs/r21-history-acquisition-20261017/RESULTS.md).

Next pursue the roadmap's recurring sensor/classifier demo instead of another
MiniGrid dose/architecture variant. It needs genuine feature-only prediction
before label arrival using the current shadow candidate; the existing arrived
observer is retrospective, and the deployed actor remains frozen. Qualify a
small label-preserving synthetic sensor stream with ordinary BP before a frozen
comparative protocol. FullR4.6, joint MiniGrid acquisition and old/new/return
scientific requirements remain open.
