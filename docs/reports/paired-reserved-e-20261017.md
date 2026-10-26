# Paired recurrent learner: reserved-E probe

One frozen seed17 case completed using the existing paired stream and isolated
probe observer. Eight probe updates/144rows were applied separately to copies
of the original0 and64A-training checkpoints. Source/deployed learners stayed
unchanged. No learner live rollout or final evaluation was run.

| Measure | Cold checkpoint | After64A updates |
|---|---:|---:|
| Initial E mean CE | 1.947311 | .360654 |
| E mean CE after8probe updates | 1.411635 | .289687 |
| CE improvement | +.535676 | +.070967 |
| Action accuracy, initial→endpoint | 11.11%→55.56% | 94.44%→94.44% |
| Exact teacher sequences, initial→endpoint | 0/2→0/2 | 1/2→1/2 |

Warm-minus-cold CE improvement is−.464709; accuracy improvement difference−4/9.
Warm absolute fit is better throughout, so the smaller improvement does not
establish plasticity loss. The warm model still lowers E CE but learns no
additional correct action under this dose. Neither endpoint fits both sequences.

E is a reserved context from the same task family, with separate training and
test collections in the same deterministic domain. This does not establish
generalization, useful live skill, a causal effect of pairing, or a circadian
advantage. Probe step/row dose matches; history, momentum and total compute differ.
Old joint-A acquisition failed, so full retention/recovery remains inconclusive.

Costs:80physicalSGD1440rows including64A updates and16probe updates;176forward
calls1584rows/80backward calls1440rows;4teacher episodes36physics;12.867seconds
parent time, later observed RSS370802688bytes. Teacher returns are not learner
returns. All intermediate points, negative differences and prior failed cases
are preserved. Permanent MiniGrid1362updates15078rows; synthetic units separate.

26focused checks/scoped statics and independent source/qualification/GO/native
review support this single case. Earlier tooling failures remain recorded.
[Frozen protocol, commands, complete costs and limits](../../artifacts/runs/r21-paired-fresh-task-20261017/RESULTS.md).

Next address joint acquisition with an explicit public-initial-frame history
control using existing encoder/learner/stream ports. Preserve the raw baseline
and require unchanged initial predictions before training. Extra capacity and
compute must be reported; no new native comparison is admitted by this result.
The encoder already exists; native recurrent policy currently rejects296, so
explicit bounded admission and a cold zero-added-input initializer are the next
code prerequisite. Existing148-only experiment validators remain unchanged.
