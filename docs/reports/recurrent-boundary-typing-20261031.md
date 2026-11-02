# Recurrent learner/readout boundary typing

R0.3-TYPE.RR / [issue393](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/393).
The targeted typing repair is implemented and its existing scripted behavior
checks pass. This closes the two-file quality defect, not the repository goal.

Initial scoped mypy confirmed exactly nine historical errors: three ambiguous
optional-keyword expansions in `src/adapters/return_policy_evaluation.py`, and
six nullable-value/callback/work-inference errors in the inherited untracked
`tests/test_recurrent_readout.py`. Initial exit1/6.4504273s is retained.

A private TypedDict names the sole optional `declared_layouts` field. Why this:
the old route must omit that keyword entirely, while the four-phase route must
pass its exact declared tuple. Passing None unconditionally would change the
existing call contract. No roles, defaults, guards, scoring, learner math or
budgets change. Tests add non-None prerequisites, separate callback names and
the existing heterogeneous fake-work annotation; no ignore/cast/config change.

Executed from physical D:\Dev\Repositories\Circadian with TEMP/TMP/TMPDIR at
D:\Dev\_Temp. Before copies and exact delta are retained in
`artifacts/runs/recurrent-boundary-typing-20261031/`.

```powershell
$files = @('src/adapters/return_policy_evaluation.py', 'tests/test_recurrent_readout.py')
.venv/Scripts/python.exe -B -m mypy --follow-imports=silent --cache-dir artifacts/runs/recurrent-boundary-typing-20261031/mypy-cache @files
.venv/Scripts/python.exe -B -m ruff format --check @files
.venv/Scripts/python.exe -B -m ruff check @files
./artifacts/runs/recurrent-boundary-typing-20261031/run-check.ps1 -Attempt final
```

Final two-file format/lint/mypy all0/.930991s. The reused bounded pytest helper
also passes Ruff format/lint and separate mypy0/.5244692s; three total mypy
invocations are spent, no new retry. Types use `--follow-imports=silent`, not
repository-wide checking.

One combined invocation ran the unchanged selections
`tests/test_recurrent_readout.py` and `tests/test_four_phase_policy_boundary.py`:
65 successful progress outcomes, exit0, no failures/errors/skips. Configured -q
plus runner -q suppresses the textual summary. Body2.8128467s, peak82137088B;
PID36216 host2.9962663s/0/no expiry; blocked runtime imports[]. The sampler starts
before pytest/NumPy imports. Literal arrays/canonical snapshot containers,
scripted feature/fit/scorer ports and fake environments are not native learning.

Existing tests check exact old keyword omission, four-phase final_test roles,
five checkpoints/40 slots, refusal before final collection on incomplete
authority, paid partial collections/cached failures, readout ownership/mutation,
lost returns, source preservation and bounded sessions. No new learner,
solver/activation/model prediction, SGD, physics or scientific raw input ran.

Lead source/procedure/result self-review is distinct from the finite peer's
scoped typing-delta/regression ACCEPT. One read/.380196s; full actual patch and
relevant assertions/receipts were visible. Output attempted6651 tokens at6000,
omitting651 tokens of the unchanged evaluation body. No further read/renewal;
do not claim complete independent inspection of that body. The peer authored
some legacy #377 tests: review is independent of this typing edit, not independent
test design. `REVIEW.md` and `reviewer-timings.json` retain these qualifications;
the exception ENDED. No numerical replication or repository-wide acceptance.

Skipped full suite, unrelated type errors, Linux, clean clone, wheel/install,
native games/models, browser/device, GPU/remote CI and release. Historical
#390X FAILED/SPENT, #391 PARTIAL, #392 authentication/browser gaps and every
negative result remain unchanged. No dependency/download/publication/deletion.

Orientation-only missing-path reads were not successful verification: the old
downloaded multi-agent prompt is absent and superseded by actual AGENT_PROMPT;
the guessed return-policy test name, guessed learning-pilot entry and a Windows
wildcard path failed. Explicit current-file discovery replaced those guesses.
No execution allowance or source was changed by these read errors.

Critical path now returns to the learner: [planned #394](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/394)
R2.1-RNN.RO.P will reuse the successful #377 A readout control and existing
generic fresh-task probe to ask about F acquisition, old-A retention and A-return
recoverability/cost on an isolated fork. Current readout owner/first-party script
has no such integration; existing joint probe uses a different128-SGD owner.
The generic short-return contract is raw296-only, whereas this readout is raw148.
Exact next action: inspect readout train_batch/fork/copy and existing short-return
validation/helper, then freeze an explicit two-cue raw148 probe/return contract
before implementation/tests. Preserve old296/default16-update behavior. No new
native allowance, scientific comparison, reliability or CPC benefit is created.

393 canonical6554646047/CLOSED4criteria; parent18receipt6554648751/21receipt
6554648897/23receipt6554649065. SESSION-END saved, stage17651482B<64MiB,
cache17600741B<32MiB, peerENDED/source unchanged since finalchecks.
Next394PLANNED/no allowance: explicit148isolated readout probe/return contract
inspection/freeze above. FullgoalACTIVE;392auth/browser stillOPEN.
