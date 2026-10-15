# Delayed native feedback across serving transitions

This fixed engineering integration composes the existing native learners, inbox
and serving controller. It exercises a label from the original actor arriving
while a different actor version is serving, then restores the original serving
bundle without losing the candidate's completed update or renewing its quota.

## Run and check

```powershell
.venv/Scripts/python.exe -m scripts.run_coupled_delayed_feedback --describe
.venv/Scripts/python.exe -m scripts.run_coupled_delayed_feedback
.venv/Scripts/python.exe -m pytest -q tests/test_coupled_delayed_feedback.py
.venv/Scripts/python.exe -m ruff check src/adapters/coupled_delayed_feedback.py scripts/run_coupled_delayed_feedback.py tests/test_coupled_delayed_feedback.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/coupled_delayed_feedback.py scripts/run_coupled_delayed_feedback.py tests/test_coupled_delayed_feedback.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/coupled_delayed_feedback.py scripts/run_coupled_delayed_feedback.py tests/test_coupled_delayed_feedback.py
```

Repeat the type check with `--platform linux` for that platform. Description
constructs no models and performs no training, serving, filesystem or RSS work.
The full invocation is a separately budgeted local run; original development
allowances close after acceptance. There are no arbitrary seed/threshold/quota
flags, new dependencies or new environment settings. OMP/MKL/OpenBLAS use one
thread before NumPy import.

## Protocol

PC, matched neutral CPC and active CPC use seed23, mean-field raw scaling,
input dimension2, width2/two hidden layers, precision1 and gamma1.5. Two native
warm updates precede independent actor/candidate forks. The candidate splits to
width4, retaining its full origin and structural tag, but does not train before
promotion. The sole experience was produced by `actor-0` at tick2. Its original
candidate budget permits one update and one consolidation.
Warm-up uses the same fixed synthetic batch as a fixture; the chronology checks
runtime admission and makes no claim about unseen online labels or learning quality.

Separate synthetic inner guards at tick7 use the same fixed broad engineering
bounds as the cache transaction control. Guard data and identities are disjoint
from training. No final evaluation data is loaded; the invalid final-role probe
uses synthetic metadata and a target that refuses copying. The broad bounds
exercise complete publication and rollback, with no learning-benefit claim.

While `candidate-0` is serving, a label for the original experience is correctly
registered with producing version `actor-0` and arrival10. Empty drains before
10 leave native state and budget untouched. At tick10 the label updates the
original candidate exactly once, recording original producer, arrival/application
times and update number1. The already promoted actor's model and cache remain
exactly unchanged by candidate learning. The candidate's CPC count advances2->3;
the actor retains count2 and its earlier full split state.

Rollback then restores the exact prior actor model/version/configuration/
metadata/cache/tick with generation2. It keeps the candidate's updated state,
applied identity and spent allowance. Repeated drains do no additional native
work. Original and promoted actors each have a real cold prediction followed
by cache hits; full predictions and cache entries are retained.

Six refusal probes cover role, producing-version mismatch, impossible chronology,
using the current actor version instead of the actual producer, reused event ID
on another sample, and duplicate applied-label identity. Each target's copy hook
must stay uncalled. Why distinguish cursor metadata: runtime registration spends
one revision before delegating to inbox validation. Refusals preserve complete
native/serving/inbox state and update budget while recording that revision+1.

## Structure and boundaries

```text
src/adapters/coupled_delayed_feedback.py  fixed orchestration and observations
scripts/run_coupled_delayed_feedback.py  describe/run and bounded JSON output
tests/test_coupled_delayed_feedback.py   independent pure/native controls
```

There is no new feedback protocol, native objective or persistent authority. The
trusted local demo reads the existing inbox cursor for evidence; no public API is
expanded for observation alone. Its report retains21 complete finite observations
per family, including native states, inbox identities/payloads, serving bundles,
cursor revisions, update counts and refusal messages. PC/neutral raw parameters
are checked exactly at corresponding boundaries; their full CPC metadata differs.
Reported object identities are process-local checks of the original budget
session, not persistent or transferable authority.

Body wall time is bounded at60seconds, output at2MiB including platform newline
bytes, and observed RSS at512MiB. RSS is sampled at boundaries; it is not a hard,
periodic or indefinite memory guarantee. Full durable recovery and long-running
service criteria remain open. See ADR0282 and the scope/results/review in
`artifacts/runs/coupled-delayed-feedback-20261015/`.

For a new scenario, declare producing-version and arrival semantics before any
execution, preserve identity indexes and original quota, and add focused tests
without changing this frozen protocol or using final data for promotion.
