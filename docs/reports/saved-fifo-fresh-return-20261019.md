# Saved FIFO36 learns F while retaining recent A

The existing isolated probe now accepts genuine saved invariant FIFO36 states.
It runs F16 then A16 on one disposable fork per cohort, with separate budgets and
read-only old-A monitors. Original benchmark states remain unchanged.

| Saved source | F after four updates | A during F | A return | F after A |
|---|---|---|---|---|
| Initial 0 | Exact, both cues | Unacquired: 1/9 per cue | Learns both cues | Retained |
| Mature 128 | Exact, both cues | Retained exactly | Remains exact | Retained |

Both F cross-entropy gains are 1.9459101490553137. Mature A loss, A return gain,
and F retention loss are zero. The cold cohort's A gain is acquisition. Mature A
never degraded, so this assay does **not** show recovery from mature loss. Older C
entries were retired from the disposable mature table. The earlier recurring study
observed A loss and return separately; these are different experiments.

This is single-seed known-A recall and reserved F fit on a finite count control.
It establishes no neural, circadian, heldout, gameplay or equal-cost advantage.
All prior partial neural fits and negative comparisons remain unchanged.

## Run and extend

```powershell
.venv/Scripts/python.exe -m scripts.run_minigrid_memory_probe --saved-fifo-return --describe
.venv/Scripts/python.exe -m scripts.run_minigrid_memory_probe --saved-fifo-return `
  --source-artifact artifacts/runs/minigrid-recurring-fifo-20261017/native.json `
  --output artifacts/runs/my-separately-budgeted-probe.json --max-seconds 120
```

Native runs require a prospective local protocol and supervisor. The new route
validates full snapshots, roles, clocks and charges, accepts bounded 16 MiB finite
JSON with unique keys, and publishes exclusively. The CLI output ceiling is 32 MiB;
this accepted case had a separate 16 MiB ceiling. Old E16/E8A8 routes remain intact.
Monitor labels never train; only disjoint TRAIN templates update during A return.
Supplied JSON consistency does not authenticate arbitrary external history.

The adapter owns admission and composition; the existing fresh-task helper owns
the loop; the invariant learner owns projection/counts; the CLI owns IO. Extend
explicit admission and isolation tests before adding another family.

## Checks, costs and limits

202 distinct scoped cases passed through composed checks, plus Ruff, format,
Windows/Linux source types and independent SOURCE/ACTUAL review. Two 100s checks
still timed out. The old run's 44 completed cases were retained; its unfinished
45th case passed separately. No repository-wide, clean-clone, release or UI gate
is claimed.

The case used 64 disposable updates/576 rows, 64 predictions/576 rows and four
teachers/36 steps, with no benchmark update, replay or SGD. Adapter time was
53.4211s; outer time 55.2991s; external RSS 105,111,552 bytes. Input arrays occupy
706,464 bytes; returned native projections add 2,727,936 bytes. Copies, snapshots,
guards, validation, Python and JSON add costs. Original training, the failed 180s
run, its censored eight steps and later readouts remain charged.

The case ran once and is spent. Engineering timeouts, type/reader errors and the
original root closure overrun remain recorded. Full research and release are open.

[Exact commands, failures and budgets](../../artifacts/runs/saved-fifo-fresh-return-20261019/RESULTS.md) ·
[Curves and independent arithmetic](../../artifacts/runs/saved-fifo-fresh-return-20261019/analysis.json) ·
[Independent review](../../artifacts/runs/saved-fifo-fresh-return-20261019/REVIEW.md).
