# P6.7c1c2: Bound scored worker and artifact lifecycle

Implemented and verified: 2026-10-01 (ADR-0161). **C1c2, c1c and c1 correctness
are complete.** No reserved final value or full reserved scored process was
executed for these correctness gates. The subsequent
[complete actual confirmation/repeat](p67-confirmation-scored-results.md)
now completes c2/P6.7c; P6.11b remains unchecked.

## Boundaries

```text
src/app/continual_confirmation_scoring_execution.py  closed request/worker/audit links
src/infra/continual_confirmation_scoring_bindings.py current source/reference/request bytes
src/infra/continual_confirmation_scoring_worker.py   held training/scoring and observers
src/infra/continual_confirmation_scoring_artifacts.py exclusive parent IO and readback
scripts/run_p67_confirmation_scoring.py              CLI and complete-reader composition
tests/fixtures/p67_scoring_training_references.json  verified historical metadata only
```

The pure app module validates the full unchanged scientific manifest, analysis,
reference report, exact source map, command/environment and original limits.
It independently verifies every worker observation and saved audit field. It
does not inspect files, run models, measure resources or grant live authority.

The parent rederives the exact 72,050-byte reference report through both
unchanged complete training readers sequentially before launching a child.
The report SHA is `cc1c1deb4c721af5d8250f17783c9daada2501c108be91f825fa37324626b001`.
The child verifies current bytes/markers for all six original files; it does
not decode another 134-MB training graph beside live held models.

The worker composes unchanged training and optimizer observation, globally
checks complete training facts and every held A/B state, then observes actual
final release/field/prediction/example work through c1a/b. Current bindings
are checked before data, before release/evaluation and after scoring and
serialization. It retains final views and their original source-returned
arrays through serialization, repeats the complete live-state gate after
observer callbacks/restoration, and checks every retained content/model/
observation/serialized endpoint link without rereleasing fields.

All original controls remain fixed: 560 cells, 580 pairs, 1,680 predictions,
67,200 examples, 16,000 optimizer calls, 600 seconds, 5-ms sampling and an
observed absolute 512-MiB process RSS cap. RSS is a sampled observation, not a
hard allocator reservation. Stdout framing and parent publication remain
outside child RSS sampling, matching the original resource contract. Failed
or nonfinite predictions retain the already frozen explicit null policy;
contract, IO, state, source, resource and cancellation errors abort.

Exclusive ownership protects request/result/audit/failure files and refuses
all occupied outputs before complete-reader work. Intended canonical bytes
are compared with actual published bytes before success. Independent readback
requires all three successful files, no failure/claim, current complete
references and every scientific/observed/resource/audit link. V1 had a
regression: if late validation failed after audit publication and the failure
marker write also failed, a readable success could remain. V2 revokes only
its verified, still-identical completion audit in that case. Request/result
evidence and changed or foreign audit bytes are retained.

## Prospective freezes and evidence

Every old scientific/algorithm/source pin remains unchanged. The conservative
97-file closure is the union of the new CLI and historical reference-inspection
producer, preserving all 92 observer V2 pins. The binding module and new CLI
bind their own current bytes in each request to avoid self-referential constants.
The full request template binds actual command/environment before scored
fixtures; each future execution publishes its exact request before training.

| Local ignored record | SHA-256 | Bytes |
| --- | --- | ---: |
| `p67-confirmation-scored-worker-source.json` (preserved V1) | `961d85cae91010b51ff665a1929cca51af538c581bf40d3990829bf68b33c520` | 202,702 |
| `p67-confirmation-scored-worker-source-v2.json` (authoritative) | `0dc9b7fd3c67d5ba3059324645e7c731f7153795a0ded4eb7c5fb731bbb3bf9c` | 203,060 |
| `p67-confirmation-scored-binding-preflight.json` (V1) | `b5ab82a71cc5978e383e7632609a2b9e6382f66bdb30efb96871fd7554fc821f` | 984 |
| `p67-confirmation-scored-binding-preflight-v2.json` | `96342728ebbc932fd3513ba24f1123d21b35cc2e3536cd9ffe39a487783dd1b4` | 983 |
| `p67-confirmation-scored-development-worker-validation.json` | `29f29eb8d65209c17915e5efdef4b3b449e1f768a56ec9ebc0750c6ca504de58` | 104,982 |
| `p67-confirmation-scored-worker-correctness-validation.json` | `e50d63f4e33e90f19fe81f8afa3ba5731bd96b44e7097367372edb71c42d23fe` | 5,543 |

Paths above are under `artifacts/runs/`. V2 map SHA is
`e0cfd897867786271974bcc971b647084f7fcf34b28e8db8a140a2d8c3f7ff8c`.
V2 request template SHA is
`8f6febfc75d1d030219eebea9a85f9352722329ae5f81506f26e4517f63a2874`,
183,568 bytes. V2 changes only the new artifact module and its binding-module
pin; V1 is preserved. Scientific manifest stays
`76cf873e5942a661bdb76e6fd7f28fc490fc6b8001e2ae0ccc4afe87063a4223`;
analysis stays `5e33ef28862bcdf9d92fe14dd6cf6b71672a2336ffd760a1214ef04666b594b1`.
Historical source-freeze records retain their prospective incomplete flags;
the separate final correctness record records accepted gates, not science.

**203 new / 984 related tests passed in 321.40 s, zero skipped.** New cases:
83 pure execution, 19 genuine development/private kernel, 29 binding IO,
53 artifact lifecycle and 19 CLI/early-child failures. Every prior selected
state/scoring/final/reference/analysis/training/resource gate remains included.
Ruff, ten-file format, mypy 422 files and diff checks pass. Tests cover late
parameters, full selector RNG, facts, train roles, resealed final content,
held model identity, counters, endpoints, serialization, current source/
request/reference bytes, original resource limits, cancellation, output
collisions, failed failure-marker publication and complete readback drift.

Actual V2 metadata preflight called both complete historical readers under
a 180-s validation timeout in 30.4984472 s (parent 30.9592891 s), exit 0,
zero forbidden source/model/train/prediction/original-final/outer accesses.
It exactly rederives the full request template. This is metadata evidence,
with no new scientific execution or resource measurement.

A real separate child then composed the six fixed first development seeds
with both original observers, the complete development state gate and
fabricated final fields. Under the unchanged caps it observed 1,518 optimizer
calls (1,344 wake + 166 applied + 8 rejected-executed replay; BP/PC/CPC+parent
370/394/754), 12 releases/24 fields/168 predictions/6,720 examples, six exact
binding rechecks and restored hooks. Complete development fact bytes remain
`f28a4391...8edee`, 13,523,045 bytes. Child elapsed was 11.2071609 s, parent
11.8479239 s; development peak RSS 126,111,744 bytes, 4,847 samples at 5 ms.
The private JSON port is an explicit development spy; the public fixed-scope
validator rejects that partial body. These are development-only observations,
not reserved reproduction, a confirmation resource claim or a model comparison.

Full CPU/CUDA suite, clean-clone/actual CI, new reserved final execution,
large sweeps and actual statistical intervals were skipped. No new dependency,
environment change, baseline/seed/metric/cap tuning or favorable stopping.
See the log for every initial failure, repair and exact command.

## Run and safely continue

Help never opens a source or launches work:

```powershell
python -m scripts.run_p67_confirmation_scoring --help
python -m ruff check .
python -m mypy
```

For **c2**, use the unchanged existing scope and train bundles and distinct
fresh local outputs. The CLI publishes each full request before training;
the complete parent reader and bounded child enforce all fixed bindings/caps.
Do not overwrite existing evidence or tune anything after results:

```powershell
python -m scripts.run_p67_confirmation_scoring --execute --output-dir artifacts/runs/p67-confirmation-scored
python -m scripts.run_p67_confirmation_scoring --execute --output-dir artifacts/runs/p67-confirmation-scored-repeat
python -m scripts.run_p67_confirmation_scoring --read-only --output-dir artifacts/runs/p67-confirmation-scored
python -m scripts.run_p67_confirmation_scoring --read-only --output-dir artifacts/runs/p67-confirmation-scored-repeat
```

Then require exact canonical scientific result bytes and every complete
metric/cost/resource/source/artifact link. Retain all failure/null/negative/
inactive rows. If a cap stops execution, preserve failure evidence, leave c2
unchecked and record the exact next action. **P6.11b** follows verified complete
actual executions with every original seed/contrast/predeclared interval.
Any future production change needs a separate justified task and prospective
linked freeze; do not repin original algorithms or weaken accepted gates.

The two official c2 bundles and all four public commands have now passed,
with exact scientific result-byte equality. Their occupied outputs are
preserved. Current next action is P6.11b's exhaustive frozen seed/interval
report and original raw cost join; see the complete results report.
