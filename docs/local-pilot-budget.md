# First local pilot resource preflight

Budget ID: `local_small_head_pilot_budget_v1`. This new planning interface is
independent of model objectives. It does not launch training, download weights,
select evaluation data, change permissions, or enforce measured runtime use.
Existing scientific protocols and exhausted budgets retain their original scope.

| Planned resource | Default and maximum |
| --- | ---: |
| Wall duration | 60 seconds |
| Training updates | 512 |
| Simulated environment steps | 1,024 |
| CPU threads | 1 |
| Process memory | 256 MiB |
| Replay storage | 1 MiB |
| Downloaded model bytes | 0 |
| Output and temporary storage | 32 MiB |
| GPU memory | 0 |

Only local execution, simulated environments and CPU devices are accepted.
Every excess is reported together. Counts/bytes must be nonnegative integers;
threads/process memory must be positive; duration must be positive and finite.
These fixed caps cannot be overridden through request arguments.

Why this: a small NumPy decision head needs neither a pretrained model nor an
external dataset. The generous host resources do not justify a large first run.
On 2026-10-06 the host exposed20 logical CPUs,68,399,599,616 RAM bytes and an
NVIDIA RTX3080 with10,240MiB VRAM. WSL exposed20 CPUs and32,711,224KiB RAM.
No GPU computation is used by the preflight or its validation.

From the repository root:

```powershell
python -m scripts.run_local_pilot_preflight
python -m scripts.run_local_pilot_preflight --training-updates 513
python -m pytest -q tests/test_local_pilot_budget.py tests/test_local_pilot_cli.py
python -m ruff check src/core/local_pilot_budget.py src/adapters/local_pilot_cli.py scripts/run_local_pilot_preflight.py tests/test_local_pilot_budget.py tests/test_local_pilot_cli.py
python -m ruff format --check src/core/local_pilot_budget.py src/adapters/local_pilot_cli.py scripts/run_local_pilot_preflight.py tests/test_local_pilot_budget.py tests/test_local_pilot_cli.py
python -m mypy --platform linux src tests scripts
python -m mypy --platform win32 src tests scripts
```

The first command emits accepted JSON and exits0. The second emits rejected JSON
and exits2. Neither creates a run artifact. Byte flags use bytes, not MiB.
Argparse rejects malformed flags; validly parsed resource refusals contain JSON.
No environment variables or dependencies are added.

`src/core/local_pilot_budget.py` owns immutable request values, validation and
typed excess errors. `src/adapters/local_pilot_cli.py` translates flags to those
values and presents JSON/status; the script delegates to the adapter. Dependency
direction is script → adapter → core; core uses only the standard library.
There is no app/infra execution path in this increment.

Extend safely by versioning an evidence-backed policy and its boundary tests,
then composing a separate measured execution port in R3. A planned request does
not prove wall/RSS/storage enforcement. Do not connect a learned controller until
the executor owns non-overridable limits and the separate promotion/safety gates.
Keep a native ranking/contrastive/world-model/diffusion objective in its adapter;
an explicit PC-objective experiment must carry its own protocol/version.
