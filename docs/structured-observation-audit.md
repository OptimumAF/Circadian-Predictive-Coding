# V14 observed-record projection (P5.2a)

The fixed v14 study stores two raw seed-level JSON files in a completed
[P5.1 run bundle](versioned-run-manifest.md). The train-only file has six
seed/arm rows, each with 24 ordered wake opportunities. Its opportunity
records contain train-role hash, replay retention and selection, full
typed sleep event, actual matched replay work, width, parameter count,
and cumulative split/prune counts. It also stores twelve inner-guard
decisions and 516 role-access audit entries. The scored file contains
six seed/arm outcomes with three method rows each, final-role hashes
and IDs, phase replay retention, exposure, capacity, accuracy/BCE,
forgetting, balanced score, and work.

**Observed gap:** the v14 runner calls
`base._train_named_model_epoch`, which discards the learning methods'
`train_epoch` return values. Neither source JSON file contains genuine
per-epoch loss or energy values. Every projected wake row says
`wake_metrics_status: unavailable_not_recorded`; there is no fabricated
loss/energy field. Guard accuracy and final-test metrics retain their
own roles and are not presented as wake metrics. These historical
values cannot be recovered from saved v14 outputs.
The separate [measured wake workflow](measured-wake-observations.md)
now provides those diagnostics on newly run studies. This P5.2a view
continues to describe the original raw v14 payload faithfully.

## Local workflow

From the repository root after creating a completed P5.1 bundle:

```powershell
.\.venv\Scripts\python.exe -m scripts.project_v14_observations --run artifacts/runs/p51-v14-local
.\.venv\Scripts\python.exe -m scripts.project_v14_observations --verify-run artifacts/runs/p51-v14-local
```

The command writes one exclusive `observations-v1/` directory beside
the raw files. Its `projection-manifest.json` binds the source run ID,
source manifest digest, both raw file digests, every output digest,
and its record count. `verify-run` first verifies the completed P5.1
bundle, then recomputes every projection byte from the source and
rejects missing, changed, extra, reordered, or rehashed forged output.
The source manifest is provenance metadata rather than a signed
attestation. P5.3a now stages and atomically publishes the complete
projection directory; interrupted stages remain hidden and fail
public-path verification. Checked recovery remains P5.3b.

| Derived file | Data rows | Origin |
|---|---:|---|
| `wake-epochs.jsonl` | 144 | Phase/epoch, train-role hash, explicit missing-metric status |
| `sleep-events.jsonl` | 144 | Complete typed event, including 12 accepted and 132 skipped |
| `topology.jsonl` | 144 | Width, parameter count, cumulative changes, stable-ID change record |
| `replay.jsonl` | 144 | Retention/order, offered IDs, per-method applied work, event replay work |
| `validation.jsonl` | 12 | Inner-guard decision and scores at its original role |
| `role-access.jsonl` | 516 | Unscored train-only role audit; zero final-test accesses |
| `final-results.jsonl` | 18 | Every full method result plus trial final roles, replay, capacity, sleep facts |
| `summary.csv` | 18 | Flat method-level final metrics and work derived only from final rows |

The original `training.json` and `outcomes.json` remain the raw seed-level
records and are never rewritten by the projector. The CSV is a view of
the final JSONL, with no seed aggregation, seed selection, or arm choice.
Per-epoch replay rows show offered IDs and committed work; the saved
final method rows carry circadian observed/exposed IDs. The v14 source
does not contain a per-epoch exposure snapshot, so none is inferred.
All streams use fixed order and UTF-8 LF line endings. Two independent
P5.1 bundles (`p51-v14-schema-c/d`) produced identical hashes for all
eight derived data files; only their run-bound projection manifests
differ. The fixed source training and scored payload hashes remain
`174ee7941c0b1e2489783f43b0b481db11f402c4ea55001888c7998cfb28b324`
and `ea11fc7cc0ac8113eec2fc5512bb28044b99d0885627813c92aad80cf0f2501f`.

Why this: exposing recorded data through narrow streams supports audit
and later reports while preserving the original v14 evaluation and its
negative or mixed findings. The next metric-capture work must describe
metric timing and method semantics before adding new rows.
