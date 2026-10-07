"""Project a verified v14 summary into static plots and dashboard HTML.

Inputs are the P5.6a summary object after report verification. Outputs are
deterministic standalone bytes. This module does not read files, train,
rank cells, infer missing attempts, or modify the historical dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import isclose, isfinite
from typing import Any

from src.app.continual_trigger_replay_outcomes import TRIGGER_REPLAY_OUTCOMES_PROTOCOL
from src.app.v14_artifact_report import BENCHMARK_TRACK, METRICS, REPORT_SCHEMA_ID
from src.app.v14_report_plot import render_metric_plot


DASHBOARD_ID = "v14_verified_dashboard_v1"
_METRIC_VIEW = (
    ("balanced_score", "Balanced score", "balanced-score.png"),
    ("signed_forgetting", "Signed forgetting", "signed-forgetting.png"),
    ("a_after_b_accuracy", "A after B accuracy", "a-after-b-accuracy.png"),
    ("b_after_b_accuracy", "B after B accuracy", "b-after-b-accuracy.png"),
)


@dataclass(frozen=True)
class V14DashboardProjection:
    """Static dashboard and standalone plots derived from one table."""

    files: dict[str, bytes]


def _validate_summary(summary: dict[str, Any]) -> None:
    if (
        summary.get("schema_id") != REPORT_SCHEMA_ID
        or summary.get("benchmark_track") != BENCHMARK_TRACK
        or summary.get("protocol_id") != TRIGGER_REPLAY_OUTCOMES_PROTOCOL
        or summary.get("interpretation_scope") != "descriptive_only_no_causal_attribution"
        or summary.get("failure_scope") != "published_completed_bundle_only"
        or summary.get("external_attempt_failures") != "not_recorded_in_bundle"
        or summary.get("failed_cells_in_bundle") != 0
    ):
        raise ValueError("v14 dashboard requires the fixed descriptive report identity")
    seeds, rows = summary.get("seeds"), summary.get("rows")
    if (
        type(seeds) is not list
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or type(rows) is not list
        or len(rows) != 9
        or summary.get("seed_count") != len(seeds)
        or summary.get("expected_cells") != len(seeds) * len(rows)
        or summary.get("completed_cells") != summary.get("expected_cells")
    ):
        raise ValueError("v14 dashboard report seed or cell grid is incomplete")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if type(row) is not dict or row.get("seed_count") != len(seeds):
            raise ValueError("v14 dashboard row seed count differs")
        arm, method = row.get("arm"), row.get("method")
        if type(arm) is not str or not arm or type(method) is not str or not method:
            raise ValueError("v14 dashboard rows require arm/method identities")
        key = (arm, method)
        if key in seen:
            raise ValueError("v14 dashboard rows require distinct arm/method identities")
        seen.add(key)
        for metric in METRICS:
            stats = row.get(metric)
            if type(stats) is not dict or set(stats) != {"mean", "min", "max", "range"}:
                raise ValueError(f"v14 dashboard metric {metric} is incomplete")
            if any(
                type(value) not in {int, float} or not isfinite(value) for value in stats.values()
            ):
                raise ValueError(f"v14 dashboard metric {metric} must be finite")
            if (
                stats["min"] > stats["mean"]
                or stats["mean"] > stats["max"]
                or not isclose(stats["range"], stats["max"] - stats["min"], abs_tol=1e-12)
            ):
                raise ValueError(f"v14 dashboard metric {metric} range is inconsistent")
    source = summary.get("source")
    if type(source) is not dict or (
        source.get("commit_sha") is None and not source.get("unavailable_reason")
    ):
        raise ValueError("v14 dashboard source commit or unavailable reason is missing")


def _stat_cell(stats: dict[str, float]) -> str:
    return (
        f"<strong>{stats['mean']:.4f}</strong>"
        f"<small>min {stats['min']:.4f} · max {stats['max']:.4f}"
        f" · range {stats['range']:.4f}</small>"
    )


def _render_table(rows: list[dict[str, Any]]) -> str:
    header = "".join(f'<th scope="col">{escape(label)}</th>' for _, label, _ in _METRIC_VIEW)
    body = []
    for row in rows:
        cells = "".join(f"<td>{_stat_cell(row[metric])}</td>" for metric, _, _ in _METRIC_VIEW)
        body.append(
            '<tr class="data-row">'
            f'<th scope="row">{escape(row["arm"])}<small>{escape(row["method"])}</small></th>'
            f"{cells}</tr>"
        )
    return (
        '<div class="table-scroll"><table><thead><tr><th scope="col">Arm / method</th>'
        f"{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _render_page(summary: dict[str, Any]) -> bytes:
    source = summary["source"]
    commit = source["commit_sha"] or f"Unavailable: {source['unavailable_reason']}"
    dirty = (
        "workspace status unavailable"
        if source["dirty"] is None
        else "dirty workspace snapshot"
        if source["dirty"]
        else "clean workspace snapshot"
    )
    seeds = ", ".join(str(seed) for seed in summary["seeds"])
    figures = "".join(
        '<figure class="plot">'
        f'<img src="{filename}" alt="{escape(label)}: means and observed min-max ranges '
        f'across {summary["seed_count"]} seeds for all nine arm and method cells">'
        f"<figcaption>{escape(label)} <code>{escape(metric)}</code></figcaption>"
        "</figure>"
        for metric, label, filename in _METRIC_VIEW
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Verified v14 outcome dashboard · {escape(summary["source_run_id"])}</title>
<style>
:root{{--ink:#172f40;--muted:#526877;--paper:#f4f5f1;--panel:#fff;--line:#d9e0df;--accent:#b55d33}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,"Segoe UI",sans-serif}}
main{{max-width:1320px;margin:auto;padding:28px 22px 64px}} a{{color:#285b78}} .eyebrow{{font-size:.75rem;letter-spacing:.16em;text-transform:uppercase;font-weight:700;color:var(--accent)}}
h1{{font-size:clamp(2rem,4vw,3.6rem);line-height:1.08;margin:.2em 0}} h2{{font-size:1.45rem;margin:0 0 .5rem}} p{{max-width:80ch}}
.hero{{border-bottom:2px solid var(--ink);padding:24px 0 26px}} .intro{{font-size:1.1rem;color:var(--muted)}}
.facts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:12px;margin:24px 0}} .fact{{background:var(--panel);border:1px solid var(--line);padding:18px}}
.fact b{{display:block;font-size:1.55rem;line-height:1.2}} .fact span{{font-size:.82rem;color:var(--muted)}}
section{{margin-top:36px}} .meta{{background:#e8eeed;padding:20px;border-left:4px solid var(--accent)}}
dl{{display:grid;grid-template-columns:max-content 1fr;gap:7px 20px;margin:0}} dt{{font-weight:700}} dd{{margin:0;overflow-wrap:anywhere}} code{{font-size:.88em}}
.plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}} .plot{{margin:0;background:var(--panel);border:1px solid var(--line);padding:12px}}
.plot img{{display:block;width:100%;height:auto}} figcaption{{padding:8px 4px 2px;color:var(--muted)}}
.table-scroll{{overflow-x:auto;background:var(--panel);border:1px solid var(--line)}} table{{border-collapse:collapse;width:100%;min-width:1000px}}
th,td{{padding:12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} thead{{background:#e8eeed}} tbody tr:nth-child(even){{background:#fafbf9}}
td strong{{display:block;font-variant-numeric:tabular-nums}} td small,th small{{display:block;color:var(--muted);font-weight:400;white-space:nowrap}}
.note{{color:var(--muted);font-size:.93rem}} footer{{margin-top:38px;border-top:1px solid var(--line);padding-top:18px;color:var(--muted)}}
@media(max-width:620px){{main{{padding:18px 12px 42px}}.plots{{display:block}}.plot{{margin-bottom:14px}}dl{{display:block}}dt{{margin-top:10px}}}}
</style>
</head>
<body><main>
<header class="hero"><div class="eyebrow">Source verified · fixed NumPy v14</div>
<h1>Outcome atlas</h1><p class="intro">Every configured arm and method from the completed synthetic continual study. Descriptive across the recorded learning rules; no cell was selected for this view.</p></header>
<div class="facts" aria-label="Run facts">
<div class="fact"><b>{summary["seed_count"]}</b><span>seeds: {escape(seeds)}</span></div>
<div class="fact"><b>{summary["completed_cells"]} of {summary["expected_cells"]}</b><span>published method cells</span></div>
<div class="fact"><b>{summary["failed_cells_in_bundle"]}</b><span>failed cells · published completed bundle only</span></div>
<div class="fact"><b>Unrecorded</b><span>external attempts not recorded in this bundle</span></div>
</div>
<section class="meta" aria-labelledby="provenance"><h2 id="provenance">Provenance and limits</h2><dl>
<dt>Run</dt><dd>{escape(summary["source_run_id"])}</dd>
<dt>Track</dt><dd><code>{escape(summary["benchmark_track"])}</code></dd>
<dt>Outcome protocol</dt><dd><code>{escape(summary["protocol_id"])}</code></dd>
<dt>Source commit</dt><dd><code>{escape(commit)}</code> · {escape(dirty)}</dd>
<dt>Failure scope</dt><dd>Zero missing or failed cells in the published completed bundle; external attempt failures are not recorded in this bundle.</dd>
<dt>Interpretation</dt><dd>Descriptive only. Two-seed minimum–maximum bars are observed spread, not uncertainty intervals or causal effects.</dd>
</dl></section>
<section aria-labelledby="plots"><h2 id="plots">Observed final outcomes</h2><div class="plots">{figures}</div></section>
<section aria-labelledby="data"><h2 id="data">Every arm and method</h2>
<p class="note">Values are rounded for display. Full precision and seed-level facts remain in the <a href="../summary-report-v1/summary.json">verified report JSON</a> and <a href="../summary-report-v1/summary.csv">CSV</a>.</p>
{_render_table(summary["rows"])}</section>
<footer>The historical dashboard remains a separate, uncorrected snapshot. This page is derived from the verified v14 report and leaves that dashboard and the fixed outcome files unchanged.</footer>
</main></body></html>
"""
    return page.encode("utf-8")


def build_v14_dashboard(summary: dict[str, Any]) -> V14DashboardProjection:
    """Render the complete descriptive grid after its identity is checked."""
    try:
        _validate_summary(summary)
        files = {
            filename: render_metric_plot(
                summary["rows"], metric=metric, title=label, seed_count=summary["seed_count"]
            )
            for metric, label, filename in _METRIC_VIEW
        }
        files["dashboard.html"] = _render_page(summary)
        return V14DashboardProjection(files)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("v14 dashboard report fields are malformed") from exc
