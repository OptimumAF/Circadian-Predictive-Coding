"""Saved four-family comparisons are source-linked tables, not new experiments.

Only scalar JSON projection, temporary files and inert HTML execute. No runner,
learner, nearest kernel, diagnostics, environment or numerical oracle is called.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any
from zipfile import Path as ZipPath, ZipFile

import pytest

from scripts import build_key_door_inspector as cli
from src.app.continuous_comparison_inspector import project_continuous_comparison_report
from src.app.key_door_inspector import project_key_door_report
from src.infra import key_door_inspector_files as files
from test_key_door_inspector_files import Page


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/runs/continuous-replay-comparison-20261020/comparison-summary.json"
FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")
BENCHMARK = (
    ("warm_acquisition_gain", True),
    ("new_acquisition_gain", True),
    ("warm_retention_loss", False),
    ("warm_return_recovery", True),
    ("warm_return_deficit", False),
    ("new_retention_loss", False),
)
PROBES = (
    ("fresh_acquisition_gain", True),
    ("fresh_training_gain", True),
    ("old_retention_loss/old_0", False),
    ("old_retention_loss/old_1", False),
)


@pytest.fixture(scope="module")
def saved():
    return json.loads(SOURCE.read_text(encoding="utf8"))


def tiny():
    return dict(
        complete=False,
        comparisons={
            "bp": dict(
                benchmark={
                    "new_acquisition_gain": dict(
                        baseline=0.0,
                        replay=-0.25,
                        replay_minus_baseline=-0.25,
                        higher_is_better=True,
                    )
                },
                probes=[
                    dict(
                        wake_updates=0,
                        baseline_native_updates=0,
                        replay_native_updates=0,
                        views={
                            "inherited": {
                                "fresh_acquisition_gain": dict(
                                    baseline=0.0,
                                    replay=0.0,
                                    replay_minus_baseline=0.0,
                                    higher_is_better=True,
                                )
                            }
                        },
                    )
                ],
            )
        },
        observed_work={"parameter_updates": 0},
        baseline_source={"path": "saved-control.json", "original_disposition": "FAILED/SPENT"},
        qualification="unequal replay compute; isolated probe clones; no benefit threshold",
    )


def section(view, identity):
    return next(item for item in view["sections"] if item["id"] == identity)


def cells(view):
    result = [view[key] for key in ("protocol_id", "status", "reason")]
    for item in view["sections"]:
        for row in item["rows"]:
            assert len(row["cells"]) == len(item["columns"])
            result.extend(row["cells"])
    return result


def at(view, pointer, origin="report"):
    return next(
        cell for cell in cells(view) if cell["pointer"] == pointer and cell["source"] == origin
    )


def resolve(value, pointer):
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if type(value) is list else value[token]
    return value


def assert_sources(view):
    for cell in cells(view):
        assert set(cell) == {"value", "pointer", "source", "missing"}
        if cell["missing"]:
            assert cell["value"] is None and cell["pointer"] is None
        elif cell["pointer"] is None:
            assert type(cell["value"]) is str
        else:
            original = view["source"] if cell["source"] == "report" else view["resources"]
            assert cell["value"] == resolve(original, cell["pointer"])


def test_should_route_all_saved_families_and_preserve_every_benchmark_and_probe_scalar(saved):
    before = deepcopy(saved)
    view = project_continuous_comparison_report(saved, source_label=SOURCE.name)
    assert view == project_key_door_report(saved, source_label=SOURCE.name)
    assert view["schema_id"] == "continuous_comparison_inspector_v1"
    assert view["source"] == before == saved
    assert view["status"]["pointer"] == "/complete" and view["status"]["value"] is True
    assert [item["id"] for item in view["sections"]] == [
        "benchmark",
        "probes",
        "work",
        "provenance",
    ]
    assert len(section(view, "benchmark")["rows"]) == 24
    assert len(section(view, "probes")["rows"]) == 128
    for family in FAMILIES:
        for name, direction in BENCHMARK:
            root = f"/comparisons/{family}/benchmark/{name}"
            for key in ("baseline", "replay", "replay_minus_baseline", "higher_is_better"):
                assert at(view, root + "/" + key)["value"] == resolve(saved, root + "/" + key)
            assert at(view, root + "/higher_is_better")["value"] is direction
        for index in range(4):
            for selected in ("inherited", "zero_velocity"):
                for name, direction in PROBES:
                    root = f"/comparisons/{family}/probes/{index}/views/{selected}/{name}"
                    for key in ("baseline", "replay", "replay_minus_baseline", "higher_is_better"):
                        assert at(view, root + "/" + key)["value"] == resolve(
                            saved, root + "/" + key
                        )
                    assert at(view, root + "/higher_is_better")["value"] is direction
    assert_sources(view)


def test_should_preserve_negative_tradeoffs_extra_costs_and_original_failed_control(saved):
    view = project_continuous_comparison_report(saved, source_label="accepted.json")
    pointer = "/comparisons/bp/benchmark/new_acquisition_gain/replay_minus_baseline"
    assert (
        at(view, pointer)["value"]
        == saved["comparisons"]["bp"]["benchmark"]["new_acquisition_gain"]["replay_minus_baseline"]
        < 0
    )
    assert any(
        cell["pointer"] and cell["pointer"].startswith("/observed_work/") for cell in cells(view)
    )
    assert any(
        cell["pointer"] and cell["pointer"].startswith("/baseline_source") for cell in cells(view)
    )
    assert any(cell["pointer"] == "/qualification" for cell in cells(view))
    assert_sources(view)


def test_should_keep_partial_null_missing_and_numeric_zero_distinct():
    source = tiny()
    metric = source["comparisons"]["bp"]["benchmark"]["new_acquisition_gain"]
    metric["baseline"] = None
    metric["replay"] = 0.0
    del metric["replay_minus_baseline"]
    view = project_continuous_comparison_report(source, source_label="partial.json")
    root = "/comparisons/bp/benchmark/new_acquisition_gain"
    assert at(view, root + "/baseline") == dict(
        value=None, pointer=root + "/baseline", source="report", missing=False
    )
    assert at(view, root + "/replay")["value"] == 0.0
    assert not any(cell["pointer"] == root + "/replay_minus_baseline" for cell in cells(view))
    assert any(cell["missing"] for cell in cells(view))
    assert len(section(view, "benchmark")["rows"]) == 24
    assert len(section(view, "probes")["rows"]) == 128
    assert_sources(view)


@pytest.mark.parametrize("wakes", [(), (2,), (3, 1)])
def test_should_keep_every_partial_probe_slot_and_map_recorded_wakes_to_original_indices(wakes):
    source = tiny()
    original = source["comparisons"]["bp"]["probes"][0]
    if not wakes:
        source["comparisons"] = {}
    else:
        source["comparisons"]["bp"]["probes"] = [
            dict(deepcopy(original), wake_updates=wake) for wake in wakes
        ]
    view = project_continuous_comparison_report(source, source_label="partial-wakes.json")
    table = section(view, "probes")
    assert len(table["rows"]) == 128
    assert "Wake slot (not evidence)" in table["columns"]
    expected_slots = [
        (family, str(wake)) for family in FAMILIES for wake in range(4) for _ in range(8)
    ]
    actual_slots = []
    for row in table["rows"]:
        fields = dict(zip(table["columns"], row["cells"], strict=True))
        family = fields["Method"]["value"]
        slot = fields["Wake slot (not evidence)"]
        wake = int(slot["value"])
        actual_slots.append((family, slot["value"]))
        assert slot["pointer"] is None and slot["missing"] is False
        observed = fields["Wake"]
        if family == "bp" and wake in wakes:
            index = wakes.index(wake)
            assert observed == dict(
                value=wake,
                pointer=f"/comparisons/bp/probes/{index}/wake_updates",
                source="report",
                missing=False,
            )
        else:
            for column in (
                "Wake",
                "Baseline native updates",
                "Replay native updates",
                "No replay",
                "Replay",
                "Replay minus baseline",
                "Higher is better",
            ):
                assert fields[column]["missing"] is True
                assert fields[column]["pointer"] is None and fields[column]["value"] is None
    assert actual_slots == expected_slots
    assert_sources(view)


@pytest.mark.parametrize("origin,value", [("report", 2**53), ("resources", -(2**53))])
def test_should_refuse_json_integers_that_the_browser_cannot_represent_exactly(origin, value):
    source = tiny()
    options: dict[str, Any] = {}
    if origin == "report":
        source["environment"] = {"oversized_integer": value}
    else:
        options = dict(resources={"oversized_integer": value}, resource_label="unsafe.json")
    with pytest.raises(ValueError):
        project_continuous_comparison_report(source, source_label="exactness.json", **options)


def test_should_preserve_both_exact_browser_integer_boundaries_in_source_and_resources():
    bounds = {"negative": -(2**53 - 1), "positive": 2**53 - 1}
    source = tiny()
    source["environment"] = deepcopy(bounds)
    view = project_continuous_comparison_report(
        source,
        source_label="safe-boundaries.json",
        resources=deepcopy(bounds),
        resource_label="safe.json",
    )
    published = Page(files.render_key_door_inspector(view).decode("utf8")).payload()
    assert published["source"]["environment"] == bounds
    assert published["resources"] == bounds
    assert_sources(published)


def test_should_detach_source_and_each_projected_metric_cell():
    source = tiny()
    view = project_continuous_comparison_report(source, source_label="owned.json")
    pointer = "/comparisons/bp/benchmark/new_acquisition_gain/baseline"
    source["comparisons"]["bp"]["benchmark"]["new_acquisition_gain"]["baseline"] = 8.0
    assert at(view, pointer)["value"] == resolve(view["source"], pointer) == 0.0
    view["source"]["comparisons"]["bp"]["benchmark"]["new_acquisition_gain"]["baseline"] = 9.0
    assert at(view, pointer)["value"] == 0.0
    at(view, pointer)["value"] = 10.0
    assert resolve(view["source"], pointer) == 9.0


def test_should_keep_separate_resource_costs_unverified_and_source_resolvable():
    resource: dict[str, Any] = {"runtime/seconds~": 0.0, "unknown": None}
    view = project_continuous_comparison_report(
        tiny(), source_label="source.json", resources=resource, resource_label="separate.json"
    )
    assert (
        "UNVERIFIED" in section(view, "resources")["note"]
        or "UNVERIFIED" in section(view, "resources")["title"]
    )
    assert at(view, "", "resources")["value"] == {"runtime/seconds~": 0.0, "unknown": None}
    resource["runtime/seconds~"] = 99.0
    assert view["resources"]["runtime/seconds~"] == 0.0
    assert_sources(view)


@pytest.mark.parametrize(
    "fault",
    [
        "complete-bool",
        "baseline-bool",
        "replay-string",
        "delta-nan",
        "wrong-direction",
        "direction-integer",
        "unknown-family",
        "unknown-view",
        "duplicate-wake",
        "wake-bool",
        "negative-wake",
        "too-many-wakes",
        "metric-shape",
        "unknown-metric",
        "unknown-old",
        "comparison-shape",
        "nonjson",
    ],
)
def test_should_refuse_malformed_metric_authority_or_ambiguous_probe_records(fault):
    source = tiny()
    family = source["comparisons"]["bp"]
    metric = family["benchmark"]["new_acquisition_gain"]
    if fault == "complete-bool":
        source["complete"] = 1
    elif fault == "baseline-bool":
        metric["baseline"] = True
    elif fault == "replay-string":
        metric["replay"] = "0.5"
    elif fault == "delta-nan":
        metric["replay_minus_baseline"] = float("nan")
    elif fault == "wrong-direction":
        metric["higher_is_better"] = False
    elif fault == "direction-integer":
        metric["higher_is_better"] = 1
    elif fault == "unknown-family":
        source["comparisons"]["invented"] = {}
    elif fault == "unknown-view":
        family["probes"][0]["views"]["favorable"] = {}
    elif fault == "duplicate-wake":
        family["probes"].append(deepcopy(family["probes"][0]))
    elif fault == "wake-bool":
        family["probes"][0]["wake_updates"] = True
    elif fault == "negative-wake":
        family["probes"][0]["wake_updates"] = -1
    elif fault == "too-many-wakes":
        family["probes"] = [dict(wake_updates=i) for i in range(5)]
    elif fault == "metric-shape":
        family["benchmark"]["new_acquisition_gain"] = []
    elif fault == "unknown-metric":
        family["benchmark"]["selected_win"] = {}
    elif fault == "unknown-old":
        family["probes"][0]["views"]["inherited"]["old_retention_loss"] = {"old_2": {}}
    elif fault == "comparison-shape":
        source["comparisons"] = []
    else:
        source["qualification"] = ("not", "native JSON")
    with pytest.raises(ValueError):
        project_continuous_comparison_report(source, source_label="refused.json")


def test_should_publish_hostile_source_text_only_as_inert_json(tmp_path):
    source = tiny()
    danger = '</script><img id="injected" src="https://example.invalid" onerror="alert(1)"> &'
    source["qualification"] = danger
    path, output = tmp_path / "saved.json", tmp_path / "comparison.html"
    path.write_text(json.dumps(source), encoding="utf8")
    files.write_key_door_inspector(path, output)
    page = Page(output.read_text(encoding="utf8"))
    assert page.payload()["source"]["qualification"] == danger
    assert page.payload()["schema_id"] == "continuous_comparison_inspector_v1"
    assert "<" not in "".join(page.payload_parts)
    assert not any(attrs.get("id") == "injected" for _, attrs in page.tags)
    assert all(
        not (attrs.get(key) or "").startswith(("http:", "https:", "//"))
        for _, attrs in page.tags
        for key in ("href", "src")
    )
    assert_sources(page.payload())


def test_should_preserve_source_and_existing_output_when_publication_is_repeated(tmp_path):
    path, output = tmp_path / "saved.json", tmp_path / "comparison.html"
    raw = json.dumps(tiny()).encode("utf8")
    path.write_bytes(raw)
    files.write_key_door_inspector(path, output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        files.write_key_door_inspector(path, output)
    assert path.read_bytes() == raw and output.read_bytes() == original


def test_should_use_one_byte_identical_checkout_example_without_package_resource_lookup(
    monkeypatch,
):
    def forbidden(*args, **options):
        pytest.fail("checkout example unexpectedly requested an installed resource")

    monkeypatch.setattr(cli.resources, "files", forbidden)
    with cli._example_comparison() as path:
        assert path.name == "comparison-summary.json"
        assert path.read_bytes() == SOURCE.read_bytes()


def test_should_hold_an_installed_zip_example_open_for_publication_then_remove_its_tempfile(
    monkeypatch, tmp_path
):
    raw = json.dumps(tiny()).encode("utf8")
    archive_path = tmp_path / "installed.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("comparison-summary.json", raw)
    calls: list[str] = []

    with ZipFile(archive_path) as archive:

        def resource_files(package):
            calls.append(package)
            assert package == "src.infra.continuous_comparison_data"
            return ZipPath(archive)

        monkeypatch.setattr(cli, "__file__", str(tmp_path / "scripts" / "builder.py"))
        monkeypatch.setattr(cli.resources, "files", resource_files)
        with cli._example_comparison() as path:
            assert path.is_file() and path.read_bytes() == raw
            output = tmp_path / "installed-example.html"
            files.write_key_door_inspector(path, output)
            assert Page(output.read_text(encoding="utf8")).payload()["source"] == tiny()
        assert not path.exists()
    assert calls == ["src.infra.continuous_comparison_data"]


def test_should_export_the_bundled_example_through_the_public_cli_without_training(
    monkeypatch, tmp_path, capsys, saved
):
    output = tmp_path / "bundled.html"
    monkeypatch.setattr(sys, "argv", ["builder", "--example-comparison", "--output", str(output)])
    cli.main()
    response = json.loads(capsys.readouterr().out)
    assert response["output"] == str(output.resolve())
    assert response["source"] == "bundled saved continuous comparison"
    view = Page(output.read_text(encoding="utf8")).payload()
    assert view["source"] == saved and view["schema_id"] == "continuous_comparison_inspector_v1"
    assert len(section(view, "benchmark")["rows"]) == 24
    assert len(section(view, "probes")["rows"]) == 128
    assert_sources(view)
