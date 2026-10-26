"""Portable saved-control selection and metadata-only public preflight.

The original validator may construct frozen plan/episode metadata records. All
study execution, runtime learning plans and learner constructors are forbidden.
"""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from zipfile import Path as ZipPath, ZipFile

import pytest

from scripts import continuous_replay_example as study


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/runs/continuous-probe-continuation-20261020/remaining-arms.log"
PACKAGE_LABEL = "package:src.infra.continuous_comparison_data/controls.json"


def forbidden(*args, **options):
    pytest.fail("saved-control input test attempted new study, model or runtime learning work")


@pytest.fixture(autouse=True)
def no_scientific_execution(monkeypatch):
    for name in (
        "run_study",
        "run_continuous_learnability",
        "ContinuousReplayPlan",
        "NumpyContinuousBCLearner",
        "NumpyContinuousPCLearner",
    ):
        monkeypatch.setattr(study, name, forbidden)


@pytest.fixture(scope="module")
def saved():
    raw = SOURCE.read_bytes()
    assert len(raw) == 1647144
    return json.loads(raw), raw


def test_should_preserve_the_original_default_control_path_and_label(monkeypatch):
    monkeypatch.setattr(study.resources, "files", forbidden)
    with study._baseline_source(None, False) as (path, label):
        assert path == Path(study.__file__).resolve().parents[1] / study.BASELINE_PATH
        assert label == str(path)


def test_should_prefer_the_exact_explicit_path_without_default_or_resource_lookup(
    monkeypatch, tmp_path
):
    selected = tmp_path / "selected.json"
    monkeypatch.setattr(study.resources, "files", forbidden)
    with study._baseline_source(selected, False) as (path, label):
        assert path == selected and label == str(selected.resolve())


def test_should_resolve_one_byte_identical_checkout_example_with_stable_package_provenance(
    monkeypatch,
):
    monkeypatch.setattr(study.resources, "files", forbidden)
    with study._baseline_source(None, True) as (path, label):
        assert path.name == "controls.json" and path.read_bytes() == SOURCE.read_bytes()
        assert label == PACKAGE_LABEL


def test_should_keep_installed_zip_controls_available_for_the_entire_use_scope(
    monkeypatch, tmp_path
):
    raw = b'{"original_invocation":{"disposition":"FAILED/SPENT"},"reports":{}}'
    zip_path = tmp_path / "installed.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("controls.json", raw)
    calls: list[str] = []
    with ZipFile(zip_path) as archive:

        def resource_files(package):
            calls.append(package)
            return ZipPath(archive)

        monkeypatch.setattr(study, "__file__", str(tmp_path / "scripts" / "producer.py"))
        monkeypatch.setattr(study.resources, "files", resource_files)
        with study._baseline_source(None, True) as (path, label):
            assert path.is_file() and label == PACKAGE_LABEL
            assert str(path) not in label
            assert study._load_baseline(path) == json.loads(raw)
        assert not path.exists()
    assert calls == ["src.infra.continuous_comparison_data"]


def test_should_refuse_explicit_missing_controls_without_any_fallback(monkeypatch, tmp_path):
    selected = tmp_path / "missing.json"
    monkeypatch.setattr(study.resources, "files", forbidden)
    with study._baseline_source(selected, False) as (path, label):
        assert path == selected
        with pytest.raises(FileNotFoundError):
            study._load_baseline(path)


def test_should_load_owned_native_json_without_modifying_source_bytes(tmp_path):
    path = tmp_path / "controls.json"
    raw = b'{"original_invocation":{"disposition":"FAILED/SPENT"},"reports":{}}'
    path.write_bytes(raw)
    loaded = study._load_baseline(path)
    loaded["original_invocation"]["disposition"] = "changed local copy"
    assert path.read_bytes() == raw
    assert study._load_baseline(path)["original_invocation"]["disposition"] == "FAILED/SPENT"


@pytest.mark.parametrize(
    "kind", ["duplicate", "nan", "infinity", "float-overflow", "utf8", "array", "oversized"]
)
def test_should_refuse_malformed_or_unbounded_controls_before_validation(
    kind, tmp_path, monkeypatch
):
    values = {
        "duplicate": b'{"x":1,"x":2}',
        "nan": b'{"x":NaN}',
        "infinity": b'{"x":Infinity}',
        "float-overflow": b'{"x":1e999}',
        "utf8": b"\xff",
        "array": b"[]",
    }
    raw = b" " * (6 * 1024**2 + 1) if kind == "oversized" else values[kind]
    path = tmp_path / "refused.json"
    path.write_bytes(raw)
    monkeypatch.setattr(study, "_validate_baseline", forbidden)
    with pytest.raises(ValueError):
        study._load_baseline(path)


def test_should_refuse_mutually_exclusive_source_flags_before_source_or_science(monkeypatch):
    monkeypatch.setattr(study, "_baseline_source", forbidden)
    monkeypatch.setattr(study, "_require_producer_environment", forbidden)
    with pytest.raises(SystemExit) as error:
        study.main(
            ["--baseline", "explicit.json", "--example-baseline", "--validate-baseline-only"]
        )
    assert error.value.code == 2


def test_should_enforce_the_original_normal_producer_profile_before_any_source_selection(
    monkeypatch, capsys
):
    monkeypatch.setattr(study.np, "__version__", "invalid fixture version")
    monkeypatch.setattr(study, "_baseline_source", forbidden)
    monkeypatch.setattr(study, "_load_baseline", forbidden)
    with pytest.raises(SystemExit) as error:
        study.main(["--baseline", "missing.json"])
    assert error.value.code == 2
    assert "fixed comparison requires" in capsys.readouterr().err


@pytest.mark.parametrize("selection", ["default", "explicit", "example"])
def test_should_run_only_original_validation_and_print_stable_preflight_metadata(
    monkeypatch, tmp_path, capsys, selection
):
    baseline = {"original_invocation": {"disposition": "FAILED/SPENT"}, "reports": {}}
    original = deepcopy(baseline)
    validations: list[Any] = []
    loads: list[Path] = []
    selected = tmp_path / "selected.json"

    def load(path):
        loads.append(path)
        return baseline

    def validate(value):
        assert value is baseline
        validations.append(value)

    monkeypatch.setattr(study, "_load_baseline", load)
    monkeypatch.setattr(study, "_validate_baseline", validate)
    monkeypatch.setattr(study, "_require_producer_environment", forbidden)
    monkeypatch.setattr(study, "_run_comparison", forbidden)
    monkeypatch.setattr(study, "ContinuousProbePlan", forbidden)
    arguments = ["--validate-baseline-only"]
    if selection == "explicit":
        arguments += ["--baseline", str(selected)]
    elif selection == "example":
        arguments += ["--example-baseline"]
    study.main(arguments)
    result = json.loads(capsys.readouterr().out)
    expected = (
        str(selected.resolve())
        if selection == "explicit"
        else PACKAGE_LABEL
        if selection == "example"
        else str(ROOT / study.BASELINE_PATH)
    )
    assert len(loads) == len(validations) == 1 and baseline == original
    assert result["scenario"] == "continuous_saved_control_preflight_v1"
    assert result["baseline_validated"] is True and result["training_executed"] is False
    assert result["baseline_source"] == dict(
        path=expected, original_disposition="FAILED/SPENT", parameter_archive=False
    )
    assert result["families"] == ["bp", "pc", "neutral-cpc", "active-cpc"]
    assert not any(key in result for key in ("candidate", "learner", "reports", "final_states"))


def test_should_admit_the_original_saved_metadata_without_mutation_or_runtime_learning(saved):
    baseline, raw = saved
    before = deepcopy(baseline)
    result = study._preflight(baseline, "preserved original controls")
    assert result["baseline_validated"] is True and result["training_executed"] is False
    assert result["baseline_source"]["original_disposition"] == "FAILED/SPENT"
    assert baseline == before and SOURCE.read_bytes() == raw


@pytest.mark.parametrize("complete", [True, False])
def test_should_route_normal_execution_once_after_profile_and_source_loading(
    monkeypatch, tmp_path, capsys, complete
):
    selected = tmp_path / "selected.json"
    baseline = {"original_invocation": {"disposition": "FAILED/SPENT"}}
    calls: list[str] = []

    def profile():
        calls.append("profile")

    def load(path):
        assert path == selected and calls == ["profile"]
        calls.append("load")
        return baseline

    def comparison(value, label):
        assert value is baseline and label == str(selected.resolve())
        calls.append("comparison")
        print(json.dumps(dict(complete=complete, baseline_source=label)))
        if not complete:
            raise SystemExit(1)

    monkeypatch.setattr(study, "_require_producer_environment", profile)
    monkeypatch.setattr(study, "_load_baseline", load)
    monkeypatch.setattr(study, "_run_comparison", comparison)
    if complete:
        study.main(["--baseline", str(selected)])
    else:
        with pytest.raises(SystemExit) as error:
            study.main(["--baseline", str(selected)])
        assert error.value.code == 1
    assert calls == ["profile", "load", "comparison"]
    assert json.loads(capsys.readouterr().out)["complete"] is complete


def test_should_refuse_invalid_preflight_structure_without_falling_through_to_study(
    monkeypatch, tmp_path, capsys
):
    path = tmp_path / "invalid.json"
    path.write_text("{}", encoding="utf8")
    monkeypatch.setattr(study, "_run_comparison", forbidden)
    monkeypatch.setattr(study, "_require_producer_environment", forbidden)
    with pytest.raises(SystemExit) as error:
        study.main(["--baseline", str(path), "--validate-baseline-only"])
    assert error.value.code == 2
    assert "saved-control" in capsys.readouterr().err
