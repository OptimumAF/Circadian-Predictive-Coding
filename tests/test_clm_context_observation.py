"""Finite167 observer qualification; no native weights or old scientific inputs."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import scripts.run_clm_ticket_workflow as study
from src.adapters.clm_head_learner import _equal
from test_clm_linear_context_workflow import loaded_case

STAGE = Path(__file__).resolve().parents[1] / "artifacts/runs/clm-linear-context-native-20261025"


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, STAGE / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


observer = load_module("context_observation167", "observe.py")
executor = load_module("context_execution167", "execute.py")
literal_capture: dict[str, Any] = {}


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    loaded = loaded_case()
    heads = deepcopy(
        (loaded.state_head.state_dict(), loaded.action_head.state_dict(), loaded.logit_scale)
    )
    rng, dumps = torch.get_rng_state().clone(), json.dumps
    folder = tmp_path_factory.mktemp("observed-context")
    actual = folder / "actual.json"
    ledger: dict[str, Any] = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "load_local_clm", lambda *a, **k: loaded)
        patch.setattr(
            "sys.argv",
            [
                "workflow",
                "--model-dir",
                "unused",
                "--head",
                "unused",
                "--output",
                str(actual),
                "--linear-context-updates",
                "--public-observation-train",
            ],
        )
        with observer.observe(
            study, ledger, persist=lambda: observer.write_observation(folder, ledger)
        ):
            study.main()
        ledger["complete"] = True
        observer.write_observation(folder, ledger)
    assert json.dumps is dumps
    report = json.loads(actual.read_text())
    literal_capture.update(
        definition="literal_observed_context_CLI_not_native_v1", report=report, observer=ledger
    )
    return loaded, heads, rng, report, ledger, folder


def test_should_observe_actual_cli_without_adding_model_work(completed):
    _, _, _, report, ledger, _ = completed
    c = ledger["counts"]
    assert report["complete"] is ledger["complete"] is True
    assert c["loads"] == c["load_returns"] == 1
    assert c["fits"] == c["fit_returns"] == 3 and c["fit_rows"] == 36
    assert (
        c["encoder_calls"]
        == c["encoder_returns"]
        == c["encoder_forwards"]
        == c["forward_returns"]
        == 52
    )
    assert c["sequences"] == c["padded_slots"] == c["token_ids"] == 68
    assert c["readouts"] == c["readout_returns"] == c["readout_rows"] == 50
    assert c["episodes"] == c["episode_returns"] == 14
    assert c["dispatches"] == c["dispatch_returns"] == 74  # 24teacher+50live.
    assert (
        ledger["wrappers_restored"]
        and not ledger["quota_stopped"]
        and not ledger["unreturned_work_unknown"]
    )


def test_should_count_report_guards_and_explicit_utf8_copy(completed):
    counts = completed[4]["counts"]
    assert counts["json_serialized_bytes"] > 24_000_000
    assert counts["json_utf8_copy_bytes"] == counts["json_serialized_bytes"]
    assert counts["json_serialized_bytes"] < 128 * 1024**2


def test_should_preserve_original_source_rng_and_zero_native_head_work(completed):
    loaded, heads, rng, _, ledger, _ = completed
    assert _equal(
        heads, (loaded.state_head.state_dict(), loaded.action_head.state_dict(), loaded.logit_scale)
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert all(
        ledger["counts"][k] == 0
        for k in ("head_forwards", "backwards", "random_physical_elements", "random_meta_elements")
    )


def test_should_stop_at_json_quota_and_restore_original_functions():
    before = json.dumps
    ledger: dict[str, Any] = {}
    limits = dict(observer.CAPS, json_serialized_bytes=16, json_utf8_copy_bytes=16)
    with pytest.raises(ValueError, match="quota"):
        with observer.observe(study, ledger, caps=limits):
            item = torch.nn.Identity()
            deepcopy(item)
            json.dumps("x" * 32)
    assert json.dumps is before and ledger["wrappers_restored"] and ledger["quota_stopped"]
    assert ledger["counts"]["json_serialized_bytes"] > 16
    assert ledger["counts"]["modules"] == 2


def test_should_refuse_extra_solve_before_entering_original(monkeypatch):
    monkeypatch.setattr(np.linalg, "lstsq", lambda *a, **k: pytest.fail("entered denied solve"))
    ledger: dict[str, Any] = {}
    with pytest.raises(ValueError, match="fits"):
        with observer.observe(study, ledger, caps=dict(observer.CAPS, fits=0)):
            np.linalg.lstsq(np.eye(2), np.eye(2), rcond=1e-12)
    assert ledger["counts"]["fits"] == 1 and ledger["counts"]["fit_returns"] == 0
    assert ledger["quota_stopped"] and ledger["wrappers_restored"]


def test_should_retain_lost_solve_return_without_retry(monkeypatch):
    original = np.linalg.lstsq

    def lost(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("controlled missing numerical return")

    monkeypatch.setattr(np.linalg, "lstsq", lost)
    ledger: dict[str, Any] = {}
    with pytest.raises(OSError, match="missing"):
        with observer.observe(study, ledger):
            np.linalg.lstsq(np.eye(2), np.eye(2), rcond=1e-12)
    assert ledger["counts"]["fits"] == 1 and ledger["counts"]["fit_rows"] == 2
    assert ledger["counts"]["fit_returns"] == 0 and ledger["unreturned_work_unknown"]
    assert ledger["wrappers_restored"]


def test_should_read_literal_result_and_preserve_early_negative_tasks(completed, monkeypatch):
    folder = completed[5]
    monkeypatch.setattr(executor, "stage", folder)
    assert executor.read_result(0) is True
    summary = json.loads((folder / "result-summary.json").read_text())
    assert summary["ordinary_successes"]["after_A"] == [True, True, False, False]
    assert summary["new_v2_success_gain"] == 2 and summary["old_v1_success_change"] == 0
    assert summary["learning_recovery_demonstrated"] is False
    assert summary["observed_episodes"][-1]["stopped_reason"] == "permission_denied"


def test_should_record_incomplete_result_without_inventing_success(tmp_path, monkeypatch):
    monkeypatch.setattr(executor, "stage", tmp_path)
    (tmp_path / "actual.json").write_text(
        json.dumps(dict(complete=False, episodes=[], training=[], error="controlled incomplete"))
    )
    (tmp_path / "case-work.json").write_text(json.dumps(dict(complete=False, counts={})))
    assert executor.read_result(1) is False
    summary = json.loads((tmp_path / "result-summary.json").read_text())
    assert summary["native_complete"] is False and summary["error"] == "controlled incomplete"
