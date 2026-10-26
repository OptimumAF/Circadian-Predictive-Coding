"""Pure CLI controls plus one separately budgeted real native stream case."""

from dataclasses import asdict
import json
import pickle
import sys
from typing import Any

import pytest

import scripts.run_native_stream_demo as cli
import src.adapters.native_stream_demo as demo


def forbidden(*args, **kwargs):
    pytest.fail("a refused or unselected native callback was reached")


@pytest.mark.parametrize("value", (None, 1, "true", (True,)))
def test_should_refuse_nonboolean_diagnostic_mode_before_model_construction(
    monkeypatch, value: Any
):
    monkeypatch.setattr(demo, "BackpropMLP", forbidden)
    monkeypatch.setattr(demo, "CircadianPredictiveCodingNetwork", forbidden)
    with pytest.raises(ValueError, match="exact boolean"):
        demo.run_demo(include_unit_diagnostics=value)


def test_should_refuse_foreign_private_stream_learner_without_callbacks():
    with pytest.raises(ValueError, match="legacy circadian"):
        demo._run_stream("foreign", None, diagnose_units=True)  # type: ignore[arg-type]


def test_should_route_unit_diagnostics_only_to_stream_with_explicit_keyword(monkeypatch, capsys):
    calls = []

    def selected(seed, *, include_unit_diagnostics):
        calls.append((seed, include_unit_diagnostics))
        return {"scope": "stub"}

    monkeypatch.setattr(cli, "run_demo", selected)
    for name in ("run_rejection_demo", "run_coupled_demo", "run_matched_demo"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["stream", "--seed", "23", "--diagnose-units"])
    cli.main()
    assert calls == [(23, True)]
    assert json.loads(capsys.readouterr().out) == {"scope": "stub"}


@pytest.mark.parametrize(
    "scenario", ("promotion-rejection", "coupled-neutral", "matched-baselines")
)
def test_should_refuse_unit_diagnostics_for_unsupported_scenarios_before_callbacks(
    monkeypatch, scenario
):
    for name in ("run_demo", "run_rejection_demo", "run_coupled_demo", "run_matched_demo"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["stream", "--scenario", scenario, "--diagnose-units"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "scenario,handler",
    (
        ("stream", "run_demo"),
        ("promotion-rejection", "run_rejection_demo"),
        ("coupled-neutral", "run_coupled_demo"),
        ("matched-baselines", "run_matched_demo"),
    ),
)
def test_should_preserve_every_previous_cli_route_without_new_keyword(
    monkeypatch, capsys, scenario, handler
):
    calls = []

    def selected(seed):
        calls.append(seed)
        return {"route": scenario}

    for name in ("run_demo", "run_rejection_demo", "run_coupled_demo", "run_matched_demo"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(cli, handler, selected)
    monkeypatch.setattr(sys, "argv", ["stream", "--scenario", scenario])
    cli.main()
    assert calls == [23]
    assert json.loads(capsys.readouterr().out) == {"route": scenario}


def test_should_run_actual_unit_diagnostics_cli_without_changing_native_models(
    monkeypatch, capsys, record_testsuite_property
):
    import src.adapters.native_unit_diagnostics as units

    original = units.read_legacy_unit_diagnostics
    observations = []

    def observed(model):
        before = pickle.dumps(vars(model))
        identity = {key: id(value) for key, value in vars(model).items()}
        result = original(model)
        assert pickle.dumps(vars(model)) == before
        assert {key: id(value) for key, value in vars(model).items()} == identity
        observations.append(asdict(result))
        return result

    monkeypatch.setattr(units, "read_legacy_unit_diagnostics", observed)
    monkeypatch.setattr(sys, "argv", ["stream", "--seed", "23", "--diagnose-units"])
    cli.main()
    output = capsys.readouterr()
    assert not output.err and len(output.out.encode("utf8")) <= 65536
    report = json.loads(output.out)
    assert report["protocol_id"] == "native_adapter_stream_v1"
    assert report["scope"] == "integration_smoke"
    assert report["seed"] == 23
    assert report["settings"]["sleep_calls"] == report["settings"]["promotion_calls"] == 0
    bp, cpc = report["methods"]
    original_fields = {
        "method",
        "experiment_state",
        "model_state",
        "events",
        "initial_candidate_predictions",
        "final_candidate_predictions",
        "diagnostics",
    }
    assert bp.keys() == original_fields
    assert cpc.keys() == original_fields | {
        "native_unit_diagnostics",
        "native_unit_diagnostic_seconds",
    }
    assert len(observations) == 2
    saved = cpc["native_unit_diagnostics"]
    assert saved == json.loads(json.dumps(dict(initial=observations[0], final=observations[1])))
    initial, final = saved["initial"], saved["final"]
    assert initial["family"] == final["family"] == "legacy_cpc"
    assert initial["layer"] == final["layer"] == "adaptive_hidden"
    assert [u["neuron_id"] for u in final["history"]["units"]] == [0, 1, 2, 3]
    assert all(u["parent_id"] is None for u in final["history"]["units"])
    assert initial["history"]["age"]["minimum"] == initial["history"]["age"]["maximum"] == 0
    assert final["history"]["age"]["minimum"] == final["history"]["age"]["maximum"] == 4
    assert initial["history"]["importance"]["maximum"] == 0
    assert final["history"]["importance"]["maximum"] > 0
    assert cpc["native_unit_diagnostic_seconds"] >= 0
    for method in (bp, cpc):
        assert method["experiment_state"]["completed_updates"] == 4
        assert [e["applied_sample_ids"] for e in method["events"]] == [
            [],
            [],
            ["s0"],
            ["s1"],
            ["s2"],
            ["s3"],
        ]
        actor = method["events"][0]["actor_predictions"]
        assert all(e["actor_predictions"] == actor for e in method["events"])
        assert method["initial_candidate_predictions"] != method["final_candidate_predictions"]
        assert len(method["diagnostics"]) == 4
    record_testsuite_property("native_unit_stdout", output.out)
    record_testsuite_property(
        "native_unit_observations",
        json.dumps(
            dict(
                reader_calls=2,
                full_native_state_and_rng_unchanged=True,
                output_bytes=len(output.out.encode("utf8")),
                report=report,
            ),
            sort_keys=True,
        ),
    )
