"""Question driver through the actual32-step stream with scripted native heads."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.adapters import minigrid_learning_pilot as shared
from src.adapters import minigrid_public_cue_health as health
from src.app.coupled_attribution import _same_value
from src.infra.local_result_json import encode_bounded_local_json
from test_public_cue_recurrent_learner import configure_scripted_native, ScriptedNative
from test_public_cue_health import sources, hidden


@pytest.fixture
def case():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/recurrent-health-dynamics-20261025/run_case.py"
    )
    spec = importlib.util.spec_from_file_location("recurrent_health_dynamics_case", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scripted(monkeypatch, case):
    configure_scripted_native(monkeypatch)
    monkeypatch.setattr(shared, "_same_state", _same_value)

    class Rng:
        def clone(self):
            return self

    rng = Rng()
    monkeypatch.setattr(
        case,
        "require_torch",
        lambda: SimpleNamespace(get_rng_state=lambda: rng, equal=lambda a, b: a is b),
    )
    seen = []

    def read(head, features):
        # The real stream has already returned32fits before any diagnostics.
        assert len(ScriptedNative.calls) == 32
        seen.append(head.completed_updates)
        return hidden()

    monkeypatch.setattr(health, "read_native_activations", read)
    return seen


def test_should_run_original_prefix_then_both_health_points_and_export_no_model(case, scripted):
    result = case.run_case(tuple(p.episode for p in sources()))
    assert result["status"] == "complete", result["reason"]
    assert scripted == [0] * 6 + [16] * 6
    row = result["training"]
    assert row["charged_updates"] == row["wake_calls_attempted"] == row["wake_calls_returned"] == 32
    assert row["native_training_examples"] == 288
    assert len(row["stream_result"]["applied_updates"]) == 32
    assert row["stream_result"]["score_forward_calls"] == 0
    assert row["source_unchanged"] and row["actor_unchanged"]
    assert (
        result["preservation"] and result["numpy_rng_unchanged"] and result["torch_rng_unchanged"]
    )
    assert result["training_budget"]["updates"] == 32
    assert result["readout_budget"]["updates"] == result["readout_budget"]["replay"] == 0
    assert len(result["deltas"]) == 6
    payload = encode_bounded_local_json(result, max_bytes=1024**2)
    assert json.loads(payload)["status"] == "complete"
    assert all(all(v == 0 for v in r["changes"].values()) for r in result["deltas"])


@pytest.mark.parametrize("fault", ["lost", "prior"])
def test_should_retain_paid_fits_and_first_health_but_refuse_more_work(
    case, scripted, monkeypatch, fault
):
    original = health.read_checkpoint_health

    def read(*a, **k):
        if k["updates"] == 32:
            if fault == "lost":

                def fail(*args):
                    raise RuntimeError("lost next native hidden return")

                monkeypatch.setattr(health, "read_native_activations", fail)
            else:
                guard = k["check_boundary"]
                cells = dict(
                    zip(guard.__code__.co_freevars, (c.cell_contents for c in guard.__closure__))
                )
                boundary = cells["boundary"]
                owned = dict(
                    zip(
                        boundary.__code__.co_freevars,
                        (c.cell_contents for c in boundary.__closure__),
                    )
                )
                owned["records"][0]["scores"][0]["head"] = 99
        return original(*a, **k)

    monkeypatch.setattr(health, "read_checkpoint_health", read)
    result = case.run_case(tuple(p.episode for p in sources()))
    assert result["status"] == "incomplete" and result["deltas"] is None
    assert result["training"]["charged_updates"] == 32
    assert result["training_budget"]["updates"] == 32
    assert len(result["health"]) == 1 and result["health"][0]["scores"][0]["head"] == 0
    assert len(scripted) == 6 and len(ScriptedNative.calls) == 32
    work = result["health_failure"]["work"]
    assert work["activation_reads_attempted"] == int(fault == "lost")
    assert work["failure_work_may_be_unobserved"] is True


def test_should_describe_without_source_or_native_work(case, monkeypatch, capsys):
    monkeypatch.setattr(case, "require_torch", lambda: pytest.fail("native reached"))
    assert case.main(["--describe"]) == 0
    p = json.loads(capsys.readouterr().out)
    assert p["updates"] == 32 and p["health_points"] == [0, 32]
    assert p["planned_work"]["health_forward_calls"] == 108
    assert p["planned_work"]["games"] == p["planned_work"]["final"] == 0


@pytest.mark.parametrize("role", ["final_test", "probe_train"])
def test_should_refuse_forbidden_roles_before_native(case, monkeypatch, role):
    from dataclasses import replace

    packets = sources()
    episodes = tuple(p.episode for p in packets)
    episodes = (replace(episodes[0], role=role),) + episodes[1:]
    monkeypatch.setattr(case, "require_torch", lambda: pytest.fail("native reached"))
    with pytest.raises(ValueError):
        case.run_case(episodes)


@pytest.mark.parametrize("fault", ["missing", "status", "identity"])
def test_should_not_make_deltas_from_missing_or_incompatible_metrics(case, fault):
    fields = (
        "active_fraction",
        "dead_fraction",
        "saturated_fraction",
        "mean_absolute_activity",
        "feature_effective_rank",
    )
    rows = [
        dict(
            context=("A", "B", "C_train")[i // 2],
            head=i % 2,
            episode_id=str(i),
            metrics=[dict.fromkeys(fields, 0.0)],
        )
        for i in range(6)
    ]
    records: list[dict[str, Any]] = [
        dict(updates=n, status="complete", scores=deepcopy(rows)) for n in (0, 32)
    ]
    if fault == "missing":
        records.pop()
    elif fault == "status":
        records[-1]["status"] = "incomplete"
    else:
        records[-1]["scores"][0]["head"] = 1
    with pytest.raises(ValueError):
        case.deltas(records)
