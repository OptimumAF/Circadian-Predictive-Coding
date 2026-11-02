"""Ordinary shared rehearsal dispatch and scripted end-to-end ownership."""

import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_native_probe as probe
from src.adapters.linear_fit_learner import LinearFitLearner
from test_shared_linear_callback import scripted

OPTIONS = dict(
    method="linear_fit",
    context_return=True,
    reserved_probe=True,
    separate_context_heads=False,
    probe_update="minimum_change",
    train_update="minimum_change",
)


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_shared_rehearsal_api_admits_explicit_controls_before_data(control, monkeypatch):
    def stop(**kwargs):
        raise RuntimeError("shared rehearsal data")

    monkeypatch.setattr(entry, "prepare_training", stop)
    result = entry.run_example(**OPTIONS, probe_rehearsal=control)
    assert not result["complete"] and "shared rehearsal data" in result["error"]


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_shared_rehearsal_cli_forwards_original_flags_and_protects_output(
    control, tmp_path, monkeypatch
):
    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", record)
    flags = [
        "--method",
        "linear_fit",
        "--context-return",
        "--reserved-probe",
        "--probe-update",
        "minimum_change",
        "--train-update",
        "minimum_change",
        "--probe-rehearsal",
        control,
        "--output",
        str(tmp_path / "result.json"),
    ]
    assert entry.main(flags) == 0
    assert calls == [OPTIONS | dict(probe_rehearsal=control)]
    with pytest.raises(FileExistsError):
        entry.main(flags)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "change",
    [
        dict(separate_context_heads=True),
        dict(replay="coverage"),
        dict(probe_update=None),
        dict(private_adaptation=True),
        dict(probe_rehearsal=True),
    ],
)
def test_shared_rehearsal_refuses_unsupported_compositions_before_data(change, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("data"))
    options = OPTIONS | dict(probe_rehearsal="joint") | change
    with pytest.raises(ValueError):
        entry.run_example(**options)


def test_actual_ordinary_runner_keeps_consent_source_and_callback_costs_separate(monkeypatch):
    monkeypatch.setattr(LinearFitLearner, "train_batch_minimum_change", scripted)
    monkeypatch.setattr(
        entry,
        "run_workflow_episode",
        lambda *a, **kw: dict(
            complete=True,
            success=False,
            scenario=kw.get("scenario", "ordinary"),
            training_updates=0,
            work=dict(prediction_calls_validated=0),
        ),
    )

    def reads(learner, phase, spec, require, row, **kwargs):
        require()
        row["episodes"][phase] = [
            dict(
                phase=phase,
                policy_version=v,
                concept=c,
                role="controller_train",
                complete=True,
                episode=dict(complete=True, success=False, scenario="ordinary", training_updates=0),
            )
            for v, c in probe.PAIRED_TASKS
        ]
        row["task_readout_work"][phase] = dict(decision_attempts=0, decision_returns=0)

    monkeypatch.setattr(probe, "_read_tasks", reads)
    result = entry.run_example(**OPTIONS, probe_rehearsal="joint")
    assert result["complete"], result.get("error")
    assert result["budget"]["updates_completed"] == 12 and len(result["applied_updates"]) == 12
    assert result["probe_updates"] == 32 and result["probe_replay_examples"] == 384
    private = result["reserved_probe"]
    assert private["complete"] and private["replay_examples"] == 384
    assert private["preserved_training_work"] == dict(
        attempted_updates=0, attempted_rows=0, returned_updates=0
    )
    paid = result["probe_auxiliary_updates"]
    assert (
        paid["applied_updates"],
        paid["additional_rows_charged"],
        paid["native_sgd_updates"],
    ) == (32, 384, 0)
    summary = result["reserved_probe_summary"]["reported_costs"]["linear_rehearsal"]
    assert summary["applied_updates"] == 32 and summary["additional_rows_charged"] == 384
    assert all(
        result[k] is True
        for k in (
            "source_unchanged",
            "actor_unchanged",
            "numpy_rng_unchanged",
            "python_rng_unchanged",
        )
    )
