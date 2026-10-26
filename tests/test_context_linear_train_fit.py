"""Small real affine fits, TRAIN admission and paid-prefix failures; no games."""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
from typing import Any

import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_linear_fit_control as control
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition


def frame(cue, step=0):
    values = [-0.8, -1.0, -1.0] * 49 + [-1.0]
    if cue is not None:
        values[:3] = [0.2, 2.0 * cue / 5 - 1, -1.0]
    values[4] = 2.0 * (step % 6) / 5 - 1
    return tuple(values)


def episode(identity, cue, actions):
    return DemonstrationEpisode(
        identity,
        "metadata-is-not-a-route",
        "train",
        tuple(
            DemonstrationTransition(
                i,
                frame(cue) + frame(cue, i),
                action,
                0.0,
                frame(cue) + frame(cue, i + 1),
                i == len(actions) - 1,
                7,
            )
            for i, action in enumerate(actions)
        ),
        7,
    )


def sources():
    # Deliberately opposite metadata names and red-first order. Only the image
    # selects heads; final scores must still retain this original source order.
    return (episode("named-blue", 0, (2, 1)), episode("named-red", 2, (1, 2)))


def test_should_fit_both_actual_cues_and_restore_original_source_order():
    e = sources()
    before = pickle.dumps(e)
    r = control.diagnose_context_train_linear_fit(e)
    assert r["status"] == "complete", r["reason"]
    assert r["fit_witness"] and r["aggregate_scores"]["accuracy"] == 1
    assert [x["episode_id"] for x in r["scores"]] == [x.episode_id for x in e]
    assert r["heads"]["blue"]["scores"][0]["episode_id"] == "named-red"
    assert r["heads"]["red"]["scores"][0]["episode_id"] == "named-blue"
    for h in r["heads"].values():
        assert h["fit"]["feature_dim"] == 296 and h["fit"]["fit_rows"] == 2
        assert h["source_unchanged"] and h["role"] == "train"
    w = r["work"]
    assert w["fit_calls_attempted"] == w["fit_calls_completed"] == 2
    assert w["fit_rows_attempted"] == w["fit_rows_completed"] == 4
    assert w["prediction_calls_completed"] == 2 and w["prediction_rows_completed"] == 4
    assert w["coefficient_array_bytes"] == 33264
    assert w["score_reductions_completed"] == 5 and w["score_rows_completed"] == 12
    assert w["admission_projection_rows"] == w["projection_rows_completed"] == 8
    assert w["routing_projection_rows"] == 2
    assert not w["failure_work_may_be_unobserved"] and r["source_unchanged"]
    assert pickle.dumps(e) == before


def test_should_keep_a_complete_negative_fit_without_promoting_it_to_a_witness():
    e = tuple(
        replace(
            x,
            transitions=tuple(
                replace(
                    s, observation=frame(cue) + frame(cue), next_observation=frame(cue) + frame(cue)
                )
                for s in x.transitions
            ),
        )
        for x, cue in zip(sources(), (0, 2), strict=True)
    )
    r = control.diagnose_context_train_linear_fit(e)
    assert r["status"] == "complete" and not r["fit_witness"]
    assert r["aggregate_scores"]["accuracy"] == 0.5
    assert len(r["scores"]) == 2 and r["work"]["fit_calls_completed"] == 2


@pytest.mark.parametrize("fault", ["final_test", "probe_train", "probe_test", "cue", "initial"])
def test_should_refuse_whole_invalid_catalogue_before_either_fit(fault, monkeypatch):
    e = sources()
    if fault in ("final_test", "probe_train", "probe_test"):
        e = e[:1] + (replace(e[1], role=fault),)
    elif fault == "cue":
        e = e[:1] + (episode("missing", None, (1, 2)),)
    else:
        s = e[-1].transitions[-1]
        object.__setattr__(s, "next_observation", frame(0) + s.next_observation[148:])
    monkeypatch.setattr(control, "fit_categorical_linear", lambda *a: pytest.fail("fit"))
    monkeypatch.setattr(control, "predict_categorical_linear", lambda *a: pytest.fail("predict"))
    with pytest.raises(ValueError):
        control.diagnose_context_train_linear_fit(e)


def test_should_require_both_public_heads_before_fitting(monkeypatch):
    monkeypatch.setattr(control, "fit_categorical_linear", lambda *a: pytest.fail("fit"))
    with pytest.raises(ValueError, match="both"):
        control.diagnose_context_train_linear_fit(sources()[:1])


@pytest.mark.parametrize("failure", ["before_blue", "after_blue", "before_red"])
def test_should_retain_failed_paid_head_and_stop_following_work(failure, monkeypatch):
    original = control.fit_categorical_linear
    attempts = []

    def fit(*args):
        attempts.append(1)
        if failure == "before_blue" or failure == "before_red" and len(attempts) == 2:
            raise RuntimeError("lost fit")
        model = original(*args)
        if failure == "after_blue":
            raise RuntimeError("lost fit return")
        return model

    monkeypatch.setattr(control, "fit_categorical_linear", fit)
    r = control.diagnose_context_train_linear_fit(sources())
    assert r["status"] == "failed" and r["aggregate_scores"] is None and not r["scores"]
    assert not r["fit_witness"] and r["work"]["failure_work_may_be_unobserved"]
    assert len(attempts) == len(r["heads"]) == (2 if failure == "before_red" else 1)
    if failure == "before_red":
        assert r["heads"]["blue"]["status"] == "complete"


@pytest.mark.parametrize("failed_call", [1, 2])
def test_should_retain_paid_admission_exception_before_or_after_success(failed_call, monkeypatch):
    original = control.diagnose_train_linear_fit
    calls = []
    partial: dict[str, Any] = dict(
        status="failed", reason="lost projection", source_unchanged=True, work=control._work()
    )
    partial["work"]["projection_rows_attempted"] = 1

    def run(e):
        calls.append(1)
        if len(calls) == failed_call:
            raise control.LinearFitAdmissionFailed(partial)
        return original(e)

    monkeypatch.setattr(control, "diagnose_train_linear_fit", run)
    r = control.diagnose_context_train_linear_fit(sources())
    assert r["status"] == "failed" and len(calls) == failed_call
    assert r["heads"]["blue" if failed_call == 1 else "red"] is partial
    assert r["work"]["projection_rows_attempted"] == (1 if failed_call == 1 else 5)
    assert r["aggregate_scores"] is None and r["work"]["failure_work_may_be_unobserved"]


@pytest.mark.parametrize("fault", ["value", "replacement"])
def test_should_restore_and_refuse_changed_previous_success_before_aggregate(fault, monkeypatch):
    original = control.diagnose_train_linear_fit
    returned: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []

    def run(e):
        r = original(e)
        if returned:
            if fault == "value":
                returned[0]["scores"][0]["accuracy"] = 0.0
            else:
                returned[0]["scores"] = deepcopy(returned[0]["scores"])
        else:
            expected.append(deepcopy(r))
        returned.append(r)
        return r

    monkeypatch.setattr(control, "diagnose_train_linear_fit", run)
    r = control.diagnose_context_train_linear_fit(sources())
    assert len(returned) == 2 and r["heads"]["blue"] is returned[0]
    assert returned[0] == expected[0]
    assert r["status"] == "failed" and r["aggregate_scores"] is None
    assert "preceding paid" in r["reason"] and r["work"]["fit_calls_completed"] == 2


def test_should_preserve_paid_fit_when_original_source_is_changed(monkeypatch):
    e = sources()
    original = control.fit_categorical_linear
    calls = []

    def fit(*args):
        calls.append(1)
        model = original(*args)
        object.__setattr__(e[1], "context", "corrupt")
        return model

    monkeypatch.setattr(control, "fit_categorical_linear", fit)
    r = control.diagnose_context_train_linear_fit(e)
    assert len(calls) == 1 and r["status"] == "failed" and not r["source_unchanged"]
    assert r["work"]["fit_calls_completed"] == 1 and r["work"]["prediction_calls_attempted"] == 0


def test_should_keep_coefficients_when_only_unused_terminal_next_view_and_reward_change():
    e = sources()
    a = control.diagnose_context_train_linear_fit(e)
    last = e[-1].transitions[-1]
    changed = replace(last, reward=-0.5, next_observation=frame(2) + frame(2, 5))
    e = e[:-1] + (replace(e[-1], transitions=e[-1].transitions[:-1] + (changed,)),)
    b = control.diagnose_context_train_linear_fit(e)
    for cue in ("blue", "red"):
        assert a["heads"][cue]["fit"] == b["heads"][cue]["fit"]
        assert (
            a["heads"][cue]["prediction_probabilities"]
            == b["heads"][cue]["prediction_probabilities"]
        )


def test_should_keep_original_pooled_model_definition_and_one_fit():
    r = control.diagnose_train_linear_fit(sources())
    assert r["status"] == "complete" and r["definition"] == control.DEFINITION
    assert r["work"]["fit_calls_completed"] == 1 and r["fit"]["fit_rows"] == 4
    assert r["protocol"] == control.describe_protocol()


@pytest.mark.parametrize("option", ["--context-linear-fit", "--bounds-only", None])
def test_should_describe_without_source_or_model(option, monkeypatch, capsys):
    args = ["fit", "--describe"] + ([option] if option else [])
    monkeypatch.setattr("sys.argv", args)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source"))
    monkeypatch.setattr(cli, "diagnose_context_train_linear_fit", lambda *a: pytest.fail("fit"))
    cli.main()
    r = json.loads(capsys.readouterr().out)
    assert r["role"] == "train"
    if option == "--context-linear-fit":
        assert r == control.describe_context_linear_protocol()


@pytest.mark.parametrize("option", ["--context-bounds", "--context-perceptron", "--fit-updates"])
def test_should_refuse_other_modes_or_fit_overrides_before_source(option, monkeypatch):
    args = ["fit", "--context-linear-fit", option] + (["16"] if option == "--fit-updates" else [])
    monkeypatch.setattr("sys.argv", args)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_dispatch_only_context_fitter_and_create_exclusive_output(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "new.json"
    monkeypatch.setattr(
        "sys.argv", ["fit", "--context-linear-fit", "--source", "unused", "--output", str(path)]
    )
    e = sources()
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: e)

    def run(value):
        assert value is e
        return dict(status="complete", definition="literal-dispatch")

    monkeypatch.setattr(cli, "diagnose_context_train_linear_fit", run)
    monkeypatch.setattr(cli, "diagnose_train_linear_fit", lambda *a: pytest.fail("pooled"))
    cli.main()
    assert json.loads(path.read_text(encoding="utf8"))["definition"] == "literal-dispatch"
    capsys.readouterr()
    with pytest.raises(FileExistsError):
        cli.main()
