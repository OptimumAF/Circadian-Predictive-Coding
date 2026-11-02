"""Real stream/CLI with SCRIPTED fit, prediction and game delegates; zero LS."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from scripts import minigrid_affine_workflow as workflow
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_context_linear_learner as port
from src.app.coupled_attribution import AttributionStream
from src.core.categorical_linear_fit import CategoricalLinearFit

DATA = Path(__file__).resolve().parents[1] / "examples/minigrid_affine_train/episodes.json"
CHARGE = lambda values: None
RETURN = lambda values: None


@pytest.fixture
def scripted(monkeypatch):
    holder: dict[str, Any] = dict(
        fits=[], operations=[], streams=[], effect=None, predict_calls=0, game_calls=0
    )

    def fit(kind, x, y):
        work = dict(script_fits=1, script_fit_rows=len(x))
        CHARGE(work)
        holder["fits"].append((kind, len(x)))
        model = CategoricalLinearFit(296, 7, len(x), ((0.0,) * 7,) * 297, 0, (0.0,) * len(x), 0.0)
        if holder["effect"] is not None:
            holder["effect"](model, x, y)
        RETURN(work)
        return model

    monkeypatch.setattr(port, "fit_categorical_linear", lambda x, y, k: fit("cold", x, y))
    monkeypatch.setattr(port, "update_categorical_linear", lambda prior, x, y: fit("warm", x, y))

    def predict(model, x):
        work = dict(script_predict=1, script_predict_rows=len(x))
        CHARGE(work)
        holder["predict_calls"] += 1
        value = np.full((len(x), 7), 1 / 7, dtype=np.float64)
        RETURN(work)
        return value

    monkeypatch.setattr(port, "predict_categorical_linear", predict)

    def operation(original, name):
        def run(self, x, y):
            holder["operations"].append(name)
            return original(self, x, y)

        return run

    for name in ("train_batch", "train_batch_minimum_change"):
        monkeypatch.setattr(
            port.ContextLinearLearner,
            name,
            operation(getattr(port.ContextLinearLearner, name), name),
        )

    original_init = AttributionStream.__init__

    def initialize(self, source, training, heldout, *args, **kwargs):
        assert heldout == () and kwargs["training_only"] is True
        budget = kwargs["budget"]
        holder["streams"].append(
            (
                self,
                source,
                source.snapshot_state(),
                budget,
                id(budget.budget),
                asdict(budget.budget),
                budget.started_at,
            )
        )
        return original_init(self, source, training, heldout, *args, **kwargs)

    monkeypatch.setattr(AttributionStream, "__init__", initialize)
    monkeypatch.setattr(workflow, "_require_minigrid", lambda: None)

    def games(source, endpoint, **options):
        work = dict(script_game_batches=1, script_game_episodes=4)
        CHARGE(work)
        holder["game_calls"] += 1
        options["check_boundary"]()
        result = dict(
            status="complete",
            state_unchanged=True,
            scripted=True,
            episodes=[
                dict(status="complete", success=False, scripted=True, environment=dict(closed=True))
                for _ in range(4)
            ],
        )
        RETURN(work)
        return result

    monkeypatch.setattr(workflow, "evaluate_context_linear_stream_checkpoint", games)

    def collect(check_boundary, reports):
        # Explicit synthetic collection records, not teacher/environment runs.
        original = cli.read_training_sources(DATA)
        for cue in ("blue", "red"):
            reports.append(
                dict(
                    status="complete",
                    closed=True,
                    primary_error=None,
                    cleanup_error=None,
                    scripted=True,
                    cue=cue,
                )
            )
            check_boundary()
        return tuple(
            replace(e, episode_id="minigrid-source-G_train-" + cue, context="G_train")
            for e, cue in zip(original[:2], ("blue", "red"), strict=True)
        )

    monkeypatch.setattr(workflow, "_collect_recurring_train_sources", collect)
    return holder


def packets(recurring=False):
    original = cli.read_training_sources(DATA)
    return (
        (
            *original,
            *(
                replace(e, episode_id="minigrid-source-G_train-" + cue, context="G_train")
                for e, cue in zip(original[:2], ("blue", "red"), strict=True)
            ),
        )
        if recurring
        else original
    )


def require_original(holder):
    for stream, source, before, budget, limits_id, limits, birth in holder["streams"]:
        assert source.snapshot_state() == before
        assert stream._runtime.actor.snapshot_state().state == before
        assert stream._budget is budget and id(budget.budget) == limits_id
        assert asdict(budget.budget) == limits and budget.started_at == birth


@pytest.mark.parametrize("mode", ["default", "minimum", "abc", "recurring"])
def test_existing_stream_preserves_choice_across_forks_and_original_paid_receipts(scripted, mode):
    options: dict[str, Any] = {} if mode == "default" else dict(train_update="minimum_change")
    if mode == "abc":
        options.update(retain_arrived=True, abc_catalogue=True)
    if mode == "recurring":
        options.update(retain_arrived=True, recurring_catalogue=True)
    endpoints: list[Any] = []
    r = cli.run_context_linear_stream(
        packets(mode == "recurring"), endpoint_observer=endpoints.append, **options
    )
    assert r["status"] == "complete", r["reason"]
    count, rows = (10, 252) if mode == "recurring" else (6, 108) if mode == "abc" else (6, 54)
    assert len(scripted["fits"]) == count and sum(n for _, n in scripted["fits"]) == rows
    expected = "train_batch" if mode == "default" else "train_batch_minimum_change"
    assert scripted["operations"] == [expected] * count
    assert (endpoints[0].completed_updates, endpoints[0].training_examples) == (count, rows)
    if mode == "default":
        assert "train_update" not in r and "train_update" not in r["protocol"]
    else:
        assert r["train_update"] == r["protocol"]["train_update"] == "minimum_change"
    assert len(r["committed_receipts"]) == r["budget"]["updates_completed"] == count
    for i, paid in enumerate(r["committed_receipts"]):
        assert (
            paid["observed_at"],
            paid["arrived_at"],
            paid["applied_at"],
            paid["update_number"],
        ) == (i, i + 1, i + 1, i + 1)
    require_original(scripted)


@pytest.mark.parametrize(
    "options",
    [
        dict(train_update=True),
        dict(train_update="cold"),
        dict(train_update=1),
        dict(train_update="minimum_change", reference_policy=object()),
        dict(train_update="minimum_change", a4_training=True),
    ],
)
def test_unsupported_selection_is_refused_before_payload_or_owner_work(scripted, options):
    with pytest.raises(ValueError):
        cli.run_context_linear_stream(object(), **options)
    assert scripted["fits"] == [] and scripted["streams"] == []


@pytest.mark.parametrize("role", ["final_test", "probe_train"])
def test_explicit_update_does_not_admit_foreign_roles(scripted, role):
    source = packets()
    foreign = (replace(source[0], role=role), *source[1:])
    with pytest.raises(ValueError):
        cli.run_context_linear_stream(foreign, train_update="minimum_change")
    assert not scripted["fits"]


def test_lost_update_keeps_preceding_receipt_and_original_lifetime_without_retry(scripted):
    def lost(model, x, y):
        if len(scripted["fits"]) == 2:
            raise RuntimeError("lost scripted fit return")

    scripted["effect"] = lost
    r = cli.run_context_linear_stream(packets(), train_update="minimum_change")
    assert r["status"] == "failed" and "lost scripted" in r["reason"]
    assert len(scripted["fits"]) == 2 and len(r["committed_receipts"]) == 1
    assert r["budget"]["updates_completed"] == 1
    require_original(scripted)


def test_changed_original_budget_counter_refuses_next_work_without_reset(scripted, monkeypatch):
    original = AttributionStream.__init__

    def initialize(self, *args, **kwargs):
        observe = kwargs["observe_checkpoint"]

        def changed(n, state, batch):
            observe(n, state, batch)
            if n == 2:
                self._budget.updates_completed += 1

        kwargs["observe_checkpoint"] = changed
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AttributionStream, "__init__", initialize)
    r = cli.run_context_linear_stream(packets(), train_update="minimum_change")
    assert r["status"] == "failed" and len(scripted["fits"]) == 2
    assert scripted["streams"][0][3].updates_completed == 3  # foreign change not reset/adopted
    require_original(scripted)


@pytest.mark.parametrize("bad", ["cold", True])
def test_changed_update_intent_is_refused_before_another_fit(scripted, monkeypatch, bad):
    original = cli._train_update_protocol
    selected = []

    def describe(protocol, update):
        result = original(protocol, update)
        if result.get("definition") and update is not None:
            selected.append(result)
        return result

    monkeypatch.setattr(cli, "_train_update_protocol", describe)

    def boundary():
        if scripted["fits"]:
            selected[-1]["train_update"] = bad

    r = cli.run_context_linear_stream(
        packets(), train_update="minimum_change", check_boundary=boundary
    )
    assert r["status"] == "failed" and "intent" in r["reason"]
    assert len(scripted["fits"]) == 1
    require_original(scripted)


@pytest.mark.parametrize("recurring", [False, True])
@pytest.mark.parametrize("update", [None, "minimum_change"])
def test_first_party_workflow_hands_off_only_actual_scripted_endpoint(scripted, recurring, update):
    r = workflow.run_workflow(recurring_training=recurring, train_update=update)
    assert r["status"] == "complete", r["reason"]
    assert r["games"]["scripted"] and all(not e["success"] for e in r["games"]["episodes"])
    count, rows = (10, 252) if recurring else (6, 108)
    assert len(scripted["fits"]) == count and sum(n for _, n in scripted["fits"]) == rows
    expected = "train_batch_minimum_change" if update else "train_batch"
    assert scripted["operations"] == [expected] * count and scripted["game_calls"] == 1
    if update:
        assert r["protocol"]["train_update"] == r["training"]["train_update"] == update
    else:
        assert "train_update" not in r["protocol"] and "train_update" not in r["training"]
    require_original(scripted)


@pytest.mark.parametrize(
    "flag", ["--context-linear-stream", "--example-affine-play", "--example-affine-recurring"]
)
def test_cli_describes_explicit_choice_without_reading_or_executing(monkeypatch, capsys, flag):
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source reached"))
    monkeypatch.setattr(workflow, "run_workflow", lambda **k: pytest.fail("work reached"))
    monkeypatch.setattr(
        sys, "argv", ["control", flag, "--train-update", "minimum_change", "--describe"]
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["train_update"] == "minimum_change"


@pytest.mark.parametrize(
    "flag",
    [
        "--example-affine-adaptation",
        "--example-affine-a4",
        "--context-linear-fit",
        "--context-linear-reference",
        "--context-linear-catalogue",
    ],
)
def test_cli_refuses_other_profiles_before_io(monkeypatch, flag):
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source reached"))
    argv = ["control", flag] + (["periodic"] if flag == "--context-linear-reference" else [])
    monkeypatch.setattr(sys, "argv", argv + ["--train-update", "minimum_change", "--describe"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "flag", ["--context-linear-stream", "--example-affine-play", "--example-affine-recurring"]
)
def test_cli_forwards_explicit_choice_to_existing_route(monkeypatch, capsys, flag):
    calls = []
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: ())

    def stream(*args, **kwargs):
        calls.append(kwargs)
        return dict(status="complete")

    monkeypatch.setattr(cli, "run_context_linear_stream", stream)
    monkeypatch.setattr(workflow, "run_workflow", stream)
    argv = ["control", flag, "--train-update", "minimum_change"]
    if flag == "--context-linear-stream":
        argv += ["--source", "unused"]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert calls == [
        dict(
            train_update="minimum_change",
            **(dict(recurring_training=True) if flag == "--example-affine-recurring" else {}),
        )
    ]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_cli_keeps_existing_output_protected_before_training(monkeypatch, tmp_path):
    output = tmp_path / "original.json"
    output.write_text("original")
    monkeypatch.setattr(workflow, "run_workflow", lambda **k: pytest.fail("training reached"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "control",
            "--example-affine-play",
            "--train-update",
            "minimum_change",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(FileExistsError):
        cli.main()
    assert output.read_text() == "original"
