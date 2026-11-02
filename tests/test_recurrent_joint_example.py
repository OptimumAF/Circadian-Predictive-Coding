"""Real stream and literal native ports; no genuine model, solver or physics."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import inspect
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import recurrent_joint_example as example
from src.app.coupled_attribution import _same_value
from src.adapters import minigrid_public_cue_recurrent_learner as public
from test_public_cue_recurrent_learner import configure_scripted_native, ScriptedNative
from test_recurrent_readout import nine

WORK = dict(constructors=0, updates=0, rows=0, readouts=0)


def episodes():
    return tuple(nine(cue, context) for context in ("A", "B", "C_train") for cue in ("blue", "red"))


def observed(point):
    roster = [(c, cue) for c in ("A", "B", "C_train") for cue in ("blue", "red")]
    return dict(
        method="recurrent_public_cue",
        role="train",
        updates=point,
        status="complete",
        state_unchanged=True,
        scores=[
            dict(context=c, cue=cue, ce=1.0, accuracy=0.0, exact_sequence_accuracy=0.0)
            for c, cue in roster
        ],
        episodes=[]
        if point == 0
        else [
            dict(
                context=c,
                cue=cue,
                source_role="train",
                status="complete",
                success=False,
                state_unchanged=True,
                environment=dict(closed=True),
                **{"return": 0.0},
            )
            for c, cue in roster
        ],
    )


@pytest.fixture
def app(monkeypatch):
    configure_scripted_native(monkeypatch)
    init, train = ScriptedNative.__init__, ScriptedNative.train_batch

    def counted_init(self, *args):
        WORK["constructors"] += 1
        init(self, *args)

    def counted_train(self, x, y):
        WORK["updates"] += 1
        WORK["rows"] += len(x)
        return train(self, x, y)

    monkeypatch.setattr(ScriptedNative, "__init__", counted_init)
    monkeypatch.setattr(ScriptedNative, "train_batch", counted_train)
    monkeypatch.setattr(example, "_same_state", _same_value)
    monkeypatch.setattr(example, "_require_minigrid", lambda: None)

    class Rng:
        def clone(self):
            return self

    rng = Rng()
    monkeypatch.setattr(
        example,
        "require_torch",
        lambda: SimpleNamespace(get_rng_state=lambda: rng, equal=lambda a, b: a is b),
    )
    monkeypatch.setattr(example, "_episodes", episodes)

    def evaluate(source, state, *, updates, check_boundary, **kwargs):
        WORK["readouts"] += 1
        check_boundary()
        # Existing evaluator requires a cold source, separate from its scorer.
        assert (source.completed_updates, source.training_examples) == (0, 0)
        assert state.completed_updates == updates
        assert kwargs["recurring_training"] is True and len(kwargs["sources"]) == 6
        return observed(updates)

    monkeypatch.setattr(example, "evaluate_public_cue_recurrent_checkpoint", evaluate)
    return evaluate


def test_describe_and_public_cli_are_metadata_only(monkeypatch, capsys):
    monkeypatch.setattr(example, "_episodes", lambda: pytest.fail("payload read"))
    monkeypatch.setattr(sys, "argv", ["joint", "--describe"])
    example.main()
    assert "joint_train_recurrent_bc128_v1" in capsys.readouterr().out
    assert example.describe()["presentations"] == dict(A=44, B=42, C_train=42)


def test_joint_recipe_preserves_raw_complete_episodes_unique_ids_and_fixed_exposures():
    source = episodes()
    original = deepcopy(source)
    sources, packets = example.joint_packets(source)
    assert source == original and len(sources) == 6 and len(packets) == 128
    assert Counter(p.context for p in packets) == dict(A=44, B=42, C_train=42)
    assert len({row for p in packets for row in p.row_ids}) == 1152
    for i, packet in enumerate(packets):
        assert packet.episode.episode_id == f"joint-rnn-arrival-{i}"
        assert _same_value(
            (packet.features, packet.targets), (sources[i % 6].features, sources[i % 6].targets)
        )


@pytest.mark.parametrize("fault", ["probe_train", "final_test", "order", "partial"])
def test_bad_joint_inputs_refused_before_native_admission(monkeypatch, fault):
    data = list(episodes())
    if fault in ("probe_train", "final_test"):
        data[0] = replace(data[0], role=fault)
    elif fault == "order":
        data.reverse()
    else:
        data.pop()
    monkeypatch.setattr(example, "_require_minigrid", lambda: pytest.fail("native admission"))
    result = example.run_example(tuple(data))
    assert result["status"] == "incomplete" and result["reason"]
    assert result["training_budget"] is None


def test_real_stream_completes128_scripted_updates_and_retains_negative_outcomes(app):
    result = example.run_example()
    assert result["status"] == "complete", result["reason"]
    assert result["training"]["completed_updates"] == 128
    assert result["training"]["training_examples"] == 1152
    assert [r["updates"] for r in result["readouts"]] == [0, 128]
    assert all(e["success"] is False for e in result["readouts"][1]["games"]["episodes"])
    assert all(result["preservation"].values())
    assert result["training_budget"]["updates"] == 128 and result["readout_budget"]["updates"] == 0
    assert [r["applied_at"] for r in result["training"]["applied_updates"]] == list(range(1, 129))


def test_replaced_prior_success_refuses_second_training_update(app, monkeypatch):
    previous = ScriptedNative.train_batch
    captured: dict[str, Any] = {}

    def evaluate(*args, **kwargs):
        captured.update(inspect.getclosurevars(kwargs["check_boundary"]).nonlocals)
        return app(*args, **kwargs)

    def corrupt(self, x, y):
        result = previous(self, x, y)
        evidence = inspect.getclosurevars(captured["check_pure"]).nonlocals["check_evidence"]
        outer = inspect.getclosurevars(evidence).nonlocals["report"]
        outer["readouts"][0] = deepcopy(outer["readouts"][0])
        return result

    monkeypatch.setattr(example, "evaluate_public_cue_recurrent_checkpoint", evaluate)
    monkeypatch.setattr(ScriptedNative, "train_batch", corrupt)
    before = WORK["updates"]
    result = example.run_example()
    assert result["status"] == "incomplete" and "readout changed" in result["reason"]
    assert WORK["updates"] == before + 1 and result["training_budget"]["updates"] == 1


def test_post_training_failure_keeps128_receipts_charges_and_first_readout(app, monkeypatch):
    def fail(*args, **kwargs):
        if kwargs["updates"] == 128:
            raise ValueError("scripted final readout failure")
        return app(*args, **kwargs)

    monkeypatch.setattr(example, "evaluate_public_cue_recurrent_checkpoint", fail)
    result = example.run_example()
    assert result["status"] == "incomplete" and len(result["training"]["applied_updates"]) == 128
    assert len(result["readouts"]) == 1 and result["training_budget"]["updates"] == 128


@pytest.mark.parametrize("fault", ["closed", "proof", "role", "nan", "omitted"])
def test_unqualified_native_outcome_never_becomes_complete(fault):
    result = observed(128)
    first = result["episodes"][0]
    if fault == "closed":
        first["environment"]["closed"] = False
    elif fault == "proof":
        first["state_unchanged"] = False
    elif fault == "role":
        first["source_role"] = "final_test"
    elif fault == "nan":
        first["return"] = float("nan")
    else:
        result["episodes"].pop()
    with pytest.raises(ValueError):
        example._qualify(128, result)


def test_existing_paid_output_refuses_before_run(monkeypatch, tmp_path):
    output = tmp_path / "prior.json"
    output.write_text("paid")
    monkeypatch.setattr(example, "run_example", lambda: pytest.fail("run admitted"))
    monkeypatch.setattr(sys, "argv", ["joint", "--output", str(output)])
    with pytest.raises(FileExistsError):
        example.main()
    assert output.read_text() == "paid"


def test_mutated_permanent_charge_refuses_before_training(app, monkeypatch):
    def changed(*args, **kwargs):
        pure = inspect.getclosurevars(kwargs["check_boundary"]).nonlocals["check_pure"]
        budget = inspect.getclosurevars(pure).nonlocals["train_budget"]
        budget.updates_completed = 1
        return observed(kwargs["updates"])

    monkeypatch.setattr(example, "evaluate_public_cue_recurrent_checkpoint", changed)
    before = WORK["updates"]
    result = example.run_example()
    assert result["status"] == "incomplete" and "permanent charge" in result["reason"]
    assert WORK["updates"] == before and result["training_budget"]["updates"] == 1


def test_unchanged_original_recurrent_policy_and_readout_example_metadata():
    from scripts.recurrent_readout_example import describe

    policy = public.PublicCueRecurrentPolicy().native_policy
    assert (policy.hidden_dim, policy.seed, policy.learning_rate, policy.momentum) == (
        8,
        17,
        0.03,
        0.9,
    )
    assert describe()["definition"] == "recurrent_readout_train_A_native_v1"


def test_missing_output_refuses_without_run(monkeypatch):
    monkeypatch.setattr(example, "run_example", lambda: pytest.fail("run admitted"))
    monkeypatch.setattr(sys, "argv", ["joint"])
    with pytest.raises(SystemExit) as failure:
        example.main()
    assert failure.value.code == 2


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_unqualified_prefit_metric_refuses_all_training(app, monkeypatch, bad):
    def invalid(*args, **kwargs):
        record = app(*args, **kwargs)
        record["scores"][0]["ce"] = bad
        return record

    monkeypatch.setattr(example, "evaluate_public_cue_recurrent_checkpoint", invalid)
    before = WORK["updates"]
    result = example.run_example()
    assert result["status"] == "incomplete"
    assert WORK["updates"] == before and result["training_budget"]["updates"] == 0
