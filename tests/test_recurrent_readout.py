"""Scripted boundaries only; no solver, native feature/model or physics work."""

from copy import deepcopy
from dataclasses import replace
from time import monotonic
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_recurrent_readout as readout
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_learning_pilot import _batch
from src.app.coupled_attribution import _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.learner_ports import TrainingUpdateFailed
from test_public_cue_recurrent_learner import configure_scripted_native, data, episode


@pytest.fixture
def scripted(monkeypatch):
    """Canonical containers exercise real ownership pins, not native arithmetic."""
    configure_scripted_native(monkeypatch)
    native = readout.public.native

    class ColdNative:
        constructed = 0

        def __init__(self, policy):
            marker = ColdNative.constructed
            ColdNative.constructed += 1
            shapes = native._shapes(policy)
            self.state = native.TorchRecurrentSnapshot(
                1,
                policy,
                {k: np.full(shape, marker, dtype=np.float32) for k, shape in shapes.items()},
                {
                    "state": {},
                    "param_groups": [dict(native._sgd_options(policy), params=list(range(5)))],
                },
                {k: None for k in shapes},
                0,
                0,
            )

        def snapshot_state(self):
            return deepcopy(self.state)

    monkeypatch.setattr(native, "TorchRecurrentLearner", ColdNative)
    monkeypatch.setattr(readout, "_same_state", _same_value)
    monkeypatch.setattr(evaluation, "_same_state", _same_value)
    calls = SimpleNamespace(
        features=[],
        fits=[],
        predictions=[],
        feature_override=None,
        fit_override=None,
        predict_override=None,
    )

    def features(state, raw):
        calls.features.append((deepcopy(state), raw.copy()))
        return np.zeros((len(raw), 8), dtype=np.float64)

    def fit(hidden, targets, classes):
        calls.fits.append((hidden.copy(), targets.copy(), classes))
        return CategoricalLinearFit(
            8,
            7,
            len(hidden),
            tuple((0.0,) * 7 for _ in range(9)),
            0,
            (0.0,) * min(len(hidden), 9),
            0.25,
        )

    def predict(model, hidden):
        calls.predictions.append((model, hidden.copy()))
        row = np.array([0.07 + 1e-12, 0.11, 0.13, 0.17, 0.19, 0.23, 0.1 - 1e-12], dtype=np.float64)
        return np.tile(row, (len(hidden), 1))

    calls.feature_port, calls.fit_port, calls.predict_port = features, fit, predict
    monkeypatch.setattr(
        readout, "read_native_activations", lambda *a: (calls.feature_override or features)(*a)
    )
    monkeypatch.setattr(
        readout, "fit_categorical_linear", lambda *a: (calls.fit_override or fit)(*a)
    )
    monkeypatch.setattr(
        readout, "predict_categorical_linear", lambda *a: (calls.predict_override or predict)(*a)
    )
    return calls


def test_returned_head_mutation_preserves_known_native_failure_prefix(scripted):
    owner = readout.RecurrentReadoutLearner()
    before = owner.snapshot_state()

    def mutate(state, raw):
        result = scripted.feature_port(state, raw)
        state.parameters["weight_input"][0, 0] += 1
        return result

    scripted.feature_override = mutate
    with pytest.raises(TrainingUpdateFailed):
        owner.train_episode(episode())
    failed = owner.snapshot_state()
    assert _same_value(failed.native, before.native)
    assert failed.failure is not None
    assert failed.failure.feature_calls_returned == 1
    assert failed.failure.fit_calls_attempted == 0


def test_session_snapshot_copy_cannot_change_prior_owner_and_return_proof(scripted, monkeypatch):
    session = fitted().start_episode()
    original = readout.copy_snapshot

    def changed(state):
        result = original(state)
        if state is session._state:
            object.__setattr__(state, "readouts", tuple(list(state.readouts)))
        return result

    monkeypatch.setattr(readout, "copy_snapshot", changed)
    with pytest.raises(ValueError):
        session.snapshot_state()


def test_old_generic_admission_refuses_new_readout_name_before_any_state_work():
    with pytest.raises(ValueError, match="unsupported"):
        evaluation._admit(readout.METHOD, 0, None, "train", None, lambda: None)


def fitted():
    owner = readout.RecurrentReadoutLearner()
    owner.train_episode(episode("blue"))
    owner.train_episode(episode("red"))
    return owner


def nine(cue, context="A"):
    original = episode(cue)
    return replace(
        original,
        episode_id=f"minigrid-source-{context}-{cue}",
        context=context,
        transitions=tuple(
            replace(original.transitions[min(i, 1)], step=i, terminal=i == 8) for i in range(9)
        ),
    )


def budget(updates=0):
    return ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=updates, max_replay_examples=0, max_wall_seconds=30
        ),
        monotonic,
    )


def test_routes_by_public_pixels_and_owns_source_sibling_restore_and_other_head(scripted):
    owner = readout.RecurrentReadoutLearner()
    before = owner.snapshot_state()
    diagnostic = owner.train_episode(replace(episode("blue"), context="misleading-red"))
    one = owner.snapshot_state()
    assert diagnostic.definition == readout.DEFINITION and diagnostic.value == 0.25
    assert one.fit_updates == (1, 0) and one.fit_rows == (2, 0)
    assert one.readouts[1] is None and _same_value(one.native, before.native)
    sibling = owner.fork()
    owner.train_episode(replace(episode("red"), context="misleading-blue"))
    assert owner.completed_updates == 2 and owner.training_examples == 4
    assert sibling.snapshot_state().fit_updates == (1, 0)
    sibling.restore_state(owner.snapshot_state())
    assert _same_value(sibling.snapshot_state(), owner.snapshot_state())
    one.native.heads[0].parameters["weight_input"][0, 0] = 99
    assert owner.snapshot_state().native.heads[0].parameters["weight_input"][0, 0] != 99
    assert [float(s.parameters["weight_input"][0, 0]) for s, _ in scripted.features] == [0, 1]
    assert all(
        h.completed_updates == h.training_examples == 0 for h in owner.snapshot_state().native.heads
    )


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_refuses_nontrain_episode_before_feature_or_fit_work(scripted, role):
    owner = readout.RecurrentReadoutLearner()
    with pytest.raises(ValueError):
        owner.train_episode(replace(episode(), role=role))
    assert not scripted.features and not scripted.fits
    assert owner.snapshot_state().failure is None


@pytest.mark.parametrize("fault", ["dtype", "no_cue", "mixed_cue", "targets"])
def test_refuses_bad_raw_or_labels_before_feature_work(scripted, fault):
    owner = readout.RecurrentReadoutLearner()
    x, y = data()
    if fault == "dtype":
        x = x.astype(np.float32)
    elif fault == "no_cue":
        x[0] = -1
    elif fault == "mixed_cue":
        x[1] = data("red")[0][0]
    else:
        y[0, 0] = 7
    with pytest.raises(ValueError):
        owner.train_batch(x, y)
    assert not scripted.features and not scripted.fits
    assert owner.snapshot_state().failure is None


@pytest.mark.parametrize(
    "fault", ["value", "equal_parameter", "equal_container", "equal_fit", "clocks"]
)
def test_refuses_corrupted_or_equal_replaced_prior_state_before_more_work(scripted, fault):
    owner = readout.RecurrentReadoutLearner()
    owner.train_episode(episode())
    head = owner._state.native.heads[0]
    if fault == "value":
        head.parameters["weight_input"][0, 0] += 1
    elif fault == "equal_parameter":
        head.parameters["weight_input"] = head.parameters["weight_input"].copy()
    elif fault == "equal_container":
        object.__setattr__(head, "parameters", dict(head.parameters))
    elif fault == "equal_fit":
        fitted_readout = owner._state.readouts[0]
        assert fitted_readout is not None
        object.__setattr__(owner._state, "readouts", (replace(fitted_readout), None))
    else:
        object.__setattr__(owner._state, "fit_rows", (3, 0))
    counts = (len(scripted.features), len(scripted.fits))
    with pytest.raises(ValueError):
        owner.train_episode(episode("red"))
    assert (len(scripted.features), len(scripted.fits)) == counts


@pytest.mark.parametrize("phase", ["features", "fit"])
def test_keeps_lost_return_charges_and_permanently_refuses_restore_or_continue(
    scripted, monkeypatch, phase
):
    owner = readout.RecurrentReadoutLearner()
    before = owner.snapshot_state()

    def lost(*args):
        raise RuntimeError("lost return")

    if phase == "features":
        scripted.feature_override = lost
    else:
        scripted.fit_override = lost
    with pytest.raises(TrainingUpdateFailed) as failure:
        owner.train_episode(episode())
    work = failure.value.work
    assert work["feature_calls_attempted"] == 1
    assert work["feature_calls_returned"] == int(phase == "fit")
    assert work["fit_calls_attempted"] == int(phase == "fit")
    assert work["fit_calls_returned"] == 0
    failed = owner.snapshot_state()
    assert failed.failure is not None and failed.completed_updates == 0
    assert _same_value(failed.native, before.native)
    for operation in (
        lambda: owner.restore_state(before),
        owner.fork,
        lambda: owner.train_episode(episode()),
        lambda: owner.predict(data()[0]),
    ):
        with pytest.raises(TrainingUpdateFailed):
            operation()


@pytest.mark.parametrize("fault", ["raw", "targets", "hidden", "fit_geometry"])
def test_detects_returned_callback_mutation_and_retains_known_paid_prefix(
    scripted, monkeypatch, fault
):
    owner = readout.RecurrentReadoutLearner()
    if fault == "raw":

        def mutate_features(state, raw):
            result = scripted.feature_port(state, raw)
            raw[0, 2] = -0.5
            return result

        scripted.feature_override = mutate_features
    else:

        def mutate_fit(hidden, targets, classes):
            result = scripted.fit_port(hidden, targets, classes)
            if fault == "targets":
                targets[0, 0] = 6
            elif fault == "hidden":
                hidden[0, 0] = 0.5
            else:
                result = replace(result, fit_rows=1, singular_values=(0.0,))
            return result

        scripted.fit_override = mutate_fit
    with pytest.raises(TrainingUpdateFailed) as failure:
        owner.train_episode(episode())
    assert failure.value.work["feature_calls_returned"] == 1
    assert failure.value.work["fit_calls_returned"] == int(fault != "raw")
    assert owner.snapshot_state().failure is not None
    with pytest.raises(TrainingUpdateFailed):
        owner.train_episode(episode("red"))


def test_batch_prediction_keeps_all_seven_float64_outputs_and_detaches_results(scripted):
    owner = readout.RecurrentReadoutLearner()
    with pytest.raises(ValueError):
        owner.predict(data()[0])
    with pytest.raises(ValueError):
        owner.start_episode()
    owner.train_episode(episode())
    owner.train_episode(episode("red"))
    before = owner.snapshot_state()
    x = data()[0]
    result = owner.predict(x)
    assert result.shape == (2, 7) and result.dtype == np.float64
    assert result[0, 0] == 0.07 + 1e-12
    assert int(np.argmax(result[0])) == 5  # No action mask or teacher fallback.
    result[:] = 0
    assert owner.predict(x)[0, 5] == 0.23
    assert _same_value(before, owner.snapshot_state())


def test_sessions_route_once_reset_and_charge_every_recomputed_prefix(scripted):
    owner = fitted()
    before = owner.snapshot_state()
    calls = len(scripted.features)
    blue, red = owner.start_episode(), owner.start_episode()
    bx, rx = data()[0], data("red")[0]
    blue.predict(bx[:1])
    red.predict(rx[:1])
    blue.predict(bx[1:])
    assert (blue._selected, red._selected) == (0, 1)
    assert (blue.completed_steps, red.completed_steps) == (2, 1)
    assert [len(x) for _, x in scripted.features[calls:]] == [1, 1, 2]
    assert (blue.feature_rows_attempted, blue.feature_rows_returned) == (3, 3)
    assert (red.feature_rows_attempted, red.feature_rows_returned) == (1, 1)
    for _ in range(14):
        blue.predict(bx[1:])
    assert blue.completed_steps == 16
    assert blue.feature_rows_attempted == blue.feature_rows_returned == 136
    with pytest.raises(ValueError):
        blue.predict(bx[1:])
    assert owner.start_episode().completed_steps == 0
    assert _same_value(owner.snapshot_state(), before)


@pytest.mark.parametrize(
    "fault", ["lost_feature", "returned_prefix_mutation", "returned_hidden_mutation"]
)
def test_paid_session_failure_is_terminal_and_retains_returned_feature_costs(
    scripted, monkeypatch, fault
):
    owner = fitted()
    session = owner.start_episode()
    if fault == "returned_hidden_mutation":

        def predict(model, hidden):
            result = scripted.predict_port(model, hidden)
            hidden[0, 0] = 0.5
            return result

        scripted.predict_override = predict
    else:

        def features(state, raw):
            if fault == "lost_feature":
                raise RuntimeError("lost feature return")
            result = scripted.feature_port(state, raw)
            raw[0, 2] = -0.5
            return result

        scripted.feature_override = features
    with pytest.raises((ValueError, RuntimeError)):
        session.predict(data()[0][:1])
    assert session.completed_steps == 0 and session.feature_rows_attempted == 1
    assert session.feature_rows_returned == int(fault != "lost_feature")
    counts = (len(scripted.features), len(scripted.predictions))
    for operation in (session.snapshot_state, lambda: session.predict(data()[0][:1])):
        with pytest.raises(ValueError):
            operation()
    assert (len(scripted.features), len(scripted.predictions)) == counts


def facade_setup():
    owner = readout.RecurrentReadoutLearner()
    sources = tuple(_batch(nine(cue)) for cue in ("blue", "red"))
    for packet in sources:
        assert packet.episode is not None
        owner.train_episode(packet.episode)
    return owner, owner.snapshot_state(), sources


@pytest.mark.parametrize("fault", ["checkpoint", "budget", "sources"])
def test_facade_refuses_broad_checkpoint_or_readout_authority(scripted, monkeypatch, fault):
    owner, state, sources = facade_setup()
    limits = budget()
    if fault == "checkpoint":
        state = readout.RecurrentReadoutLearner().snapshot_state()
    elif fault == "budget":
        limits = budget(1)
    else:
        sources = list(sources)
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("loop admitted"))
    with pytest.raises(ValueError):
        evaluation.evaluate_recurrent_readout_checkpoint(
            owner, state, sources=sources, budget=limits, check_boundary=lambda: None
        )


def test_facade_passes_exact_custom_copy_equality_hooks_and_refuses_equal_replacement(
    scripted, monkeypatch
):
    owner, state, sources = facade_setup()
    limits, checked = budget(), []

    def loop(actual, actual_state, **options):
        assert actual is owner and actual_state is state
        assert options["method"] == readout.METHOD
        assert options["updates"] == 2 and options["native_training_examples"] == 18
        assert options["sources"] is sources and options["public_history"] is False
        assert options["memory_copier"] is readout.copy_snapshot
        assert options["state_equal"] is _same_value
        assert [x[1:] for x in options["live_plan"]] == [("blue", "train"), ("red", "train")]
        copied = options["memory_copier"](state)
        assert _same_value(copied, state) and copied is not state
        options["extra_check"]()
        object.__setattr__(state, "readouts", tuple(list(state.readouts)))
        with pytest.raises(ValueError, match="replaced"):
            options["extra_check"]()
        return {"status": "scripted", "metadata": options["return_metadata"]}

    monkeypatch.setattr(evaluation, "_run_checkpoint", loop)
    result = evaluation.evaluate_recurrent_readout_checkpoint(
        owner, state, sources=sources, budget=limits, check_boundary=lambda: checked.append(True)
    )
    assert result["status"] == "scripted" and checked
    assert result["metadata"]["native_recurrent_updates"] == 0
    assert limits.updates_completed == limits.replay_examples_completed == 0


@pytest.mark.parametrize("replace_training", [False, True])
def test_firstparty_composes_actual_stream_then_scripted_games_and_guards_paid_prefix(
    scripted, monkeypatch, replace_training
):
    from scripts import recurrent_readout_example as example

    assert example.describe()["final_available"] is example.describe()["probe_available"] is False
    assert not scripted.features and not scripted.fits
    monkeypatch.setattr(example, "_require_minigrid", lambda: None)
    monkeypatch.setattr(
        example,
        "_episodes",
        lambda: tuple(
            nine(cue, context) for context in ("A", "B", "C_train") for cue in ("blue", "red")
        ),
    )
    monkeypatch.setattr(example, "_same_state", _same_value)

    class Rng:
        def clone(self):
            return self

    rng = Rng()
    monkeypatch.setattr(
        example,
        "require_torch",
        lambda: SimpleNamespace(get_rng_state=lambda: rng, equal=lambda a, b: a is b),
    )
    seen = []

    def games(reader, state, **options):
        seen.append(state)
        assert state.fit_updates == (1, 1) and state.fit_rows == (9, 9)
        assert reader.completed_updates == 2
        assert state.native.completed_updates == state.native.training_examples == 0
        assert (
            options["budget"].budget.max_training_updates
            == options["budget"].budget.max_replay_examples
            == 0
        )
        options["check_boundary"]()
        if replace_training:
            report = next(
                cell.cell_contents
                for cell in options["check_boundary"].__closure__
                if type(cell.cell_contents) is dict and "protocol" in cell.cell_contents
            )
            report["training"] = deepcopy(report["training"])
            options["check_boundary"]()
        return {"status": "scripted", "episodes": []}

    monkeypatch.setattr(example, "evaluate_recurrent_readout_checkpoint", games)
    result = example.run_example()
    assert seen and result["training"]["completed_updates"] == 2
    assert len(result["training"]["applied_updates"]) == 2
    assert result["training_budget"]["updates"] == 2
    assert result["readout_budget"]["updates"] == result["readout_budget"]["replay"] == 0
    if replace_training:
        assert result["status"] == "incomplete" and "replaced" in result["reason"]
    else:
        assert result["status"] == "complete" and all(result["preservation"].values())


@pytest.fixture
def fake_loop(scripted, monkeypatch):
    from test_public_cue_probe_games import CueEnvironment
    from test_minigrid_learning_pilot import Environment

    work: dict[str, Any] = {}
    monkeypatch.setattr(Environment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", CueEnvironment)
    return work


def test_actual_generic_loop_uses_frozen_readout_sessions_and_closes_fake_environments(
    scripted, fake_loop
):
    work = fake_loop
    owner, state, sources = facade_setup()
    result = evaluation.evaluate_recurrent_readout_checkpoint(
        owner, state, sources=sources, budget=budget(), check_boundary=lambda: None
    )
    assert result["status"] == "complete" and result["state_unchanged"] is True
    assert len(result["scores"]) == len(result["episodes"]) == 2
    assert [e["cue"] for e in result["episodes"]] == ["blue", "red"]
    assert all(e["status"] == "complete" and e["environment"]["closed"] for e in result["episodes"])
    assert all(len(e["predictions"]) == 2 for e in result["episodes"])
    assert len(work["environments"]) == 2 and len(work["steps"]) == 4
    assert all(e.closed for e in work["environments"])
    assert owner.completed_updates == 2 and owner.training_examples == 18


@pytest.mark.parametrize("fault", ["columns", "dtype"])
def test_bad_prediction_result_refuses_and_closes_session(scripted, monkeypatch, fault):
    owner = fitted()
    session = owner.start_episode()
    bad = np.full(
        (1, 6 if fault == "columns" else 7),
        1 / (6 if fault == "columns" else 7),
        dtype=np.float32 if fault == "dtype" else np.float64,
    )
    scripted.predict_override = lambda *a: bad
    with pytest.raises(ValueError):
        session.predict(data()[0][:1])
    assert session.feature_rows_attempted == session.feature_rows_returned == 1
    with pytest.raises(ValueError):
        session.predict(data()[0][:1])
