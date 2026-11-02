"""Actual ordinary lifecycle/controller/guards; explicitly scripted source fits."""

from dataclasses import replace
import json

import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import shared_linear_publication as publication
from src.adapters.linear_fit_learner import LinearFitLearner
from src.adapters.linear_fit_serving import LinearFitServingPredictor
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.serving_promotion import ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.ticket_workflow import workflow_request
import test_shared_joint_train as shared

OPTIONS = dict(
    method="linear_fit",
    context_return=True,
    train_update="minimum_change",
    replay="coverage",
    replay_joint="joint",
    shared_serving=True,
)
SCRIPTED_EPISODES = 0
SCRIPTED_EPISODE_RETURNS = 0


def fixture(monkeypatch, *, accept=True):
    holder = shared.scripted(monkeypatch)
    original_init = publication.SharedLinearPublication.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        holder["publication"] = self

    monkeypatch.setattr(publication.SharedLinearPublication, "__init__", initialize)

    # Numeric guard utility is scripted, but metadata/state/copy/controller paths
    # are real. Genuine guard mathematics is separately accepted in increment A.
    def measure(self, model, guard, policy, last):
        return (float(model.snapshot_state().model is not None) if accept else 0.0, True, True, 0.0)

    monkeypatch.setattr(PromotionGuardEvaluator, "_measure", measure)

    def episode(state, predict, *, capabilities, scenario, **kwargs):
        global SCRIPTED_EPISODES, SCRIPTED_EPISODE_RETURNS
        SCRIPTED_EPISODES += 1
        predict(workflow_request(state, capabilities, "scripted", include_policy_context=True))
        SCRIPTED_EPISODE_RETURNS += 1
        return dict(
            complete=True,
            success=False,
            scenario=scenario,
            training_updates=0,
            steps=[],
            work={"prediction_calls_validated": 1},
            stopped_reason="scripted_false",
        )

    monkeypatch.setattr(publication, "run_workflow_episode", episode)
    return holder


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_ordinary_stream_publishes_shared_candidate_and_restores_exact_source(monkeypatch, control):
    holder = fixture(monkeypatch)
    r = entry.run_example(**{**OPTIONS, "replay_joint": control})
    assert r["complete"] is True, r.get("error")
    row = r["shared_serving"]
    session = holder["publication"]
    assert row["complete"] and row["status"] == "published_and_rolled_back"
    assert row["fit_admissions"] == row["fit_returns"] == 16
    assert row["commit_returns"] == row["rollback_returns"] == 1
    assert row["promotion_receipt"] == dict(generation=1, previous_version="a0", actor_version="c0")
    assert row["sharing"]["admitted_updates"] == 16 and row["sharing"]["paused"]
    assert row["requests"] == row["request_returns"] == 16 and len(row["episodes"]) == 16
    assert all(e["episode"]["success"] is False for e in row["episodes"])
    assert session.actor.serving_snapshot().generation == 2
    assert session.actor.snapshot_state().state == holder["source"].snapshot_state()
    assert r["source_unchanged"] and r["actor_unchanged"]
    assert holder["wake"].started_at == holder["birth"] and holder["wake"].updates_completed == 12
    assert r["replay_examples"] == 48 and len(r["published_replay_receipts"]) == 4
    assert r["final_examples"] == r["probe_updates"] == 0


def test_measured_rejection_is_complete_and_does_not_issue_or_renew_publication(monkeypatch):
    holder = fixture(monkeypatch, accept=False)
    r = entry.run_example(**OPTIONS)
    assert r["complete"], r.get("error")
    row = r["shared_serving"]
    assert row["status"] == "rejected" and row["complete"]
    assert row["commit_attempts"] == row["requests"] == row["rollback_attempts"] == 0
    assert holder["publication"].actor.serving_snapshot().generation == 0
    assert holder["wake"].updates_completed == 12 and r["replay_examples"] == 48


@pytest.mark.parametrize(
    "fault",
    [
        "row",
        "gate",
        "budget",
        "replay_budget",
        "source",
        "prior_episode",
        "spec",
        "actor_metadata",
        "gate_type",
    ],
)
def test_refuses_changed_or_replaced_paid_source_before_guard_or_publication(monkeypatch, fault):
    holder = fixture(monkeypatch)
    original = publication.SharedLinearPublication.run

    def corrupt(self, stream, require_source, context_guard):
        if fault == "row":
            holder["report"]["shared_serving"] = dict(self.row)
        elif fault == "gate":
            self.gate._admitted += 1
        elif fault == "gate_type":
            self.gate._serving = False
        elif fault == "budget":
            stream._budget.updates_completed = 0
        elif fault == "replay_budget":
            stream._rehearsal._replay_budget.replay_examples_completed = 0
        elif fault == "source":
            stream._learner = LinearFitLearner(holder["source"].policy)
        elif fault == "spec":
            object.__setattr__(
                self.spec, "vocabulary", tuple("x" + w for w in self.spec.vocabulary)
            )
        elif fault == "actor_metadata":
            self.actor._slot.bundle.metadata["mode"] = "foreign"
        else:
            holder["report"]["episodes"][0]["episode"]["success"] = True
        return original(self, stream, require_source, context_guard)

    monkeypatch.setattr(publication.SharedLinearPublication, "run", corrupt)
    r = entry.run_example(**OPTIONS)
    assert not r["complete"]
    paid = r["shared_serving_preserved_work"]
    assert paid["fit_returns"] == 16 and paid["guard_attempts"] == paid["commit_attempts"] == 0
    assert r["training_work"]["attempted_rows"] == 120
    assert holder["publication"].actor.serving_snapshot().generation == 0


@pytest.mark.parametrize("fault", ["lost", "candidate", "clock", "guard_role"])
def test_real_guard_read_refuses_loss_or_changed_state_without_more_work(monkeypatch, fault):
    holder = fixture(monkeypatch)
    # Reenable actual native predictions for this boundary fault, no LS fitting.
    # Restore the ordinary measurement method saved from the class definition.
    monkeypatch.setattr(PromotionGuardEvaluator, "_measure", REAL_MEASURE)
    original = LinearFitLearner.predict

    def predict(self, features):
        result = original(self, features)
        if fault == "lost":
            raise RuntimeError("lost guard return")
        if fault == "candidate":
            owner = holder["stream"]._runtime._candidate
            owner._state = replace(owner.snapshot_state(), training_examples=121)
        elif fault == "clock":
            holder["wake"].last_clock -= 1
        return result

    if fault == "guard_role":
        run = publication.SharedLinearPublication.run

        def foreign(self, *args):
            self.guards = (replace(self.guards[0], role="final_test"), self.guards[1])
            return run(self, *args)

        monkeypatch.setattr(publication.SharedLinearPublication, "run", foreign)
    else:
        monkeypatch.setattr(LinearFitLearner, "predict", predict)
    r = entry.run_example(**OPTIONS)
    assert not r["complete"]
    row = r["shared_serving_preserved_work"]
    assert row["guard_attempts"] == (0 if fault == "guard_role" else 1), r.get("error")
    assert row["guard_returns"] == (0 if fault in ("lost", "guard_role") else 1)
    assert row["commit_attempts"] == 0 and row["fit_returns"] == 16


REAL_MEASURE = PromotionGuardEvaluator._measure


def test_real_guard_completes_all_paid_reads_and_preserves_measured_rejection(monkeypatch):
    holder = fixture(monkeypatch)
    monkeypatch.setattr(PromotionGuardEvaluator, "_measure", REAL_MEASURE)
    r = entry.run_example(**OPTIONS)
    assert r["complete"], r.get("error")
    row = r["shared_serving"]
    assert row["complete"] and row["status"] == "rejected"
    assert row["guard_attempts"] == row["guard_returns"] == 4 and row["guard_rows"] == 24
    assert (
        row["commit_attempts"] == 0
        and holder["publication"].actor.serving_snapshot().generation == 0
    )


@pytest.mark.parametrize(
    "port", ["commit", "rollback", "read", "commit_receipt", "rollback_receipt"]
)
def test_lost_opaque_return_preserves_paid_work_and_never_adopts_unobserved_actor(
    monkeypatch, port
):
    holder = fixture(monkeypatch)
    cls = LinearFitServingPredictor if port == "read" else ServingPromotionController
    name = "serve" if port == "read" else port.split("_")[0]
    original = getattr(cls, name)

    def lost(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if port == "commit_receipt":
            object.__setattr__(result, "generation", True)
            return result
        if port == "rollback_receipt":
            return float(result)
        raise RuntimeError("lost " + port)

    monkeypatch.setattr(cls, name, lost)
    r = entry.run_example(**OPTIONS)
    assert not r["complete"]
    row = r["shared_serving_preserved_work"]
    session = holder["publication"]
    assert row["fit_returns"] == 16 and holder["wake"].updates_completed == 12
    if port in ("commit", "commit_receipt"):
        assert row["commit_unknown"] and row["commit_returns"] == row["requests"] == 0
        assert session.expected_actor.actor_version == "a0"
        assert session.actor.version == "c0"
    elif port in ("rollback", "rollback_receipt"):
        assert (
            row["rollback_unknown"]
            and row["rollback_returns"] == 0
            and row["request_returns"] == 12
        )
        assert session.expected_actor.actor_version == "c0" and session.actor.version == "a0"
    else:
        assert (
            row["requests"] == 1 and row["request_returns"] == 0 and row["rollback_attempts"] == 0
        )
        assert row["typed_work"]["decision_returns"] == 1
    assert not row["complete"] and r["actor_unchanged"] is None


@pytest.mark.parametrize(
    "options",
    [
        dict(shared_serving=1),
        dict(replay="none"),
        dict(method="bp"),
        dict(reserved_probe=True),
        dict(separate_context_heads=True),
        dict(train_update=None),
    ],
)
def test_foreign_mode_refuses_before_data(monkeypatch, options):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("data opened"))
    with pytest.raises(ValueError):
        entry.run_example(**{**OPTIONS, **options})


def test_cli_exposes_only_explicit_shared_mode_and_refuses_existing_output(monkeypatch, tmp_path):
    fixture(monkeypatch)
    output = tmp_path / "shared.json"
    args = [
        "--method",
        "linear_fit",
        "--context-return",
        "--train-update",
        "minimum_change",
        "--replay",
        "coverage",
        "--replay-joint",
        "joint",
        "--shared-serving",
        "--output",
        str(output),
    ]
    assert entry.main(args) == 0
    r = json.loads(output.read_text())
    assert r["shared_serving"]["complete"]
    with pytest.raises(FileExistsError):
        entry.main(args)


def test_default_shared_replay_keeps_no_publication_and_original_actor(monkeypatch):
    holder = fixture(monkeypatch)
    r = entry.run_example(**{**OPTIONS, "shared_serving": False})
    assert r["complete"] and "shared_serving" not in r and "publication" not in holder
    assert r["source_unchanged"] and r["actor_unchanged"] and r["replay_examples"] == 48


def test_changed_preceding_serving_success_refuses_the_next_read(monkeypatch):
    holder = fixture(monkeypatch)
    original = publication.run_workflow_episode
    calls: list[int] = []

    def corrupt(*args, **kwargs):
        session = holder["publication"]
        if calls:
            session.row["episodes"][0]["episode"]["success"] = True
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(publication, "run_workflow_episode", corrupt)
    r = entry.run_example(**OPTIONS)
    assert not r["complete"]
    paid = r["shared_serving_preserved_work"]
    assert paid["requests"] == paid["request_returns"] == 1 and len(paid["episodes"]) == 1
    assert paid["episodes"][0]["episode"]["success"] is False
    assert r["shared_serving"]["episodes"][0]["episode"]["success"] is True


def test_published_actor_refuses_stale_candidate_before_more_guard_work(monkeypatch):
    holder = fixture(monkeypatch)
    original = publication.SharedLinearPublication._episodes

    def stale(self, phase, scenarios):
        if phase == "published":
            with pytest.raises(ValueError, match="stale"):
                self.controller.prepare(
                    self.stream._runtime,
                    *self.guards,
                    now=self.stream._next_tick,
                    training_ids=frozenset(),
                )
        return original(self, phase, scenarios)

    monkeypatch.setattr(publication.SharedLinearPublication, "_episodes", stale)
    r = entry.run_example(**OPTIONS)
    assert r["complete"], r.get("error")
    assert r["shared_serving"]["commit_returns"] == 1
    assert holder["wake"].updates_completed == 12


def test_paused_source_refusal_preserves_exact_owned_deferral_without_more_fit(monkeypatch):
    holder = fixture(monkeypatch)
    original = publication.SharedLinearPublication._episodes

    def denied(self, phase, scenarios):
        if phase == "published":
            with pytest.raises(ValueError, match="deferred: paused"):
                self.before_fit()
        return original(self, phase, scenarios)

    monkeypatch.setattr(publication.SharedLinearPublication, "_episodes", denied)
    r = entry.run_example(**OPTIONS)
    assert r["complete"], r.get("error")
    row = r["shared_serving"]
    assert row["fit_admissions"] == row["fit_returns"] == 16
    assert row["sharing"]["deferrals"] == (("paused", 1),)
    assert holder["source"].snapshot_state().model is None


@pytest.mark.parametrize("point", ["before_guard", "during_guard"])
def test_original_replay_bank_is_checked_before_any_more_publication_work(monkeypatch, point):
    holder = fixture(monkeypatch)

    def change():
        bank = holder["stream"]._rehearsal._buffer
        first = next(iter(bank._rows.values()))
        first[0][0, 0] += 1.0

    if point == "before_guard":
        original = publication.SharedLinearPublication.run

        def run(self, *args):
            change()
            return original(self, *args)

        monkeypatch.setattr(publication.SharedLinearPublication, "run", run)
    else:
        monkeypatch.setattr(PromotionGuardEvaluator, "_measure", REAL_MEASURE)
        original_predict = LinearFitLearner.predict

        def predict(self, features):
            result = original_predict(self, features)
            change()
            return result

        monkeypatch.setattr(LinearFitLearner, "predict", predict)
    r = entry.run_example(**OPTIONS)
    assert not r["complete"]
    paid = r["shared_serving_preserved_work"]
    assert paid["guard_attempts"] == paid["guard_returns"] == (0 if point == "before_guard" else 1)
    assert paid["commit_attempts"] == 0 and r["replay_examples"] == 48
    assert paid["fit_returns"] == 16


def test_published_native_probability_port_preserves_source_clock_and_state(monkeypatch):
    holder = fixture(monkeypatch)
    original = publication.SharedLinearPublication._episodes

    def native(self, phase, scenarios):
        if phase == "published":
            clock = self.budget.last_clock
            result = self.actor.predict(self.guards[0].features[:1])
            assert result.actor_version == "c0" and result.prediction.shape == (1, 6)
            assert self.budget.last_clock == clock
            self._readonly()
        return original(self, phase, scenarios)

    monkeypatch.setattr(publication.SharedLinearPublication, "_episodes", native)
    r = entry.run_example(**OPTIONS)
    assert r["complete"], r.get("error")
    assert holder["publication"].actor.serving_snapshot().generation == 2


@pytest.mark.parametrize("point", ["before_guard", "during_guard"])
def test_replaced_helper_budget_cannot_bypass_original_read_checks(monkeypatch, point):
    holder = fixture(monkeypatch)

    def change():
        owner = holder["wake"]
        # A frozen clock equals the original cached value. Value-only source
        # checks cannot detect that original per-read wall/RSS checks were lost.
        holder["publication"].budget = ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=12, max_wall_seconds=30),
            lambda: owner.last_clock,
        )

    if point == "before_guard":
        original = publication.SharedLinearPublication.run

        def run(self, *args):
            change()
            return original(self, *args)

        monkeypatch.setattr(publication.SharedLinearPublication, "run", run)
    else:
        monkeypatch.setattr(PromotionGuardEvaluator, "_measure", REAL_MEASURE)
        original_predict = LinearFitLearner.predict

        def predict(self, features):
            result = original_predict(self, features)
            change()
            return result

        monkeypatch.setattr(LinearFitLearner, "predict", predict)
    r = entry.run_example(**OPTIONS)
    assert not r["complete"] and "budget replaced" in r["error"]
    paid = r["shared_serving_preserved_work"]
    assert paid["fit_returns"] == 16 and paid["commit_attempts"] == 0
    assert paid["guard_attempts"] == paid["guard_returns"] == (0 if point == "before_guard" else 1)
    assert holder["wake"].started_at == holder["birth"] and holder["wake"].updates_completed == 12
    assert (
        r["replay_examples"] == 48
        and holder["publication"].actor.serving_snapshot().generation == 0
    )
