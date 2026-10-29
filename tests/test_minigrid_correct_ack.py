"""Full frozen chooser composition with literal native shapes and game IO."""

from copy import deepcopy
import pytest

from scripts import minigrid_controller_learning as learning
from src.adapters import minigrid_context_linear_learner as native
from src.adapters import minigrid_policy_evaluation as evaluation
from tests import test_minigrid_controller_validation as old
from tests import test_minigrid_controller_bank as bank


class Harness(old.Harness):
    def game(
        self,
        cold,
        state,
        *,
        budget,
        check_boundary,
        reference_window,
        reference_catalogue=False,
        reference_catalogue_cues=None,
        reference_applied_cues=None,
    ):
        if reference_applied_cues is None:
            return super().game(
                cold,
                state,
                budget=budget,
                check_boundary=check_boundary,
                reference_window=reference_window,
                reference_catalogue=reference_catalogue,
                reference_catalogue_cues=reference_catalogue_cues,
            )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(native, "ContextLinearLearner", bank.BankLiteral)
            patch.setattr(evaluation, "_run_checkpoint", self.literal_checkpoint)
            return evaluation.evaluate_context_linear_stream_checkpoint(
                cold,
                state,
                budget=budget,
                check_boundary=check_boundary,
                reference_window=reference_window,
                reference_catalogue=reference_catalogue,
                reference_applied_cues=reference_applied_cues,
            )


@pytest.fixture(scope="module")
def integrated():
    with pytest.MonkeyPatch.context() as patch:
        h = Harness(patch)
        original = learning.run_controller_learning

        def run(*args, **kwargs):
            return original(*args, **kwargs, acknowledge_correct=True)

        patch.setattr(learning, "run_controller_learning", run)
        r = h.run_validation()
        assert r["status"] == "complete", r.get("error")
        return r, h


def test_should_reach_later_new_cue_without_redundant_correct_packet_fit(integrated):
    r, _ = integrated
    v = r["validation"]
    assert v["protocol"]["acknowledge_correct"] is True and r["controller"]["attempts"] == 4
    assert r["application"] is None
    for i in (1, 2, 4):
        row, e = v["candidates"][i], v["endpoints"][i]
        w = row["window"]
        assert row["budget"]["updates"] == 2 and row["budget"]["replay"] == 0
        assert w["applied_packet_indices"] == [0, 2] and row["applied_cues"] == [0, 1]
        assert (
            w["processed_packets"] == 4
            and w["consumption_complete"]
            and not w["training_complete"]
            and not w["all_packets_fitted"]
        )
        assert len(w["acknowledged_experiences"]) == 2 and w["pending_unapplied"] == 0
        assert w["acknowledgement"]["proof"]["prediction_rows_completed"] == 36
        assert (
            e["after"] == dict(old_A3=2, new_A4=2)
            and e["native_updates"] == 4
            and e["native_rows"] == 54
        )
        assert e["games"]["reference_applied_cues"] == [0, 1] and e["recovery"] is None


def test_should_keep_noop_costs_and_all_five_windows_before_readouts(integrated):
    r, h = integrated
    for i in (0, 3):
        row = r["validation"]["candidates"][i]
        assert row["budget"]["updates"] == 0 and row["window"]["processed_packets"] == 0
        assert row["window"]["acknowledged_experiences"] == []
    assert len(set(h.endpoint_train_counts)) == 1 and h.endpoint_train_counts[0] == h.train_calls
    assert len(r["trials"]) == 4 and r["source_unchanged"]


@pytest.mark.parametrize("cues", [(0,), (0, 0), (True, 1)])
def test_should_refuse_inconsistent_actual_fit_cues_before_native_io(integrated, monkeypatch, cues):
    _, h = integrated
    state = deepcopy(
        next(
            s
            for s in h.states
            if s.completed_updates == 4 and all(x.fit_rows == 18 for x in s.heads)
        )
    )
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.shared.process_memory import ProcessRssSampler

    with ProcessRssSampler(interval_seconds=0.01) as memory:
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=0,
                max_replay_examples=0,
                max_wall_seconds=150,
                max_process_rss_bytes=512 * 1024**2,
            ),
            lambda: h.time,
        )
        budget.attach_memory(memory)
        with pytest.raises(ValueError):
            h.game(
                bank.BankLiteral(),
                state,
                budget=budget,
                check_boundary=lambda: None,
                reference_window=True,
                reference_catalogue=True,
                reference_applied_cues=cues,
            )
