"""Linear details for the existing recurring recipe, not a second runner.

Counts the actual owned forks, freezes typed reads and reports distinct work.
Existing stream/task code owns roles, chronology, guards and original budgets.
"""

from copy import deepcopy
from dataclasses import asdict, fields

from src.adapters.linear_fit_decision import LinearFitDecisionPredictor, LinearDecisionWork
from src.adapters.linear_fit_learner import LinearFitLearner, LINEAR_RESIDUAL_ID
from src.core.categorical_linear_state import LinearFitPolicy
from src.core.context_linear_state import ContextLinearFitPolicy, ContextLinearFitSnapshot
from src.adapters.context_linear_fit_learner import (
    ContextLinearFitLearner,
    CONTEXT_LINEAR_RESIDUAL_ID,
)
from src.adapters.context_linear_fit_decision import (
    ContextLinearFitDecisionPredictor,
    ContextLinearDecisionWork,
)
from src.app.coupled_attribution import _same_value


def counted_learner(before_update, after_update, *, separate=False, spec=None):
    class CountedLinearLearner(LinearFitLearner):
        def fork(self):
            child = CountedLinearLearner(self.policy)
            child.restore_state(self.snapshot_state())
            return child

        def train_batch(self, features, targets):
            before_update(features)
            diagnostic = super().train_batch(features, targets)
            after_update()
            return diagnostic

    class CountedContextLinearLearner(ContextLinearFitLearner):
        def fork(self):
            child = CountedContextLinearLearner(self.policy)
            child.restore_state(self.snapshot_state())
            return child

        def train_batch(self, features, targets):
            before_update(features)
            diagnostic = super().train_batch(features, targets)
            after_update()
            return diagnostic

    if separate:
        return CountedContextLinearLearner(
            ContextLinearFitPolicy(
                9, 6, (spec.vocabulary.index("policy_v1"), spec.vocabulary.index("policy_v2"))
            )
        )
    return CountedLinearLearner(LinearFitPolicy(9, 6))


def checkpoint(point, state):
    heads = state.heads if type(state) is ContextLinearFitSnapshot else (state,)
    return dict(
        source_updates=point,
        source_rows=state.training_examples,
        source_mistakes=None,
        state=asdict(state),
        coefficient_bytes=sum(
            (h.policy.feature_dim + 1) * h.policy.num_classes * 8
            for h in heads
            if h.model is not None
        ),
    )


def readout(spec, state):
    if type(state) is ContextLinearFitSnapshot:
        return ContextLinearFitDecisionPredictor(spec, state)
    return LinearFitDecisionPredictor(spec, state.model)


def require_readout(reader, spec, state):
    if type(state) is ContextLinearFitSnapshot:
        if (
            type(reader) is not ContextLinearFitDecisionPredictor
            or reader.spec != spec
            or reader.state != state
        ):
            raise ValueError("context frozen linear two-head binding changed")
        reader.assert_unchanged()
        return
    if (
        type(reader) is not LinearFitDecisionPredictor
        or reader.spec != spec
        or reader.model != state.model
    ):
        raise ValueError("context frozen linear readout binding changed")
    reader.assert_unchanged()


def record_readout_work(report, owner, pins, reader):
    """Persist owned paid work before post-call guards, refusing changed metadata."""
    row, original = pins[-1]
    expected = deepcopy(original)
    expected["work"] = asdict(reader.work)
    unchanged = (
        report.get("linear_readout_work") is owner
        and len(owner) == len(pins)
        and owner[-1] is row
        and _same_value(row, original)
    )
    row.clear()
    row.update(expected)
    pins[-1] = (row, deepcopy(expected))
    if not unchanged:
        owner[:] = [original for original, _ in pins]
        for original, value in pins:
            original.clear()
            original.update(deepcopy(value))
        report["linear_readout_work"] = owner
        raise ValueError("linear pending readout metadata changed")


def costs(report):
    records = report.get("linear_readout_work")
    separate = report.get("separate_context_heads") is True
    names = tuple(
        f.name for f in fields(ContextLinearDecisionWork if separate else LinearDecisionWork)
    )
    valid = type(records) is list and all(
        type(row) is dict
        and type(row.get("work")) is dict
        and all(type(row["work"].get(n)) is int and row["work"][n] >= 0 for n in names)
        for row in records
    )
    totals = {n: sum(row["work"][n] for row in records) if valid else None for n in names}
    work = report["training_work"]
    known = work["attempted_updates"] == work["returned_updates"]
    return dict(
        objective=CONTEXT_LINEAR_RESIDUAL_ID if separate else LINEAR_RESIDUAL_ID,
        learner_fit_calls_attempted=work["attempted_updates"],
        fit_rows_attempted=work["attempted_rows"],
        fit_calls_returned=work["returned_updates"],
        physical_lstsq_calls=work["returned_updates"] if known else None,
        solver_call_count_qualification="one unchanged core LS solve per returned fit; uncertain failed physical calls remain null, owned failure receipt separate",
        linear_readout_totals=totals,
        linear_readout_work_unknown=not valid
        or any(
            totals[a] != totals[b]
            for a, b in (
                ("affine_attempts", "affine_returns"),
                ("unfitted_attempts", "unfitted_returns"),
                ("decision_attempts", "decision_returns"),
            )
            + ((("routing_attempts", "routing_returns"),) if separate else ())
        ),
        retained_checkpoint_coefficient_bytes=sum(
            row["coefficient_bytes"] for row in report["checkpoints"]
        ),
        fitted_coefficient_bytes=768 if separate else 480,
        qualification="fixed LS residual/SVD work is not a single affine SGD step; reader nominal terms exclude fit/copies/validation/softmax; no matched compute claim",
    )
