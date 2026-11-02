"""Explicit TRAIN reuse inside an existing disposable full-network probe fit.

Owns a fixed TRAIN bank and its original replay exposure budget. Reserved inputs
remain separately bound; no role conversion, retention of probe rows, model
selection, runtime, IO or promotion belongs here. The native fitter is unchanged.
"""

from copy import deepcopy
from dataclasses import asdict

import numpy as np

from src.adapters.continuous_probe_fit import _FullProbeBinding
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.app.continuous_stream_rehearsal import _BudgetBinding, _signature, prepare_replay_bank
from src.app.toy_execution_budget import ToyBudgetSession


class _ProbeRehearsalBinding:
    def __init__(self, training, evaluation, plan, episodes, budget, readouts):
        self.plan, self.bank, self.rows, self.numeric_bytes = prepare_replay_bank(plan, episodes)
        if self.plan.every_updates != 1:
            raise ValueError("joint probe rehearsal requires every_updates=1")
        if any(
            item.spec != training.spec
            or item.episode_id in (training.episode_id, evaluation.episode_id)
            or item not in readouts
            for item in self.bank
        ):
            raise ValueError("replay requires separately supplied original old TRAIN readouts")
        if (
            type(budget) is not ToyBudgetSession
            or budget.budget.max_replay_examples != self.plan.max_replay_examples
            or budget.updates_completed != 0
            or budget.replay_examples_completed != 0
        ):
            raise ValueError("probe rehearsal requires a fresh original separate replay budget")
        self._budget = _BudgetBinding(budget)
        new = _FullProbeBinding.from_training(training)
        self.training_id = training.episode_id
        self.features, self.targets = new.features, new.targets
        from src.core.continuous_demonstration import continuous_training_pairs

        old = [continuous_training_pairs(item, role="train") for item in self.bank]
        self._joint = _FullProbeBinding(
            self.features + tuple(tuple(row) for inputs, _ in old for row in inputs),
            self.targets + tuple(tuple(row) for _, targets in old for row in targets),
        )
        self.training_rows = len(self._joint.features)
        self._report_header = deepcopy(self._metadata())
        self._birth = self._configuration()
        self.charged = self.attempts = self.returns = 0
        self.failed = False
        self.require()

    def _configuration(self):
        return (
            id(self.plan),
            id(self.bank),
            id(self.features),
            id(self.targets),
            id(self._joint),
            id(self._budget),
            self.rows,
            self.numeric_bytes,
            self.training_rows,
            self.training_id,
            _signature((self.plan, self.bank, self.features, self.targets, self._joint)),
        )

    def require(self):
        if self.failed or self._configuration() != self._birth:
            raise RuntimeError("probe rehearsal stopped or original inputs changed")
        self._budget.check(0, self.charged)

    def check_boundary(self):
        try:
            self.require()
            self._budget.session.before_final()
            self.require()
        except BaseException:
            self.failed = True
            raise

    def update(self, learner, features, targets):
        self.require()
        if type(learner) is not NumpyContinuousBCLearner:
            raise ValueError("joint full-network probe requires exact BP")
        for actual, expected in ((features, self.features), (targets, self.targets)):
            if (
                type(actual) is not np.ndarray
                or actual.dtype != np.float64
                or actual.shape != (len(expected), len(expected[0]))
                or not np.array_equal(actual, expected)
            ):
                raise ValueError("joint probe wake arrays differ from reserved packet")
        # Why this: one native joint fit uses two separately authorized roles.
        # Charge old TRAIN exposure before any kernel; a failed fit never refunds it.
        try:
            self._budget.session.before_sleep()
            self.require()
            remaining = self._budget.session.remaining_replay_examples
            if type(remaining) is not int or remaining < self.rows:
                self._budget.session.stop_replay()
            previous = self.charged
            try:
                self._budget.session.record_replay(self.rows)
            finally:
                observed = self._budget.session.replay_examples_completed
                if type(observed) is int and observed >= previous:
                    self.charged = observed
            if self.charged != previous + self.rows:
                raise RuntimeError("joint probe replay charge differs from original budget")
            self.require()
            x = np.asarray(self._joint.features, dtype=np.float64)
            y = np.asarray(self._joint.targets, dtype=np.float64)
            self.attempts += 1
            result = self._joint.update(learner, x, y)
            self.returns += 1
            # The existing recorded port must save this paid result BEFORE any
            # post-native observer can refuse it. The outer boundary checks next.
            return result
        except BaseException:
            self.failed = True
            raise

    def _metadata(self):
        offset = len(self.features)
        spans = [dict(role="probe_train", episode_id=self.training_id, start=0, stop=offset)]
        for item in self.bank:
            spans.append(
                dict(
                    role="train",
                    episode_id=item.episode_id,
                    start=offset,
                    stop=offset + len(item.transitions),
                )
            )
            offset += len(item.transitions)
        return dict(
            mode="joint_old_train",
            plan=asdict(self.plan),
            bank_episode_ids=[e.episode_id for e in self.bank],
            bank_rows=self.rows,
            retained_numeric_bytes=self.numeric_bytes,
            joint_training_rows=self.training_rows,
            row_spans=spans,
            joint_array_numeric_bytes_per_fit=8
            * self.training_rows
            * (len(self.features[0]) + len(self.targets[0])),
            original_limits=self._budget.limit_report,
            scope="Old TRAIN bank only; probe_train rows are transient. Joint fits add rows/compute, not extra wake updates. Payload excludes containers/copies/RSS; native work is in native_updates. Charges include failed attempts, not just commits.",
        )

    def report(self):
        # Preserve admitted metadata even when a later callback corrupts live inputs.
        result = deepcopy(self._report_header)
        observed = self._budget.session.replay_examples_completed
        result.update(
            charged_exposures=self.charged,
            native_attempts=self.attempts,
            native_returns=self.returns,
            stopped=self.failed,
            observed_budget_exposures=observed if type(observed) is int else None,
            charge_count_unknown=type(observed) is not int or observed != self.charged,
            unreturned_native_work_unknown=self.attempts != self.returns,
        )
        return result
