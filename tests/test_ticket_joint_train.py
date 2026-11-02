"""Fixed TRAIN cohort behavior; no native model, fitting, probe or final data."""

from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np

from scripts.run_ticket_workflow import CONTEXT_FEATURES, training_cases
from scripts.ticket_train_fit import context_teacher, joint_old_fresh_teacher
from src.app.decision_reference_evaluation import DecisionCase
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import CATALOG


class JointTrainTeacherTests(unittest.TestCase):
    teachers: dict[str, tuple[DecisionCase, ...]]
    spec: DecisionFeatureSpec

    @classmethod
    def setUpClass(cls):
        cls.teachers = {
            v: training_cases(policy_version=v, include_policy_context=True) for v in ("v1", "v2")
        }
        cls.spec = DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)

    def factory(self, teachers=None):
        source = self.teachers if teachers is None else teachers

        def build(*, policy_version, include_policy_context):
            self.assertIs(include_policy_context, True)
            return source[policy_version]

        return build

    def test_should_preserve_original_old_packet_and_append_fixed_fresh_tasks(self):
        before = deepcopy(self.teachers)
        old_cases, old = context_teacher(self.spec, self.factory())
        cases, batch = joint_old_fresh_teacher(self.spec, self.factory())
        self.assertEqual(cases[:12], old_cases)
        np.testing.assert_array_equal(batch.features[:12], old.features)
        np.testing.assert_array_equal(batch.targets[:12], old.targets)
        self.assertEqual(batch.features.shape, (18, 9))
        self.assertEqual(batch.role, "train")
        self.assertEqual(
            [c.target_id for c in cases[12:]],
            ["open", "route", "commit", "open", "inspect", "commit"],
        )
        self.assertTrue(all(c.role == "train" for c in cases))
        self.assertEqual(before, self.teachers)

    def test_should_refuse_fresh_probe_role(self):
        teachers = deepcopy(self.teachers)
        rows = list(teachers["v1"])
        rows[6] = replace(rows[6])
        object.__setattr__(rows[6], "role", "probe_test")
        teachers["v1"] = tuple(rows)
        with self.assertRaisesRegex(ValueError, "TRAIN teacher"):
            joint_old_fresh_teacher(self.spec, self.factory(teachers))

    def test_should_refuse_foreign_fresh_source_group(self):
        teachers = deepcopy(self.teachers)
        rows = list(teachers["v2"])
        rows[9] = replace(rows[9], source_group="foreign")
        teachers["v2"] = tuple(rows)
        with self.assertRaisesRegex(ValueError, "TRAIN teacher"):
            joint_old_fresh_teacher(self.spec, self.factory(teachers))

    def test_should_refuse_missing_teacher_rows(self):
        teachers = {**self.teachers, "v1": self.teachers["v1"][:-1]}
        with self.assertRaisesRegex(ValueError, "TRAIN teacher"):
            joint_old_fresh_teacher(self.spec, self.factory(teachers))

    def test_should_refuse_duplicate_fresh_request_id(self):
        teachers = deepcopy(self.teachers)
        rows = list(teachers["v1"])
        rows[6] = replace(
            rows[6], request=replace(rows[6].request, request_id=rows[0].request.request_id)
        )
        teachers["v1"] = tuple(rows)
        with self.assertRaisesRegex(ValueError, "duplicate training request"):
            joint_old_fresh_teacher(self.spec, self.factory(teachers))

    def test_should_refuse_changed_feature_schema_before_teacher_release(self):
        def forbidden(**kwargs):
            self.fail("teacher was released before schema admission")

        with self.assertRaisesRegex(ValueError, "exact D9"):
            joint_old_fresh_teacher(DecisionFeatureSpec(CONTEXT_FEATURES[:-1], CATALOG), forbidden)


if __name__ == "__main__":
    unittest.main()
