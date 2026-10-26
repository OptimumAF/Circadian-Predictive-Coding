"""Bounded prediction/selected-command/native-return receipts for opt-in decoding.

Own at most50 controls and retain paid counts on later integrity failures. The
caller supplies native records and keeps original port work if a return fails.
No learner, observation encoding, native action, budget, clock, IO or hash store.
"""

from dataclasses import asdict
import json
import math
import struct

from src.core.continuous_demonstration import (
    ContinuousEpisode,
    ContinuousTransition,
    _copy_transition,
    copy_continuous_episode,
)
from src.core.pick_cube_action_decoding import (
    PickCubeBinaryGripperRecipe,
    _normalized_action,
    decode_gripper_action,
    validate_gripper_recipe,
)

_Control = tuple[int, tuple[float, ...], tuple[float, ...], bool, tuple[float, ...] | None]


def _native_payload(value):
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("decoding evidence requires finite native scalars")
    elif type(value) in (str, int, bool, type(None)):
        return
    elif type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("decoding report keys must be native strings")
            _native_payload(item)
    elif type(value) in (tuple, list):
        for item in value:
            _native_payload(item)
    else:
        raise ValueError("decoding evidence requires exact native containers/scalars")


def _signature(value):
    _native_payload(value)
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _record_payload(record):
    owned = _copy_transition(record)
    _normalized_action(record.action)
    original = tuple(
        getattr(record, name)
        for name in (
            "step",
            "observation",
            "action",
            "reward",
            "next_observation",
            "terminated",
            "truncated",
        )
    )
    copied = tuple(
        getattr(owned, name)
        for name in (
            "step",
            "observation",
            "action",
            "reward",
            "next_observation",
            "terminated",
            "truncated",
        )
    )
    if _signature(original) != _signature(copied):
        raise ValueError("native decoding record contains normalized or replaced scalar types")
    return copied


class ActionDecodingLedger:
    def __init__(self, recipe: PickCubeBinaryGripperRecipe):
        self.request = self._request_owner = recipe
        self.recipe = self._recipe_owner = validate_gripper_recipe(recipe)
        self._recipe_metadata = _signature(asdict(self.recipe))
        self._controls: tuple[_Control, ...] = ()
        self._accepted_controls: tuple[_Control, ...] = self._controls
        # decode attempts/returns, control attempts/returns, validated returns.
        self._counts: tuple[int, ...] = (0, 0, 0, 0, 0)
        self._accepted_counts: tuple[int, ...] = self._counts
        self._records: tuple[ContinuousTransition, ...] = ()
        self._record_owners: tuple[ContinuousTransition, ...] = self._records
        self._payloads: tuple[tuple, ...] = ()
        self._accepted_payloads: tuple[tuple, ...] = self._payloads
        self._evidence = self._accepted_signature()
        self._failed = False

    def _accepted_signature(self):
        return _signature((self._accepted_controls, self._accepted_counts, self._accepted_payloads))

    def _pay(self, index):
        # A native return stays paid even if a preceding private receipt was
        # corrupted. Updating a charge must never bless that corrupted receipt.
        original = self._accepted_counts
        intact = self._evidence == self._accepted_signature()
        changed = list(original)
        changed[index] += 1
        self._accepted_counts = tuple(changed)
        if self._counts is original:
            self._counts = self._accepted_counts
        if intact:
            self._evidence = self._accepted_signature()

    def _guard(self):
        if self._failed:
            raise ValueError("failed action decoding ledger cannot resume")
        if (
            self.request is not self._request_owner
            or self.recipe is not self._recipe_owner
            or _signature(asdict(validate_gripper_recipe(self.request))) != self._recipe_metadata
            or _signature(asdict(validate_gripper_recipe(self.recipe))) != self._recipe_metadata
            or self._controls is not self._accepted_controls
            or self._counts is not self._accepted_counts
            or self._records is not self._record_owners
            or self._payloads is not self._accepted_payloads
            or self._accepted_signature() != self._evidence
            or len(self._accepted_controls) > 50
            or len(self._records) != len(self._accepted_payloads)
        ):
            raise ValueError("original decoding recipe, receipts or paid counters changed")
        if _signature(tuple(_record_payload(record) for record in self._records)) != _signature(
            self._accepted_payloads
        ):
            raise ValueError("preceding native decoding records changed")

    def _stable(self, *, paid_return=False):
        attempts, decoded, attempted, returned, validated = self._accepted_counts
        controls = self._accepted_controls
        if (
            any(type(count) is not int or not 0 <= count <= 50 for count in self._accepted_counts)
            or attempts != decoded
            or decoded != len(controls)
            or attempted != sum(row[3] for row in controls)
            or validated != sum(row[4] is not None for row in controls)
            or returned != validated + int(paid_return)
            or validated != len(self._records)
            or not 0 <= attempted - validated <= 1
            or any(row[0] != index for index, row in enumerate(controls))
            or any(not row[3] or row[4] is None for row in controls[:-1])
        ):
            raise ValueError("decoding progress differs from its bounded control prefix")

    def report(self):
        # Observation must remain available when the original request has been
        # mutated; only the admitted metadata and original paid prefix are read.
        a, d, c, r, v = self._accepted_counts
        return dict(
            recipe=json.loads(self._recipe_metadata),
            controls=[
                dict(
                    step=step,
                    prediction=prediction,
                    selected_command=selected,
                    control_attempted=attempted,
                    returned_action=returned,
                )
                for step, prediction, selected, attempted, returned in self._accepted_controls
            ],
            work=dict(
                decode_attempts=a,
                decode_returns=d,
                prediction_values=4 * d,
                copied_xyz_values=3 * d,
                gripper_decisions=d,
                control_attempts=c,
                control_returns=r,
                validated_control_returns=v,
                returned_action_values=4 * v,
            ),
        )

    def check(self, actual, records):
        try:
            self._guard()
            self._stable()
            if _signature(actual) != _signature(self.report()):
                raise ValueError("preceding public decoding report changed")
            if (
                type(records) is not tuple
                or len(records) != len(self._records)
                or any(
                    record is not original
                    for record, original in zip(records, self._records, strict=True)
                )
            ):
                raise ValueError("native decoding prefix was removed or replaced")
            self._guard()
        except Exception:
            self._failed = True
            raise

    def _publish_controls(self, controls):
        self._controls = self._accepted_controls = controls
        self._evidence = self._accepted_signature()

    def select(self, step, prediction, actual):
        try:
            self.check(actual, self._records)
            if (
                type(step) is not int
                or step != len(self._controls)
                or not 0 <= step < 50
                or (self._controls and self._controls[-1][4] is None)
            ):
                raise ValueError("decoding requires the next step with no pending control")
            self._pay(0)
            selected = decode_gripper_action(prediction, self.recipe)
            self._pay(1)
            owned = _normalized_action(prediction)
            _normalized_action(selected)
            expected = (*owned[:3], -1.0 if owned[3] < 0.0 else 1.0)
            if _signature(selected) != _signature(expected):
                raise ValueError("decoder output differs from the fixed endpoint rule")
            self._guard()
            self._publish_controls((*self._controls, (step, owned, selected, False, None)))
            self._stable()
            return selected
        except Exception:
            self._failed = True
            raise

    def attempt(self, actual):
        try:
            self.check(actual, self._records)
            if not self._controls or self._controls[-1][3]:
                raise ValueError("decoding requires one selected unattempted command")
            step, prediction, selected, _, returned = self._controls[-1]
            self._pay(2)
            self._publish_controls(
                (*self._controls[:-1], (step, prediction, selected, True, returned))
            )
            self._stable()
        except Exception:
            self._failed = True
            raise

    def returned(self, record, actual):
        previous = self.report()
        self._pay(3)  # Count the observed native return before ANY acceptance check.
        try:
            self._guard()
            self._stable(paid_return=True)
            if _signature(actual) != _signature(previous) or not self._controls:
                raise ValueError("native return lost its preceding public decoding receipt")
            step, prediction, selected, attempted, returned = self._controls[-1]
            if not attempted or returned is not None:
                raise ValueError("native return requires one attempted pending command")
            payload = _record_payload(record)
            native = tuple(struct.unpack("!f", struct.pack("!f", value))[0] for value in selected)
            if (
                record.step != step
                or _signature(record.action) != _signature(native)
                or (self._records and self._records[-1].next_observation != record.observation)
            ):
                raise ValueError("native return differs from its selected float32 command/step")
            self._guard()
            self._records = self._record_owners = (*self._records, record)
            self._payloads = self._accepted_payloads = (*self._payloads, payload)
            self._pay(4)
            self._publish_controls(
                (*self._controls[:-1], (step, prediction, selected, True, native))
            )
            self._guard()
            self._stable()
        except Exception:
            self._failed = True
            raise

    def verify_episode(self, episode: ContinuousEpisode, actual):
        try:
            self.check(actual, self._records)
            if type(episode) is not ContinuousEpisode:
                raise ValueError("decoding verification requires an exact native episode")
            original = tuple(_record_payload(row) for row in episode.transitions)
            owned = copy_continuous_episode(episode)
            if (
                not self._controls
                or len(owned.transitions) != len(self._controls)
                or any(not row[3] or row[4] is None for row in self._controls)
                or _signature(original) != _signature(self._accepted_payloads)
            ):
                raise ValueError("complete native episode differs from decoded control receipts")
            self.check(actual, self._records)
        except Exception:
            self._failed = True
            raise
