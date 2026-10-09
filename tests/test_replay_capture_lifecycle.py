"""Real lifecycle leases and fake learner rows; no native model operations."""

from unittest.mock import patch
import pytest
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_replay_origins import ManagedReplayOrigins
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_retention import DataRetentionPolicy
from src.core.payload_ownership import PayloadOwnershipLimits
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_managed_replay_origins import setup as fake_setup, queue, PORTS, LIMITS, WINDOW, Payload
from test_replay_capture_origins import leased


def measure_payload_bytes(payload: object) -> int:
    if type(payload) is not Payload:
        raise ValueError("installed lifecycle fixture requires its exact fake payload")
    return len(payload.value.encode())


def setup():
    seconds = [0.0]
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=4), lambda: seconds[0])
    owner, unused, runtime, clock, _, _ = fake_setup(budget=budget)
    # No work/admission was spent by the helper's provisional ledger.
    del unused
    life = ManagedDataLifecycle(
        owner,
        policy=DataRetentionPolicy(
            4096, 120, PayloadOwnershipLimits(8, 12), max_retention_seconds=1000.0
        ),
        measure_payload_bytes=measure_payload_bytes,
        native_footprint=lambda learner: ReplayPayloadErasure(0, 0, 0),
        native_erase=lambda _: pytest.fail("erasure outside fake scope"),
        measure_auxiliary_bytes=lambda _: 0,
    )
    ledger = ManagedReplayOrigins(owner, PORTS, LIMITS, WINDOW)
    queue(owner, clock)
    ledger.train_ready()
    return owner, ledger, runtime, clock, life, seconds


def test_should_use_leased_elapsed_inside_capture_and_ordinary_elapsed_for_public_reads():
    owner, ledger, runtime, _, life, _ = setup()
    with (
        life._time_gate,
        patch.object(
            ManagedDataLifecycle, "_elapsed", side_effect=AssertionError("ordinary elapsed reentry")
        ),
    ):
        with leased(owner, runtime, ledger) as check:
            check()
    original = ManagedDataLifecycle._elapsed
    reads = []

    def ordinary(value):
        reads.append(value)
        return original(value)

    with patch.object(ManagedDataLifecycle, "_elapsed", ordinary):
        assert ledger.origins()[0].key == ("e1", "s1")
    assert reads and all(value is life for value in reads)
    assert ledger.accounting().invocations_started == 2


@pytest.mark.parametrize(
    "failure",
    ["revoked", "optout", "tick_age", "elapsed_age", "rewound_elapsed", "guard", "cleanup"],
)
def test_should_preserve_original_lifecycle_consent_and_elapsed_refusals_under_lease(failure):
    owner, ledger, runtime, clock, life, seconds = setup()
    before = ledger.accounting()
    if failure == "revoked":
        owner._revoked_keys.add(("e1", "s1"))
    elif failure == "optout":
        owner._opted_out.add("person-1")
    elif failure == "tick_age":
        clock.advance_to(120)
    elif failure == "elapsed_age":
        seconds[0] = 1000.0
    elif failure == "rewound_elapsed":
        seconds[0] = -1.0
    elif failure == "guard":
        runtime._inbox._training_guard = lambda *args: True
    else:
        life._failed = True
    with (
        life._time_gate,
        patch.object(
            ManagedDataLifecycle, "_elapsed", side_effect=AssertionError("ordinary elapsed reentry")
        ),
    ):
        with pytest.raises(ValueError):
            with leased(owner, runtime, ledger):
                pytest.fail("projection/copy after refusal")
    assert runtime._budget.updates_completed == 1
    assert len(runtime.applied_updates) == len(runtime._candidate.model.rows) == 1
    after = ledger.accounting()
    assert after.records_created == before.records_created
    admitted = int(failure in ("revoked", "optout"))
    assert after.invocations_started == before.invocations_started + admitted
    assert after.metadata_bytes_charged == before.metadata_bytes_charged + 1024 * admitted
