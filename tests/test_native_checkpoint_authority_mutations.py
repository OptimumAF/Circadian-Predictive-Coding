"""Four additional separately budgeted last-probe authority/history refusals."""

import pytest

from test_native_checkpoint_publication_mutations import assert_last_mutation_refused
from test_native_managed_replay_checkpoints import observe_native_work


@pytest.fixture(scope="module", autouse=True)
def bounded_authority_work(tmp_path_factory):
    limits = dict(
        models=4,
        learners=16,
        forks=12,
        wakes=4,
        steps=8,
        stores=4,
        predicts=4,
        array_copies=20,
        graph_events=64,
        source_array_bytes=1024 * 1024,
        captures=4,
        preparations=4,
        native_restores=4,
        handoffs=0,
    )
    yield from observe_native_work(tmp_path_factory, limits)


@pytest.mark.parametrize(
    "mutation",
    [
        "nan_clock",
        "renewed_tick",
        "weakref_callback",
        "history_both",
    ],
)
def test_should_refuse_mutated_original_authority_before_publication(mutation):
    assert_last_mutation_refused(mutation)
