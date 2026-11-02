"""Bounded content coverage for the explicit retained TRAIN-only probe.

Uses existing content IDs and returned nearest indices; no extra prediction,
training, evaluation, role conversion, runner, selection authority or history.
"""

from dataclasses import dataclass

import numpy as np

from src.adapters import numpy_retained_nearest_bc as retained


@dataclass(frozen=True)
class RetainedTrainingContent:
    source_ids: tuple[str, ...]
    incoming_ids: tuple[str, ...]


def _hash_ids(features, targets, work, publish):
    counts = work.setdefault("retained_content", {})
    # Array casts and tobytes payloads only; not Python concatenations/objects.
    numeric_copies = 2 * (features.nbytes + targets.nbytes)
    size = features.nbytes + targets.nbytes + 24 * len(features)
    for key, amount in (
        ("hash_calls_attempted", 1),
        ("hash_rows_attempted", len(features)),
        ("hash_bytes_attempted", size),
        ("hash_numeric_materialization_bytes_attempted_upper_bound", numeric_copies),
    ):
        counts[key] = counts.get(key, 0) + amount
    publish()
    try:
        ids = retained._ids(features, targets)
        for key, amount in (
            ("hash_calls_returned", 1),
            ("hash_rows_returned", len(features)),
            ("hash_bytes_returned", size),
            ("hash_numeric_materialization_bytes_returned_upper_bound", numeric_copies),
        ):
            counts[key] = counts.get(key, 0) + amount
        return ids
    finally:
        publish()


def bind_content(before, features, targets, work, publish):
    """Keep only this source bank and incoming packet, not lifetime provenance."""
    return RetainedTrainingContent(
        before.prototype_ids, _hash_ids(features, targets, work, publish)
    )


def describe_selection(state, content, features, targets, indices, work, publish):
    """Describe actual selections, preserving overlapping duplicate content origins."""
    counts = work.setdefault("retained_content", {})
    rows = len(features)
    # Semantic row visits only, not Python hash-table probes or collision costs.
    # Why: these are bounded upper bounds, charged before any mapping or hash.
    for key, amount in (
        ("mapping_attempts", 1),
        ("mapping_rows_attempted", rows),
        (
            "set_input_id_occurrences_upper_bound",
            len(content.source_ids) + len(content.incoming_ids) + len(state.prototype_ids),
        ),
        ("membership_tests_upper_bound", 5 * len(state.prototype_ids) + 4 * rows),
        ("feature_values_compared_upper_bound", features.size),
    ):
        counts[key] = counts.get(key, 0) + amount
    publish()
    try:
        source, incoming, bank = (
            set(content.source_ids),
            set(content.incoming_ids),
            set(state.prototype_ids),
        )
        if not bank <= source | incoming:
            raise RuntimeError("retained training bank has unobserved content")
        query_ids = _hash_ids(features, targets, work, publish)
        selected = []
        for row, (index, query_id) in enumerate(zip(indices, query_ids, strict=True)):
            content_id = state.prototype_ids[index]
            selected.append(
                dict(
                    index=index,
                    content_id=content_id,
                    matches_source=content_id in source,
                    matches_incoming=content_id in incoming,
                    query_content_retained=query_id in bank,
                    selected_content_exact=content_id == query_id,
                    selected_input_equal=bool(
                        np.array_equal(features[row], state.prototype_features[index])
                    ),
                )
            )
        exact_rows = sum(item["selected_content_exact"] for item in selected)
        result = dict(
            selected=selected,
            content_coverage=dict(
                query_rows=rows,
                distinct_query_ids=len(set(query_ids)),
                query_content_rows_retained=sum(
                    item["query_content_retained"] for item in selected
                ),
                distinct_query_ids_retained=len(set(query_ids) & bank),
                selected_content_exact_rows=exact_rows,
                selected_input_equal_rows=sum(item["selected_input_equal"] for item in selected),
                distinct_selected_ids=len({item["content_id"] for item in selected}),
                prototype_rows=len(bank),
                source_content_rows_retained=len(bank & source),
                incoming_content_ids=len(incoming),
                incoming_occurrences=len(content.incoming_ids),
                incoming_content_rows_retained=len(bank & incoming),
                shared_content_rows_retained=len(bank & source & incoming),
            ),
        )
        counts["mapping_returns"] = counts.get("mapping_returns", 0) + 1
        counts["mapping_rows_returned"] = counts.get("mapping_rows_returned", 0) + rows
        return result
    finally:
        publish()
