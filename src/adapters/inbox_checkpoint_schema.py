"""Exact complete inbox metadata without inspecting opaque feature/label payloads.

Convert supported records and validate existing native relationships before the
outer NumPy payload stage. No native work, payload copy, authority, IO or restore.
"""

from math import isfinite

from src.adapters.numpy_checkpoint_frames import exact_object
from src.core.inbox_cursor import InboxCursor
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, AppliedExperience
from src.core.data_erasure import ErasedExperience
from src.core.learner_ports import TrainingDiagnostic

CURSOR = {
    "format_version",
    "learner_version",
    "capacity",
    "experiences",
    "labels",
    "applied",
    "last_tick",
    "stopped",
    "completed_updates",
    "erased",
}
SOURCE = {
    "sample_id",
    "episode_id",
    "observed_at",
    "model_version",
    "features",
    "role",
    "permissions",
    "candidate_ids",
    "action_id",
    "reward",
}
LABEL = {"event_id", "sample_id", "episode_id", "arrived_at", "model_version", "targets", "role"}
APPLIED = {
    "sample_id",
    "episode_id",
    "event_id",
    "actor_version",
    "learner_version",
    "observed_at",
    "arrived_at",
    "applied_at",
    "update_number",
    "diagnostic",
}
ERASED = {"key", "actor_version", "observed_at", "event_id", "arrived_at", "erased_at", "reason"}
PERMISSIONS = {"training", "replay", "evaluation"}
DIAGNOSTIC = {"definition", "value"}
RECORDS = {
    "experiences": (Experience, SOURCE),
    "labels": (LabelArrival, LABEL),
    "applied": (AppliedExperience, APPLIED),
    "erased": (ErasedExperience, ERASED),
}


def native_record(value, kind, keys):
    if type(value) is not kind:
        raise ValueError("inbox record requires exact supported native type")
    return exact_object(vars(value), keys).copy()


def native_data(cursor, policy):
    data = native_record(cursor, InboxCursor, CURSOR)
    for name, (kind, keys) in RECORDS.items():
        values = data[name]
        if type(values) is not tuple or len(values) > policy.max_records:
            raise ValueError("native histories require bounded exact immutable tuples")
        rows = [native_record(value, kind, keys) for value in values]
        for row in rows:
            if name == "experiences":
                row["permissions"] = native_record(
                    row["permissions"], ExperiencePermissions, PERMISSIONS
                )
                if (
                    type(row["candidate_ids"]) is not tuple
                    or len(row["candidate_ids"]) > policy.max_candidate_ids
                ):
                    raise ValueError("native candidate IDs require bounded exact tuple")
                row["candidate_ids"] = list(row["candidate_ids"])
            elif name == "applied":
                row["diagnostic"] = native_record(row["diagnostic"], TrainingDiagnostic, DIAGNOSTIC)
            elif name == "erased":
                if type(row["key"]) is not tuple or len(row["key"]) != 2:
                    raise ValueError("native erased identity requires exact pair tuple")
                row["key"] = list(row["key"])
        data[name] = rows
    return data


def bounded_metadata(value, policy):
    if type(value) is dict:
        for child in value.values():
            bounded_metadata(child, policy)
    elif type(value) is list:
        if len(value) > max(policy.max_records, policy.max_candidate_ids, 2):
            raise ValueError("metadata collection exceeds original bound")
        for child in value:
            bounded_metadata(child, policy)
    elif type(value) is str:
        if (
            len(value) > policy.max_identifier_bytes
            or len(value.encode("utf8")) > policy.max_identifier_bytes
        ):
            raise ValueError("metadata string exceeds original UTF8 bound")
    elif type(value) is int:
        if not -(2**63) < value < 2**63:
            raise ValueError("metadata integer exceeds supported finite wire bound")
    elif type(value) is float:
        if not isfinite(value):
            raise ValueError("metadata value must be finite")
    elif value is not None and type(value) is not bool:
        raise ValueError("unsupported exact metadata scalar")


def metadata_record(row, name, kind, keys, policy):
    exact_object(row, keys)
    payload = "features" if name == "experiences" else "targets" if name == "labels" else None
    values = {key: row[key] for key in keys if key != payload}
    bounded_metadata(values, policy)
    if name == "experiences":
        exact_object(values["permissions"], PERMISSIONS)
        values["permissions"] = ExperiencePermissions(**values["permissions"])
        candidates = values["candidate_ids"]
        if type(candidates) is not list or len(candidates) > policy.max_candidate_ids:
            raise ValueError("candidate IDs exceed original bound")
        values["candidate_ids"] = tuple(candidates)
    elif name == "applied":
        exact_object(values["diagnostic"], DIAGNOSTIC)
        values["diagnostic"] = TrainingDiagnostic(**values["diagnostic"])
    elif name == "erased":
        if type(values["key"]) is not list or len(values["key"]) != 2:
            raise ValueError("wire erased identity requires exact pair list")
        values["key"] = tuple(values["key"])
    if payload is not None:
        values[payload] = None  # native validation does not inspect payloads
    return kind(**values)


def metadata_cursor(data, policy):
    exact_object(data, CURSOR)
    if type(data["capacity"]) is not int or not 0 < data["capacity"] <= policy.max_records:
        raise ValueError("inbox capacity exceeds original record bound")
    meta = {name: value for name, value in data.items() if name not in RECORDS}
    bounded_metadata(meta, policy)
    for name, (kind, keys) in RECORDS.items():
        rows = data[name]
        if type(rows) is not list or len(rows) > policy.max_records:
            raise ValueError("wire histories require bounded exact lists")
        meta[name] = tuple(metadata_record(row, name, kind, keys, policy) for row in rows)
    return InboxCursor(**meta)


def materialize_cursor(meta, data):
    from dataclasses import replace

    sources = tuple(
        replace(row, features=raw["features"])
        for row, raw in zip(meta.experiences, data["experiences"])
    )
    labels = tuple(
        replace(row, targets=raw["targets"]) for row, raw in zip(meta.labels, data["labels"])
    )
    return replace(meta, experiences=sources, labels=labels)
