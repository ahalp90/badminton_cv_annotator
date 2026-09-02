"""Reader for ShuttleSet human contact annotations into the frozen v1 schema.

Builds the ``schema_v1.SOURCE_CONTACTS`` table and usable rally spans for one
match's ShuttleSet set CSVs. Reproduces the rally-usability rules from the
benchmark-only reader
``annotator.calibration.shuttleset22_features.load_annotation_rallies``: a
rally is unusable if any of its rows has an invalid frame or a marked flaw,
or if its contact frames are not strictly increasing in (ball_round,
frame_num) order. See issue #18.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import NamedTuple

import pandas as pd

from classifier_shared.taxonomy import ZH_TO_EN
from dataset_builder.schema_v1 import SOURCE_CONTACTS, validate_table


_SET_FILENAME = re.compile(r"^set(\d+)$")
_REQUIRED_COLUMNS = ("rally", "ball_round", "frame_num", "type", "flaw")
# ShuttleSet22 writes passive drop with a homophone character (過度 for 過渡).
# The verbatim label stays in contact_type; only the English mapping uses this.
SOURCE_LABEL_ALIASES: dict[str, str] = {"過度切球": "過渡切球"}


def english_label(label: object) -> object:
    """Map a verbatim source stroke label to its English taxonomy name, or NA."""
    if not isinstance(label, str):
        return pd.NA
    return ZH_TO_EN.get(SOURCE_LABEL_ALIASES.get(label, label), pd.NA)


class SourceRally(NamedTuple):
    """One usable ShuttleSet rally, rebuilt from its human contact rows."""

    source_set: int
    source_rally: int
    start_frame: int
    end_frame: int
    contact_rows: tuple[int, ...]


class SourceAnnotations(NamedTuple):
    """One match's frozen ``source_contacts`` rows and its usable rallies."""

    contacts: pd.DataFrame
    rallies: tuple[SourceRally, ...]
    population: dict[str, int]


def set_number(path: Path) -> int:
    """Parse the set number from a ShuttleSet filename, e.g. set1.csv -> 1."""
    match = _SET_FILENAME.match(Path(path).stem)
    if match is None:
        raise ValueError(f"not a ShuttleSet set filename: {path}")
    return int(match.group(1))


def _read_set(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = [name for name in _REQUIRED_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    table["source_set"] = set_number(path)
    table["source_row"] = range(len(table))
    return table


def _usable_rallies(
    raw: pd.DataFrame,
    source_rally: pd.Series,
    ball_round: pd.Series,
    frame_num: pd.Series,
    invalid_frame: pd.Series,
    flaw_marked: pd.Series,
) -> tuple[list[SourceRally], pd.Series, dict[str, int]]:
    """Group contact rows into rallies and split usable from unusable ones."""
    rallies: list[SourceRally] = []
    rally_id = pd.Series(pd.NA, index=raw.index, dtype="Int64")
    counts = {
        "excluded_incomplete_rallies": 0,
        "excluded_incomplete_rally_rows": 0,
        "excluded_non_monotonic_rallies": 0,
        "excluded_non_monotonic_rally_rows": 0,
    }
    groups = raw.groupby([raw["source_set"], source_rally], sort=True).groups
    for (set_value, rally_value), index in groups.items():
        if bool((invalid_frame.loc[index] | flaw_marked.loc[index]).any()):
            counts["excluded_incomplete_rallies"] += 1
            counts["excluded_incomplete_rally_rows"] += len(index)
            continue
        order = pd.DataFrame(
            {"ball_round": ball_round.loc[index], "frame_num": frame_num.loc[index]},
            index=index,
        ).sort_values(["ball_round", "frame_num"], kind="stable")
        contact_frames = frame_num.loc[order.index].to_list()
        strictly_increasing = all(
            right > left for left, right in zip(contact_frames, contact_frames[1:])
        )
        if not contact_frames or not strictly_increasing:
            counts["excluded_non_monotonic_rallies"] += 1
            counts["excluded_non_monotonic_rally_rows"] += len(index)
            continue
        rally_id.loc[order.index] = len(rallies)
        rallies.append(
            SourceRally(
                source_set=int(set_value),
                source_rally=int(rally_value),
                start_frame=int(contact_frames[0]),
                end_frame=int(contact_frames[-1]) + 1,
                contact_rows=tuple(raw.loc[order.index, "source_row"]),
            )
        )
    return rallies, rally_id, counts


def load_source_annotations(
    set_dir: Path, *, source_dataset: str, video_id: str, frame_count: int
) -> SourceAnnotations:
    """Load one match's ShuttleSet set CSVs into the frozen source_contacts table."""
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    paths = sorted(Path(set_dir).glob("set*.csv"))
    if not paths:
        raise ValueError(f"no ShuttleSet set tables under {set_dir}")
    raw = pd.concat([_read_set(path) for path in paths], ignore_index=True)

    source_rally = pd.to_numeric(raw["rally"], errors="coerce")
    ball_round = pd.to_numeric(raw["ball_round"], errors="coerce")
    frame_num = pd.to_numeric(raw["frame_num"], errors="coerce")
    flaw_marked = raw["flaw"].notna()
    invalid_frame = frame_num.isna() | (frame_num < 0) | (frame_num >= frame_count)

    rallies, rally_id, rally_counts = _usable_rallies(
        raw, source_rally, ball_round, frame_num, invalid_frame, flaw_marked
    )

    contacts = pd.DataFrame(
        {
            "source_dataset": source_dataset,
            "video_id": video_id,
            "source_set": raw["source_set"],
            "source_row": raw["source_row"],
            "source_rally": source_rally,
            "ball_round": ball_round,
            "frame_num": frame_num,
            "contact_type": raw["type"],
            "contact_type_en": raw["type"].map(english_label),
            "flaw_marked": flaw_marked,
            "rally_id": rally_id,
        }
    )
    population = {
        "source_contact_rows": len(raw),
        "usable_contact_rows": int(rally_id.notna().sum()),
        "usable_rallies": len(rallies),
        "excluded_flaw_rows": int(flaw_marked.sum()),
        "excluded_invalid_frame_rows": int((invalid_frame & ~flaw_marked).sum()),
        "excluded_incomplete_rallies": rally_counts["excluded_incomplete_rallies"],
        "excluded_incomplete_rally_rows": rally_counts["excluded_incomplete_rally_rows"],
        "excluded_non_monotonic_rallies": rally_counts["excluded_non_monotonic_rallies"],
        "excluded_non_monotonic_rally_rows": rally_counts["excluded_non_monotonic_rally_rows"],
    }
    return SourceAnnotations(
        contacts=validate_table(SOURCE_CONTACTS, contacts),
        rallies=tuple(rallies),
        population=population,
    )
