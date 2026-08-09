"""Deterministic visual selection kept separate from commentary availability."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
import csv
from dataclasses import dataclass, replace
import gzip
from io import StringIO
import os
from pathlib import Path
import tempfile


SELECTED_VIDEOS_FILENAME = "selected_videos.csv.gz"
SELECTION_COLUMNS = (
    "video_id",
    "visual_selected",
    "selection_source",
    "selection_reason",
    "source_order",
    "commentary_status",
)

COMMENTARY_AVAILABLE = "available"
COMMENTARY_INELIGIBLE = "ineligible_video"
COMMENTARY_UNAVAILABLE_TRANSCRIPT = "unavailable_transcript"
COMMENTARY_UNAVAILABLE_TRIAGE = "unavailable_triage"
COMMENTARY_NO_RETAINED_CHUNK = "no_retained_chunk"
COMMENTARY_NO_PAIR = "no_pair"
COMMENTARY_FAILED = "failed_commentary_processing"
COMMENTARY_STATUSES = frozenset({
    COMMENTARY_AVAILABLE,
    COMMENTARY_INELIGIBLE,
    COMMENTARY_UNAVAILABLE_TRANSCRIPT,
    COMMENTARY_UNAVAILABLE_TRIAGE,
    COMMENTARY_NO_RETAINED_CHUNK,
    COMMENTARY_NO_PAIR,
    COMMENTARY_FAILED,
})


@dataclass(frozen=True)
class SelectionDecision:
    """One source-ordered visual decision and independent commentary status."""

    video_id: str
    visual_selected: bool
    selection_source: str
    selection_reason: str
    source_order: int
    commentary_status: str

    def __post_init__(self) -> None:
        for name in ("video_id", "selection_source", "selection_reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"selection {name} must be a non-empty string")
        if not isinstance(self.visual_selected, bool):
            raise ValueError("selection visual_selected must be boolean")
        if (
            isinstance(self.source_order, bool)
            or not isinstance(self.source_order, int)
            or self.source_order < 0
        ):
            raise ValueError("selection source_order must be a non-negative integer")
        if self.commentary_status not in COMMENTARY_STATUSES:
            raise ValueError(
                f"selection commentary_status is unsupported: {self.commentary_status!r}"
            )

    def to_row(self) -> dict[str, str]:
        """Return the fixed compressed-CSV representation."""
        return {
            "video_id": self.video_id,
            "visual_selected": str(self.visual_selected),
            "selection_source": self.selection_source,
            "selection_reason": self.selection_reason,
            "source_order": str(self.source_order),
            "commentary_status": self.commentary_status,
        }


def resolve_visual_selection(
    candidates: Sequence[Mapping[str, object]],
    *,
    max_videos: int,
    transcript_video_ids: Collection[str] = (),
) -> tuple[SelectionDecision, ...]:
    """Resolve triage-first match selection and bounded metadata fallback."""
    if isinstance(max_videos, bool) or not isinstance(max_videos, int) or max_videos < 0:
        raise ValueError("max_videos must be a non-negative integer")
    transcripts = _video_id_set(transcript_video_ids, "transcript_video_ids")
    normalized = [
        _candidate(candidate, source_order=index, transcripts=transcripts)
        for index, candidate in enumerate(candidates)
    ]
    ids = [decision.video_id for decision, _keep, _fallback in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("visual selection candidates contain duplicate video_id values")

    triage_kept = [item for item in normalized if item[1]]
    fallback = [item for item in normalized if item[2]]
    fallback.sort(key=lambda item: (item[0].video_id not in transcripts, item[0].source_order))
    selected_ids = {
        item[0].video_id
        for item in [*triage_kept, *fallback][:max_videos]
    }
    decisions: list[SelectionDecision] = []
    for decision, triage_selected, fallback_eligible in normalized:
        if decision.video_id in selected_ids:
            source = "triage" if triage_selected else "metadata_fallback"
            reason = "triage_keep" if triage_selected else "commentary_unavailable"
            decisions.append(replace(
                decision,
                visual_selected=True,
                selection_source=source,
                selection_reason=reason,
            ))
        elif triage_selected or fallback_eligible:
            decisions.append(replace(decision, selection_reason="video_cap_reached"))
        else:
            decisions.append(decision)
    return tuple(decisions)


def with_commentary_statuses(
    decisions: Sequence[SelectionDecision],
    statuses: Mapping[str, str],
) -> tuple[SelectionDecision, ...]:
    """Return decisions with validated late commentary statuses applied."""
    known = {decision.video_id for decision in decisions}
    unknown = set(statuses) - known
    if unknown:
        raise ValueError(f"commentary statuses reference unknown videos: {sorted(unknown)}")
    return tuple(
        replace(decision, commentary_status=statuses.get(decision.video_id, decision.commentary_status))
        for decision in decisions
    )


def selected_video_ids(decisions: Sequence[SelectionDecision]) -> tuple[str, ...]:
    """Return selected IDs in persisted candidate order."""
    return tuple(
        decision.video_id
        for decision in sorted(decisions, key=lambda item: item.source_order)
        if decision.visual_selected
    )


def write_selection(path: Path, decisions: Sequence[SelectionDecision]) -> Path:
    """Atomically write the complete source-ordered decision table as gzip CSV."""
    materialized = _validate_decisions(decisions)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SELECTION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(decision.to_row() for decision in materialized)
    encoded = gzip.compress(buffer.getvalue().encode("utf-8"), compresslevel=9, mtime=0)
    destination = Path(path)
    if not destination.name.endswith(".csv.gz"):
        raise ValueError("selection path must end in .csv.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_selection(path: Path) -> tuple[SelectionDecision, ...]:
    """Load and validate one compressed visual-selection table."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SELECTION_COLUMNS:
                raise ValueError("selection CSV header differs")
            rows = list(reader)
            if any(set(row) != set(SELECTION_COLUMNS) for row in rows):
                raise ValueError("selection CSV row fields differ")
    except (gzip.BadGzipFile, UnicodeDecodeError, OSError) as error:
        raise ValueError(f"could not read visual selection {path}: {error}") from error
    decisions = [
        SelectionDecision(
            video_id=_video_id(row["video_id"], "selection video_id"),
            visual_selected=_csv_bool(row["visual_selected"], "selection visual_selected"),
            selection_source=_nonempty(row["selection_source"], "selection source"),
            selection_reason=_nonempty(row["selection_reason"], "selection reason"),
            source_order=_csv_integer(row["source_order"], "selection source_order"),
            commentary_status=_nonempty(row["commentary_status"], "commentary status"),
        )
        for row in rows
    ]
    return _validate_decisions(decisions)


def _candidate(
    candidate: Mapping[str, object],
    *,
    source_order: int,
    transcripts: set[str],
) -> tuple[SelectionDecision, bool, bool]:
    video_id = _video_id(candidate.get("video_id"), "candidate video_id")
    keep = candidate.get("keep", "")
    if keep not in ("", "True", "False"):
        raise ValueError(f"candidate {video_id} keep must be blank, 'True', or 'False'")
    is_match = candidate.get("substream") == "match"
    clean_metadata = not any(
        _candidate_bool(candidate.get(name, False), f"candidate {video_id} {name}")
        for name in ("doubles_suspect", "duration_suspect", "upload_date_suspect")
    )
    triage_selected = is_match and keep == "True"
    fallback_eligible = is_match and keep == "" and clean_metadata
    commentary_status = _commentary_status(video_id, is_match, keep, transcripts)
    if not is_match:
        reason = "not_match_candidate"
    elif keep == "False":
        reason = "triage_rejected"
    elif keep == "" and not clean_metadata:
        reason = "metadata_suspect"
    else:
        reason = "not_selected"
    return SelectionDecision(
        video_id=video_id,
        visual_selected=False,
        selection_source="none",
        selection_reason=reason,
        source_order=source_order,
        commentary_status=commentary_status,
    ), triage_selected, fallback_eligible


def _commentary_status(
    video_id: str,
    is_match: bool,
    keep: object,
    transcripts: set[str],
) -> str:
    if not is_match:
        return COMMENTARY_INELIGIBLE
    if video_id not in transcripts:
        return COMMENTARY_UNAVAILABLE_TRANSCRIPT
    if keep == "":
        return COMMENTARY_UNAVAILABLE_TRIAGE
    if keep == "False":
        return COMMENTARY_NO_RETAINED_CHUNK
    return COMMENTARY_AVAILABLE


def _validate_decisions(
    decisions: Sequence[SelectionDecision],
) -> tuple[SelectionDecision, ...]:
    materialized = tuple(decisions)
    if any(not isinstance(decision, SelectionDecision) for decision in materialized):
        raise TypeError("visual selection rows must be SelectionDecision values")
    orders = [decision.source_order for decision in materialized]
    ids = [decision.video_id for decision in materialized]
    if orders != list(range(len(materialized))):
        raise ValueError("selection source_order must be unique and contiguous from zero")
    if len(ids) != len(set(ids)):
        raise ValueError("visual selection contains duplicate video_id values")
    return materialized


def _video_id_set(values: Collection[str], name: str) -> set[str]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a collection, not one string")
    normalized = {_nonempty(value, name) for value in values}
    if len(normalized) != len(values):
        raise ValueError(f"{name} contains duplicate values")
    return normalized


def _candidate_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value in ("False", ""):
        return False
    raise ValueError(f"{name} must be boolean or a CSV boolean string")


def _csv_bool(value: object, name: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{name} must be 'True' or 'False'")


def _csv_integer(value: object, name: str) -> int:
    if not isinstance(value, str) or not value.isdecimal():
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _video_id(value: object, name: str) -> str:
    result = _nonempty(value, name)
    if result in {".", ".."} or "/" in result or "\\" in result:
        raise ValueError(f"{name} must be a path-safe basename")
    return result
