"""Rally-start event truth, persistence, and review-session state."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass, replace
from enum import StrEnum
import gzip
import io
import math
import os
from pathlib import Path
import tempfile
from typing import TypeVar

from annotator.broadcast_timeline_labels import VideoMetadata


CONTRACT_PATH = Path(__file__).resolve()
EnumT = TypeVar("EnumT", bound=StrEnum)
EXPECTED_COMMITTED_STATUS = "serve_missed_later_strokes_matched"
CANONICAL_TIMELINE_SUFFIX = "_broadcast_timeline_labels.csv.gz"

DECISION_COLUMNS = (
    "video_id",
    "set_id",
    "rally",
    "review_status",
    "serve_visibility",
    "visible_serve_frame",
    "first_visible_rally_frame",
    "broadcast_return_frame",
    "confidence",
    "review_note",
)
DECISION_VALUE_COLUMNS = DECISION_COLUMNS[3:]

TARGET_COLUMNS = (
    "video_id",
    "fps",
    "frame_count",
    "set_id",
    "rally",
    "gt_first_frame",
    "gt_first_ball_round",
    "gt_first_type_raw",
    "gt_first_type_en",
    "gt_first_flaw",
    "gt_first_server",
    "committed_status",
    "later_strokes_matched",
    "timeline_truth",
    "timeline_interval_start",
    "timeline_interval_end",
    "preceding_truth",
    "live_transition_frame",
    "frames_from_live_transition",
    "review_start_frame",
    "review_end_frame",
    "pilot_stratum",
    "note",
    "review_status",
    "serve_visibility",
    "visible_serve_frame",
    "first_visible_rally_frame",
    "broadcast_return_frame",
    "confidence",
    "review_note",
)


class ReviewStatus(StrEnum):
    """Completion state for one rally-start decision."""

    PENDING = "pending"
    REVIEWED = "reviewed"


class ServeVisibility(StrEnum):
    """Human result for the physical service action."""

    VISIBLE = "visible"
    BROADCAST_OMITTED = "broadcast-omitted"
    OFF_FRAME = "off-frame"
    UNCERTAIN = "uncertain"


class Confidence(StrEnum):
    """Allowed confidence values for reviewed decisions."""

    CERTAIN = "certain"
    UNCERTAIN = "uncertain"


class UndoResult(StrEnum):
    """Effect of one session undo request."""

    DRAFT_CLEARED = "draft-cleared"
    SAVED_ROW_RESTORED = "saved-row-restored"
    NOTHING_TO_UNDO = "nothing-to-undo"


@dataclass(frozen=True, order=True)
class RallyStartKey:
    """Stable identity of one ShuttleSet rally-start target."""

    video_id: str
    set_id: str
    rally: int

    def __post_init__(self) -> None:
        if not self.video_id or not self.set_id:
            raise ValueError("video_id and set_id must not be empty")
        if isinstance(self.rally, bool) or not isinstance(self.rally, int) or self.rally <= 0:
            raise ValueError(f"rally must be a positive integer, got {self.rally!r}")


@dataclass(frozen=True)
class RallyStartTarget:
    """Immutable source context needed to review one rally start."""

    key: RallyStartKey
    metadata: VideoMetadata
    gt_first_frame: int
    gt_first_type_en: str
    gt_first_flaw: bool
    timeline_truth: str
    timeline_interval_start: int
    timeline_interval_end: int
    preceding_truth: str
    live_transition_frame: int
    review_start_frame: int
    review_end_frame: int
    pilot_stratum: str
    note: str

    def __post_init__(self) -> None:
        if self.metadata.video_id != self.key.video_id:
            raise ValueError(
                f"target {self.key}: metadata video_id {self.metadata.video_id!r} differs"
            )
        frame_values = (
            self.gt_first_frame,
            self.timeline_interval_start,
            self.timeline_interval_end,
            self.live_transition_frame,
            self.review_start_frame,
            self.review_end_frame,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in frame_values):
            raise ValueError(f"target {self.key}: frame fields must be integers")
        if not 0 <= self.review_start_frame <= self.live_transition_frame <= self.gt_first_frame:
            raise ValueError(f"target {self.key}: transition and GT frames precede the review window")
        if not self.gt_first_frame < self.review_end_frame <= self.metadata.frame_count:
            raise ValueError(f"target {self.key}: GT frame or review end is outside video bounds")
        if not 0 <= self.timeline_interval_start <= self.gt_first_frame < self.timeline_interval_end:
            raise ValueError(f"target {self.key}: GT frame is outside its timeline interval")
        if self.timeline_interval_end > self.metadata.frame_count:
            raise ValueError(f"target {self.key}: timeline interval exceeds the video")
        if self.timeline_interval_start != self.live_transition_frame:
            raise ValueError(f"target {self.key}: timeline start and live transition differ")
        if (
            not self.gt_first_type_en
            or not self.timeline_truth
            or not self.preceding_truth
            or not self.pilot_stratum
        ):
            raise ValueError(f"target {self.key}: timeline context and pilot stratum are required")


@dataclass(frozen=True)
class RallyStartDecision:
    """Compact human decision stored separately from source-derived context."""

    key: RallyStartKey
    review_status: ReviewStatus
    serve_visibility: ServeVisibility | None = None
    visible_serve_frame: int | None = None
    first_visible_rally_frame: int | None = None
    broadcast_return_frame: int | None = None
    confidence: Confidence | None = None
    review_note: str = ""

    @classmethod
    def pending(cls, key: RallyStartKey) -> RallyStartDecision:
        """Create one blank pending decision."""
        return cls(key=key, review_status=ReviewStatus.PENDING)


@dataclass(frozen=True)
class SavedDecisionAction:
    """The one persisted row that can be restored in this process."""

    row_index: int
    previous: RallyStartDecision
    saved: RallyStartDecision


DecisionWriter = Callable[
    [Path, Sequence[RallyStartDecision], Sequence[RallyStartTarget]],
    None,
]


def _parse_int(value: object, field: str, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source}: {field} must be an integer, got {value!r}")
    try:
        parsed = float(str(value))
    except ValueError as exc:
        raise ValueError(f"{source}: {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(f"{source}: {field} must be an integer, got {value!r}")
    return int(parsed)


def _parse_float(value: object, field: str, source: str) -> float:
    try:
        parsed = float(str(value))
    except ValueError as exc:
        raise ValueError(f"{source}: {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{source}: {field} must be finite, got {value!r}")
    return parsed


def _parse_bool(value: object, field: str, source: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{source}: {field} must be true or false, got {value!r}")


def _parse_optional_int(value: object, field: str, source: str) -> int | None:
    if not str(value).strip():
        return None
    return _parse_int(value, field, source)


def _parse_enum(
    enum_type: type[EnumT],
    value: object,
    field: str,
    source: str,
) -> EnumT:
    try:
        return enum_type(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{source}: invalid {field} {value!r}") from exc


def _parse_optional_enum(
    enum_type: type[EnumT],
    value: object,
    field: str,
    source: str,
) -> EnumT | None:
    if not str(value).strip():
        return None
    return _parse_enum(enum_type, value, field, source)


def _key_from_row(row: Mapping[str, object], source: str) -> RallyStartKey:
    video_id = str(row.get("video_id", "")).strip()
    set_id = str(row.get("set_id", "")).strip()
    rally = _parse_int(row.get("rally", ""), "rally", source)
    try:
        return RallyStartKey(video_id, set_id, rally)
    except ValueError as exc:
        raise ValueError(f"{source}: {exc}") from exc


def target_from_row(row: Mapping[str, object], source: str) -> RallyStartTarget:
    """Parse immutable review context from one full target row."""
    key = _key_from_row(row, source)
    metadata = VideoMetadata(
        key.video_id,
        _parse_float(row.get("fps", ""), "fps", source),
        _parse_int(row.get("frame_count", ""), "frame_count", source),
    )
    try:
        return RallyStartTarget(
            key=key,
            metadata=metadata,
            gt_first_frame=_parse_int(row.get("gt_first_frame", ""), "gt_first_frame", source),
            gt_first_type_en=str(row.get("gt_first_type_en", "")),
            gt_first_flaw=_parse_bool(row.get("gt_first_flaw", ""), "gt_first_flaw", source),
            timeline_truth=str(row.get("timeline_truth", "")),
            timeline_interval_start=_parse_int(
                row.get("timeline_interval_start", ""), "timeline_interval_start", source
            ),
            timeline_interval_end=_parse_int(
                row.get("timeline_interval_end", ""), "timeline_interval_end", source
            ),
            preceding_truth=str(row.get("preceding_truth", "")),
            live_transition_frame=_parse_int(
                row.get("live_transition_frame", ""), "live_transition_frame", source
            ),
            review_start_frame=_parse_int(
                row.get("review_start_frame", ""), "review_start_frame", source
            ),
            review_end_frame=_parse_int(
                row.get("review_end_frame", ""), "review_end_frame", source
            ),
            pilot_stratum=str(row.get("pilot_stratum", "")),
            note=str(row.get("note", "")),
        )
    except ValueError as exc:
        raise ValueError(f"{source}: {exc}") from exc


def decision_from_row(row: Mapping[str, object], source: str) -> RallyStartDecision:
    """Parse one compact decision from a CSV-shaped mapping."""
    return RallyStartDecision(
        key=_key_from_row(row, source),
        review_status=_parse_enum(
            ReviewStatus, row.get("review_status", ""), "review_status", source
        ),
        serve_visibility=_parse_optional_enum(
            ServeVisibility, row.get("serve_visibility", ""), "serve_visibility", source
        ),
        visible_serve_frame=_parse_optional_int(
            row.get("visible_serve_frame", ""), "visible_serve_frame", source
        ),
        first_visible_rally_frame=_parse_optional_int(
            row.get("first_visible_rally_frame", ""), "first_visible_rally_frame", source
        ),
        broadcast_return_frame=_parse_optional_int(
            row.get("broadcast_return_frame", ""), "broadcast_return_frame", source
        ),
        confidence=_parse_optional_enum(
            Confidence, row.get("confidence", ""), "confidence", source
        ),
        review_note=str(row.get("review_note", "")),
    )


def _validate_full_target_row(
    row: Mapping[str, object],
    target: RallyStartTarget,
    source: str,
) -> None:
    if _parse_int(row.get("gt_first_ball_round", ""), "gt_first_ball_round", source) != 1:
        raise ValueError(f"{source}: gt_first_ball_round must be 1")
    if _parse_int(row.get("gt_first_server", ""), "gt_first_server", source) != 1:
        raise ValueError(f"{source}: gt_first_server must be 1")
    if str(row.get("committed_status", "")) != EXPECTED_COMMITTED_STATUS:
        raise ValueError(f"{source}: committed_status differs from the target contract")
    if _parse_int(row.get("later_strokes_matched", ""), "later_strokes_matched", source) <= 0:
        raise ValueError(f"{source}: later_strokes_matched must be positive")
    expected_offset = target.gt_first_frame - target.live_transition_frame
    actual_offset = _parse_int(
        row.get("frames_from_live_transition", ""),
        "frames_from_live_transition",
        source,
    )
    if actual_offset != expected_offset:
        raise ValueError(
            f"{source}: frames_from_live_transition {actual_offset} != {expected_offset}"
        )


def validate_decision(
    decision: RallyStartDecision,
    target: RallyStartTarget | None = None,
    *,
    source: str = "decision",
) -> None:
    """Validate pending blanks or the conditional four-state review contract."""
    if target is not None and decision.key != target.key:
        raise ValueError(f"{source}: decision key {decision.key} != target key {target.key}")
    fields = (
        decision.serve_visibility,
        decision.visible_serve_frame,
        decision.first_visible_rally_frame,
        decision.broadcast_return_frame,
        decision.confidence,
    )
    if decision.review_status is ReviewStatus.PENDING:
        if any(value is not None for value in fields) or decision.review_note != "":
            raise ValueError(f"{source}: pending rows require blank decision fields")
        return

    visibility = decision.serve_visibility
    if visibility is None:
        raise ValueError(f"{source}: reviewed rows require serve_visibility")
    expected_confidence = (
        Confidence.UNCERTAIN if visibility is ServeVisibility.UNCERTAIN else Confidence.CERTAIN
    )
    if decision.confidence is not expected_confidence:
        raise ValueError(
            f"{source}: {visibility.value} requires confidence={expected_confidence.value}"
        )
    if not decision.review_note.strip():
        raise ValueError(f"{source}: reviewed rows require review_note")

    visible = decision.visible_serve_frame
    first_visible = decision.first_visible_rally_frame
    broadcast_return = decision.broadcast_return_frame
    if visibility is ServeVisibility.VISIBLE:
        if visible is None or first_visible is not None or broadcast_return is not None:
            raise ValueError(f"{source}: visible requires only visible_serve_frame")
    elif visibility is ServeVisibility.BROADCAST_OMITTED:
        if (
            visible is not None
            or first_visible is None
            or broadcast_return is None
            or broadcast_return > first_visible
        ):
            raise ValueError(
                f"{source}: broadcast-omitted requires ordered return and visible-rally frames"
            )
    elif any(marker is not None for marker in (visible, first_visible, broadcast_return)):
        raise ValueError(f"{source}: {visibility.value} leaves all frame markers blank")

    for field, marker in (
        ("visible_serve_frame", visible),
        ("first_visible_rally_frame", first_visible),
        ("broadcast_return_frame", broadcast_return),
    ):
        if marker is None:
            continue
        if marker < 0:
            raise ValueError(f"{source}: {field} must not be negative")
        if target is not None and not target.review_start_frame <= marker < target.review_end_frame:
            raise ValueError(f"{source}: {field} falls outside the review window")
        if target is not None and marker >= target.metadata.frame_count:
            raise ValueError(f"{source}: {field} falls outside the video")


def decision_to_row(decision: RallyStartDecision) -> dict[str, str]:
    """Serialize one decision using the compact stable schema."""
    return {
        "video_id": decision.key.video_id,
        "set_id": decision.key.set_id,
        "rally": str(decision.key.rally),
        "review_status": decision.review_status.value,
        "serve_visibility": "" if decision.serve_visibility is None else decision.serve_visibility.value,
        "visible_serve_frame": "" if decision.visible_serve_frame is None else str(decision.visible_serve_frame),
        "first_visible_rally_frame": (
            "" if decision.first_visible_rally_frame is None else str(decision.first_visible_rally_frame)
        ),
        "broadcast_return_frame": (
            "" if decision.broadcast_return_frame is None else str(decision.broadcast_return_frame)
        ),
        "confidence": "" if decision.confidence is None else decision.confidence.value,
        "review_note": decision.review_note,
    }


def _read_rows(path: Path, expected_columns: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.name.endswith(".csv.gz"):
        raise ValueError(f"CSV path must end in .csv.gz: {path}")
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        if columns != expected_columns:
            raise ValueError(f"{path}: columns {columns!r} != expected {expected_columns!r}")
        rows = list(reader)
    for row_number, row in enumerate(rows, start=2):
        if set(row) != set(expected_columns) or any(value is None for value in row.values()):
            raise ValueError(f"{path} row {row_number}: row width does not match header")
    return rows


def validate_targets(targets: Sequence[RallyStartTarget], source: str = "targets") -> VideoMetadata:
    """Require non-empty, unique, one-video targets with shared metadata."""
    if not targets:
        raise ValueError(f"{source}: no target rows")
    metadata = targets[0].metadata
    keys: set[RallyStartKey] = set()
    for row_number, target in enumerate(targets, start=2):
        if target.key in keys:
            raise ValueError(f"{source} row {row_number}: duplicate target key {target.key}")
        if target.metadata != metadata:
            raise ValueError(f"{source} row {row_number}: target metadata differs")
        keys.add(target.key)
    return metadata


def read_target_csv(path: Path) -> list[RallyStartTarget]:
    """Read a strict immutable full-target table in source order."""
    path = Path(path)
    rows = _read_rows(path, TARGET_COLUMNS)
    targets: list[RallyStartTarget] = []
    for row_number, row in enumerate(rows, start=2):
        source = f"{path} row {row_number}"
        target = target_from_row(row, source)
        _validate_full_target_row(row, target, source)
        template = decision_from_row(row, source)
        validate_decision(template, target, source=source)
        if template.review_status is not ReviewStatus.PENDING:
            raise ValueError(f"{source}: full targets must contain pending event templates")
        targets.append(target)
    validate_targets(targets, str(path))
    return targets


def _decisions_in_target_order(
    decisions: Sequence[RallyStartDecision],
    targets: Sequence[RallyStartTarget],
    source: str,
) -> list[RallyStartDecision]:
    validate_targets(targets)
    by_key: dict[RallyStartKey, RallyStartDecision] = {}
    for row_number, decision in enumerate(decisions, start=2):
        if decision.key in by_key:
            raise ValueError(f"{source} row {row_number}: duplicate decision key {decision.key}")
        by_key[decision.key] = decision
    target_keys = {target.key for target in targets}
    decision_keys = set(by_key)
    if decision_keys != target_keys:
        missing = sorted(target_keys - decision_keys)
        extra = sorted(decision_keys - target_keys)
        raise ValueError(f"{source}: decision key mismatch; missing={missing}, extra={extra}")
    ordered = [by_key[target.key] for target in targets]
    for row_number, (decision, target) in enumerate(zip(ordered, targets), start=2):
        validate_decision(decision, target, source=f"{source} row {row_number}")
    return ordered


def read_decision_csv(
    path: Path,
    targets: Sequence[RallyStartTarget] | None = None,
) -> list[RallyStartDecision]:
    """Read compact decisions and optionally reconcile them to full targets."""
    path = Path(path)
    decisions: list[RallyStartDecision] = []
    for row_number, row in enumerate(_read_rows(path, DECISION_COLUMNS), start=2):
        source = f"{path} row {row_number}"
        decision = decision_from_row(row, source)
        validate_decision(decision, source=source)
        decisions.append(decision)
    if targets is None:
        keys: set[RallyStartKey] = set()
        for row_number, decision in enumerate(decisions, start=2):
            if decision.key in keys:
                raise ValueError(f"{path} row {row_number}: duplicate decision key {decision.key}")
            keys.add(decision.key)
        return decisions
    return _decisions_in_target_order(decisions, targets, str(path))


def build_decision_seed(
    targets: Sequence[RallyStartTarget],
    reviewed_decisions: Sequence[RallyStartDecision],
) -> list[RallyStartDecision]:
    """Combine reviewed pilot rows with pending rows for every other target."""
    validate_targets(targets)
    reviewed_by_key: dict[RallyStartKey, RallyStartDecision] = {}
    target_keys = {target.key for target in targets}
    for row_number, decision in enumerate(reviewed_decisions, start=2):
        if decision.review_status is not ReviewStatus.REVIEWED:
            raise ValueError(f"reviewed decisions row {row_number}: expected reviewed status")
        if decision.key in reviewed_by_key:
            raise ValueError(f"reviewed decisions row {row_number}: duplicate key {decision.key}")
        if decision.key not in target_keys:
            raise ValueError(f"reviewed decisions row {row_number}: unknown target key {decision.key}")
        reviewed_by_key[decision.key] = decision
    seed = [reviewed_by_key.get(target.key, RallyStartDecision.pending(target.key)) for target in targets]
    return _decisions_in_target_order(seed, targets, "decision seed")


def _write_candidate(
    destination: Path,
    decisions: Sequence[RallyStartDecision],
    targets: Sequence[RallyStartTarget],
) -> Path:
    ordered = _decisions_in_target_order(decisions, targets, str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name[:-7]}.",
        suffix=".tmp.csv.gz",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_handle:
                    writer = csv.DictWriter(
                        text_handle,
                        fieldnames=DECISION_COLUMNS,
                        lineterminator="\n",
                    )
                    writer.writeheader()
                    writer.writerows(decision_to_row(decision) for decision in ordered)
        reloaded = read_decision_csv(temporary, targets)
        if reloaded != ordered:
            raise RuntimeError(f"decision CSV round trip changed values: {destination}")
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_output_path(path: Path) -> Path:
    path = Path(path)
    if not path.name.endswith(".csv.gz"):
        raise ValueError(f"decision path must end in .csv.gz: {path}")
    if path.name.endswith(CANONICAL_TIMELINE_SUFFIX):
        raise ValueError(f"decision path uses a protected canonical timeline name: {path}")
    return path


def validate_decision_output_path(path: Path) -> Path:
    """Reject non-gzip and canonical-timeline decision destinations."""
    return _validate_output_path(path)


def write_decision_csv(
    path: Path,
    decisions: Sequence[RallyStartDecision],
    targets: Sequence[RallyStartTarget],
) -> None:
    """Atomically replace a compact decision table after a strict round trip."""
    path = _validate_output_path(path)
    temporary = _write_candidate(path, decisions, targets)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def initialize_decision_csv(
    path: Path,
    decisions: Sequence[RallyStartDecision],
    targets: Sequence[RallyStartTarget],
) -> None:
    """Atomically create a missing decision table without replacing human work."""
    path = _validate_output_path(path)
    temporary = _write_candidate(path, decisions, targets)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RallyStartAuditSession:
    """Proposal-keyed event state with failure-safe save and one-step undo."""

    def __init__(
        self,
        targets: Sequence[RallyStartTarget],
        decisions: Sequence[RallyStartDecision],
        decision_path: Path | None = None,
    ) -> None:
        self._targets = tuple(targets)
        self._decisions = _decisions_in_target_order(decisions, targets, "session decisions")
        self._index = self._first_pending_index()
        self._draft = self._decisions[self._index]
        self._undo_action: SavedDecisionAction | None = None
        self._decision_path = (
            None
            if decision_path is None
            else _validate_output_path(Path(decision_path)).resolve()
        )

    @property
    def targets(self) -> tuple[RallyStartTarget, ...]:
        return self._targets

    @property
    def decisions(self) -> tuple[RallyStartDecision, ...]:
        return tuple(self._decisions)

    @property
    def row_index(self) -> int:
        return self._index

    @property
    def current_target(self) -> RallyStartTarget:
        return self._targets[self._index]

    @property
    def current_decision(self) -> RallyStartDecision:
        return self._decisions[self._index]

    @property
    def draft(self) -> RallyStartDecision:
        return self._draft

    @property
    def dirty(self) -> bool:
        return self._draft != self.current_decision

    @property
    def reviewed_count(self) -> int:
        return sum(decision.review_status is ReviewStatus.REVIEWED for decision in self._decisions)

    @property
    def pending_count(self) -> int:
        return len(self._decisions) - self.reviewed_count

    @property
    def decision_path(self) -> Path | None:
        return self._decision_path

    def _first_pending_index(self) -> int:
        return next(
            (
                index
                for index, decision in enumerate(self._decisions)
                if decision.review_status is ReviewStatus.PENDING
            ),
            0,
        )

    def _set_draft(self, **changes: object) -> None:
        self._draft = replace(self._draft, **changes)

    def select_visibility(self, visibility: ServeVisibility) -> None:
        """Select one state and clear marker fields that it cannot use."""
        confidence = Confidence.UNCERTAIN if visibility is ServeVisibility.UNCERTAIN else Confidence.CERTAIN
        changes: dict[str, object] = {
            "review_status": ReviewStatus.REVIEWED,
            "serve_visibility": visibility,
            "confidence": confidence,
        }
        if visibility is ServeVisibility.VISIBLE:
            changes.update(first_visible_rally_frame=None, broadcast_return_frame=None)
        elif visibility is ServeVisibility.BROADCAST_OMITTED:
            changes.update(visible_serve_frame=None)
        else:
            changes.update(
                visible_serve_frame=None,
                first_visible_rally_frame=None,
                broadcast_return_frame=None,
            )
        self._set_draft(**changes)

    def _require_frame(self, frame: int) -> None:
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError(f"frame must be an integer, got {frame!r}")
        target = self.current_target
        if not target.review_start_frame <= frame < target.review_end_frame:
            raise ValueError(
                f"frame {frame} is outside review window "
                f"[{target.review_start_frame}, {target.review_end_frame})"
            )

    def capture_visible_serve(self, frame: int) -> None:
        if self._draft.serve_visibility is not ServeVisibility.VISIBLE:
            raise ValueError("select visible before capturing visible_serve_frame")
        self._require_frame(frame)
        self._set_draft(visible_serve_frame=frame)

    def capture_broadcast_return(self, frame: int) -> None:
        if self._draft.serve_visibility is not ServeVisibility.BROADCAST_OMITTED:
            raise ValueError("select broadcast-omitted before capturing broadcast_return_frame")
        self._require_frame(frame)
        self._set_draft(broadcast_return_frame=frame)

    def capture_first_visible_rally(self, frame: int) -> None:
        if self._draft.serve_visibility is not ServeVisibility.BROADCAST_OMITTED:
            raise ValueError("select broadcast-omitted before capturing first_visible_rally_frame")
        self._require_frame(frame)
        self._set_draft(first_visible_rally_frame=frame)

    def set_note(self, note: str) -> None:
        self._set_draft(review_note=str(note))

    def validate_draft(self) -> None:
        validate_decision(self._draft, self.current_target, source=f"draft row {self._index + 1}")

    def clear_draft(self) -> bool:
        if not self.dirty:
            return False
        self._draft = self.current_decision
        return True

    def move_row(self, offset: int) -> int:
        if self.dirty:
            raise ValueError("save or clear the unsaved draft before changing rows")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset == 0:
            raise ValueError(f"row offset must be a non-zero integer, got {offset!r}")
        self._index = min(max(self._index + offset, 0), len(self._targets) - 1)
        self._draft = self.current_decision
        return self._index

    def _output_path_for(self, path: Path) -> Path:
        resolved = _validate_output_path(Path(path)).resolve()
        if self._decision_path is not None and resolved != self._decision_path:
            raise ValueError(
                f"session decision path is {self._decision_path}, not {resolved}"
            )
        return resolved

    def save(
        self,
        path: Path,
        writer: DecisionWriter | None = None,
    ) -> RallyStartDecision:
        """Persist the draft before changing live session state."""
        if not self.dirty:
            raise ValueError("draft has no changes")
        self.validate_draft()
        output_path = self._output_path_for(path)
        row_index = self._index
        previous = self.current_decision
        saved = self._draft
        candidate = list(self._decisions)
        candidate[row_index] = saved
        active_writer = write_decision_csv if writer is None else writer
        active_writer(output_path, candidate, self._targets)

        self._decisions = candidate
        self._decision_path = output_path
        self._undo_action = SavedDecisionAction(row_index, previous, saved)
        pending_after = next(
            (
                index
                for index in range(row_index + 1, len(candidate))
                if candidate[index].review_status is ReviewStatus.PENDING
            ),
            None,
        )
        if pending_after is not None:
            self._index = pending_after
        else:
            first_pending = next(
                (
                    index
                    for index, decision in enumerate(candidate)
                    if decision.review_status is ReviewStatus.PENDING
                ),
                None,
            )
            self._index = row_index if first_pending is None else first_pending
        self._draft = self.current_decision
        return saved

    def undo(
        self,
        path: Path,
        writer: DecisionWriter | None = None,
    ) -> UndoResult:
        """Clear an unsaved draft or atomically restore the last saved row."""
        if self.clear_draft():
            return UndoResult.DRAFT_CLEARED
        if self._undo_action is None:
            return UndoResult.NOTHING_TO_UNDO
        output_path = self._output_path_for(path)
        action = self._undo_action
        candidate = list(self._decisions)
        candidate[action.row_index] = action.previous
        active_writer = write_decision_csv if writer is None else writer
        active_writer(output_path, candidate, self._targets)

        self._decisions = candidate
        self._index = action.row_index
        self._draft = action.previous
        self._undo_action = None
        return UndoResult.SAVED_ROW_RESTORED
