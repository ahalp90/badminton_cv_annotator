"""Typed input loading for the corrected contact and refit EDA."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, TypedDict

import numpy as np
import pandas as pd

from annotator import point_winner
from annotator.calibration.fixtures import FIXTURES, Fixture
from annotator.calibration.gt_scoring import load_gt_tables
from annotator.calibration.scoring import (
    GtRally,
    RallyBoundary,
    classify_all,
    load_gt_rallies,
)
from annotator.fps_constants import scale_for_fps
from annotator.inpaint_guard import grade_track
from annotator.rally_segmentation import build_sticky_result, tracker_segments
from annotator.replay_mask import _read_homography_rows
from annotator.types import StickyResult
from classifier_shared.dataset import SET_INFO_DIR
from classifier_shared.player_mapping import collect_shots
from classifier_shared.taxonomy import STROKE_TYPES_19_ZH

RUN_DIR = Path(__file__).resolve().parent
FIXTURE_ROOT = RUN_DIR / "inputs/fixtures"
ASSET_ROOT = RUN_DIR / "assets/shuttleset-current-annotator-reference-v1"
RELEASE_RESULTS = ASSET_ROOT / "measurement/current_annotator_8config_288p/static_shuttleset_homography"
PRODUCER_EVIDENCE = ASSET_ROOT / "inputs/tracknet_producer_evidence"

Half = point_winner.Half
OTHER_HALF = point_winner.OTHER_HALF
GtTables: TypeAlias = tuple[pd.DataFrame, pd.DataFrame, dict[int, Any], pd.DataFrame]
Boundary = tuple[RallyBoundary, int | None]
TruthKey = tuple[str, int]


class FirstSecondTruth(TypedDict):
    """Exact first and second ShuttleSet contacts for one rally."""

    gt_first_frame: int
    gt_second_frame: int | None
    gt_server: Half | None
    gt_receiver: Half | None


@dataclass(frozen=True, slots=True)
class RawContact:
    """One released impulse candidate and its independent rejection fields."""

    rally_id: int
    contact_frame: int
    proximity_ok: bool | None
    wrist_near: bool | None
    suppressed: bool | None
    definitive_exclusion: bool

    @property
    def accepted(self) -> bool:
        """Whether the candidate survives the scorer's contact gates and mask."""
        return (
            self.wrist_near is not False
            and self.suppressed is not True
            and not self.definitive_exclusion
        )


@dataclass(frozen=True, slots=True)
class VideoData:
    """Frame-aligned release data and GT joins for one fixture."""

    fixture: Fixture
    track: np.ndarray
    bboxes: np.ndarray
    sticky: StickyResult
    segments: list[tuple[int, int]]
    guard_codes: np.ndarray
    producer_inpaint: np.ndarray
    court_present: np.ndarray
    keep_vote: np.ndarray
    raw_replay: np.ndarray
    definitive_exclusion: np.ndarray
    annotations: dict[str, Any]
    spans: list[tuple[int, int]]
    accepted_by_span: dict[int, list[int]]
    raw_contacts_by_span: dict[int, list[RawContact]]
    gt_rallies: list[GtRally]
    boundaries: list[Boundary]
    truth_first_second: dict[TruthKey, FirstSecondTruth]

    def segment_for_frame(self, frame: int) -> tuple[int, int] | None:
        """Return the court/tracker segment containing ``frame``, if any."""
        return segment_for_frame(self.segments, frame)

    @property
    def boundary_by_rally(self) -> dict[TruthKey, Boundary]:
        """Return boundary classifications keyed by the stable GT rally key."""
        return {
            (rally.set_id, rally.rally): boundary
            for rally, boundary in zip(self.gt_rallies, self.boundaries)
        }


def normalise_half(value: object) -> Half | None:
    """Map a repository half enum or its string spelling to ``Half``."""
    if isinstance(value, Half):
        return value
    if isinstance(value, str):
        if value.upper() == Half.TOP.value.upper():
            return Half.TOP
        if value.upper() in {Half.BOT.value.upper(), "BOTTOM"}:
            return Half.BOT
    return None


def other_half(half: Half) -> Half:
    """Return the opposite court half."""
    return OTHER_HALF[half]


def load_sidecar_mask(path: Path, n_frames: int) -> np.ndarray:
    """Load and validate a stride-8 producer inpaint mask."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"producer sidecar is not an object: {path}")
    if payload.get("schema") != "inpaint_fill_mask/1" or payload.get("index_space") != "frame":
        raise ValueError(f"unsupported producer sidecar: {path}")
    if payload.get("stride") != 8 or payload.get("n_rows") != n_frames:
        raise ValueError(f"misaligned stride-8 producer sidecar: {path}")

    selected = payload.get("inpaint_selected", [])
    if not isinstance(selected, list):
        raise TypeError(f"malformed producer spans: {path}")
    mask = np.zeros(n_frames, dtype=bool)
    previous_end = 0
    for span in selected:
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError(f"malformed producer span: {span!r}")
        start, end = span
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError(f"non-integer producer span: {span!r}")
        if start < previous_end or start < 0 or end > n_frames or end <= start:
            raise ValueError(f"unsorted producer span: {span!r}")
        mask[start:end] = True
        previous_end = end
    return mask


def read_annotations(path: Path) -> dict[str, Any]:
    """Read a frozen result payload and validate its span-aligned fields."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: annotations payload is not an object")
    try:
        n_spans = len(payload["spans"])
    except (KeyError, TypeError):
        raise ValueError(f"{path}: annotations payload has no valid spans") from None
    for field in ("fitted_first_all", "striker_halves", "n_strokes_list"):
        try:
            field_length = len(payload[field])
        except (KeyError, TypeError):
            raise ValueError(f"{path}: {field} is missing or not sized") from None
        if field_length != n_spans:
            raise ValueError(f"{path}: {field} is not span-aligned")
    return payload


def truth_first_second(fixture: Fixture) -> dict[TruthKey, FirstSecondTruth]:
    """Load exact ShuttleSet first and second rows with mapped court halves."""
    match = pd.read_csv(SET_INFO_DIR / "match.csv").set_index("id")
    if fixture.video_id not in match.index:
        raise ValueError(f"{fixture.name}: match.csv has no video id {fixture.video_id}")
    shots = collect_shots(SET_INFO_DIR, match.loc[fixture.video_id], STROKE_TYPES_19_ZH)
    result: dict[TruthKey, FirstSecondTruth] = {}
    for (set_number, rally), group in shots.groupby(["set", "rally"]):
        first = group[group["ball_round"] == 1]
        second = group[group["ball_round"] == 2]
        if len(first) != 1:
            raise ValueError(
                f"{fixture.name} set{set_number}/r{rally}: expected one first contact, got {len(first)}"
            )
        if len(second) > 1:
            raise ValueError(f"{fixture.name} set{set_number}/r{rally}: expected at most one second contact")
        first_row = first.iloc[0]
        second_row = second.iloc[0] if len(second) else None
        result[(f"set{int(set_number)}", int(rally))] = {
            "gt_first_frame": int(first_row["frame_num"]),
            "gt_second_frame": int(second_row["frame_num"]) if second_row is not None else None,
            "gt_server": normalise_half(first_row["player"]),
            "gt_receiver": normalise_half(second_row["player"]) if second_row is not None else None,
        }
    return result


def segment_for_frame(segments: Iterable[tuple[int, int]], frame: int) -> tuple[int, int] | None:
    """Return the half-open segment containing ``frame``, or ``None``."""
    for start, end in segments:
        if start <= frame < end:
            return start, end
    return None


def _release_dir(fixture: Fixture) -> Path:
    """Return the static stride-8 release directory for one fixture."""
    return RELEASE_RESULTS / fixture.name / "tracknet-stride-8"


def _load_frame_array(path: Path, n_frames: int, *, require_bool: bool = False) -> np.ndarray:
    """Load one non-object array and check its frame axis."""
    array = np.load(path, allow_pickle=False)
    if array.ndim == 0 or array.shape[0] != n_frames:
        raise ValueError(f"{path}: first dimension {array.shape} is not frame-aligned to {n_frames}")
    if require_bool and (array.ndim != 1 or array.dtype != np.bool_):
        raise ValueError(f"{path}: expected one-dimensional bool array, got {array.shape} {array.dtype}")
    return array


def _parse_spans(values: object, path: Path) -> list[tuple[int, int]]:
    """Parse half-open rally spans from the frozen JSON payload."""
    if not isinstance(values, list):
        raise TypeError(f"{path}: spans is not a list")
    spans: list[tuple[int, int]] = []
    for span in values:
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError(f"{path}: malformed span {span!r}")
        start, end = span
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            raise ValueError(f"{path}: malformed span {span!r}")
        spans.append((start, end))
    return spans


def _parse_accepted_by_span(values: object, n_frames: int, n_spans: int, path: Path) -> dict[int, list[int]]:
    """Parse the released scorer's accepted frame lists."""
    if not isinstance(values, dict):
        raise TypeError(f"{path}: filtered_by_rally is not an object")
    accepted: dict[int, list[int]] = {}
    for raw_id, raw_frames in values.items():
        try:
            span_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid rally id {raw_id!r}") from error
        if span_id < 0 or span_id >= n_spans or not isinstance(raw_frames, list):
            raise ValueError(f"{path}: invalid filtered frames for span {raw_id!r}")
        frames: list[int] = []
        for frame in raw_frames:
            if not isinstance(frame, int) or frame < 0 or frame >= n_frames:
                raise ValueError(f"{path}: invalid accepted frame {frame!r}")
            frames.append(frame)
        accepted[span_id] = sorted(frames)
    return accepted


def _optional_bool(value: object, field: str) -> bool | None:
    """Validate a nullable boolean field from a contact record."""
    if value is None or isinstance(value, bool):
        return value
    raise TypeError(f"contact field {field} must be bool or null, got {value!r}")


def _parse_raw_contacts(
    values: object,
    definitive_exclusion: np.ndarray,
    n_spans: int,
    path: Path,
) -> dict[int, list[RawContact]]:
    """Parse released raw contacts and attach the definitive-mask verdict."""
    if not isinstance(values, list):
        raise TypeError(f"{path}: contacts is not a list")
    contacts_by_span: dict[int, list[RawContact]] = {}
    for raw_contact in values:
        if not isinstance(raw_contact, Mapping):
            raise TypeError(f"{path}: malformed contact {raw_contact!r}")
        try:
            rally_id = int(raw_contact["rally_id"])
            frame = int(raw_contact["contact_frame"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: malformed contact identifiers {raw_contact!r}") from error
        if rally_id < 0 or rally_id >= n_spans or frame < 0 or frame >= len(definitive_exclusion):
            raise ValueError(f"{path}: contact is outside the released frame/span ranges: {raw_contact!r}")
        contact = RawContact(
            rally_id=rally_id,
            contact_frame=frame,
            proximity_ok=_optional_bool(raw_contact.get("proximity_ok"), "proximity_ok"),
            wrist_near=_optional_bool(raw_contact.get("wrist_near"), "wrist_near"),
            suppressed=_optional_bool(raw_contact.get("suppressed"), "suppressed"),
            definitive_exclusion=bool(definitive_exclusion[frame]),
        )
        contacts_by_span.setdefault(rally_id, []).append(contact)
    for contacts in contacts_by_span.values():
        contacts.sort(key=lambda contact: contact.contact_frame)
    return contacts_by_span


def _validate_accepted_contacts(
    accepted_by_span: dict[int, list[int]], raw_contacts_by_span: dict[int, list[RawContact]], path: Path,
) -> None:
    """Check that released accepted frames match the raw gate and mask fields."""
    raw_accepted = {
        span_id: [contact.contact_frame for contact in contacts if contact.accepted]
        for span_id, contacts in raw_contacts_by_span.items()
    }
    span_ids = set(accepted_by_span) | set(raw_accepted)
    for span_id in span_ids:
        if accepted_by_span.get(span_id, []) != raw_accepted.get(span_id, []):
            raise ValueError(f"{path}: filtered_by_rally disagrees with raw contact gates for span {span_id}")


def _shared_gt_tables(gt_tables: GtTables | None) -> GtTables:
    """Load or return the shared GT and court gate tables."""
    return load_gt_tables() if gt_tables is None else gt_tables


def load_video_data(fixture: Fixture, gt_tables: GtTables | None = None) -> VideoData:
    """Load one prepared fixture and all GT joins needed by the EDA."""
    master, _homography, courts, resolution_table = _shared_gt_tables(gt_tables)
    release_dir = _release_dir(fixture)
    track = np.load(FIXTURE_ROOT / fixture.track_path, allow_pickle=False)
    if track.ndim != 2 or track.shape[1] < 3:
        raise ValueError(f"{fixture.name}: track must have shape (n_frames, 3+), got {track.shape}")
    n_frames = track.shape[0]
    bboxes = _load_frame_array(FIXTURE_ROOT / fixture.pose_path("bboxes"), n_frames)
    scores = _load_frame_array(FIXTURE_ROOT / fixture.pose_path("scores"), n_frames)
    kps = _load_frame_array(FIXTURE_ROOT / fixture.pose_path("kps"), n_frames)
    ndet = _load_frame_array(FIXTURE_ROOT / fixture.pose_path("ndet"), n_frames)
    court_present = _load_frame_array(release_dir / "court_present.npy", n_frames, require_bool=True)
    keep_vote = _load_frame_array(release_dir / "keep_vote.npy", n_frames, require_bool=True)
    raw_replay = _load_frame_array(release_dir / "raw_replay_mask.npy", n_frames, require_bool=True)
    definitive_exclusion = _load_frame_array(
        release_dir / "definitive_exclusion_mask.npy", n_frames, require_bool=True
    )

    scene_rows = _read_homography_rows(release_dir / "scene_rows.csv", str(fixture.video_id))
    if not scene_rows:
        raise ValueError(f"{fixture.name}: scene rows are missing")
    segments = tracker_segments(scene_rows, court_present, n_frames)
    constants = scale_for_fps(fixture.fps)
    gate_courts = {str(video_id): court for video_id, court in courts.items()}
    gate_resolution = resolution_table.copy()
    gate_resolution.index = gate_resolution.index.astype(str)
    sticky = build_sticky_result(
        track,
        segments,
        bboxes,
        scores,
        kps,
        ndet,
        str(fixture.video_id),
        gate_courts,
        gate_resolution,
        fixture.resolution,
        constants.body_unit_half_window,
    )
    guard_codes, _guard_info = grade_track(track)
    sidecar = (
        PRODUCER_EVIDENCE / fixture.name / "tracknet-stride-8"
        / f"{fixture.video_id}_stride8_inpaint_mask.json.gz"
    )
    producer_inpaint = load_sidecar_mask(sidecar, n_frames)

    annotations_path = release_dir / "annotations.json"
    annotations = read_annotations(annotations_path)
    spans = _parse_spans(annotations["spans"], annotations_path)
    accepted_by_span = _parse_accepted_by_span(
        annotations["filtered_by_rally"], n_frames, len(spans), annotations_path
    )
    raw_contacts_by_span = _parse_raw_contacts(
        annotations["contacts"], definitive_exclusion, len(spans), annotations_path
    )
    _validate_accepted_contacts(accepted_by_span, raw_contacts_by_span, annotations_path)

    gt_rallies = load_gt_rallies(master, fixture.video_id)
    boundaries = classify_all(spans, gt_rallies)
    truth = truth_first_second(fixture)
    gt_keys = {(rally.set_id, rally.rally) for rally in gt_rallies}
    if gt_keys != set(truth):
        raise ValueError(f"{fixture.name}: ShuttleSet truth keys do not match GT rallies")
    return VideoData(
        fixture=fixture,
        track=track,
        bboxes=bboxes,
        sticky=sticky,
        segments=segments,
        guard_codes=guard_codes,
        producer_inpaint=producer_inpaint,
        court_present=court_present,
        keep_vote=keep_vote,
        raw_replay=raw_replay,
        definitive_exclusion=definitive_exclusion,
        annotations=annotations,
        spans=spans,
        accepted_by_span=accepted_by_span,
        raw_contacts_by_span=raw_contacts_by_span,
        gt_rallies=gt_rallies,
        boundaries=boundaries,
        truth_first_second=truth,
    )


def load_all_video_data(
    fixtures: Iterable[Fixture] = FIXTURES, gt_tables: GtTables | None = None,
) -> list[VideoData]:
    """Load every fixture while sharing the verified GT and gate tables."""
    shared_tables = _shared_gt_tables(gt_tables)
    return [load_video_data(fixture, shared_tables) for fixture in fixtures]
