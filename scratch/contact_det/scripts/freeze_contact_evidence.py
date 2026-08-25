"""Freeze label-blind contact and player-geometry evidence.

The freezer reads only the standard vision stages and the saved annotation result.
It deliberately has no ShuttleSet-table import.  The scorer consumes the resulting
gzip JSON after checking its manifest and checksum.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


ANNOTATOR_RESULT_SCHEMA = "annotator-result/0.1"
EVIDENCE_SCHEMA = "contact-evidence-freeze/1"
MANIFEST_SCHEMA = "contact-evidence-manifest/1"
EVIDENCE_FILENAME = "contact_evidence.json.gz"
MANIFEST_FILENAME = "contact_evidence_manifest.json"
RESOLUTION = (1920.0, 1080.0)
FORBIDDEN_KEY_TOKENS = ("gt", "label", "truth", "correct", "score")
FIXTURE_SPECS = {
    "sset_01": (1, 25.0),
    "sset_15": (15, 25.0),
    "sset_21": (21, 30.0),
}


@dataclass(frozen=True)
class FixtureSpec:
    """Fixed identity and capture metadata for one standard stage fixture."""

    name: str
    video_id: int
    fps: float
    width: float = RESOLUTION[0]
    height: float = RESOLUTION[1]


@dataclass(frozen=True)
class AnnotationData:
    """Label-blind fields restored from one saved annotation result."""

    spans: tuple[tuple[int, int], ...]
    contacts: tuple[dict[str, Any], ...]
    filtered_contacts: tuple[dict[str, Any], ...]
    filtered_by_rally: dict[int, tuple[int, ...]]
    striker_halves: tuple[str | None, ...]
    fitted_first_all: tuple[str | None, ...]


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _normalise_half(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be Top, Bot, Bottom, or null")
    if value == "Top":
        return value
    if value in {"Bot", "Bottom"}:
        return "Bot"
    raise ValueError(f"{name} has unexpected half {value!r}")


def _optional_bool(value: object, name: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be bool or null")


def _parse_spans(value: object, name: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    spans: list[tuple[int, int]] = []
    previous_end = -1
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{name} contains a malformed span {item!r}")
        start, end = item
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ValueError(f"{name} contains invalid span {item!r}")
        if start < previous_end:
            raise ValueError(f"{name} must be ordered and non-overlapping")
        spans.append((start, end))
        previous_end = end
    return tuple(spans)


def _parse_contact(value: object, name: str) -> dict[str, Any]:
    record = _mapping(value, name)
    required = ("rally_id", "contact_frame")
    if any(field not in record for field in required):
        raise ValueError(f"{name} is missing a contact identity")
    rally_id = record["rally_id"]
    frame = record["contact_frame"]
    if (
        isinstance(rally_id, bool)
        or not isinstance(rally_id, int)
        or isinstance(frame, bool)
        or not isinstance(frame, int)
        or rally_id < 0
        or frame < 0
    ):
        raise ValueError(f"{name} has invalid contact identity")
    return {
        "rally_id": rally_id,
        "contact_frame": frame,
        "proximity_ok": _optional_bool(record.get("proximity_ok"), f"{name}.proximity_ok"),
        "wrist_near": _optional_bool(record.get("wrist_near"), f"{name}.wrist_near"),
        "suppressed": _optional_bool(record.get("suppressed"), f"{name}.suppressed"),
    }


def _parse_frame_map(value: object, n_spans: int, name: str) -> dict[int, tuple[int, ...]]:
    record = _mapping(value, name)
    parsed: dict[int, tuple[int, ...]] = {}
    for raw_id, raw_frames in record.items():
        try:
            span_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} has invalid span id {raw_id!r}") from error
        if span_id < 0 or span_id >= n_spans or not isinstance(raw_frames, list):
            raise ValueError(f"{name} has invalid frames for span {raw_id!r}")
        frames: list[int] = []
        for frame in raw_frames:
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                raise ValueError(f"{name}[{raw_id!r}] has invalid frame {frame!r}")
            frames.append(frame)
        parsed[span_id] = tuple(frames)
    return parsed


def read_annotation(path: Path, fixture: FixtureSpec) -> AnnotationData:
    """Read and validate only the saved, label-blind annotation result."""
    from dataset_builder.vision import load_json_gz

    payload = load_json_gz(path)
    if payload.get("schema") != ANNOTATOR_RESULT_SCHEMA or payload.get("video_id") != fixture.name:
        raise ValueError(f"{path}: annotation identity or schema differs")
    result = _mapping(payload.get("result"), "annotation.result")
    spans = _parse_spans(result.get("spans"), "annotation.spans")
    contacts = tuple(
        _parse_contact(item, f"annotation.contacts[{index}]")
        for index, item in enumerate(result.get("contacts", []))
    )
    filtered_contacts = tuple(
        _parse_contact(item, f"annotation.filtered_contacts[{index}]")
        for index, item in enumerate(result.get("filtered_contacts", []))
    )
    filtered_by_rally = _parse_frame_map(
        result.get("filtered_by_rally"), len(spans), "annotation.filtered_by_rally"
    )
    stored_striker = result.get("striker_halves")
    stored_first = result.get("fitted_first_all")
    if not isinstance(stored_striker, list) or len(stored_striker) != len(spans):
        raise ValueError(f"{path}: striker_halves is not span-aligned")
    if stored_first is None:
        stored_first = [None] * len(spans)
    if not isinstance(stored_first, list) or len(stored_first) != len(spans):
        raise ValueError(f"{path}: fitted_first_all is not span-aligned")
    striker_halves = tuple(
        _normalise_half(value, f"annotation.striker_halves[{index}]")
        for index, value in enumerate(stored_striker)
    )
    fitted_first_all = tuple(
        _normalise_half(value, f"annotation.fitted_first_all[{index}]")
        for index, value in enumerate(stored_first)
    )

    if any(contact["rally_id"] >= len(spans) for contact in contacts + filtered_contacts):
        raise ValueError(f"{path}: contact references a span outside spans")
    raw_keys = Counter((row["rally_id"], row["contact_frame"]) for row in contacts)
    filtered_keys = Counter((row["rally_id"], row["contact_frame"]) for row in filtered_contacts)
    if filtered_keys - raw_keys:
        raise ValueError(f"{path}: filtered_contacts contains a row absent from contacts")
    listed_keys = Counter(
        (span_id, frame)
        for span_id, frames in filtered_by_rally.items()
        for frame in frames
    )
    if listed_keys != filtered_keys:
        raise ValueError(f"{path}: filtered_by_rally disagrees with filtered_contacts")
    for row in contacts + filtered_contacts:
        span_start, span_end = spans[row["rally_id"]]
        if not span_start <= row["contact_frame"] < span_end:
            raise ValueError(f"{path}: contact lies outside its span")
    return AnnotationData(
        spans,
        contacts,
        filtered_contacts,
        filtered_by_rally,
        striker_halves,
        fitted_first_all,
    )


def _stage_paths(data_root: Path, fixture: FixtureSpec) -> dict[str, Path]:
    root = Path(data_root) / "stages"
    fixture_root = fixture.name
    return {
        "shuttle_track": root / "shuttle" / fixture_root / "shuttle_track.npy.xz",
        "pose_kps": root / "pose" / fixture_root / "pose_kps.npy.xz",
        "pose_bboxes": root / "pose" / fixture_root / "pose_bboxes.npy.xz",
        "pose_scores": root / "pose" / fixture_root / "pose_scores.npy.xz",
        "pose_kp_scores": root / "pose" / fixture_root / "pose_kp_scores.npy.xz",
        "pose_ndet": root / "pose" / fixture_root / "pose_ndet.npy.xz",
        "court_evidence": root / "court" / fixture_root / "court_evidence.json.gz",
        "court_keep_vote": root / "court" / fixture_root / "court_keep_vote.npy.xz",
        "court_present": root / "court" / fixture_root / "court_present.npy.xz",
        "annotation": root / "annotation" / fixture_root / "annotator_result.json.gz",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_inputs(
    data_root: Path, fixture: FixtureSpec
) -> tuple[np.ndarray, Any, Any, list[tuple[int, int]], Any, AnnotationData]:
    """Restore arrays, court inputs, sticky evidence, and saved annotation."""
    from annotator.fps_constants import scale_for_fps
    from annotator.rally.evidence import build_sticky_result, tracker_segments
    from dataset_builder.vision import load_court_vision, load_npy_xz, load_pose_arrays

    paths = _stage_paths(data_root, fixture)
    track = load_npy_xz(paths["shuttle_track"])
    if track.ndim != 2 or track.shape[1] != 3:
        raise ValueError(f"{fixture.name}: shuttle track must have shape (n, 3), got {track.shape}")
    frame_count = int(track.shape[0])
    pose = load_pose_arrays(Path(data_root) / "stages" / "pose" / fixture.name, frame_count)
    court = load_court_vision(
        Path(data_root) / "stages" / "court" / fixture.name,
        video_id=fixture.name,
        frame_count=frame_count,
        resolution=RESOLUTION,
    )
    court_inputs = court.evidence.inputs
    if court_inputs is None:
        raise ValueError(f"{fixture.name}: court inputs are unavailable")
    court_present = court.evidence.court_present
    homography_rows = court_inputs.homography_rows.to_dict("records")
    segments = tracker_segments(homography_rows, court_present, frame_count)
    sticky = build_sticky_result(
        track,
        segments,
        pose.bboxes,
        pose.scores,
        pose.kps,
        pose.ndet,
        fixture.name,
        court_inputs.gate_court_info,
        court_inputs.gate_resolution_table,
        RESOLUTION,
        scale_for_fps(fixture.fps).body_unit_half_window,
    )
    annotation = read_annotation(paths["annotation"], fixture)
    return track, pose, court, segments, sticky, annotation


def _selected_wrist_slot(frame: int, sticky: Any) -> int | None:
    distances = np.asarray(sticky.distances_per_slot[frame], dtype=float)
    finite = np.isfinite(distances)
    if not finite.any():
        return None
    # This is intentionally the same nearest-slot choice used by attribute_half.
    return int(np.nanargmin(distances))


def ankle_half(
    frame: int,
    sticky: Any,
    net_band: tuple[float, float],
    image_height: float,
) -> str | None:
    """Apply the frozen image-y ankle rule to the nearest-wrist slot.

    The selected slot is chosen from sticky's body-unit wrist distances.  The
    rule has no inside-band exception: one finite ankle compares directly with
    the calibrated net-band midpoint.
    """
    selected = _selected_wrist_slot(frame, sticky)
    if selected is None:
        return None
    picks = np.asarray(sticky.picks[frame], dtype=int)
    ankles = np.asarray(sticky.ankle_pos[frame, :, 1], dtype=float)
    if selected >= len(ankles) or picks[selected] < 0 or not np.isfinite(ankles[selected]):
        return None
    valid = np.isfinite(ankles) & (picks >= 0)
    valid_count = int(np.count_nonzero(valid))
    if valid_count >= 2:
        other = 1 - selected
        if not valid[other] or ankles[selected] == ankles[other]:
            return None
        return "Top" if ankles[selected] < ankles[other] else "Bot"
    if valid_count == 1:
        midpoint = (float(net_band[0]) + float(net_band[1])) / (2.0 * image_height)
        return "Top" if ankles[selected] < midpoint else "Bot"
    return None


def _valid_ankle_slots(frame: int, sticky: Any) -> int:
    picks = np.asarray(sticky.picks[frame], dtype=int)
    ankles = np.asarray(sticky.ankle_pos[frame, :, 1], dtype=float)
    return int(np.count_nonzero(np.isfinite(ankles) & (picks >= 0)))


def _preceding_scene_distance(frame: int, raw_cuts: Sequence[tuple[int, int]]) -> int | None:
    starts = [start for start, _end in raw_cuts if start <= frame]
    if not starts:
        return None
    return frame - max(starts)


def _half_value(value: Any) -> str | None:
    return None if value is None else str(getattr(value, "value", value))


def _first_half(final_half: str | None, n_contacts: int) -> str | None:
    if final_half is None or n_contacts <= 0:
        return None
    if (n_contacts - 1) % 2 == 0:
        return final_half
    return "Bot" if final_half == "Top" else "Top"


def _forbidden_key_error(value: object, path: str = "evidence") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                return f"{path}: non-string key"
            key_lower = key.casefold()
            if any(token in key_lower for token in FORBIDDEN_KEY_TOKENS):
                return f"{path}.{key}: forbidden frozen key"
            error = _forbidden_key_error(child, f"{path}.{key}")
            if error is not None:
                return error
    elif isinstance(value, list):
        for index, child in enumerate(value):
            error = _forbidden_key_error(child, f"{path}[{index}]")
            if error is not None:
                return error
    return None


def _fixture_evidence(
    data_root: Path, fixture: FixtureSpec
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one fixture's evidence and its path-free input digest rows."""
    from annotator import point_winner

    track, pose, court, segments, sticky, annotation = _load_inputs(data_root, fixture)
    court_inputs = court.evidence.inputs
    assert court_inputs is not None
    n_frames = int(track.shape[0])
    contact_evidence: dict[tuple[int, int], dict[str, Any]] = {}
    for row_index, row in enumerate(annotation.contacts):
        key = (int(row["rally_id"]), int(row["contact_frame"]))
        frame = key[1]
        current = _half_value(
            point_winner.attribute_half(
                frame,
                track,
                sticky,
                pose.bboxes,
                court_inputs.net_band,
            )
        )
        ankle = ankle_half(frame, sticky, court_inputs.net_band, fixture.height)
        contact_evidence[key] = {
            "rally_id": key[0],
            "contact_frame": frame,
            "proximity_ok": row["proximity_ok"],
            "wrist_near": row["wrist_near"],
            "suppressed": row["suppressed"],
            "filtered": False,
            "nearest_wrist_slot": _selected_wrist_slot(frame, sticky),
            "current_half": current,
            "ankle_half": ankle,
            "valid_ankle_slots": _valid_ankle_slots(frame, sticky),
            "preceding_scene_distance_frames": _preceding_scene_distance(frame, court.raw_cuts),
            "_row_index": row_index,
        }

    remaining = Counter((row["rally_id"], row["contact_frame"]) for row in annotation.filtered_contacts)
    for row in annotation.contacts:
        key = (int(row["rally_id"]), int(row["contact_frame"]))
        if remaining[key] > 0:
            contact_evidence[key]["filtered"] = True
            remaining[key] -= 1
    if any(value > 0 for value in remaining.values()):
        raise ValueError(f"{fixture.name}: filtered membership could not be assigned")

    # Contact frames are unique in standard annotations.  Keep a stable row list
    # and reject a duplicate key rather than silently changing its membership.
    if len(contact_evidence) != len(annotation.contacts):
        raise ValueError(f"{fixture.name}: duplicate raw contact identity is unsupported")
    filtered_by_span: dict[int, list[dict[str, Any]]] = {}
    for row in annotation.filtered_contacts:
        key = (int(row["rally_id"]), int(row["contact_frame"]))
        filtered_by_span.setdefault(key[0], []).append(contact_evidence[key])

    spans: list[dict[str, Any]] = []
    for span_id, (start, end) in enumerate(annotation.spans):
        rows = [
            contact_evidence[(int(row["rally_id"]), int(row["contact_frame"]))]
            for row in annotation.contacts
            if int(row["rally_id"]) == span_id
        ]
        rows.sort(key=lambda row: (row["contact_frame"], row["_row_index"]))
        filtered_rows = filtered_by_span.get(span_id, [])
        current_guesses = [
            point_winner.attribute_half(
                int(row["contact_frame"]),
                track,
                sticky,
                pose.bboxes,
                court_inputs.net_band,
            )
            for row in filtered_rows
        ]
        ankle_guesses = [
            ankle_half(int(row["contact_frame"]), sticky, court_inputs.net_band, fixture.height)
            for row in filtered_rows
        ]
        current_fit = _half_value(point_winner.fit_alternation(current_guesses))
        ankle_fit = _half_value(point_winner.fit_alternation(ankle_guesses))
        stored_fit = annotation.striker_halves[span_id]
        if current_fit != stored_fit:
            raise AssertionError(
                f"{fixture.name} span {span_id}: recomputed current fit {current_fit!r} "
                f"differs from stored striker half {stored_fit!r}"
            )
        stored_first = annotation.fitted_first_all[span_id]
        current_first = _first_half(current_fit, len(filtered_rows))
        if stored_first is not None and current_first != stored_first:
            raise AssertionError(
                f"{fixture.name} span {span_id}: recomputed current first half {current_first!r} "
                f"differs from stored fitted first half {stored_first!r}"
            )
        for row in rows:
            row.pop("_row_index", None)
        spans.append(
            {
                "span_id": span_id,
                "start_frame": start,
                "end_frame": end,
                "raw_contact_count": len(rows),
                "filtered_contact_count": len(filtered_rows),
                "stored_striker_half": stored_fit,
                "current_striker_half": current_fit,
                "geometry_striker_half": ankle_fit,
                "stored_server_half": stored_first,
                "current_server_half": current_first,
                "geometry_server_half": _first_half(ankle_fit, len(filtered_rows)),
                "contacts": rows,
            }
        )
    for row in contact_evidence.values():
        row.pop("_row_index", None)
    evidence_fixture = {
        "fixture": fixture.name,
        "video_id": fixture.video_id,
        "fps": fixture.fps,
        "frame_count": n_frames,
        "resolution": [fixture.width, fixture.height],
        "tracker_segment_count": len(segments),
        "spans": spans,
    }
    error = _forbidden_key_error(evidence_fixture)
    if error is not None:
        raise AssertionError(error)

    paths = _stage_paths(data_root, fixture)
    input_rows = [
        {
            "role": role,
            "stage": path.parent.parent.name,
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for role, path in paths.items()
    ]
    return evidence_fixture, {"fixture": fixture.name, "files": input_rows}


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def deterministic_gzip_bytes(payload: Mapping[str, object]) -> bytes:
    """Return canonical gzip bytes with a zero timestamp."""
    import io

    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=9, mtime=0) as zipped:
        zipped.write(_json_bytes(payload))
    return output.getvalue()


def freeze(data_root: Path, output_dir: Path, source_commit: str) -> tuple[Path, Path]:
    """Freeze all fixed fixtures and return evidence and manifest paths."""
    if not isinstance(source_commit, str) or not source_commit.strip():
        raise ValueError("source_commit must be a non-empty declared commit")
    fixture_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for name, (video_id, fps) in FIXTURE_SPECS.items():
        fixture = FixtureSpec(name, video_id, fps)
        evidence_fixture, inputs = _fixture_evidence(Path(data_root), fixture)
        fixture_rows.append(evidence_fixture)
        input_rows.append(inputs)
    evidence = {"schema": EVIDENCE_SCHEMA, "fixtures": fixture_rows}
    error = _forbidden_key_error(evidence)
    if error is not None:
        raise AssertionError(error)
    evidence_bytes = deterministic_gzip_bytes(evidence)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_path = output_root / EVIDENCE_FILENAME
    evidence_path.write_bytes(evidence_bytes)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "labels_read": False,
        "source_commit": source_commit,
        "fixture_set": list(FIXTURE_SPECS),
        "inputs": input_rows,
        "evidence_schema": EVIDENCE_SCHEMA,
        "evidence_file": EVIDENCE_FILENAME,
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
    }
    manifest_path = output_root / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence_path, manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the label-blind freezer CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the freezer."""
    arguments = parse_args(argv)
    evidence_path, manifest_path = freeze(
        arguments.data_root,
        arguments.output_dir,
        arguments.source_commit,
    )
    print(f"wrote {evidence_path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
