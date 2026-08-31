"""Build, run, and score the 32-case rally-start VLM gate."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import lzma
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .backends import BACKEND_KEYS, QWEN_BACKEND_KEYS, load_backend

MANIFEST_SCHEMA = "vlm-rally-start-manifest-v1"
TRUTH_SCHEMA = "vlm-rally-start-truth-v1"
PROVENANCE_SCHEMA = "vlm-rally-start-provenance-v1"
ATTEMPT_SCHEMA = "vlm-rally-start-attempt-v1"
SCORE_SCHEMA = "vlm-rally-start-score-v1"

VIDEO_IDS = {"sset_01": 1, "sset_15": 15, "sset_21": 21}
EXPECTED_FRAMES = 120
WIDTH = 512
HEIGHT = 288
FIRST_CONTACTS = 3
CUT_TO_CONTACT_SECONDS = 2.0
COURT_CONFIRM_SECONDS = 1.0
COURT_CONFIRM_FRACTION = 0.8
CUT_PRE_FRAMES = 40
CONTACT_PRE_FRAMES = 80
MAX_NEW_TOKENS = 128
QWEN_MAX_MODEL_LEN = 16_384
GOLD = (30, 190, 240)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
_LEADING_ZERO_CONTACT_FRAME = re.compile(
    r'("contact_frame"\s*:\s*)(0[0-9]+)(?=\s*[,}])'
)

REVIEW_DIR = Path(
    "docs/scraper_pipeline/serve_prepend_lookback/data/"
    "rally_start_visibility_audit_20260809"
)
COMMITTED_DIR = Path(
    "docs/scraper_pipeline/serve_prepend_lookback/data/"
    "serve_prepend_lookback_189c5af_20260808"
)

PROMPT = """You are reviewing a short consecutive badminton broadcast clip near an automatically proposed rally start.

Every video frame is labelled with a CLIP index from 000 to 119 and its original SOURCE frame number. A gold border, when present, marks an automatically selected camera cut. The cut is a navigation hint and can be wrong.

Identify the server and decide how the current rally's physical service contact is shown. Do not mistake a replay, cutaway, warm-up action, or a later rally contact for the current serve.

TOP means the player on the far or top half of the court in the normal full-court view. BOTTOM means the player on the near or bottom half. If an unusual camera view makes the court side impossible to map, answer unclear.

Use these serve_state values:
- visible: the physical racket-shuttle service contact is visible in this clip;
- off_frame: the current-rally service action is shown, but physical contact occurs outside the image;
- broadcast_omitted: the broadcast does not show the service action and returns with the rally already underway;
- unclear: the clip does not support one of the other states.

contact_frame must be the CLIP index from 0 to 119 at physical contact when serve_state is visible. It must be null for every other serve_state. Do not return the SOURCE frame number. Do not infer an exact frame from preparation, a camera cut, or the first return shot.

Return a bare JSON object with exactly three keys: server, serve_state, contact_frame. server must be top, bottom, or unclear. serve_state must be visible, off_frame, broadcast_omitted, or unclear. contact_frame must be an integer or null. Do not use a Markdown fence."""

_FORBIDDEN_INFERENCE_KEYS = {
    "set_id",
    "rally",
    "truth",
    "serve_visibility",
    "visible_serve_frame",
    "expected_server",
    "expected_serve_state",
    "review_note",
    "confidence",
}


@dataclass(frozen=True)
class RallyStartCase:
    """One truth-blind clip supplied to both models."""

    case_id: str
    video_id: str
    clip_path: Path
    source_start_frame: int
    source_end_frame: int
    sample_fps: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _load_npy_xz(path: Path) -> np.ndarray:
    with lzma.open(path, "rb") as stream:
        return np.load(stream)


def _load_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _single_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected one {pattern!r} under {directory}, found {len(matches)}"
        )
    return matches[0]


def _reject_truth_keys(value: Any, location: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _FORBIDDEN_INFERENCE_KEYS:
                raise ValueError(f"{location} contains forbidden key {key!r}")
            _reject_truth_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_truth_keys(child, f"{location}[{index}]")


def _exact_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{location} keys differ: missing={missing}, extra={extra}")


def load_manifest(path: Path, *, require_clips: bool = True) -> tuple[RallyStartCase, ...]:
    """Load the inference manifest and reject truth leakage or schema drift."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("rally-start manifest must be a JSON object")
    _reject_truth_keys(payload)
    _exact_keys(payload, {"schema", "cases"}, "manifest")
    if payload["schema"] != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported manifest schema {payload['schema']!r}")
    if not isinstance(payload["cases"], list) or not payload["cases"]:
        raise ValueError("manifest cases must be a non-empty list")

    expected_keys = {
        "case_id",
        "video_id",
        "clip_path",
        "source_start_frame",
        "source_end_frame",
        "sample_fps",
        "expected_frames",
        "width",
        "height",
    }
    cases = []
    seen = set()
    for index, raw in enumerate(payload["cases"]):
        if not isinstance(raw, dict):
            raise TypeError(f"case {index} must be an object")
        _exact_keys(raw, expected_keys, f"case {index}")
        case_id = raw["case_id"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"case {index} has an invalid case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id {case_id!r}")
        seen.add(case_id)
        if raw["expected_frames"] != EXPECTED_FRAMES:
            raise ValueError(f"{case_id}: expected_frames must be {EXPECTED_FRAMES}")
        if (raw["width"], raw["height"]) != (WIDTH, HEIGHT):
            raise ValueError(f"{case_id}: clip geometry must be {WIDTH}x{HEIGHT}")
        source_start = int(raw["source_start_frame"])
        source_end = int(raw["source_end_frame"])
        if source_end - source_start != EXPECTED_FRAMES:
            raise ValueError(f"{case_id}: source window must contain 120 frames")
        clip_path = Path(raw["clip_path"])
        if not clip_path.is_absolute():
            clip_path = path.parent / clip_path
        if require_clips and not clip_path.is_file():
            raise FileNotFoundError(f"{case_id}: clip is missing: {clip_path}")
        cases.append(
            RallyStartCase(
                case_id=case_id,
                video_id=str(raw["video_id"]),
                clip_path=clip_path,
                source_start_frame=source_start,
                source_end_frame=source_end,
                sample_fps=float(raw["sample_fps"]),
            )
        )
    return tuple(cases)


def parse_response(raw_response: str) -> dict[str, Any]:
    """Parse the exact three-field serve reply."""
    try:
        payload = json.loads(raw_response.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"reply is not bare valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise TypeError("reply must be a JSON object")
    _exact_keys(payload, {"server", "serve_state", "contact_frame"}, "reply")
    if payload["server"] not in {"top", "bottom", "unclear"}:
        raise ValueError("server must be top, bottom, or unclear")
    if payload["serve_state"] not in {
        "visible",
        "off_frame",
        "broadcast_omitted",
        "unclear",
    }:
        raise ValueError("serve_state has an unsupported value")
    contact_frame = payload["contact_frame"]
    if contact_frame is not None and (
        not isinstance(contact_frame, int) or isinstance(contact_frame, bool)
    ):
        raise TypeError("contact_frame must be an integer or null")
    if payload["serve_state"] == "visible":
        if contact_frame is None:
            raise ValueError("visible serve_state requires contact_frame")
        if not 0 <= contact_frame < EXPECTED_FRAMES:
            raise ValueError("contact_frame must be a CLIP index from 0 to 119")
    elif contact_frame is not None:
        raise ValueError("contact_frame must be null unless serve_state is visible")
    return payload


def _normalise_leading_zero_contact_frame(raw_response: str) -> str | None:
    """Repair only an unambiguous leading-zero integer in contact_frame."""

    def replace(match: re.Match[str]) -> str:
        return f"{match.group(1)}{int(match.group(2))}"

    normalised, replacements = _LEADING_ZERO_CONTACT_FRAME.subn(replace, raw_response)
    return normalised if replacements == 1 else None


def _reviewed_rows(repo_root: Path) -> list[dict[str, str]]:
    root = repo_root / REVIEW_DIR
    rows = []
    for video_name in VIDEO_IDS:
        path = root / f"{video_name}_rally_start_reviewed.csv.gz"
        video_rows = _load_csv_gz(path)
        if any(row["video_id"] != video_name for row in video_rows):
            raise ValueError(f"{path}: video_id mismatch")
        rows.extend(video_rows)
    if len(rows) != 32:
        raise ValueError(f"expected 32 reviewed rally starts, found {len(rows)}")
    return sorted(
        rows,
        key=lambda row: (
            row["video_id"],
            int(row["set_id"].removeprefix("set")),
            int(row["rally"]),
        ),
    )


def _committed_anchors(repo_root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    root = repo_root / COMMITTED_DIR
    mapped: dict[tuple[str, str, int], dict[str, Any]] = {}
    for video_name in VIDEO_IDS:
        path = root / f"{video_name}_committed_rallies.csv.gz"
        for row in _load_csv_gz(path):
            key = (video_name, row["set_id"], int(row["rally_number"]))
            if key in mapped:
                raise ValueError(f"duplicate committed rally key {key}")
            mapped[key] = {
                "anchor_frame": (
                    None
                    if not row["lookback_anchor_frame"]
                    else int(row["lookback_anchor_frame"])
                ),
                "anchor_source": row["lookback_anchor_source"],
                "recorded_span_start": (
                    None if not row["span_start"] else int(row["span_start"])
                ),
                "recorded_span_end": (
                    None if not row["span_end"] else int(row["span_end"])
                ),
                "recorded_span_category": row["span_category"],
            }
    return mapped


def _server_truth(repo_root: Path) -> dict[tuple[str, str, int], str]:
    path = repo_root / "training/data/shuttleset/annotations/shots_master.csv"
    mapped: dict[tuple[str, str, int], str] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["ball_round"]) != 1:
                continue
            video_name = f"sset_{int(row['vid']):02d}"
            if video_name not in VIDEO_IDS:
                continue
            key = (video_name, row["set_id"], int(row["rally"]))
            side = row["player_side"].lower()
            if side not in {"top", "bottom"}:
                raise ValueError(f"{path}: invalid player_side {side!r} for {key}")
            if key in mapped:
                raise ValueError(f"{path}: duplicate first stroke for {key}")
            mapped[key] = side
    return mapped


def _artifact_paths(artifacts_root: Path, video_name: str) -> dict[str, Path]:
    stages = artifacts_root / "stages"
    return {
        "source": _single_file(stages / "tracknet_input" / video_name, "*.avi"),
        "result": stages / "annotation" / video_name / "annotator_result.json.gz",
        "court_evidence": stages / "court" / video_name / "court_evidence.json.gz",
        "court_present": stages / "court" / video_name / "court_present.npy.xz",
    }


def _shift_window(anchor: int, pre_frames: int, total_frames: int) -> tuple[int, int]:
    start = min(max(anchor - pre_frames, 0), total_frames - EXPECTED_FRAMES)
    return start, start + EXPECTED_FRAMES


def _nearby_current_contacts(
    result: dict[str, Any], recorded_anchor: int, fps: float
) -> tuple[list[int], int | None]:
    """Add current contacts only when a current span still covers the frozen anchor."""
    rally_id = None
    for index, raw_span in enumerate(result["spans"]):
        start, end = map(int, raw_span)
        if start <= recorded_anchor < end:
            rally_id = index
            break
    guesses = [recorded_anchor]
    if rally_id is not None:
        current_contacts = [
            int(frame)
            for frame in result["filtered_by_rally"].get(str(rally_id), [])
            if int(frame) >= recorded_anchor
            and int(frame) - recorded_anchor <= round(CUT_TO_CONTACT_SECONDS * fps)
        ]
        guesses.extend(current_contacts)
    return sorted(set(guesses))[:FIRST_CONTACTS], rally_id


def _segment_for_frame(raw_cuts: list[list[int]], frame: int) -> tuple[int, int]:
    for raw_start, raw_end in raw_cuts:
        start = int(raw_start)
        end = int(raw_end)
        if start <= frame < end:
            return start, end
    raise ValueError(f"frame {frame} lies outside the persisted cut segments")


def _select_anchor(
    accepted_contacts: list[int],
    raw_cuts: list[list[int]],
    court_present: np.ndarray,
    fps: float,
) -> dict[str, int | float | str | None]:
    if not accepted_contacts:
        raise ValueError("automatic rally has no accepted contacts")
    first_contacts = sorted(set(accepted_contacts))[:FIRST_CONTACTS]
    max_gap = round(CUT_TO_CONTACT_SECONDS * fps)
    confirm_frames = round(COURT_CONFIRM_SECONDS * fps)
    for contact_frame in first_contacts:
        cut_frame, segment_end = _segment_for_frame(raw_cuts, contact_frame)
        gap = contact_frame - cut_frame
        confirm_end = cut_frame + confirm_frames
        if cut_frame == 0 or gap > max_gap or confirm_end > segment_end:
            continue
        court_fraction = float(np.mean(court_present[cut_frame:confirm_end]))
        if court_fraction >= COURT_CONFIRM_FRACTION:
            return {
                "anchor_frame": cut_frame,
                "selection_method": "court-supported-cut",
                "selected_cut_frame": cut_frame,
                "contact_after_cut_frame": contact_frame,
                "cut_to_contact_frames": gap,
                "court_confirm_fraction": court_fraction,
            }
    return {
        "anchor_frame": first_contacts[0],
        "selection_method": "earliest-accepted-contact",
        "selected_cut_frame": None,
        "contact_after_cut_frame": None,
        "cut_to_contact_frames": None,
        "court_confirm_fraction": None,
    }


def _letterbox(frame: np.ndarray) -> np.ndarray:
    scale = min(WIDTH / frame.shape[1], HEIGHT / frame.shape[0])
    resized_width = round(frame.shape[1] * scale)
    resized_height = round(frame.shape[0] * scale)
    resized = cv2.resize(frame, (resized_width, resized_height))
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    left = (WIDTH - resized_width) // 2
    top = (HEIGHT - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _render_clip(
    source_path: Path,
    clip_path: Path,
    *,
    fps: float,
    source_start: int,
    source_end: int,
    selected_cut_frame: int | None,
) -> None:
    capture = cv2.VideoCapture(str(source_path))
    writer = cv2.VideoWriter(
        str(clip_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not open clip writer for {clip_path}")
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video {source_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_start)
        for clip_frame, source_frame in enumerate(range(source_start, source_end)):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to read source frame {source_frame}")
            rendered = _letterbox(frame)
            at_cut = source_frame == selected_cut_frame
            if at_cut:
                cv2.rectangle(rendered, (2, 2), (WIDTH - 3, HEIGHT - 3), GOLD, 4)
            cv2.rectangle(rendered, (0, 0), (WIDTH, 30), BLACK, -1)
            label = f"CLIP {clip_frame:03d}  |  SOURCE {source_frame:06d}"
            if at_cut:
                label += "  AUTO CUT"
            cv2.putText(
                rendered,
                label,
                (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                WHITE,
                1,
                cv2.LINE_AA,
            )
            writer.write(rendered)
    finally:
        writer.release()
        capture.release()

    check = cv2.VideoCapture(str(clip_path))
    try:
        observed = (
            round(check.get(cv2.CAP_PROP_FRAME_COUNT)),
            round(check.get(cv2.CAP_PROP_FRAME_WIDTH)),
            round(check.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if observed != (EXPECTED_FRAMES, WIDTH, HEIGHT):
            raise ValueError(f"{clip_path}: clip geometry/count is {observed}")
        if abs(check.get(cv2.CAP_PROP_FPS) - fps) > 0.01:
            raise ValueError(f"{clip_path}: clip FPS differs from {fps}")
    finally:
        check.release()


def _truth_state(reviewed_state: str) -> str:
    mapping = {
        "visible": "visible",
        "off-frame": "off_frame",
        "broadcast-omitted": "broadcast_omitted",
        "uncertain": "unclear",
    }
    try:
        return mapping[reviewed_state]
    except KeyError as exc:
        raise ValueError(f"unsupported reviewed serve state {reviewed_state!r}") from exc


def build_trials(artifacts_root: Path, repo_root: Path, output_dir: Path) -> None:
    """Build one immutable truth-separated 32-case input set."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    inference_dir = output_dir / "inference"
    scoring_dir = output_dir / "scoring"
    clips_dir = inference_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    scoring_dir.mkdir()

    reviewed_rows = _reviewed_rows(repo_root)
    committed_anchors = _committed_anchors(repo_root)
    expected_servers = _server_truth(repo_root)
    shots_master_path = repo_root / "training/data/shuttleset/annotations/shots_master.csv"

    manifest_cases = []
    truth_cases = []
    selections = []
    input_paths = [shots_master_path]
    video_payloads = {}
    for video_name in VIDEO_IDS:
        paths = _artifact_paths(artifacts_root, video_name)
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        result = _load_json_gz(paths["result"])["result"]
        court_evidence = _load_json_gz(paths["court_evidence"])
        court_present = _load_npy_xz(paths["court_present"])
        capture = cv2.VideoCapture(str(paths["source"]))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"could not open source video {paths['source']}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total_frames = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if len(court_present) != total_frames:
            raise ValueError(
                f"{video_name}: court evidence has {len(court_present)} frames, "
                f"source has {total_frames}"
            )
        raw_cuts = court_evidence["raw_cuts"]
        if int(raw_cuts[0][0]) != 0 or int(raw_cuts[-1][1]) != total_frames:
            raise ValueError(f"{video_name}: cut segments do not cover the video")
        video_payloads[video_name] = (paths, result, court_present, raw_cuts, fps, total_frames)
        input_paths.extend(paths.values())
        input_paths.append(
            repo_root
            / REVIEW_DIR
            / f"{video_name}_rally_start_reviewed.csv.gz"
        )
        input_paths.append(
            repo_root
            / COMMITTED_DIR
            / f"{video_name}_committed_rallies.csv.gz"
        )

    for case_index, reviewed in enumerate(reviewed_rows):
        video_name = reviewed["video_id"]
        set_id = reviewed["set_id"]
        rally_number = int(reviewed["rally"])
        key = (video_name, set_id, rally_number)
        if key not in committed_anchors:
            raise ValueError(f"reviewed case has no committed automatic rally: {key}")
        if key not in expected_servers:
            raise ValueError(f"reviewed case has no first-stroke server truth: {key}")
        paths, result, court_present, raw_cuts, fps, total_frames = video_payloads[video_name]
        reviewed_fps = float(reviewed["fps"])
        if abs(fps - reviewed_fps) > 0.01:
            raise ValueError(f"{key}: source FPS {fps} differs from review FPS {reviewed_fps}")
        committed = committed_anchors[key]
        if (
            committed["anchor_source"] != "first_assigned_accepted_contact"
            or committed["anchor_frame"] is None
        ):
            raise ValueError(f"{key}: reviewed case lacks an accepted-contact anchor")
        recorded_anchor = int(committed["anchor_frame"])
        accepted_contacts, current_rally_id = _nearby_current_contacts(
            result, recorded_anchor, fps
        )
        selection = _select_anchor(accepted_contacts, raw_cuts, court_present, fps)
        anchor_frame = int(selection["anchor_frame"])
        pre_frames = (
            CUT_PRE_FRAMES
            if selection["selected_cut_frame"] is not None
            else CONTACT_PRE_FRAMES
        )
        source_start, source_end = _shift_window(
            anchor_frame, pre_frames, total_frames
        )
        case_id = f"rally-start-{video_name}-c{case_index:02d}"
        clip_path = clips_dir / f"{case_id}.mp4"
        selected_cut = selection["selected_cut_frame"]
        _render_clip(
            paths["source"],
            clip_path,
            fps=fps,
            source_start=source_start,
            source_end=source_end,
            selected_cut_frame=None if selected_cut is None else int(selected_cut),
        )
        manifest_cases.append(
            {
                "case_id": case_id,
                "video_id": video_name,
                "clip_path": str(Path("clips") / clip_path.name),
                "source_start_frame": source_start,
                "source_end_frame": source_end,
                "sample_fps": fps,
                "expected_frames": EXPECTED_FRAMES,
                "width": WIDTH,
                "height": HEIGHT,
            }
        )
        state = _truth_state(reviewed["serve_visibility"])
        visible_frame = (
            int(reviewed["visible_serve_frame"])
            if reviewed["visible_serve_frame"]
            else None
        )
        if (state == "visible") != (visible_frame is not None):
            raise ValueError(f"{key}: visible frame disagrees with serve state")
        truth_cases.append(
            {
                "case_id": case_id,
                "video_id": video_name,
                "set_id": set_id,
                "rally": rally_number,
                "expected_server": expected_servers[key],
                "expected_serve_state": state,
                "visible_contact_frame": visible_frame,
                "accepted_tolerance_frames": round(5 * fps / 30),
                "pilot_stratum": reviewed["pilot_stratum"],
            }
        )
        selections.append(
            {
                "case_id": case_id,
                "video_id": video_name,
                "set_id": set_id,
                "rally": rally_number,
                "recorded_anchor_frame": recorded_anchor,
                "recorded_anchor_source": committed["anchor_source"],
                "recorded_span_start": committed["recorded_span_start"],
                "recorded_span_end": committed["recorded_span_end"],
                "recorded_span_category": committed["recorded_span_category"],
                "current_rally_id_at_anchor": current_rally_id,
                "contact_guess_frames": accepted_contacts,
                "source_start_frame": source_start,
                "source_end_frame": source_end,
                **selection,
            }
        )

    manifest_path = inference_dir / "manifest.json"
    truth_path = scoring_dir / "truth.json"
    _write_new_json(manifest_path, {"schema": MANIFEST_SCHEMA, "cases": manifest_cases})
    _write_new_json(truth_path, {"schema": TRUTH_SCHEMA, "cases": truth_cases})
    loaded = load_manifest(manifest_path)
    if len(loaded) != 32:
        raise ValueError(f"built manifest has {len(loaded)} cases")
    prompt_hash = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "settings": {
            "expected_frames": EXPECTED_FRAMES,
            "width": WIDTH,
            "height": HEIGHT,
            "first_contacts_considered": FIRST_CONTACTS,
            "maximum_cut_to_contact_seconds": CUT_TO_CONTACT_SECONDS,
            "court_confirmation_seconds": COURT_CONFIRM_SECONDS,
            "minimum_court_confirmation_fraction": COURT_CONFIRM_FRACTION,
            "frames_before_selected_cut": CUT_PRE_FRAMES,
            "frames_before_fallback_contact": CONTACT_PRE_FRAMES,
            "prompt_sha256": prompt_hash,
        },
        "selections": selections,
        "inputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in input_paths
        ],
        "outputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(clips_dir.glob("*.mp4"))
        ],
        "manifest_sha256": _sha256(manifest_path),
        "truth_sha256": _sha256(truth_path),
    }
    _write_new_json(scoring_dir / "provenance.json", provenance)


def run_trials(
    backend_name: str,
    manifest_path: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
) -> None:
    """Run one resident model over the frozen rally-start clips."""
    cases = list(load_manifest(manifest_path))
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("no rally-start cases selected")
    max_model_len = QWEN_MAX_MODEL_LEN if backend_name in QWEN_BACKEND_KEYS else None
    backend = load_backend(
        backend_name,
        expected_input_frames=EXPECTED_FRAMES,
        max_model_len=max_model_len,
    )
    model_identity = asdict(backend.spec.identity(backend.backend_version))
    prompt_hash = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    for case in cases:
        attempt_path = output_dir / backend_name / f"{case.case_id}.json"
        base_payload = {
            "schema": ATTEMPT_SCHEMA,
            "backend": backend_name,
            "model": model_identity,
            "case": {
                "case_id": case.case_id,
                "video_id": case.video_id,
                "clip_path": str(case.clip_path),
                "clip_sha256": _sha256(case.clip_path),
                "source_start_frame": case.source_start_frame,
                "source_end_frame": case.source_end_frame,
            },
            "prompt": PROMPT,
            "prompt_sha256": prompt_hash,
        }
        started = perf_counter()
        try:
            evidence = backend.generate(
                case.clip_path,
                PROMPT,
                requested_fps=case.sample_fps,
                width=WIDTH,
                height=HEIGHT,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        except Exception as exc:
            _write_new_json(
                attempt_path,
                {
                    **base_payload,
                    "raw_response": None,
                    "parsed_response": None,
                    "parser_error": None,
                    "generation_error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": perf_counter() - started,
                    "sampling": {
                        "requested_fps": case.sample_fps,
                        "sampled_input_frames": None,
                        "width": None,
                        "height": None,
                        "visual_tokens": None,
                        "total_input_tokens": None,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "qwen_max_model_len": max_model_len,
                    },
                },
            )
            raise
        parsed = None
        parser_error = None
        try:
            parsed = parse_response(evidence.raw_response)
        except (TypeError, ValueError) as exc:
            parser_error = str(exc)
        _write_new_json(
            attempt_path,
            {
                **base_payload,
                "raw_response": evidence.raw_response,
                "parsed_response": parsed,
                "parser_error": parser_error,
                "generation_error": None,
                "elapsed_seconds": perf_counter() - started,
                "sampling": {
                    "requested_fps": case.sample_fps,
                    "sampled_input_frames": evidence.sampled_input_frames,
                    "width": evidence.width,
                    "height": evidence.height,
                    "visual_tokens": evidence.visual_tokens,
                    "total_input_tokens": evidence.total_input_tokens,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "qwen_max_model_len": max_model_len,
                },
            },
        )
        print(attempt_path, flush=True)


def _load_truth(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("truth sidecar must be a JSON object")
    _exact_keys(payload, {"schema", "cases"}, "truth")
    if payload["schema"] != TRUTH_SCHEMA:
        raise ValueError(f"unsupported truth schema {payload['schema']!r}")
    rows = {}
    for raw in payload["cases"]:
        if not isinstance(raw, dict):
            raise TypeError("truth case must be an object")
        case_id = raw.get("case_id")
        if not isinstance(case_id, str):
            raise TypeError("truth case_id must be a string")
        if case_id in rows:
            raise ValueError(f"duplicate truth case {case_id!r}")
        rows[case_id] = raw
    return rows


def _load_attempts(
    root: Path,
    backend_name: str,
    cases: dict[str, RallyStartCase],
) -> dict[str, dict[str, Any]]:
    attempts = {}
    backend_dir = root / backend_name
    for path in sorted(backend_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != ATTEMPT_SCHEMA:
            raise ValueError(f"{path}: unsupported attempt schema")
        if payload.get("backend") != backend_name:
            raise ValueError(f"{path}: backend mismatch")
        case_payload = payload.get("case")
        if not isinstance(case_payload, dict):
            raise TypeError(f"{path}: invalid case payload")
        case_id = case_payload.get("case_id")
        if case_id not in cases:
            raise ValueError(f"{path}: unexpected case_id {case_id!r}")
        if case_id in attempts:
            raise ValueError(f"duplicate attempt for {case_id}")
        if hashlib.sha256(payload["prompt"].encode("utf-8")).hexdigest() != payload[
            "prompt_sha256"
        ]:
            raise ValueError(f"{path}: prompt hash does not reproduce")
        parsed = None
        parser_error = None
        if payload["generation_error"] is None:
            try:
                parsed = parse_response(payload["raw_response"])
            except (TypeError, ValueError) as exc:
                parser_error = str(exc)
        if parsed != payload["parsed_response"] or parser_error != payload["parser_error"]:
            raise ValueError(f"{path}: stored parse does not reproduce")
        scoring_normalisation = None
        if parsed is None and payload["generation_error"] is None:
            normalised_response = _normalise_leading_zero_contact_frame(
                payload["raw_response"]
            )
            if normalised_response is not None:
                try:
                    parsed = parse_response(normalised_response)
                except (TypeError, ValueError):
                    pass
                else:
                    parser_error = None
                    scoring_normalisation = "contact_frame_leading_zero"
        attempts[case_id] = {
            **payload,
            "parsed_response": parsed,
            "parser_error": parser_error,
            "scoring_normalisation": scoring_normalisation,
        }
    return attempts


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _score_backend(
    cases: tuple[RallyStartCase, ...],
    truth: dict[str, dict[str, Any]],
    attempts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    server_correct = 0
    state_correct = 0
    server_unclear = 0
    state_unclear = 0
    any_abstention = 0
    false_frame_claims = 0
    visible_claims = 0
    visible_truth = 0
    exact_timing = 0
    accepted_timing = 0
    timing_at_10 = 0
    timing_at_15 = 0
    absolute_errors = []
    server_by_state: dict[str, Counter[str]] = defaultdict(Counter)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    parser_errors = Counter()
    rows = []

    for case in cases:
        truth_row = truth[case.case_id]
        attempt = attempts.get(case.case_id)
        parsed = None if attempt is None else attempt["parsed_response"]
        if parsed is None:
            predicted_server = "invalid"
            predicted_state = "invalid"
            predicted_clip_frame = None
            error = "missing attempt" if attempt is None else attempt["parser_error"]
            parser_errors[str(error)] += 1
        else:
            predicted_server = parsed["server"]
            predicted_state = parsed["serve_state"]
            predicted_clip_frame = parsed["contact_frame"]

        predicted_source_frame = (
            None
            if predicted_clip_frame is None
            else case.source_start_frame + int(predicted_clip_frame)
        )

        expected_server = truth_row["expected_server"]
        expected_state = truth_row["expected_serve_state"]
        server_is_correct = predicted_server == expected_server
        state_is_correct = predicted_state == expected_state
        server_correct += server_is_correct
        state_correct += state_is_correct
        server_unclear += predicted_server == "unclear"
        state_unclear += predicted_state == "unclear"
        any_abstention += predicted_server == "unclear" or predicted_state == "unclear"
        server_by_state[expected_state]["correct" if server_is_correct else "wrong"] += 1
        confusion[expected_state][predicted_state] += 1

        timing_error = None
        if expected_state == "visible":
            visible_truth += 1
            expected_frame = int(truth_row["visible_contact_frame"])
            if predicted_source_frame is not None:
                visible_claims += 1
                timing_error = abs(predicted_source_frame - expected_frame)
                absolute_errors.append(timing_error)
                exact_timing += timing_error == 0
                accepted_timing += timing_error <= int(
                    truth_row["accepted_tolerance_frames"]
                )
                timing_at_10 += timing_error <= round(10 * case.sample_fps / 30)
                timing_at_15 += timing_error <= round(15 * case.sample_fps / 30)
        elif predicted_source_frame is not None:
            false_frame_claims += 1

        rows.append(
            {
                "case_id": case.case_id,
                "video_id": case.video_id,
                "expected_server": expected_server,
                "predicted_server": predicted_server,
                "server_correct": server_is_correct,
                "expected_serve_state": expected_state,
                "predicted_serve_state": predicted_state,
                "serve_state_correct": state_is_correct,
                "expected_contact_frame": truth_row["visible_contact_frame"],
                "predicted_clip_contact_frame": predicted_clip_frame,
                "predicted_source_contact_frame": predicted_source_frame,
                "absolute_contact_error_frames": timing_error,
            }
        )

    total = len(cases)
    model = None
    prompt_hashes = set()
    sampled_counts = Counter()
    reply_normalisations = Counter()
    for attempt in attempts.values():
        model = attempt["model"] if model is None else model
        if attempt["model"] != model:
            raise ValueError("model identity changed within one backend run")
        prompt_hashes.add(attempt["prompt_sha256"])
        sampled_frames = attempt["sampling"]["sampled_input_frames"]
        sampled_count = "missing" if sampled_frames is None else str(len(sampled_frames))
        sampled_counts[sampled_count] += 1
        scoring_normalisation = attempt.get("scoring_normalisation")
        if scoring_normalisation is not None:
            reply_normalisations[scoring_normalisation] += 1
    return {
        "complete": len(attempts) == total,
        "parse_complete": len(attempts) == total and not parser_errors,
        "case_count": total,
        "attempt_count": len(attempts),
        "model": model,
        "prompt_sha256": next(iter(prompt_hashes)) if len(prompt_hashes) == 1 else None,
        "sampled_input_frame_counts": dict(sorted(sampled_counts.items())),
        "reply_normalisations": dict(sorted(reply_normalisations.items())),
        "parser_errors": dict(sorted(parser_errors.items())),
        "server": {
            "correct": server_correct,
            "total": total,
            "accuracy": _ratio(server_correct, total),
            "unclear": server_unclear,
            "by_truth_state": {
                state: dict(counts) for state, counts in sorted(server_by_state.items())
            },
        },
        "serve_state": {
            "correct": state_correct,
            "total": total,
            "accuracy": _ratio(state_correct, total),
            "unclear": state_unclear,
            "confusion": {
                state: dict(sorted(counts.items()))
                for state, counts in sorted(confusion.items())
            },
        },
        "contact_timing_visible_truth": {
            "truth_cases": visible_truth,
            "frame_claims": visible_claims,
            "exact": exact_timing,
            "within_project_tolerance": accepted_timing,
            "within_10_base30_frames": timing_at_10,
            "within_15_base30_frames": timing_at_15,
            "mean_absolute_error_frames_when_claimed": (
                None if not absolute_errors else sum(absolute_errors) / len(absolute_errors)
            ),
        },
        "false_exact_frame_claims_on_nonvisible_truth": false_frame_claims,
        "abstention": {
            "server_unclear": server_unclear,
            "serve_state_unclear": state_unclear,
            "either_field_unclear": any_abstention,
        },
        "rows": rows,
    }


def score_trials(
    manifest_path: Path,
    truth_path: Path,
    attempt_roots: dict[str, Path],
    output_path: Path,
) -> None:
    """Score both model runs against the separate 32-case truth sidecar."""
    cases = load_manifest(manifest_path)
    by_case = {case.case_id: case for case in cases}
    truth = _load_truth(truth_path)
    if set(truth) != set(by_case):
        raise ValueError("manifest and truth case IDs differ")
    results = {}
    for backend_name, root in sorted(attempt_roots.items()):
        attempts = _load_attempts(root, backend_name, by_case)
        results[backend_name] = _score_backend(cases, truth, attempts)
    _write_new_json(
        output_path,
        {
            "schema": SCORE_SCHEMA,
            "manifest_sha256": _sha256(manifest_path),
            "truth_sha256": _sha256(truth_path),
            "models": results,
        },
    )


def _parse_attempt_roots(values: Sequence[str]) -> dict[str, Path]:
    parsed = {}
    for value in values:
        backend_name, separator, raw_path = value.partition("=")
        if not separator or backend_name not in BACKEND_KEYS:
            raise ValueError("--attempt must be BACKEND=PATH for a supported backend")
        if backend_name in parsed:
            raise ValueError(f"duplicate attempt root for {backend_name}")
        parsed[backend_name] = Path(raw_path)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build the frozen 32-case input")
    build.add_argument("--artifacts-root", type=Path, required=True)
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)

    run = subparsers.add_parser("run", help="run one VLM backend")
    run.add_argument("--backend", choices=BACKEND_KEYS, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--limit", type=int)

    score = subparsers.add_parser("score", help="score one or both model runs")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--attempt", action="append", required=True)
    score.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        build_trials(args.artifacts_root, args.repo_root, args.out)
    elif args.command == "run":
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        run_trials(args.backend, args.manifest, args.out, limit=args.limit)
    else:
        score_trials(
            args.manifest,
            args.truth,
            _parse_attempt_roots(args.attempt),
            args.out,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
