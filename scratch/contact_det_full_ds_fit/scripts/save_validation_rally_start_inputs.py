"""Save label-free rally-start inputs for the eight validation videos."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scratch.contact_det.scripts.freeze_contact_evidence import _load_inputs
from scratch.contact_det_full_ds_fit.scripts.baseline_results import (
    VerifiedBaselineMenu,
    load_completed_baseline_menu,
)
from scratch.contact_det_full_ds_fit.scripts.save_training_rally_start_inputs import (
    InputLoader,
    SideAttributor,
    ValidationPaths,
    _build_validation_candidate_lists,
    _candidate_frames,
    _checked_side,
    _enriched_candidates,
    _file_record,
    _fixture,
    _json_bytes,
    _mapping,
    _read_json,
    _rows_for_frames,
    _saved_validation_run,
    _saved_validation_videos,
    _score_rows_for_video,
    _video_feature_rows,
    _video_record,
    _write_json,
    check_validation_reproduction,
)
from scratch.contact_det_full_ds_fit.scripts.save_validation_rally_predictions import (
    _check_centre_feature_values,
    _checked_stage_files,
    _normalise_side,
    _span_id,
    _spans,
)
from scratch.contact_det_full_ds_fit.scripts.score_training_videos import (
    CHOSEN_RUN_ID,
    SOURCE_COMMIT,
)

RESULT_SCHEMA = "contact-rally-start-validation-inputs/1"
RESULT_FILENAME = "rally_start_validation_inputs.json.gz"
EXPECTED_VIDEO_COUNT = 8
EXPECTED_CANDIDATE_LIST_COUNT = 615
EXPECTED_CANDIDATE_ENTRY_COUNT = 1_845
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ValidationInputFiles:
    """Tracked and saved files needed for the validation input."""

    config: Path
    split: Path
    raw_feature_record: Path
    common30_feature_record: Path
    contact_labels: Path
    baseline_summary: Path
    menu_result: Path
    rally_predictions: Path
    rally_result: Path
    chosen_scores: Path
    candidate_summary: Path
    frozen_candidates: Path

    @property
    def validation_paths(self) -> ValidationPaths:
        """Return the paths expected by the existing validation replay."""
        return ValidationPaths(
            self.menu_result,
            self.common30_feature_record,
            self.rally_predictions,
            self.rally_result,
            self.chosen_scores,
            self.candidate_summary,
            self.frozen_candidates,
        )


def _chosen_run(verified: VerifiedBaselineMenu) -> Any:
    matches = [run for run in verified.runs if run.run.run_id == CHOSEN_RUN_ID]
    if len(matches) != 1:
        raise ValueError("chosen validation run differs")
    return matches[0]


def _candidate_lists_by_fixture(
    candidate_lists: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    output: dict[str, list[Mapping[str, object]]] = {}
    identities: set[tuple[str, int]] = set()
    for candidate_list in candidate_lists:
        fixture = candidate_list.get("fixture")
        span_id = candidate_list.get("span_id")
        if not isinstance(fixture, str) or type(span_id) is not int:
            raise ValueError("validation candidate-list identity differs")
        identity = (fixture, span_id)
        if identity in identities:
            raise ValueError("validation candidate-list identities repeat")
        identities.add(identity)
        output.setdefault(fixture, []).append(candidate_list)
    return output


def _checked_replay_sides(
    video_name: str,
    frames: np.ndarray,
    track: np.ndarray,
    sticky: Any,
    pose_bboxes: np.ndarray,
    net_band: tuple[float, float],
    side_attributor: SideAttributor,
) -> dict[int, str | None]:
    """Replay one answer for every requested frame and no other frame."""
    if len(np.unique(frames)) != len(frames):
        raise ValueError(f"{video_name}: replay frames repeat")
    sides = {
        int(frame): _normalise_side(
            side_attributor(int(frame), track, sticky, pose_bboxes, net_band),
            f"{video_name}/{frame}",
        )
        for frame in frames
    }
    if set(sides) != set(frames.tolist()):
        raise ValueError(f"{video_name}: player-side replay coverage differs")
    return sides


def _saved_kept_contacts(
    video_name: str,
    video_rows: np.ndarray,
    spans: Sequence[Mapping[str, int]],
    raw_contacts: Sequence[Mapping[str, Any]],
    sides: Mapping[int, str | None],
) -> list[dict[str, object]]:
    rows_by_frame = {int(row["frame"]): row for row in video_rows}
    if len(rows_by_frame) != len(video_rows):
        raise ValueError(f"{video_name}: validation score frames repeat")
    output: list[dict[str, object]] = []
    previous_frame = -1
    for raw_contact in raw_contacts:
        contact = _mapping(raw_contact, f"{video_name}: saved kept contact")
        frame = int(contact["frame"])
        if frame <= previous_frame:
            raise ValueError(f"{video_name}: saved kept-contact order differs")
        previous_frame = frame
        row = rows_by_frame.get(frame)
        if row is None or not bool(row["kept"]):
            raise ValueError(f"{video_name}/{frame}: kept score row differs")
        side = _checked_side(contact.get("predicted_side"), f"{video_name}/{frame}")
        if (
            float(contact["timing_score"]) != float(row["contact_score"])
            or contact.get("span_id") != _span_id(frame, spans)
            or sides.get(frame) != side
        ):
            raise ValueError(f"{video_name}/{frame}: saved kept contact differs")
        output.append(
            {
                "frame": frame,
                "interval_id": int(row["interval_id"]),
                "contact_score": float(row["contact_score"]),
                "span_id": _span_id(frame, spans),
                "predicted_side": side,
            }
        )
    expected_frames = [int(row["frame"]) for row in video_rows if bool(row["kept"])]
    if [contact["frame"] for contact in output] != expected_frames:
        raise ValueError(f"{video_name}: kept-contact frames differ")
    return output


def _assemble_video_value(
    source_commit: str,
    video: Any,
    frame_count: int,
    checked_stage_files: Sequence[Mapping[str, object]],
    spans: Sequence[Mapping[str, int]],
    kept_contacts: Sequence[Mapping[str, object]],
    candidate_lists: Sequence[Mapping[str, object]],
    sides: Mapping[int, str | None],
    skipped_section_count: int,
) -> dict[str, object]:
    """Build one portable validation-video value from checked inputs."""
    candidate_entry_count = sum(
        len(_mapping(candidate_list, "validation candidate list")["candidates"])
        for candidate_list in candidate_lists
    )
    return {
        "fixture": video.fixture,
        "video_id": video.video_id,
        "fps": video.fps,
        "frame_count": frame_count,
        "source_commit": source_commit,
        "labels_read": False,
        "input_files": list(checked_stage_files),
        "spans": list(spans),
        "kept_contacts": list(kept_contacts),
        "candidate_lists": list(candidate_lists),
        "counts": {
            "detected_sections": len(spans),
            "sections_without_kept_contact": skipped_section_count,
            "kept_contacts": len(kept_contacts),
            "candidate_lists": len(candidate_lists),
            "candidate_entries": candidate_entry_count,
            "earlier_candidate_entries": candidate_entry_count - len(candidate_lists),
            "distinct_replayed_frames": len(sides),
        },
    }


def _assemble_result(
    source_commit: str,
    input_files: Sequence[Mapping[str, object]],
    videos: Sequence[Mapping[str, Any]],
    expected_video_names: Sequence[str],
) -> dict[str, object]:
    """Combine the eight checked videos and enforce the frozen totals."""
    video_names = [str(video.get("fixture")) for video in videos]
    if video_names != list(expected_video_names):
        raise ValueError("validation video order differs")
    candidate_identities: set[tuple[str, int]] = set()
    for video in videos:
        fixture = str(video["fixture"])
        raw_lists = video.get("candidate_lists")
        if not isinstance(raw_lists, list):
            raise TypeError(f"{fixture}: candidate lists must be a list")
        for raw_list in raw_lists:
            candidate_list = _mapping(raw_list, f"{fixture}: candidate list")
            raw_candidates = candidate_list.get("candidates")
            if not isinstance(raw_candidates, list) or len(raw_candidates) != 3:
                raise ValueError(f"{fixture}: candidate-list size differs")
            for raw_candidate in raw_candidates:
                candidate = _mapping(raw_candidate, f"{fixture}: candidate")
                identity = (fixture, int(candidate["frame"]))
                if identity in candidate_identities:
                    raise ValueError(
                        "a validation candidate appears in more than one section"
                    )
                candidate_identities.add(identity)

    counts = {
        "videos": len(videos),
        "detected_sections": sum(
            int(video["counts"]["detected_sections"]) for video in videos
        ),
        "sections_without_kept_contact": sum(
            int(video["counts"]["sections_without_kept_contact"]) for video in videos
        ),
        "kept_contacts": sum(int(video["counts"]["kept_contacts"]) for video in videos),
        "candidate_lists": sum(
            int(video["counts"]["candidate_lists"]) for video in videos
        ),
        "candidate_entries": sum(
            int(video["counts"]["candidate_entries"]) for video in videos
        ),
        "earlier_candidate_entries": sum(
            int(video["counts"]["earlier_candidate_entries"]) for video in videos
        ),
    }
    if (
        counts["videos"] != EXPECTED_VIDEO_COUNT
        or counts["candidate_lists"] != EXPECTED_CANDIDATE_LIST_COUNT
        or counts["candidate_entries"] != EXPECTED_CANDIDATE_ENTRY_COUNT
    ):
        raise ValueError("frozen validation counts differ")
    return {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "source_commit": source_commit,
        "labels_read": False,
        "validation_videos": list(expected_video_names),
        "inputs": list(input_files),
        "counts": counts,
        "videos": list(videos),
    }


def save_validation_rally_start_inputs(
    files: ValidationInputFiles,
    data_root: Path,
    output_path: Path,
    source_commit: str,
    *,
    menu_loader: Callable[..., VerifiedBaselineMenu] = load_completed_baseline_menu,
    input_loader: InputLoader = _load_inputs,
    side_attributor: SideAttributor | None = None,
) -> Path:
    """Check, replay and save every validation candidate before labels."""
    destination = Path(output_path)
    _write_json(
        destination,
        {
            "schema": RESULT_SCHEMA,
            "status": "running",
            "source_commit": source_commit,
            "labels_read": False,
        },
    )
    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a short or full Git commit")
    source_root = str(REPO_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    if side_attributor is None:
        from annotator.point_winner import attribute_half

        side_attributor = attribute_half

    verified = menu_loader(
        files.menu_result,
        files.config,
        files.split,
        files.raw_feature_record,
        files.common30_feature_record,
        files.contact_labels,
    )
    check_validation_reproduction(
        verified,
        files.validation_paths,
        files.baseline_summary,
        Path(data_root),
        source_commit,
    )
    input_records = tuple(
        _file_record(role, path)
        for role, path in (
            ("model menu", files.menu_result),
            ("model settings", files.config),
            ("development split", files.split),
            ("raw feature record", files.raw_feature_record),
            ("common-30 feature record", files.common30_feature_record),
            ("contact labels", files.contact_labels),
            ("baseline summary", files.baseline_summary),
            ("validation rally predictions", files.rally_predictions),
            ("validation rally result", files.rally_result),
            ("chosen validation scores", files.chosen_scores),
            ("validation candidate summary", files.candidate_summary),
            ("frozen validation candidates", files.frozen_candidates),
        )
    )

    saved_predictions = _read_json(files.rally_predictions, "validation predictions")
    saved_run = _saved_validation_run(saved_predictions, CHOSEN_RUN_ID)
    saved_videos = _saved_validation_videos(saved_predictions, "videos")
    saved_contacts = _saved_validation_videos(saved_run, "videos")
    checked_run = _chosen_run(verified)
    candidate_lists, skipped_total = _build_validation_candidate_lists(
        verified,
        saved_predictions,
    )
    lists_by_fixture = _candidate_lists_by_fixture(candidate_lists)

    saved_video_values: list[dict[str, object]] = []
    skipped_sum = 0
    for video in verified.split.validation_videos:
        feature_record = _video_record(verified.raw_features, video.fixture)
        feature_summary = _mapping(
            feature_record.get("feature_summary"),
            f"{video.fixture}: feature summary",
        )
        frame_count = int(feature_summary["frame_count"])
        checked_files = _checked_stage_files(
            Path(data_root),
            _fixture(video),
            feature_record,
        )
        track, pose, court, _tracker_intervals, sticky, annotation = input_loader(
            Path(data_root),
            _fixture(video),
        )
        if len(track) != frame_count:
            raise ValueError(f"{video.fixture}: replay frame count differs")
        spans = _spans(annotation, video.fixture)
        raw_saved_spans = saved_videos[video.fixture].get("spans")
        if not isinstance(raw_saved_spans, list) or spans != raw_saved_spans:
            raise ValueError(f"{video.fixture}: saved section bounds differ")

        video_rows = _score_rows_for_video(checked_run.score_rows, video.fixture)
        raw_contacts = saved_contacts[video.fixture].get("contacts")
        if not isinstance(raw_contacts, list):
            raise TypeError(f"{video.fixture}: saved contacts must be a list")
        video_lists = lists_by_fixture.get(video.fixture, [])
        kept_frames = {
            int(_mapping(contact, "saved contact")["frame"]) for contact in raw_contacts
        }
        replay_frames = np.asarray(
            sorted(kept_frames | _candidate_frames(video_lists)),
            dtype=np.int32,
        )
        feature_rows = _rows_for_frames(
            video.fixture,
            _video_feature_rows(verified.raw_features, video.fixture),
            replay_frames,
        )
        _check_centre_feature_values(
            video.fixture,
            feature_rows,
            replay_frames,
            track,
            pose,
            sticky,
            (float(video.width), float(video.height)),
        )
        court_inputs = getattr(getattr(court, "evidence", None), "inputs", None)
        if court_inputs is None:
            raise ValueError(f"{video.fixture}: court inputs are unavailable")
        net_band = tuple(float(value) for value in court_inputs.net_band)
        if (
            len(net_band) != 2
            or not np.all(np.isfinite(net_band))
            or net_band[0] > net_band[1]
        ):
            raise ValueError(f"{video.fixture}: net band differs")
        sides = _checked_replay_sides(
            video.fixture,
            replay_frames,
            track,
            sticky,
            pose.bboxes,
            net_band,
            side_attributor,
        )
        kept_contacts = _saved_kept_contacts(
            video.fixture,
            video_rows,
            spans,
            [
                _mapping(contact, f"{video.fixture}: saved contact")
                for contact in raw_contacts
            ],
            sides,
        )
        rows_by_frame = {int(row["frame"]): row for row in video_rows}
        enriched_lists = _enriched_candidates(video_lists, rows_by_frame, sides)
        skipped = len(spans) - len(video_lists)
        skipped_sum += skipped
        first = _assemble_video_value(
            source_commit,
            video,
            frame_count,
            checked_files,
            spans,
            kept_contacts,
            enriched_lists,
            sides,
            skipped,
        )
        second = _assemble_video_value(
            source_commit,
            video,
            frame_count,
            checked_files,
            spans,
            kept_contacts,
            enriched_lists,
            sides,
            skipped,
        )
        if _json_bytes(first) != _json_bytes(second):
            raise ValueError(
                f"{video.fixture}: repeated validation input build differs"
            )
        saved_video_values.append(first)
        print(f"saved {video.fixture}", flush=True)

    if skipped_sum != skipped_total:
        raise ValueError("validation skipped-section count differs")
    expected_names = [video.fixture for video in verified.split.validation_videos]
    first_result = _assemble_result(
        source_commit,
        input_records,
        saved_video_values,
        expected_names,
    )
    second_result = _assemble_result(
        source_commit,
        input_records,
        saved_video_values,
        expected_names,
    )
    if _json_bytes(first_result) != _json_bytes(second_result):
        raise ValueError("repeated validation input combination differs")
    _write_json(destination, first_result)
    saved = _read_json(destination, "saved validation rally-start inputs")
    if _json_bytes(saved) != _json_bytes(first_result):
        raise ValueError("saved validation rally-start input differs")
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--raw-feature-record", type=Path, required=True)
    parser.add_argument("--common30-feature-record", type=Path, required=True)
    parser.add_argument("--contact-labels", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--menu-result", type=Path, required=True)
    parser.add_argument("--validation-rally-predictions", type=Path, required=True)
    parser.add_argument("--validation-rally-result", type=Path, required=True)
    parser.add_argument("--chosen-validation-scores", type=Path, required=True)
    parser.add_argument("--validation-candidate-summary", type=Path, required=True)
    parser.add_argument("--frozen-validation-candidates", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    save_validation_rally_start_inputs(
        ValidationInputFiles(
            arguments.config,
            arguments.split,
            arguments.raw_feature_record,
            arguments.common30_feature_record,
            arguments.contact_labels,
            arguments.baseline_summary,
            arguments.menu_result,
            arguments.validation_rally_predictions,
            arguments.validation_rally_result,
            arguments.chosen_validation_scores,
            arguments.validation_candidate_summary,
            arguments.frozen_validation_candidates,
        ),
        arguments.data_root,
        arguments.output,
        arguments.source_commit,
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
