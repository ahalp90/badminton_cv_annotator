"""Score one freshly prepared court comparison with frozen contact choosers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from annotator.point_winner import attribute_half
from annotator.scene_courts import build_scene_courts, court_at_frame
from dataset_builder.vision import load_npy_xz, save_json_gz, save_npy_xz
from experiments.annotator.court_geometry_repair.scripts.prepare_court_comparison import (
    _global_tracker_mode,
)
from scratch.contact_det.scripts.freeze_contact_evidence import (
    FixtureSpec,
    _load_inputs,
)
from scratch.contact_det.scripts.freeze_tree_contact_features import REGION_FIELDS
from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.boundary_followup import (
    pad_contact_boundaries,
)
from scratch.contact_det_closing_pass.scripts.features import _frozen_feature_names
from scratch.contact_det_closing_pass.scripts.prepare_broader_inputs import (
    _video_record,
)
from scratch.contact_det_closing_pass.scripts.prepare_later_inputs import (
    _physical_blocks,
    _sections,
)
from scratch.contact_det_closing_pass.scripts.run_broader_comparison import (
    predict_video as predict_opening_video,
)
from scratch.contact_det_closing_pass.scripts.run_broader_comparison import (
    stream_records,
)
from scratch.contact_det_closing_pass.scripts.run_insertion_broader import (
    predict_video as predict_insertion_video,
)
from scratch.contact_det_closing_pass.scripts.run_later_broader import (
    predict_video as predict_later_video,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_full_ds_fit.scripts.check_rally_start_candidates import (
    DUPLICATE_DISTANCE_AT_30_FPS,
    build_video_candidate_lists,
)
from scratch.contact_det_full_ds_fit.scripts.prepare_shuttleset22_predictions import (
    _score_candidates,
    load_model_bundle,
)
from scratch.contact_det_full_ds_fit.scripts.save_training_rally_start_inputs import (
    _enriched_candidates,
    _kept_contacts,
)

MODEL_FILES = {
    "contact": (
        "scratch/contact_det_full_ds_fit/raw/final_contact_model/contact_model.joblib",
        "scratch/contact_det_full_ds_fit/raw/final_contact_model/final_contact_model_result.json",
        "scratch/contact_det_full_ds_fit/raw/final_contact_scores/combined_first/final_contact_setting_result.json",
    ),
    "opening": "scratch/contact_det_closing_pass/raw/broader_models.joblib",
    "policy": "scratch/contact_det_closing_pass/results/broader_action_policy.json.gz",
    "later": "scratch/contact_det_closing_pass/raw/later_run/models.joblib",
    "local": "scratch/contact_det_closing_pass/raw/followups/local_models.joblib",
}
PROBE_FRAMES = {17: (46045, 47276), 53: (83084,)}
CONTACT_MODEL_FILENAMES = (
    "contact_model.joblib",
    "final_contact_model_result.json",
    "final_contact_setting_result.json",
)


def _contact_model_paths(model_root: Path, override: Path | None) -> tuple[Path, Path, Path]:
    if override is not None:
        contact_root = override.resolve(strict=True)
        return tuple(contact_root / filename for filename in CONTACT_MODEL_FILENAMES)
    return tuple(model_root / Path(path) for path in MODEL_FILES["contact"])


def _search_intervals(features: np.ndarray, fixture: str) -> tuple[tuple[int, int], ...]:
    ids = np.unique(features["interval_id"].astype(np.int64))
    if not np.array_equal(ids, np.arange(len(ids), dtype=np.int64)):
        raise ValueError(f"{fixture}: feature interval IDs are not contiguous")
    intervals = []
    for interval_id in ids:
        frames = features[features["interval_id"] == interval_id]["frame"].astype(np.int64)
        if not len(frames):
            raise ValueError(f"{fixture}/{interval_id}: feature interval is empty")
        start, end = int(frames.min()), int(frames.max()) + 1
        if not np.array_equal(frames, np.arange(start, end, dtype=np.int64)):
            raise ValueError(f"{fixture}/{interval_id}: feature interval has gaps")
        intervals.append((start, end))
    return tuple(intervals)


def _span_records(annotation: Any, fixture: str) -> list[dict[str, int]]:
    result, previous_end = [], -1
    for span_id, raw in enumerate(annotation.spans):
        start, end = map(int, raw)
        if end <= start or start < previous_end:
            raise ValueError(f"{fixture}/{span_id}: annotation spans overlap")
        result.append({"span_id": span_id, "start_frame": start, "end_frame": end})
        previous_end = end
    return result


def _make_spans(
    fixture: str,
    raw_spans: Sequence[Mapping[str, int]],
    rows_by_frame: Mapping[int, np.void],
    sides: Mapping[int, str | None],
) -> tuple[FixedSpan, ...]:
    output = []
    for raw in raw_spans:
        start, end = int(raw["start_frame"]), int(raw["end_frame"])
        events = tuple(
            FixedEvent(fixture, frame, float(row["contact_score"]), sides[frame])
            for frame, row in sorted(rows_by_frame.items())
            if bool(row["kept"]) and start <= frame < end
        )
        output.append(FixedSpan(fixture, int(raw["span_id"]), start, end, events))
    return tuple(output)


def _replay_sides(
    score_rows: np.ndarray, track: np.ndarray, pose: Any, sticky: Any, court: Any, baseline: bool,
) -> dict[int, str | None]:
    inputs = court.evidence.inputs
    if inputs is None:
        raise ValueError("court inputs are unavailable")
    resolution = tuple(float(value) for value in inputs.resolution)
    scenes = build_scene_courts(inputs.homography_rows.to_dict("records"), resolution)
    global_band = tuple(float(value) for value in inputs.net_band)
    present = np.asarray(court.evidence.court_present, dtype=bool)
    output: dict[int, str | None] = {}
    for frame in sorted({int(row["frame"]) for row in score_rows}):
        if frame < 0 or frame >= len(track):
            raise ValueError(f"frame {frame} lies outside the shuttle timeline")
        if not present[frame]:
            output[frame] = None
            continue
        band = global_band if baseline else court_at_frame(scenes, frame).net_band
        value = attribute_half(frame, track, sticky, pose.bboxes, band)
        side = None if value is None else getattr(value, "value", value)
        if side not in {None, "Top", "Bot", "Bottom"}:
            raise ValueError(f"frame {frame}: unexpected player side {side!r}")
        output[frame] = "Bot" if side == "Bottom" else side
    return output


def _probe(sticky: Any, track: np.ndarray, video_id: int, baseline: bool) -> dict[str, object]:
    frames = {}
    for frame in PROBE_FRAMES[video_id]:
        if frame >= len(track):
            raise ValueError(f"probe frame {frame} lies outside video {video_id}")
        frames[str(frame)] = {
            "picks": [int(value) for value in sticky.picks[frame]],
            "analysed": bool(sticky.analysed[frame]),
            "standing_count": int(sticky.standing_count[frame]),
        }
    return {
        "video_id": video_id, "baseline": baseline, "frames": frames,
        "counts": {
            "analysed_frames": int(np.count_nonzero(sticky.analysed)),
            "two_player_frames": int(np.count_nonzero(sticky.standing_count == 2)),
        },
    }


def _feature_root(output: Path, prepared: Path, fixture: str) -> Path:
    source = (prepared / "contact_features.npy.xz").resolve(strict=True)
    root = output / "features"
    destination = root / "videos" / fixture / "contact_features.npy.xz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)
    return root


def score(arguments: argparse.Namespace) -> None:
    prepared = Path(arguments.prepared).resolve(strict=True)
    model_root = Path(arguments.model_repo).resolve(strict=True)
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    video_id, fixture = int(arguments.video_id), str(arguments.video_id)
    spec = FixtureSpec(fixture, video_id, 30.0, 1920.0, 1080.0)
    contact_paths = _contact_model_paths(model_root, arguments.contact_model_dir)
    bundle = load_model_bundle(*contact_paths)
    broader_models = joblib.load(model_root / MODEL_FILES["opening"])
    policy = prediction_io.read_json(model_root / MODEL_FILES["policy"])
    later_models = joblib.load(model_root / MODEL_FILES["later"])
    local_models = joblib.load(model_root / MODEL_FILES["local"])

    features = load_npy_xz(prepared / "contact_features.npy.xz")
    if features.ndim != 1 or features.dtype.names is None or not len(features):
        raise ValueError(f"{fixture}: regenerated feature rows are empty or unstructured")
    intervals = _search_intervals(features, fixture)
    mask = np.zeros(len(features), dtype=bool)
    for field in REGION_FIELDS:
        if field not in features.dtype.names:
            raise ValueError(f"{fixture}: feature rows lack region field {field}")
        mask |= features[field].astype(bool)
    candidates = features[mask]
    if not len(candidates):
        raise ValueError(f"{fixture}: no regenerated contact search rows")
    scores, _predicted_frames = _score_candidates(candidates, spec, bundle)
    save_npy_xz(output / "candidate_scores.npy.xz", scores)

    with _global_tracker_mode(bool(arguments.baseline)):
        track, pose, court, _segments, sticky, annotation = _load_inputs(prepared, spec)
    if len(track) <= int(features["frame"].max()):
        raise ValueError(f"{fixture}: feature rows exceed track timeline")
    raw_spans = _span_records(annotation, fixture)
    rows_by_frame = {int(row["frame"]): row for row in scores}
    kept_frames = [frame for frame, row in rows_by_frame.items() if bool(row["kept"])]
    candidate_lists, skipped = build_video_candidate_lists(
        fixture, 30.0, scores, kept_frames, raw_spans, intervals, DUPLICATE_DISTANCE_AT_30_FPS
    )
    if any(len(item["candidates"]) != 3 for item in candidate_lists):
        raise ValueError(f"{fixture}: regenerated chooser candidate list is not three-wide")
    sides = _replay_sides(scores, track, pose, sticky, court, bool(arguments.baseline))
    fresh_spans = _make_spans(fixture, raw_spans, rows_by_frame, sides)
    events = tuple(
        FixedEvent(fixture, frame, float(row["contact_score"]), sides[frame])
        for frame, row in sorted(rows_by_frame.items()) if bool(row["kept"])
    )
    enriched = _enriched_candidates(candidate_lists, rows_by_frame, sides)
    kept_contacts = _kept_contacts(scores, "ShuttleSet22", raw_spans, sides)
    feature_root = _feature_root(output, prepared, fixture)
    opening_video = _video_record(
        source_commit="court-comparison", fixture=fixture, video_id=video_id, fps=30.0,
        frame_count=len(track), spans=raw_spans, kept_contacts=kept_contacts,
        candidate_lists=enriched, skipped=skipped, replayed_frames=len(sides), elapsed_seconds=0.0,
    )
    _, _, opening_choices, opening_record = predict_opening_video(
        opening_video, fresh_spans, events, broader_models, feature_root, policy
    )
    save_json_gz(output / "opening_record.json.gz", opening_record)

    physical_names = _frozen_feature_names()
    later_sections, later_frames, _ = _sections(fixture, fresh_spans, scores, 30.0)
    physical, _ = _physical_blocks(fixture, feature_root, later_frames, physical_names)
    for section in later_sections:
        for candidate in section["candidates"]:
            frame = int(candidate["frame"])
            candidate.update(predicted_side=sides[frame], physical=physical[frame])
    later_video = {"fixture": fixture, "group": "ShuttleSet22", "fps": 30.0, "sections": later_sections}
    _, _, later_record = predict_later_video(
        opening_video, later_video, physical_names, fresh_spans, events, later_models,
        feature_root, opening_choices,
    )
    save_json_gz(output / "later_record.json.gz", later_record)
    local_stream, _, _, local_record = predict_insertion_video(
        opening_video, later_video, fresh_spans, events, physical_names, feature_root,
        local_models, later_record["selected_actions"], later_record["output"], "local",
        output / "insertion_scores",
    )
    save_json_gz(output / "insertion_local_record.json.gz", local_record)
    padded = pad_contact_boundaries(
        local_stream.spans, local_stream.events_by_fixture, {fixture: 30.0},
        padding_base30=10, preserve_membership=True,
    )
    boundary_record = {
        "fixture": fixture, "fps": 30.0, "labels_read": False,
        "padding_base30": 10, "preserve_membership": True, "output": stream_records(padded),
    }
    save_json_gz(output / "boundary_record.json.gz", boundary_record)
    save_json_gz(output / "final_stream_records.json.gz", stream_records(padded))
    save_json_gz(output / "sticky_probe.json.gz", _probe(sticky, track, video_id, bool(arguments.baseline)))
    save_json_gz(output / "run_metadata.json.gz", {
        "schema": "court-comparison-score-run/1", "status": "complete", "labels_read": False,
        "video_id": video_id, "fixture": fixture, "baseline": bool(arguments.baseline),
        "candidate_row_count": len(scores), "kept_contact_count": len(events),
        "span_count": len(fresh_spans), "later_candidate_count": len(later_frames),
        "models": {name: str(path) for name, path in MODEL_FILES.items()},
        "contact_model_dir": (
            None
            if arguments.contact_model_dir is None
            else str(arguments.contact_model_dir.resolve(strict=True))
        ),
    })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument(
        "--contact-model-dir",
        type=Path,
        help="optional directory containing the contact model and its two JSON receipts",
    )
    parser.add_argument("--video-id", type=int, choices=(17, 53), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    score(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
