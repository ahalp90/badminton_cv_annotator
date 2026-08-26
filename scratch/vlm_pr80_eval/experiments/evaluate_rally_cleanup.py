"""Replay VLM contact decisions through the normal annotator rally stages."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from annotator.calibration.fixtures import FIXTURES
from annotator.calibration.gt_scoring import (
    RallyRow,
    RunVideoInputs,
    load_gt_tables,
    score_video,
)
from annotator.fps_constants import ScalingKind
from annotator.point_winner import SHIPPED_LANDING_FILTER_OPTIONS
from annotator.run_video import AnnotatorResult, RunCapture, run_video
from dataset_builder.vision import (
    _court_inputs_from_payload,
    _raw_cuts_from_payload,
    annotation_result_payload,
    load_npy_xz,
    load_pose_arrays,
)

from .prompts import build_prompt
from .score_trials import _load_attempts
from .trial_schema import TrialArm, TrialKind, load_manifest

SCORE_SCHEMA = "vlm-rally-cleanup-score/0.2"
RULES = (
    "natural",
    "vlm-yes",
    "vlm-visible",
    "vlm-yes-preserve-first",
    "vlm-visible-preserve-first",
)


def _load_result(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
        raise TypeError(f"{path}: expected an annotator result object")
    return payload["result"]


def _load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _build_artifact_inputs(
    artifacts_root: Path, fixture: Any
) -> RunVideoInputs:
    """Rebuild the frozen Issue 103 call from its persisted stage inputs."""
    stages = artifacts_root / "stages"
    video_id = fixture.name
    track = load_npy_xz(stages / "shuttle" / video_id / "shuttle_track.npy.xz")
    guard_codes = load_npy_xz(
        stages / "shuttle" / video_id / "shuttle_guard_codes.npy.xz"
    )
    pose = load_pose_arrays(stages / "pose" / video_id, len(track))
    court_root = stages / "court" / video_id
    court_payload = _load_json_gz(court_root / "court_evidence.json.gz")
    court_inputs = _court_inputs_from_payload(court_payload["inputs"])
    raw_cuts = _raw_cuts_from_payload(court_payload["raw_cuts"])
    keep_vote = load_npy_xz(court_root / "court_keep_vote.npy.xz")
    court_present = load_npy_xz(court_root / "court_present.npy.xz")
    master, _homo, courts, _resolution = load_gt_tables()
    keyword: dict[str, object] = {
        "fps": fixture.fps,
        "landing_options": SHIPPED_LANDING_FILTER_OPTIONS,
        "net_band": court_inputs.net_band,
        "resolution": court_inputs.resolution,
        "video_id": video_id,
        "court_info": court_inputs.court_info,
        "homo_df": None,
        "gate_court_info": court_inputs.gate_court_info,
        "gate_resolution_table": court_inputs.gate_resolution_table,
        "ref_err_px": 3.5,
        "raw_exclusion_mask": None,
        "court_present": court_present,
        "homography_rows": court_inputs.homography_rows,
        "cut_frames": [end for _start, end in raw_cuts[:-1]],
        "keep_vote": keep_vote,
        "inpaint_codes": guard_codes,
        "court_invalid_is_excluded": True,
        "landing_error_band_m": court_inputs.landing_error_band_m,
    }
    return RunVideoInputs(
        (track, pose.bboxes, pose.scores, pose.kps, pose.ndet),
        keyword,
        master,
        courts,
    )


def _normalise_contacts(value: Mapping[Any, Sequence[Any]]) -> dict[int, list[int]]:
    return {
        int(span): sorted(map(int, frames))
        for span, frames in value.items()
    }


def select_rule_frames(
    natural_frames: Sequence[int],
    decisions: Mapping[int, dict[str, Any] | None],
    rule: str,
) -> list[int]:
    """Apply one frozen cleanup rule to a natural span's contact candidates."""
    if rule == "natural":
        return sorted(map(int, natural_frames))
    if rule not in RULES:
        raise ValueError(f"unknown cleanup rule {rule!r}")
    visible_only = "visible" in rule
    selected = {
        int(frame)
        for frame, parsed in decisions.items()
        if parsed is not None
        and parsed["contact_at_marker"] == "yes"
        and (not visible_only or parsed["evidence_kind"] == "visible-contact")
    }
    if rule.endswith("preserve-first") and natural_frames:
        selected.add(min(map(int, natural_frames)))
    return sorted(selected)


def _row_is_usable(row: RallyRow) -> bool:
    return (
        row.ball_round_correct
        and row.timing_matched_n == row.n_gt_strokes
        and row.player_correct
        and row.server_correct
        and row.getpoint_eligible
        and row.getpoint_correct is True
    )


def _selected_rows(scoring: Any, span_ids: set[int]) -> list[RallyRow]:
    rows = [row for row in scoring.rows if row.mapped_span in span_ids]
    if len(rows) != len(span_ids):
        raise ValueError(
            f"selected spans map to {len(rows)} ground-truth rallies, expected {len(span_ids)}"
        )
    return rows


def _score_rule(
    fixture: Any,
    result: AnnotatorResult,
    inputs: Any,
    span_ids: set[int],
) -> dict[str, Any]:
    by_tolerance: dict[str, Any] = {}
    for base30_tolerance in (5, 10, 15):
        native_tolerance = int(
            ScalingKind.FRAME_COUNT.scale(base30_tolerance, fixture.fps)
        )
        scoring = score_video(
            fixture,
            result,
            inputs.master,
            inputs.courts,
            native_tolerance,
        )
        rows = _selected_rows(scoring, span_ids)
        by_tolerance[str(base30_tolerance)] = {
            "native_frame_tolerance": native_tolerance,
            "rallies": len(rows),
            "exact_contact_count": sum(row.ball_round_correct for row in rows),
            "all_contacts_matched": sum(
                row.timing_matched_n == row.n_gt_strokes for row in rows
            ),
            "server_correct": sum(row.server_correct for row in rows),
            "final_actor_correct": sum(row.player_correct for row in rows),
            "all_alternating_attributions_correct": sum(
                row.ball_round_correct and row.server_correct for row in rows
            ),
            "point_outcome_correct": sum(
                row.getpoint_eligible and row.getpoint_correct is True for row in rows
            ),
            "structurally_usable": sum(_row_is_usable(row) for row in rows),
            "rows": [row._asdict() for row in rows],
        }
    return by_tolerance


def _natural_result_matches_artifact(
    natural: AnnotatorResult,
    artifact: dict[str, Any],
    capture: RunCapture,
    artifacts_root: Path,
    video_id: str,
) -> None:
    observed = annotation_result_payload(video_id, natural)["result"]
    if observed != artifact:
        differing = sorted(
            key for key in artifact if artifact.get(key) != observed.get(key)
        )
        raise ValueError(
            f"{video_id}: natural result differs from the frozen artefact in {differing}"
        )
    annotation_root = artifacts_root / "stages" / "annotation" / video_id
    expected_raw = load_npy_xz(annotation_root / "raw_replay_mask.npy.xz")
    expected_definitive = load_npy_xz(
        annotation_root / "definitive_exclusion_mask.npy.xz"
    )
    if capture.raw_exclusion_mask is None or not (
        capture.raw_exclusion_mask == expected_raw
    ).all():
        raise ValueError(f"{video_id}: rebuilt raw mask differs from the artefact")
    if capture.definitive_exclusion_mask is None or not (
        capture.definitive_exclusion_mask == expected_definitive
    ).all():
        raise ValueError(f"{video_id}: rebuilt definitive mask differs from the artefact")


def _validate_attempt_identity(
    case: Any, attempt: dict[str, Any], arm: TrialArm
) -> None:
    attempt_case = attempt["case"]
    expected = {
        "case_id": case.case_id,
        "kind": case.kind.value,
        "video_id": case.video_id,
        "clip_path": str(case.clip_path),
        "source_start_frame": case.source_start_frame,
        "source_end_frame": case.source_end_frame,
        "candidate_frame": case.candidate_frame,
    }
    for name, value in expected.items():
        if attempt_case[name] != value:
            raise ValueError(f"{case.case_id}: attempt has wrong {name}")
    clip_hash = hashlib.sha256(case.clip_path.read_bytes()).hexdigest()
    if attempt_case["clip_sha256"] != clip_hash:
        raise ValueError(f"{case.case_id}: attempt has wrong clip hash")
    prompt = build_prompt(case, arm)
    if attempt["prompt"] != prompt or attempt["prompt_sha256"] != hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest():
        raise ValueError(f"{case.case_id}: attempt has wrong prompt")


def evaluate(
    manifest_path: Path,
    attempts_root: Path,
    artifacts_root: Path,
    *,
    backend: str,
    arm: TrialArm,
) -> dict[str, Any]:
    """Evaluate selected full-span contact sets without changing production code."""
    cases = load_manifest(manifest_path)
    if any(case.kind is not TrialKind.EVENT for case in cases):
        raise ValueError("rally cleanup accepts event cases only")
    attempts = _load_attempts(attempts_root)
    expected_keys = {(backend, case.case_id, arm.value) for case in cases}
    observed_keys = set(attempts)
    if observed_keys != expected_keys:
        raise ValueError(
            f"attempt set differs: missing={sorted(expected_keys - observed_keys)}, "
            f"unexpected={sorted(observed_keys - expected_keys)}"
        )
    for case in cases:
        _validate_attempt_identity(
            case, attempts[(backend, case.case_id, arm.value)], arm
        )

    fixture_by_name = {fixture.name: fixture for fixture in FIXTURES}
    output: dict[str, Any] = {}
    for video_id in sorted({case.video_id for case in cases}):
        fixture = fixture_by_name[video_id]
        artifact_path = (
            artifacts_root
            / "stages"
            / "annotation"
            / video_id
            / "annotator_result.json.gz"
        )
        artifact = _load_result(artifact_path)
        artifact_contacts = _normalise_contacts(artifact["filtered_by_rally"])
        frame_to_span = {
            frame: span
            for span, frames in artifact_contacts.items()
            for frame in frames
        }
        video_cases = [case for case in cases if case.video_id == video_id]
        candidate_frames = {
            int(case.candidate_frame)
            for case in video_cases
            if case.candidate_frame is not None
        }
        missing_frames = candidate_frames - set(frame_to_span)
        if missing_frames:
            raise ValueError(
                f"{video_id}: cases are not natural filtered contacts: {sorted(missing_frames)}"
            )
        span_ids = {frame_to_span[frame] for frame in candidate_frames}
        expected_frames = {
            frame for span in span_ids for frame in artifact_contacts[span]
        }
        if candidate_frames != expected_frames:
            raise ValueError(f"{video_id}: cases do not cover every selected span candidate")

        inputs = _build_artifact_inputs(artifacts_root, fixture)
        capture = RunCapture()
        natural = run_video(*inputs.positional, **inputs.keyword, capture=capture)
        _natural_result_matches_artifact(
            natural, artifact, capture, artifacts_root, video_id
        )
        parsed_by_frame = {
            int(case.candidate_frame): attempts[
                (backend, case.case_id, arm.value)
            ]["parsed_response"]
            for case in video_cases
            if case.candidate_frame is not None
        }

        rule_results: dict[str, Any] = {}
        for rule in RULES:
            if rule == "natural":
                result = natural
            else:
                contacts = _normalise_contacts(natural.filtered_by_rally)
                for span in span_ids:
                    decisions = {
                        frame: parsed_by_frame[frame]
                        for frame in artifact_contacts[span]
                    }
                    contacts[span] = select_rule_frames(
                        artifact_contacts[span], decisions, rule
                    )
                result = run_video(
                    *inputs.positional,
                    **inputs.keyword,
                    spans=list(natural.spans),
                    contacts=contacts,
                )
            retained = _normalise_contacts(result.filtered_by_rally)
            rule_results[rule] = {
                "selected_contact_count": sum(
                    len(retained.get(span, [])) for span in span_ids
                ),
                "scores": _score_rule(fixture, result, inputs, span_ids),
            }
        output[video_id] = {
            "selected_spans": sorted(span_ids),
            "candidate_count": len(candidate_frames),
            "invalid_attempts": sum(
                parsed is None for parsed in parsed_by_frame.values()
            ),
            "rules": rule_results,
        }
    return {
        "schema": SCORE_SCHEMA,
        "manifest": str(manifest_path),
        "attempts_root": str(attempts_root),
        "artifacts_root": str(artifacts_root),
        "backend": backend,
        "arm": arm.value,
        "parse_complete": all(
            attempt["parsed_response"] is not None for attempt in attempts.values()
        ),
        "complete": all(
            attempt["generation_error"] is None
            and attempt["parsed_response"] is not None
            for attempt in attempts.values()
        ),
        "score_limit": (
            "ShuttleSet serve frames are structural timing. The structurally_usable "
            "field is not visible-serve usability for off-screen or inferred serves."
        ),
        "videos": output,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("qwen3-vl", "internvideo3"), required=True)
    parser.add_argument("--arm", choices=tuple(TrialArm), required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = evaluate(
        args.manifest,
        args.attempts,
        args.artifacts_root,
        backend=args.backend,
        arm=TrialArm(args.arm),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
