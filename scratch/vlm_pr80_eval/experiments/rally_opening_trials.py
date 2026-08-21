"""Build, run, and score dense rally-opening server trials."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2

from .backends import load_backend

MANIFEST_SCHEMA = "vlm-rally-opening-trial-manifest/1.0"
TRUTH_SCHEMA = "vlm-rally-opening-trial-truth/1.0"
ATTEMPT_SCHEMA = "vlm-rally-opening-attempt/1.0"
SCORE_SCHEMA = "vlm-rally-opening-score/1.0"
CLIP_SECONDS = 22.0
WIDTH = 512
HEIGHT = 288
MAX_NEW_TOKENS = 128
ARMS = {
    "clean_half_native": {"frame_step": 2, "use_navigation_cue": False},
    "cued_half_native": {"frame_step": 2, "use_navigation_cue": True},
    "cued_native": {"frame_step": 1, "use_navigation_cue": True},
}

PROMPT = """You are reviewing one continuous badminton broadcast clip centred on an automatically proposed rally opening. The clip may begin with a close-up or another camera view before returning to the full court.

Identify who served for the rally that begins in this clip. Follow the sequence from preparation through the first exchange. A player shown in close-up is not necessarily the server. Do not mistake a warm-up action, replay, return shot, or later contact for the serve.

TOP means the player on the far or top half in the normal full-court view. BOTTOM means the player on the near or bottom half. Use UNCLEAR when the sequence does not support either side.

{navigation_cue}

Return a bare JSON object with exactly two keys: server and evidence. server must be top, bottom, or unclear. evidence must be one short sentence describing the visible sequence you used. Do not use a Markdown fence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _portable_input(path: Path, label: str) -> dict[str, object]:
    return {"label": label, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _standard_window(case: Mapping[str, object]) -> tuple[int, int]:
    fps = float(case["fps"])
    target_frames = round(CLIP_SECONDS * fps)
    route_start = int(case["window_start_frame"])
    route_end = int(case["window_end_frame_exclusive"])
    if route_end - route_start > target_frames:
        raise ValueError(f"{case['case_id']}: routed window exceeds {CLIP_SECONDS} seconds")
    extra_frames = target_frames - (route_end - route_start)
    source_start = route_start - extra_frames // 2
    source_start = max(0, min(source_start, int(case["total_video_frames"]) - target_frames))
    source_end = source_start + target_frames
    if source_start > route_start or source_end < route_end:
        raise ValueError(f"{case['case_id']}: standard window does not contain routed evidence")
    return source_start, source_end


def _navigation_sentence(case: Mapping[str, object], source_start: int) -> str:
    fps = float(case["fps"])
    cut_times = [
        (int(frame) - source_start) / fps for frame in case["qualifying_cut_frames"]
    ]
    contact_times = [
        (int(frame) - source_start) / fps for frame in case["early_contact_frames"]
    ]
    cut_text = ", ".join(f"{time:.1f}" for time in cut_times)
    return (
        f"The automatic pipeline suggests a broadcast shot change at {cut_text} seconds "
        f"and places its first few possible racket contacts between {contact_times[0]:.1f} "
        f"and {contact_times[-1]:.1f} seconds. Treat this only as a place to inspect. "
        "The possible contacts may include returns or later shots and do not identify the server."
    )


def _prompt(case: Mapping[str, object], source_start: int, *, use_cue: bool) -> str:
    navigation_cue = (
        _navigation_sentence(case, source_start)
        if use_cue
        else "No automatic timing observations are provided."
    )
    return PROMPT.format(navigation_cue=navigation_cue)


def _selected_cases(
    manifest: Mapping[str, object], truth: Mapping[str, object]
) -> list[tuple[Mapping[str, object], Mapping[str, object]]]:
    truth_by_id = {str(case["case_id"]): case for case in truth["cases"]}
    selected = []
    for case in manifest["cases"]:
        case_truth = truth_by_id[str(case["case_id"])]
        committed = case_truth["committed_rallies"]
        if not case["route_selected"] or case_truth["scorable_expected_server"] is None:
            continue
        if len(committed) != 1 or committed[0]["reviewed_visibility"] is None:
            continue
        selected.append((case, case_truth))
    if len(selected) != 12:
        raise ValueError(f"expected 12 routed reviewed cases, found {len(selected)}")
    if {str(case["video_id"]) for case, _ in selected} != {
        "sset_01",
        "sset_15",
        "sset_21",
    }:
        raise ValueError("selected cases do not cover all three fixtures")
    return selected


def _render_clip(
    source_path: Path,
    output_path: Path,
    *,
    source_start: int,
    source_end: int,
    fps: float,
    frame_step: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open source video {source_path}")
    observed_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if abs(observed_fps - fps) > 0.01:
        capture.release()
        raise ValueError(f"source FPS {observed_fps} differs from expected {fps}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, source_start)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / frame_step,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not create clip {output_path}")
    written = 0
    try:
        for source_frame in range(source_start, source_end):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"source video ended at frame {source_frame}")
            if (source_frame - source_start) % frame_step:
                continue
            writer.write(cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
            written += 1
    finally:
        writer.release()
        capture.release()
    expected = (source_end - source_start) // frame_step
    if written != expected:
        raise ValueError(f"{output_path}: wrote {written} frames, expected {expected}")
    return written


def build_trials(
    join_manifest_path: Path,
    join_truth_path: Path,
    source_paths: Mapping[str, Path],
    output_dir: Path,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    manifest = _load_json_gz(join_manifest_path)
    truth = _load_json_gz(join_truth_path)
    selected = _selected_cases(manifest, truth)
    clips: dict[tuple[str, int], tuple[Path, int]] = {}
    manifest_cases: list[dict[str, object]] = []
    truth_cases: list[dict[str, object]] = []

    for case, case_truth in selected:
        video_id = str(case["video_id"])
        source_path = source_paths[video_id]
        source_start, source_end = _standard_window(case)
        for arm, settings in ARMS.items():
            frame_step = int(settings["frame_step"])
            clip_key = (str(case["case_id"]), frame_step)
            if clip_key not in clips:
                density = "native" if frame_step == 1 else "half-native"
                clip_path = output_dir / "inference" / "clips" / density / f"{case['case_id']}.mp4"
                frame_count = _render_clip(
                    source_path,
                    clip_path,
                    source_start=source_start,
                    source_end=source_end,
                    fps=float(case["fps"]),
                    frame_step=frame_step,
                )
                clips[clip_key] = (clip_path, frame_count)
            clip_path, frame_count = clips[clip_key]
            manifest_cases.append(
                {
                    "trial_id": f"{case['case_id']}--{arm}",
                    "case_id": case["case_id"],
                    "video_id": video_id,
                    "arm": arm,
                    "clip_path": str(clip_path.relative_to(output_dir / "inference")),
                    "clip_sha256": _sha256(clip_path),
                    "source_start_frame": source_start,
                    "source_end_frame_exclusive": source_end,
                    "sample_fps": float(case["fps"]) / frame_step,
                    "expected_input_frames": frame_count,
                    "prompt": _prompt(
                        case,
                        source_start,
                        use_cue=bool(settings["use_navigation_cue"]),
                    ),
                }
            )
        committed = case_truth["committed_rallies"][0]
        truth_cases.append(
            {
                "case_id": case["case_id"],
                "video_id": video_id,
                "expected_server": case_truth["scorable_expected_server"],
                "serve_visibility": committed["reviewed_visibility"]["serve_visibility"],
                "automatic_span_overlap_fraction": committed[
                    "automatic_span_overlap_fraction"
                ],
            }
        )

    inference = {
        "schema": MANIFEST_SCHEMA,
        "settings": {
            "clip_seconds": CLIP_SECONDS,
            "width": WIDTH,
            "height": HEIGHT,
            "contains_ground_truth": False,
            "selection": "routed one-to-one cases with independent visibility review",
            "selection_is_truth_filtered": True,
            "selection_labels_are_not_model_inputs": True,
        },
        "cases": manifest_cases,
        "provenance": {
            "inputs": [
                _portable_input(join_manifest_path, "rally_opening_window_manifest"),
                *[
                    _portable_input(path, f"{video_id}:source_video")
                    for video_id, path in sorted(source_paths.items())
                ],
            ]
        },
    }
    scoring = {
        "schema": TRUTH_SCHEMA,
        "cases": truth_cases,
        "provenance": {
            "inputs": [_portable_input(join_truth_path, "rally_opening_window_truth")]
        },
    }
    _write_new_json(output_dir / "inference" / "manifest.json", inference)
    _write_new_json(output_dir / "scoring" / "truth.json", scoring)


def parse_response(raw_response: str) -> dict[str, str]:
    payload = json.loads(raw_response.strip())
    if not isinstance(payload, dict) or set(payload) != {"server", "evidence"}:
        raise ValueError("response must contain exactly server and evidence")
    server = payload["server"]
    evidence = payload["evidence"]
    if server not in {"top", "bottom", "unclear"}:
        raise ValueError("server must be top, bottom, or unclear")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("evidence must be a non-empty string")
    return {"server": server, "evidence": evidence.strip()}


def run_trials(
    manifest_path: Path,
    output_dir: Path,
    *,
    arm: str,
    video_id: str,
    limit: int | None = None,
) -> None:
    manifest = _load_json(manifest_path)
    cases = [
        case
        for case in manifest["cases"]
        if case["arm"] == arm and case["video_id"] == video_id
    ]
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError(f"no cases for {arm} {video_id}")
    expected_frames = {int(case["expected_input_frames"]) for case in cases}
    if len(expected_frames) != 1:
        raise ValueError("one run group must have a fixed input-frame count")
    backend = load_backend("internvideo3", expected_input_frames=expected_frames.pop())
    model_identity = asdict(backend.spec.identity(backend.backend_version))
    failed_cases: list[str] = []
    for case in cases:
        clip_path = manifest_path.parent / case["clip_path"]
        attempt_path = output_dir / arm / video_id / f"{case['case_id']}.json"
        base = {
            "schema": ATTEMPT_SCHEMA,
            "model": model_identity,
            "trial_id": case["trial_id"],
            "case_id": case["case_id"],
            "video_id": video_id,
            "arm": arm,
            "clip_sha256": _sha256(clip_path),
            "prompt": case["prompt"],
            "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
        }
        started = perf_counter()
        try:
            evidence = backend.generate(
                clip_path,
                case["prompt"],
                requested_fps=float(case["sample_fps"]),
                width=WIDTH,
                height=HEIGHT,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            parsed = None
            parser_error = None
            try:
                parsed = parse_response(evidence.raw_response)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                parser_error = str(exc)
            payload = {
                **base,
                "raw_response": evidence.raw_response,
                "parsed_response": parsed,
                "parser_error": parser_error,
                "generation_error": None,
                "elapsed_seconds": perf_counter() - started,
                "sampling": {
                    "requested_fps": case["sample_fps"],
                    "sampled_input_frames": evidence.sampled_input_frames,
                    "width": evidence.width,
                    "height": evidence.height,
                    "visual_tokens": evidence.visual_tokens,
                    "total_input_tokens": evidence.total_input_tokens,
                    "max_new_tokens": MAX_NEW_TOKENS,
                },
            }
        # Preserve completed cases when one long-running inference fails.
        except Exception as exc:  # noqa: BLE001
            failed_cases.append(str(case["case_id"]))
            payload = {
                **base,
                "raw_response": None,
                "parsed_response": None,
                "parser_error": None,
                "generation_error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": perf_counter() - started,
                "sampling": None,
            }
        _write_new_json(attempt_path, payload)
        print(attempt_path, flush=True)
    if failed_cases:
        joined = ", ".join(failed_cases)
        raise RuntimeError(f"generation failed for {len(failed_cases)} cases: {joined}")


def _validate_attempt(
    attempt: Mapping[str, object],
    case: Mapping[str, object],
) -> None:
    expected_identity = {
        "schema": ATTEMPT_SCHEMA,
        "trial_id": case["trial_id"],
        "case_id": case["case_id"],
        "video_id": case["video_id"],
        "arm": case["arm"],
        "clip_sha256": case["clip_sha256"],
        "prompt": case["prompt"],
        "prompt_sha256": hashlib.sha256(str(case["prompt"]).encode()).hexdigest(),
    }
    for key, expected in expected_identity.items():
        if attempt.get(key) != expected:
            raise ValueError(f"{case['trial_id']}: attempt {key} differs")
    if attempt.get("generation_error") is not None:
        raise ValueError(f"{case['trial_id']}: attempt has a generation error")
    sampling = attempt.get("sampling")
    if not isinstance(sampling, Mapping):
        raise TypeError(f"{case['trial_id']}: attempt has no sampling evidence")
    frame_indices = sampling.get("sampled_input_frames")
    if not isinstance(frame_indices, list) or len(frame_indices) != int(
        case["expected_input_frames"]
    ):
        raise ValueError(f"{case['trial_id']}: sampled frame grid differs")
    if float(sampling["requested_fps"]) != float(case["sample_fps"]):
        raise ValueError(f"{case['trial_id']}: requested FPS differs")


def _paired_summary(
    rows: Sequence[Mapping[str, object]],
    left_arm: str,
    right_arm: str,
) -> dict[str, object]:
    left = {str(row["case_id"]): row for row in rows if row["arm"] == left_arm}
    right = {str(row["case_id"]): row for row in rows if row["arm"] == right_arm}
    if set(left) != set(right):
        raise ValueError(f"paired arms have different case identities: {left_arm}, {right_arm}")
    outcomes = Counter()
    changed_predictions = 0
    for case_id in sorted(left):
        left_row = left[case_id]
        right_row = right[case_id]
        left_correct = bool(left_row["server_correct"])
        right_correct = bool(right_row["server_correct"])
        if left_correct and right_correct:
            outcomes["both_correct"] += 1
        elif left_correct:
            outcomes["left_only_correct"] += 1
        elif right_correct:
            outcomes["right_only_correct"] += 1
        else:
            outcomes["both_wrong"] += 1
        changed_predictions += int(
            left_row["predicted_server"] != right_row["predicted_server"]
        )
    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "cases": len(left),
        "both_correct": outcomes["both_correct"],
        "left_only_correct": outcomes["left_only_correct"],
        "right_only_correct": outcomes["right_only_correct"],
        "both_wrong": outcomes["both_wrong"],
        "changed_predictions": changed_predictions,
    }


def score_trials(manifest_path: Path, truth_path: Path, attempts_dir: Path) -> dict[str, object]:
    manifest = _load_json(manifest_path)
    truth = _load_json(truth_path)
    truth_by_id = {str(case["case_id"]): case for case in truth["cases"]}
    expected_attempt_paths = {
        attempts_dir / case["arm"] / case["video_id"] / f"{case['case_id']}.json"
        for case in manifest["cases"]
    }
    observed_attempt_paths = set(attempts_dir.glob("*/*/*.json"))
    if observed_attempt_paths != expected_attempt_paths:
        raise ValueError("attempt file set differs from the frozen manifest")
    rows: list[dict[str, object]] = []
    model_identities: set[str] = set()
    for case in manifest["cases"]:
        attempt_path = (
            attempts_dir / case["arm"] / case["video_id"] / f"{case['case_id']}.json"
        )
        attempt = _load_json(attempt_path)
        _validate_attempt(attempt, case)
        model_identities.add(json.dumps(attempt["model"], sort_keys=True))
        expected = truth_by_id[str(case["case_id"])]
        parsed = attempt["parsed_response"]
        predicted_server = None if parsed is None else parsed["server"]
        rows.append(
            {
                "trial_id": case["trial_id"],
                "case_id": case["case_id"],
                "video_id": case["video_id"],
                "arm": case["arm"],
                "expected_server": expected["expected_server"],
                "serve_visibility": expected["serve_visibility"],
                "predicted_server": predicted_server,
                "server_correct": predicted_server == expected["expected_server"],
                "generation_error": attempt["generation_error"],
                "parser_error": attempt["parser_error"],
                "evidence": None if parsed is None else parsed["evidence"],
                "elapsed_seconds": attempt["elapsed_seconds"],
                "visual_tokens": (
                    None if attempt["sampling"] is None else attempt["sampling"]["visual_tokens"]
                ),
                "total_input_tokens": (
                    None
                    if attempt["sampling"] is None
                    else attempt["sampling"]["total_input_tokens"]
                ),
            }
        )
    if len(model_identities) != 1:
        raise ValueError("attempts do not use one model identity")
    summaries = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        summaries[arm] = {
            "cases": len(arm_rows),
            "server_correct": sum(row["server_correct"] is True for row in arm_rows),
            "predictions": dict(Counter(str(row["predicted_server"]) for row in arm_rows)),
            "generation_errors": sum(row["generation_error"] is not None for row in arm_rows),
            "parser_errors": sum(row["parser_error"] is not None for row in arm_rows),
        }
    paired = {
        "cue_effect_at_half_native": _paired_summary(
            rows, "clean_half_native", "cued_half_native"
        ),
        "density_effect_with_cues": _paired_summary(
            rows, "cued_half_native", "cued_native"
        ),
    }
    return {
        "schema": SCORE_SCHEMA,
        "summaries": summaries,
        "paired_comparisons": paired,
        "rows": rows,
    }


def _source_mapping(values: Sequence[str]) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    for value in values:
        video_id, separator, raw_path = value.partition("=")
        if not separator or video_id in mapped:
            raise ValueError(f"invalid source mapping: {value!r}")
        mapped[video_id] = Path(raw_path)
    if set(mapped) != {"sset_01", "sset_15", "sset_21"}:
        raise ValueError("source mappings must cover sset_01, sset_15, and sset_21")
    return mapped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--join-manifest", type=Path, required=True)
    build.add_argument("--join-truth", type=Path, required=True)
    build.add_argument("--source", action="append", required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--arm", choices=tuple(ARMS), required=True)
    run.add_argument("--video-id", choices=("sset_01", "sset_15", "sset_21"), required=True)
    run.add_argument("--limit", type=int)
    score = subparsers.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--attempts-dir", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        build_trials(
            args.join_manifest,
            args.join_truth,
            _source_mapping(args.source),
            args.output_dir,
        )
    elif args.command == "run":
        run_trials(
            args.manifest,
            args.output_dir,
            arm=args.arm,
            video_id=args.video_id,
            limit=args.limit,
        )
    else:
        score = score_trials(args.manifest, args.truth, args.attempts_dir)
        _write_new_json(args.output, score)


if __name__ == "__main__":
    main()
