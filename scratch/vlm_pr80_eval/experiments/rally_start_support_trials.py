"""Add compact automatic support to the frozen 32 rally-start clips."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .backends import load_backend
from .rally_start_trials import (
    ATTEMPT_SCHEMA,
    EXPECTED_FRAMES,
    HEIGHT,
    MAX_NEW_TOKENS,
    PROMPT,
    QWEN_MAX_MODEL_LEN,
    WIDTH,
    RallyStartCase,
    _load_attempts,
    _reject_truth_keys,
    _score_backend,
    _sha256,
    _write_new_json,
    load_manifest,
    parse_response,
)

SUPPORT_SCHEMA = "vlm-rally-start-support-v1"
SCORE_SCHEMA = "vlm-rally-start-support-score-v1"
INTRODUCTION = (
    "Automated video analysis produced the observations below. They may be "
    "wrong, so use them only as supporting evidence."
)


class SupportArm(StrEnum):
    OBSERVATIONS = "observations"
    OBSERVATIONS_PLUS_PROPOSALS = "observations_plus_proposals"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
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


def _load_npy_xz(path: Path) -> np.ndarray[Any, Any]:
    with lzma.open(path, "rb") as stream:
        return np.load(stream)


def _clip_hashes(provenance: Mapping[str, object]) -> dict[str, str]:
    outputs = provenance.get("outputs")
    if not isinstance(outputs, list):
        raise TypeError("provenance outputs must be a list")
    hashes: dict[str, str] = {}
    for raw in outputs:
        if not isinstance(raw, dict):
            raise TypeError("provenance output must be an object")
        path = Path(str(raw["path"]))
        case_id = path.stem
        if case_id in hashes:
            raise ValueError(f"duplicate provenance output for {case_id}")
        hashes[case_id] = str(raw["sha256"])
    return hashes


def _selections(provenance: Mapping[str, object]) -> dict[str, dict[str, Any]]:
    raw_selections = provenance.get("selections")
    if not isinstance(raw_selections, list):
        raise TypeError("provenance selections must be a list")
    selections: dict[str, dict[str, Any]] = {}
    for raw in raw_selections:
        if not isinstance(raw, dict):
            raise TypeError("provenance selection must be an object")
        case_id = raw.get("case_id")
        if not isinstance(case_id, str):
            raise TypeError("provenance selection case_id must be a string")
        if case_id in selections:
            raise ValueError(f"duplicate provenance selection for {case_id}")
        selections[case_id] = raw
    return selections


def _one_second_window(
    anchor_frame: int,
    source_start: int,
    source_end: int,
    fps: float,
) -> tuple[int, int]:
    frame_count = round(fps)
    start = max(source_start, anchor_frame - frame_count // 2)
    end = min(source_end, start + frame_count)
    if end - start < frame_count and start > source_start:
        start = max(source_start, end - frame_count)
    if not source_start <= start < end <= source_end:
        raise ValueError("automatic support window is outside the supplied clip")
    return start, end


def _count_sentence(subject: str, count: int, total: int) -> str:
    return f"{subject} in {count} of {total} frames near the inspection point."


def _current_contact(
    result: Mapping[str, Any],
    rally_id: int | None,
    anchor_frame: int,
    source_start: int,
    source_end: int,
) -> dict[str, Any] | None:
    if rally_id is None:
        return None
    frames = [
        int(frame)
        for frame in result["filtered_by_rally"].get(str(rally_id), [])
        if anchor_frame <= int(frame) < source_end
    ]
    if not frames:
        return None
    contact_frame = min(frames)
    matching = [
        row
        for row in result["filtered_contacts"]
        if int(row["rally_id"]) == rally_id
        and int(row["contact_frame"]) == contact_frame
    ]
    if len(matching) != 1:
        raise ValueError(
            f"rally {rally_id}: expected one accepted contact at {contact_frame}"
        )
    return {**matching[0], "clip_frame": contact_frame - source_start}


def _observation_sentences(
    *,
    anchor_clip_frame: int,
    cut_clip_frame: int | None,
    court_count: int,
    player_count: int,
    shuttle_count: int,
    window_frames: int,
    contact: Mapping[str, Any] | None,
) -> list[str]:
    sentences = [f"The automatic inspection point is clip frame {anchor_clip_frame}."]
    if cut_clip_frame is None:
        sentences.append("The automatic analysis did not select a camera cut in this clip.")
    else:
        sentences.append(
            f"The automatic analysis marked a hard camera cut at clip frame {cut_clip_frame}."
        )
    sentences.extend(
        [
            _count_sentence(
                "A usable full-court scene was detected", court_count, window_frames
            ),
            _count_sentence(
                "Exactly two on-court players were detected",
                player_count,
                window_frames,
            ),
            _count_sentence(
                "The shuttle tracker reported the shuttle", shuttle_count, window_frames
            ),
        ]
    )
    if contact is not None and contact["wrist_near"] is True:
        sentences.append(
            "The shuttle was detected close to a player's wrist near the inspection point."
        )
    elif contact is not None and contact["wrist_near"] is False:
        sentences.append(
            "The shuttle was not detected close to a player's wrist near the inspection point."
        )
    if contact is not None and contact["proximity_ok"] is True:
        sentences.append(
            "The contact proposal also passed the automatic player-proximity check."
        )
    elif contact is not None and contact["proximity_ok"] is False:
        sentences.append(
            "The contact proposal failed the automatic player-proximity check."
        )
    return sentences


def _proposal_sentences(
    result: Mapping[str, Any],
    rally_id: int | None,
    contact: Mapping[str, Any] | None,
) -> list[str]:
    if rally_id is None:
        return [
            "The current pipeline did not resolve a server or physical-contact proposal for this clip."
        ]
    raw_server = result["fitted_first_all"][rally_id]
    server = None if raw_server is None else str(raw_server).lower()
    if server == "bot":
        server = "bottom"
    sentences = []
    if server in {"top", "bottom"}:
        sentences.append(
            f"The current pipeline proposes {server.upper()} as the server. This proposal may be wrong."
        )
    else:
        sentences.append("The current pipeline did not resolve a server proposal.")
    if contact is None:
        sentences.append("The current pipeline did not resolve a physical-contact proposal.")
    else:
        sentences.append(
            "The current pipeline proposes clip frame "
            f"{int(contact['clip_frame'])} as physical contact. This proposal may be wrong."
        )
    return sentences


def build_prompt(case: Mapping[str, Any], arm: SupportArm) -> str:
    observations = case.get("observations")
    proposals = case.get("proposals")
    if not isinstance(observations, list) or not all(
        isinstance(sentence, str) for sentence in observations
    ):
        raise TypeError("support observations must be a list of strings")
    if not isinstance(proposals, list) or not all(
        isinstance(sentence, str) for sentence in proposals
    ):
        raise TypeError("support proposals must be a list of strings")
    sentences = observations if arm is SupportArm.OBSERVATIONS else observations + proposals
    facts = "\n".join(f"- {sentence}" for sentence in sentences)
    return f"{PROMPT}\n\n{INTRODUCTION}\n\n{facts}"


def build_support(
    base_manifest_path: Path,
    provenance_path: Path,
    artifacts_root: Path,
) -> dict[str, object]:
    """Build plain-language support without opening the scoring truth."""
    base_cases = load_manifest(base_manifest_path, require_clips=False)
    provenance = _load_json(provenance_path)
    if provenance.get("manifest_sha256") != _sha256(base_manifest_path):
        raise ValueError("provenance does not describe the base manifest")
    selections = _selections(provenance)
    clip_hashes = _clip_hashes(provenance)
    expected_ids = {case.case_id for case in base_cases}
    if set(selections) != expected_ids or set(clip_hashes) != expected_ids:
        raise ValueError("provenance case IDs differ from the base manifest")

    video_inputs: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    for video_id in sorted({case.video_id for case in base_cases}):
        annotation_path = (
            artifacts_root
            / "stages"
            / "annotation"
            / video_id
            / "annotator_result.json.gz"
        )
        court_path = (
            artifacts_root / "stages" / "court" / video_id / "court_present.npy.xz"
        )
        keep_path = (
            artifacts_root / "stages" / "court" / video_id / "court_keep_vote.npy.xz"
        )
        track_path = (
            artifacts_root / "stages" / "shuttle" / video_id / "shuttle_track.npy.xz"
        )
        payload = _load_json_gz(annotation_path)
        result = payload.get("result")
        if payload.get("video_id") != video_id or not isinstance(result, dict):
            raise ValueError(f"{video_id}: invalid annotator result")
        court_present = _load_npy_xz(court_path)
        keep_vote = _load_npy_xz(keep_path)
        track = _load_npy_xz(track_path)
        if (
            court_present.shape != keep_vote.shape
            or court_present.shape != (len(track),)
            or court_present.dtype != np.bool_
            or keep_vote.dtype != np.bool_
            or track.ndim != 2
            or track.shape[1] != 3
        ):
            raise ValueError(f"{video_id}: automatic support arrays differ in shape")
        video_inputs[video_id] = {
            "result": result,
            "court_present": court_present,
            "keep_vote": keep_vote,
            "track_visible": track[:, 2] > 0,
        }
        source_hashes[video_id] = {
            "annotator_result": _sha256(annotation_path),
            "court_present": _sha256(court_path),
            "court_keep_vote": _sha256(keep_path),
            "shuttle_track": _sha256(track_path),
        }

    cases: list[dict[str, object]] = []
    for case in base_cases:
        selection = selections[case.case_id]
        if (
            selection["video_id"] != case.video_id
            or int(selection["source_start_frame"]) != case.source_start_frame
            or int(selection["source_end_frame"]) != case.source_end_frame
        ):
            raise ValueError(f"{case.case_id}: selection identity differs")
        anchor_frame = int(selection["anchor_frame"])
        raw_cut = selection["selected_cut_frame"]
        cut_frame = None if raw_cut is None else int(raw_cut)
        window_start, window_end = _one_second_window(
            anchor_frame,
            case.source_start_frame,
            case.source_end_frame,
            case.sample_fps,
        )
        inputs = video_inputs[case.video_id]
        result = inputs["result"]
        raw_rally_id = selection["current_rally_id_at_anchor"]
        rally_id = None if raw_rally_id is None else int(raw_rally_id)
        contact = _current_contact(
            result,
            rally_id,
            anchor_frame,
            case.source_start_frame,
            case.source_end_frame,
        )
        observations = _observation_sentences(
            anchor_clip_frame=anchor_frame - case.source_start_frame,
            cut_clip_frame=(
                None if cut_frame is None else cut_frame - case.source_start_frame
            ),
            court_count=int(np.count_nonzero(inputs["court_present"][window_start:window_end])),
            player_count=int(np.count_nonzero(inputs["keep_vote"][window_start:window_end])),
            shuttle_count=int(np.count_nonzero(inputs["track_visible"][window_start:window_end])),
            window_frames=window_end - window_start,
            contact=contact,
        )
        proposals = _proposal_sentences(result, rally_id, contact)
        case_payload: dict[str, object] = {
            "case_id": case.case_id,
            "video_id": case.video_id,
            "clip_sha256": clip_hashes[case.case_id],
            "source_start_frame": case.source_start_frame,
            "source_end_frame": case.source_end_frame,
            "support_window_start_frame": window_start,
            "support_window_end_frame": window_end,
            "observations": observations,
            "proposals": proposals,
        }
        case_payload["prompt_sha256_by_arm"] = {
            arm.value: hashlib.sha256(
                build_prompt(case_payload, arm).encode("utf-8")
            ).hexdigest()
            for arm in SupportArm
        }
        cases.append(case_payload)

    output: dict[str, object] = {
        "schema": SUPPORT_SCHEMA,
        "contains_scoring_truth": False,
        "base_manifest_sha256": _sha256(base_manifest_path),
        "base_provenance_sha256": _sha256(provenance_path),
        "base_prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "source_hashes": source_hashes,
        "cases": cases,
    }
    _reject_truth_keys(output, "support manifest")
    return output


def load_support(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    _reject_truth_keys(payload, "support manifest")
    if payload.get("schema") != SUPPORT_SCHEMA:
        raise ValueError("support manifest schema differs")
    if payload.get("contains_scoring_truth") is not False:
        raise ValueError("support manifest must be explicitly truth-free")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("support manifest cases must be a non-empty list")
    cases: dict[str, dict[str, Any]] = {}
    for raw in raw_cases:
        if not isinstance(raw, dict) or not isinstance(raw.get("case_id"), str):
            raise TypeError("support case must have a string case_id")
        case_id = raw["case_id"]
        if case_id in cases:
            raise ValueError(f"duplicate support case {case_id}")
        for arm in SupportArm:
            expected_hash = hashlib.sha256(
                build_prompt(raw, arm).encode("utf-8")
            ).hexdigest()
            if raw["prompt_sha256_by_arm"][arm.value] != expected_hash:
                raise ValueError(f"{case_id}: {arm.value} prompt hash differs")
        cases[case_id] = raw
    return cases


def _attempt_base(
    backend_name: str,
    backend: Any,
    case: RallyStartCase,
    support: Mapping[str, Any],
    arm: SupportArm,
) -> tuple[dict[str, object], str]:
    if _sha256(case.clip_path) != support["clip_sha256"]:
        raise ValueError(f"{case.case_id}: clip hash differs from Follow-up 2")
    prompt = build_prompt(support, arm)
    return (
        {
            "schema": ATTEMPT_SCHEMA,
            "backend": backend_name,
            "model": asdict(backend.spec.identity(backend.backend_version)),
            "case": {
                "case_id": case.case_id,
                "video_id": case.video_id,
                "clip_path": str(case.clip_path),
                "clip_sha256": support["clip_sha256"],
                "source_start_frame": case.source_start_frame,
                "source_end_frame": case.source_end_frame,
            },
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        },
        prompt,
    )


def run_trials(
    backend_name: str,
    base_manifest_path: Path,
    support_path: Path,
    arm: SupportArm,
    output_dir: Path,
) -> None:
    base_cases = load_manifest(base_manifest_path)
    support_cases = load_support(support_path)
    if {case.case_id for case in base_cases} != set(support_cases):
        raise ValueError("base and support case IDs differ")
    max_model_len = QWEN_MAX_MODEL_LEN if backend_name == "qwen3-vl" else None
    backend = load_backend(
        backend_name,
        expected_input_frames=EXPECTED_FRAMES,
        max_model_len=max_model_len,
    )
    for case in base_cases:
        support = support_cases[case.case_id]
        base_payload, prompt = _attempt_base(
            backend_name, backend, case, support, arm
        )
        attempt_path = output_dir / backend_name / f"{case.case_id}.json"
        started = perf_counter()
        try:
            evidence = backend.generate(
                case.clip_path,
                prompt,
                requested_fps=case.sample_fps,
                width=WIDTH,
                height=HEIGHT,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        except Exception as error:
            _write_new_json(
                attempt_path,
                {
                    **base_payload,
                    "raw_response": None,
                    "parsed_response": None,
                    "parser_error": None,
                    "generation_error": f"{type(error).__name__}: {error}",
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
        except (TypeError, ValueError) as error:
            parser_error = str(error)
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


def _validated_attempts(
    base_cases: tuple[RallyStartCase, ...],
    support_cases: Mapping[str, Mapping[str, Any]],
    arm: SupportArm,
    backend_name: str,
    attempts_root: Path,
) -> dict[str, dict[str, Any]]:
    cases_by_id = {case.case_id: case for case in base_cases}
    attempts = _load_attempts(attempts_root, backend_name, cases_by_id)
    for case_id, attempt in attempts.items():
        support = support_cases[case_id]
        expected_prompt = build_prompt(support, arm)
        case_payload = attempt["case"]
        if (
            attempt["prompt"] != expected_prompt
            or attempt["prompt_sha256"]
            != hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
            or case_payload["clip_sha256"] != support["clip_sha256"]
            or int(case_payload["source_start_frame"])
            != cases_by_id[case_id].source_start_frame
            or int(case_payload["source_end_frame"])
            != cases_by_id[case_id].source_end_frame
        ):
            raise ValueError(f"{case_id}: attempt identity differs")
    return attempts


def score_trials(
    base_manifest_path: Path,
    truth_path: Path,
    support_path: Path,
    arm: SupportArm,
    backend_name: str,
    attempts_root: Path,
) -> dict[str, object]:
    base_cases = load_manifest(base_manifest_path, require_clips=False)
    support_cases = load_support(support_path)
    if {case.case_id for case in base_cases} != set(support_cases):
        raise ValueError("base and support case IDs differ")
    truth_payload = _load_json(truth_path)
    raw_truth = truth_payload.get("cases")
    if not isinstance(raw_truth, list):
        raise TypeError("truth cases must be a list")
    truth = {str(row["case_id"]): row for row in raw_truth}
    if set(truth) != set(support_cases):
        raise ValueError("truth and support case IDs differ")
    attempts = _validated_attempts(
        base_cases,
        support_cases,
        arm,
        backend_name,
        attempts_root,
    )
    return {
        "schema": SCORE_SCHEMA,
        "arm": arm.value,
        "backend": backend_name,
        "base_manifest_sha256": _sha256(base_manifest_path),
        "truth_sha256": _sha256(truth_path),
        "support_manifest_sha256": _sha256(support_path),
        "result": _score_backend(base_cases, truth, attempts),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-support")
    build.add_argument("--base-manifest", type=Path, required=True)
    build.add_argument("--base-provenance", type=Path, required=True)
    build.add_argument("--artifacts-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--backend", choices=("internvideo3", "qwen3-vl"), required=True)
    run.add_argument("--base-manifest", type=Path, required=True)
    run.add_argument("--support", type=Path, required=True)
    run.add_argument("--arm", type=SupportArm, choices=tuple(SupportArm), required=True)
    run.add_argument("--out", dest="output", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--base-manifest", type=Path, required=True)
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--support", type=Path, required=True)
    score.add_argument("--arm", type=SupportArm, choices=tuple(SupportArm), required=True)
    score.add_argument("--backend", choices=("internvideo3", "qwen3-vl"), required=True)
    score.add_argument("--attempts", type=Path, required=True)
    score.add_argument("--out", dest="output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build-support":
        payload = build_support(
            args.base_manifest,
            args.base_provenance,
            args.artifacts_root,
        )
    elif args.command == "run":
        run_trials(
            args.backend,
            args.base_manifest,
            args.support,
            args.arm,
            args.output,
        )
        return
    else:
        payload = score_trials(
            args.base_manifest,
            args.truth,
            args.support,
            args.arm,
            args.backend,
            args.attempts,
        )
    _write_new_json(args.output, payload)


if __name__ == "__main__":
    main()
