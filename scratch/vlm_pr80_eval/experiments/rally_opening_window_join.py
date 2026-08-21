"""Build truth-separated rally-opening windows from retained pipeline evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "vlm-rally-opening-window-manifest/1.0"
TRUTH_SCHEMA = "vlm-rally-opening-window-truth/1.0"
VIDEO_FPS = {"sset_01": 25.0, "sset_15": 25.0, "sset_21": 30.0}
FIRST_CONTACTS = 3
CUT_LEAD_SECONDS = 2.0
WINDOW_BUFFER_SECONDS = 5.0

COMMITTED_DIR = Path(
    "docs/scraper_pipeline/serve_prepend_lookback/data/"
    "serve_prepend_lookback_189c5af_20260808"
)
REVIEW_DIR = Path(
    "docs/scraper_pipeline/serve_prepend_lookback/data/"
    "rally_start_visibility_audit_20260809"
)
SERVER_TRUTH_PATH = Path(
    "scratch/serve_id_by_lookback_followup/results/preferred_server_rule.csv.gz"
)
SHOTS_MASTER_PATH = Path("training/data/shuttleset/annotations/shots_master.csv")

_FORBIDDEN_INFERENCE_KEYS = {
    "expected_server",
    "gt_first_contact",
    "gt_last_contact",
    "gt_serve_frame",
    "gt_server",
    "reviewed_visibility",
    "serve_visibility",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _load_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_json_gz(path: Path, payload: Mapping[str, object], *, replace: bool) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as stream,
    ):
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    path.write_bytes(buffer.getvalue())


def _reject_truth_keys(value: object, *, location: str = "manifest") -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_INFERENCE_KEYS.intersection(value)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(f"{location}: inference data contains truth fields: {names}")
        for key, child in value.items():
            _reject_truth_keys(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_truth_keys(child, location=f"{location}[{index}]")


def _scene_cut_frames(raw_cuts: Sequence[Sequence[int]], total_frames: int) -> list[int]:
    if not raw_cuts or list(raw_cuts[0])[:1] != [0]:
        raise ValueError("scene intervals must begin at frame zero")
    previous_end = 0
    cuts: list[int] = []
    for index, interval in enumerate(raw_cuts):
        if len(interval) != 2:
            raise ValueError(f"scene interval {index} does not have two endpoints")
        start, end = map(int, interval)
        if start != previous_end or end <= start:
            raise ValueError(f"scene interval {index} breaks contiguous coverage")
        if index:
            cuts.append(start)
        previous_end = end
    if previous_end != total_frames:
        raise ValueError("scene intervals do not cover the full video")
    return cuts


def opening_window(
    accepted_contacts: Sequence[int],
    scene_cut_frames: Sequence[int],
    *,
    fps: float,
    total_frames: int,
) -> dict[str, object]:
    """Return the deterministic opening-cut route for one automatic rally span."""
    contacts = [int(frame) for frame in accepted_contacts]
    if contacts != sorted(contacts):
        raise ValueError("accepted contacts must be ordered")
    if not contacts:
        return {
            "route_selected": False,
            "route_reason": "no_accepted_contacts",
            "early_contact_frames": [],
            "qualifying_cut_frames": [],
            "window_start_frame": None,
            "window_end_frame_exclusive": None,
        }

    early_contacts = contacts[:FIRST_CONTACTS]
    cut_search_start = early_contacts[0] - round(CUT_LEAD_SECONDS * fps)
    cut_search_end = early_contacts[-1]
    qualifying_cuts = [
        int(frame)
        for frame in scene_cut_frames
        if cut_search_start <= int(frame) <= cut_search_end
    ]
    if not qualifying_cuts:
        return {
            "route_selected": False,
            "route_reason": "no_qualifying_cut",
            "early_contact_frames": early_contacts,
            "qualifying_cut_frames": [],
            "window_start_frame": None,
            "window_end_frame_exclusive": None,
        }

    buffer_frames = round(WINDOW_BUFFER_SECONDS * fps)
    window_start = max(0, min(early_contacts[0], qualifying_cuts[0]) - buffer_frames)
    window_end = min(
        total_frames,
        max(early_contacts[-1], qualifying_cuts[-1]) + buffer_frames + 1,
    )
    return {
        "route_selected": True,
        "route_reason": "qualifying_opening_cut",
        "early_contact_frames": early_contacts,
        "qualifying_cut_frames": qualifying_cuts,
        "window_start_frame": window_start,
        "window_end_frame_exclusive": window_end,
    }


def _artifact_paths(artifacts_root: Path, video_id: str) -> dict[str, Path]:
    stages = artifacts_root / "stages"
    return {
        "annotation": stages / "annotation" / video_id / "annotator_result.json.gz",
        "court_evidence": stages / "court" / video_id / "court_evidence.json.gz",
    }


def _portable_input(path: Path, label: str) -> dict[str, object]:
    return {"label": label, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def build_manifest(artifacts_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Build all inference-side rows without opening any ground-truth source."""
    cases: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    video_summaries: dict[str, dict[str, object]] = {}

    for video_id, fps in VIDEO_FPS.items():
        paths = _artifact_paths(artifacts_root, video_id)
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        annotation_payload = _load_json_gz(paths["annotation"])
        court_payload = _load_json_gz(paths["court_evidence"])
        if annotation_payload.get("video_id") != video_id:
            raise ValueError(f"{video_id}: annotation identity differs")
        if court_payload.get("video_id") != video_id:
            raise ValueError(f"{video_id}: court evidence identity differs")

        result = annotation_payload["result"]
        spans = result["spans"]
        filtered_by_rally = result["filtered_by_rally"]
        total_frames = int(court_payload["raw_cuts"][-1][1])
        cut_frames = _scene_cut_frames(court_payload["raw_cuts"], total_frames)
        selected_count = 0
        for automatic_rally_index, span in enumerate(spans):
            contacts = [
                int(frame)
                for frame in filtered_by_rally.get(str(automatic_rally_index), [])
            ]
            route = opening_window(
                contacts,
                cut_frames,
                fps=fps,
                total_frames=total_frames,
            )
            selected_count += int(route["route_selected"] is True)
            window_start = route["window_start_frame"]
            window_end = route["window_end_frame_exclusive"]
            duration_frames = (
                None
                if window_start is None or window_end is None
                else int(window_end) - int(window_start)
            )
            cases.append(
                {
                    "case_id": f"rally-opening-{video_id}-p{automatic_rally_index:03d}",
                    "video_id": video_id,
                    "automatic_rally_index": automatic_rally_index,
                    "automatic_span_start_frame": int(span[0]),
                    "automatic_span_end_frame_exclusive": int(span[1]),
                    "accepted_contact_frames": contacts,
                    "automatic_server_prediction": result["next_servers"][automatic_rally_index],
                    "fps": fps,
                    "total_video_frames": total_frames,
                    "window_duration_frames": duration_frames,
                    "window_duration_seconds": (
                        None if duration_frames is None else duration_frames / fps
                    ),
                    **route,
                }
            )
        video_summaries[video_id] = {
            "automatic_rallies": len(spans),
            "scene_cuts": len(cut_frames),
            "route_selected": selected_count,
            "fps": fps,
            "total_video_frames": total_frames,
        }
        inputs.extend(
            _portable_input(path, f"{video_id}:{name}") for name, path in paths.items()
        )

    if len(cases) != 311:
        raise ValueError(f"expected 311 automatic rallies, found {len(cases)}")
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "settings": {
            "first_accepted_contacts_considered": FIRST_CONTACTS,
            "maximum_cut_lead_seconds": CUT_LEAD_SECONDS,
            "window_buffer_seconds": WINDOW_BUFFER_SECONDS,
            "contains_ground_truth": False,
            "selection_uses_ground_truth": False,
        },
        "videos": video_summaries,
        "cases": cases,
    }
    _reject_truth_keys(manifest)
    provenance = {"inference_inputs": inputs}
    return manifest, provenance


def _best_span_match(
    start: int,
    end: int,
    manifest_cases: Sequence[Mapping[str, object]],
) -> tuple[str, float] | None:
    scored: list[tuple[float, str]] = []
    for case in manifest_cases:
        span_start = int(case["automatic_span_start_frame"])
        span_end = int(case["automatic_span_end_frame_exclusive"])
        overlap = max(0, min(end, span_end) - max(start, span_start))
        if not overlap:
            continue
        union = max(end, span_end) - min(start, span_start)
        scored.append((overlap / union, str(case["case_id"])))
    if not scored:
        return None
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        raise ValueError(f"committed span [{start}, {end}) has an ambiguous best match")
    return scored[0][1], scored[0][0]


def _preferred_server_truth(repo_root: Path) -> dict[tuple[str, str, int], str]:
    rows = _load_csv_gz(repo_root / SERVER_TRUTH_PATH)
    mapped: dict[tuple[str, str, int], str] = {}
    for row in rows:
        key = (row["fixture"], row["set_id"], int(row["rally"]))
        server = {"top": "top", "bot": "bottom"}.get(row["gt_server"].lower())
        if server is None:
            raise ValueError(f"{key}: unsupported server label {server!r}")
        if key in mapped and mapped[key] != server:
            raise ValueError(f"{key}: conflicting server truth")
        mapped[key] = server
    if len(mapped) != 239:
        raise ValueError(f"expected 239 server-truth rows, found {len(mapped)}")
    return mapped


def _canonical_server_truth(repo_root: Path) -> dict[tuple[str, str, int], str]:
    mapped: dict[tuple[str, str, int], str] = {}
    for row in _load_csv(repo_root / SHOTS_MASTER_PATH):
        if int(row["ball_round"]) != 1:
            continue
        video_id = f"sset_{int(row['vid']):02d}"
        if video_id not in VIDEO_FPS:
            continue
        key = (video_id, row["set_id"], int(row["rally"]))
        server = row["player_side"].lower()
        if server not in {"top", "bottom"}:
            raise ValueError(f"{key}: unsupported canonical server label {server!r}")
        if key in mapped:
            raise ValueError(f"{key}: duplicate canonical first stroke")
        mapped[key] = server
    return mapped


def _reviewed_truth(repo_root: Path) -> dict[tuple[str, str, int], dict[str, object]]:
    mapped: dict[tuple[str, str, int], dict[str, object]] = {}
    for video_id in VIDEO_FPS:
        path = repo_root / REVIEW_DIR / f"{video_id}_rally_start_reviewed.csv.gz"
        for row in _load_csv_gz(path):
            key = (video_id, row["set_id"], int(row["rally"]))
            mapped[key] = {
                "serve_visibility": row["serve_visibility"],
                "visible_serve_frame": (
                    None if not row["visible_serve_frame"] else int(row["visible_serve_frame"])
                ),
                "first_visible_rally_frame": (
                    None
                    if not row["first_visible_rally_frame"]
                    else int(row["first_visible_rally_frame"])
                ),
                "broadcast_return_frame": (
                    None
                    if not row["broadcast_return_frame"]
                    else int(row["broadcast_return_frame"])
                ),
            }
    return mapped


def build_truth(
    repo_root: Path,
    manifest: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Join scoring labels after inference-side windows have been frozen."""
    manifest_cases = list(manifest["cases"])
    cases_by_video: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for case in manifest_cases:
        cases_by_video[str(case["video_id"])].append(case)

    server_truth = _canonical_server_truth(repo_root)
    preferred_server_truth = _preferred_server_truth(repo_root)
    for key, expected_server in preferred_server_truth.items():
        if server_truth.get(key) != expected_server:
            raise ValueError(f"{key}: preferred-rule truth differs from shots master")
    reviewed_truth = _reviewed_truth(repo_root)
    committed_by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    unmatched_committed: list[dict[str, object]] = []
    truth_inputs: list[dict[str, object]] = []
    mapped_server_keys: set[tuple[str, str, int]] = set()

    for video_id in VIDEO_FPS:
        committed_path = repo_root / COMMITTED_DIR / f"{video_id}_committed_rallies.csv.gz"
        rows = _load_csv_gz(committed_path)
        truth_inputs.append(_portable_input(committed_path, f"{video_id}:committed_rallies"))
        for row in rows:
            key = (video_id, row["set_id"], int(row["rally_number"]))
            expected_server = server_truth.get(key)
            common_truth = {
                "video_id": video_id,
                "set_id": row["set_id"],
                "rally": int(row["rally_number"]),
                "committed_status": row["status"],
                "span_category": row["span_category"],
                "gt_first_contact": int(row["gt_first_contact"]),
                "gt_last_contact": int(row["gt_last_contact"]),
                "gt_serve_frame": int(row["gt_serve_frame"]),
                "expected_server": expected_server,
                "reviewed_visibility": reviewed_truth.get(key),
            }
            if not row["span_start"]:
                unmatched_committed.append(
                    {**common_truth, "unmatched_reason": "no_committed_automatic_span"}
                )
                continue
            match = _best_span_match(
                int(row["span_start"]),
                int(row["span_end"]),
                cases_by_video[video_id],
            )
            if match is None:
                unmatched_committed.append(
                    {
                        **common_truth,
                        "unmatched_reason": "no_retained_automatic_span_overlap",
                    }
                )
                continue
            case_id, overlap_fraction = match
            if expected_server is not None:
                mapped_server_keys.add(key)
            committed_by_case[case_id].append(
                {
                    **common_truth,
                    "automatic_span_overlap_fraction": overlap_fraction,
                }
            )

    unmatched_server_keys = {
        (str(row["video_id"]), str(row["set_id"]), int(row["rally"]))
        for row in unmatched_committed
        if row["expected_server"] is not None
    }
    committed_server_keys = mapped_server_keys | unmatched_server_keys
    if committed_server_keys != set(server_truth):
        missing = sorted(set(server_truth) - committed_server_keys)
        raise ValueError(f"server truth disappeared during the join: {missing[:3]}")

    truth_cases: list[dict[str, object]] = []
    for case in manifest_cases:
        case_id = str(case["case_id"])
        committed = sorted(
            committed_by_case.get(case_id, []),
            key=lambda row: (str(row["set_id"]), int(row["rally"])),
        )
        server_labels = [row["expected_server"] for row in committed if row["expected_server"]]
        truth_cases.append(
            {
                "case_id": case_id,
                "video_id": case["video_id"],
                "automatic_rally_index": case["automatic_rally_index"],
                "committed_rallies": committed,
                "scorable_expected_server": (
                    server_labels[0] if len(committed) == 1 and len(server_labels) == 1 else None
                ),
                "server_scoring_reason": (
                    "one_to_one_server_truth"
                    if len(committed) == 1 and len(server_labels) == 1
                    else "no_one_to_one_server_truth"
                ),
            }
        )

    truth = {
        "schema": TRUTH_SCHEMA,
        "cases": truth_cases,
        "unmatched_committed_rallies": sorted(
            unmatched_committed,
            key=lambda row: (str(row["video_id"]), str(row["set_id"]), int(row["rally"])),
        ),
    }
    truth_inputs.append(_portable_input(repo_root / SERVER_TRUTH_PATH, "preferred_server_rule"))
    truth_inputs.append(_portable_input(repo_root / SHOTS_MASTER_PATH, "shots_master"))
    truth_inputs.extend(
        _portable_input(
            repo_root / REVIEW_DIR / f"{video_id}_rally_start_reviewed.csv.gz",
            f"{video_id}:reviewed_visibility",
        )
        for video_id in VIDEO_FPS
    )
    return truth, {"truth_inputs": truth_inputs}


def build_join(
    artifacts_root: Path,
    repo_root: Path,
    output_dir: Path,
    *,
    replace: bool = False,
) -> tuple[Path, Path]:
    manifest, inference_provenance = build_manifest(artifacts_root)
    truth, truth_provenance = build_truth(repo_root, manifest)
    if [case["case_id"] for case in manifest["cases"]] != [
        case["case_id"] for case in truth["cases"]
    ]:
        raise ValueError("manifest and truth case identities differ")
    manifest["provenance"] = inference_provenance
    truth["provenance"] = truth_provenance
    manifest_path = output_dir / "6_rally_opening_window_manifest.json.gz"
    truth_path = output_dir / "6_rally_opening_window_truth.json.gz"
    _write_json_gz(manifest_path, manifest, replace=replace)
    _write_json_gz(truth_path, truth, replace=replace)
    return manifest_path, truth_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest_path, truth_path = build_join(
        args.artifacts_root,
        args.repo_root,
        args.output_dir,
        replace=args.replace,
    )
    print(manifest_path)
    print(truth_path)


if __name__ == "__main__":
    main()
