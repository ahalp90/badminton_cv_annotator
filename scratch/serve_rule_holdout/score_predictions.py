"""Join frozen issue #90 predictions to ShuttleSet labels and score them."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import pandas as pd

from annotator.calibration.gt_scoring import load_gt_tables
from annotator.calibration.scoring import RallyBoundary, classify_all, load_gt_rallies
from scratch.serve_start_trajectory_exploration.trajectory_features import align_anchor_to_gt


VIDEO_FPS = {20: 25.0, 22: 30.0}
TOLERANCE_BASE30_FRAMES = 10
EXPECTED_PREDICTIONS = 153


@dataclass(frozen=True)
class TemporalScore:
    """Ground-truth alignment of one frozen temporal claim."""

    claim: str
    frame: int | None
    gt_label: str
    correct: bool
    multiple_within_tolerance: bool


def _bool_text(value: bool) -> str:
    return "True" if value else "False"


def _read_freeze(freeze_dir: Path) -> list[dict[str, str]]:
    """Verify the immutable prediction freeze before labels are loaded."""
    manifest_path = freeze_dir / "prediction_freeze.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_path = freeze_dir / str(manifest["prediction_file"])
    digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    if manifest.get("labels_read") is not False:
        raise ValueError("prediction manifest does not preserve the label-blind boundary")
    if digest != manifest.get("prediction_sha256"):
        raise ValueError("prediction freeze checksum mismatch")

    with gzip.open(prediction_path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_PREDICTIONS or manifest.get("population") != len(rows):
        raise ValueError("prediction freeze population mismatch")
    forbidden = ("gt_", "truth", "label", "correct")
    fields = set(rows[0])
    if any(token in field.lower() for field in fields for token in forbidden):
        raise ValueError("prediction freeze contains a ground-truth or score field")
    keys = [(row["video_id"], int(row["span_id"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("prediction freeze contains duplicate span keys")
    return rows


def _normalise_side(value: object) -> str:
    side = str(value)
    if side == "Top":
        return side
    if side in {"Bot", "Bottom"}:
        return "Bot"
    raise ValueError(f"unexpected court side {value!r}")


def _prediction_spans(rows: Sequence[Mapping[str, str]]) -> list[tuple[int, int]]:
    ordered = sorted(rows, key=lambda row: int(row["span_id"]))
    span_ids = [int(row["span_id"]) for row in ordered]
    if span_ids != list(range(len(ordered))):
        raise ValueError("prediction span ids are not contiguous")
    return [(int(row["span_start"]), int(row["span_end"])) for row in ordered]


def _temporal_score(
    claim: str,
    frame: int | None,
    stroke_frames: Sequence[int],
    fps: float,
) -> TemporalScore:
    if frame is None:
        return TemporalScore(claim, None, "unmatched", False, False)
    alignment = align_anchor_to_gt(
        frame,
        stroke_frames,
        fps,
        TOLERANCE_BASE30_FRAMES,
    )
    correct = (claim == "serve" and alignment.label == "contact_1") or (
        claim == "return" and alignment.label == "contact_2"
    )
    return TemporalScore(
        claim,
        frame,
        alignment.label,
        correct,
        alignment.multiple_within_tolerance,
    )


def _pr82_temporal(
    prediction: Mapping[str, str], stroke_frames: Sequence[int], fps: float
) -> TemporalScore:
    claim = "return" if prediction["pr82_pre_verdict"] == "incoming" else "serve"
    raw_frame = prediction["pr82_anchor_frame"]
    return _temporal_score(claim, int(raw_frame) if raw_frame else None, stroke_frames, fps)


def _pr88_temporal(
    prediction: Mapping[str, str],
    baseline: TemporalScore,
    stroke_frames: Sequence[int],
    fps: float,
) -> TemporalScore:
    category = prediction["pr88_category"]
    if category == "first_visible_post_serve_contact":
        return _temporal_score(
            "return", int(prediction["pr88_selected_frame"]), stroke_frames, fps
        )
    if category == "visible_serve":
        return _temporal_score(
            "serve", int(prediction["pr88_selected_frame"]), stroke_frames, fps
        )
    return baseline


def _score_row(
    prediction: Mapping[str, str],
    video_id: int,
    set_id: str,
    rally: int,
    stroke_frames: Sequence[int],
    gt_server: str,
) -> dict[str, object]:
    fps = VIDEO_FPS[video_id]
    pr82_temporal = _pr82_temporal(prediction, stroke_frames, fps)
    pr88_temporal = _pr88_temporal(prediction, pr82_temporal, stroke_frames, fps)
    pr82_server = prediction["pr82_server"]
    pr88_server = prediction["pr88_server"]
    pr82_server_correct = pr82_server == gt_server
    pr88_server_correct = pr88_server == gt_server
    if pr88_server_correct and not pr82_server_correct:
        paired_outcome = "fix"
    elif pr82_server_correct and not pr88_server_correct:
        paired_outcome = "damage"
    elif pr88_server_correct:
        paired_outcome = "both_correct"
    else:
        paired_outcome = "both_wrong"

    row: dict[str, object] = {
        "video_id": f"sset_{video_id:02d}",
        "set_id": set_id,
        "rally": rally,
        "span_id": int(prediction["span_id"]),
        "gt_server": gt_server,
        "gt_first_frame": stroke_frames[0],
        "gt_second_frame": stroke_frames[1] if len(stroke_frames) > 1 else None,
        "pr82_server": pr82_server,
        "pr82_server_correct": _bool_text(pr82_server_correct),
        "pr82_temporal_claim": pr82_temporal.claim,
        "pr82_claimed_frame": pr82_temporal.frame,
        "pr82_temporal_gt_label": pr82_temporal.gt_label,
        "pr82_visible_start_correct": _bool_text(pr82_temporal.correct),
        "pr82_joint_correct": _bool_text(pr82_server_correct and pr82_temporal.correct),
        "pr88_branch": prediction["pr88_branch"],
        "pr88_category": prediction["pr88_category"],
        "pr88_server": pr88_server,
        "pr88_server_correct": _bool_text(pr88_server_correct),
        "pr88_temporal_claim": pr88_temporal.claim,
        "pr88_claimed_frame": pr88_temporal.frame,
        "pr88_temporal_gt_label": pr88_temporal.gt_label,
        "pr88_visible_start_correct": _bool_text(pr88_temporal.correct),
        "pr88_joint_correct": _bool_text(pr88_server_correct and pr88_temporal.correct),
        "changed_vs_pr82": _bool_text(pr88_server != pr82_server),
        "paired_server_outcome": paired_outcome,
        "multiple_temporal_matches": _bool_text(
            pr82_temporal.multiple_within_tolerance
            or pr88_temporal.multiple_within_tolerance
        ),
    }
    return row


def _exact_mcnemar_p(fixes: int, damages: int) -> float:
    changed = fixes + damages
    if changed == 0:
        return 1.0
    tail = min(fixes, damages)
    probability = 2.0 * sum(math.comb(changed, k) for k in range(tail + 1)) / 2**changed
    return min(1.0, probability)


def _rule_summary(rows: Sequence[Mapping[str, object]], prefix: str) -> dict[str, int]:
    return {
        "population": len(rows),
        "server_correct": sum(row[f"{prefix}_server_correct"] == "True" for row in rows),
        "visible_start_correct": sum(
            row[f"{prefix}_visible_start_correct"] == "True" for row in rows
        ),
        "joint_correct": sum(row[f"{prefix}_joint_correct"] == "True" for row in rows),
    }


def _boundary_summary(
    boundaries: Sequence[tuple[RallyBoundary, int | None]],
    covered_per_span: Counter[int],
    overlap_per_span: Counter[int],
    spans_with_strokes: int,
) -> dict[str, int]:
    n_spans = len(overlap_per_span)
    return {
        "ground_truth_rallies": len(boundaries),
        "predicted_spans": n_spans,
        "covered_rallies": sum(boundary is RallyBoundary.COVERED for boundary, _ in boundaries),
        "split_rallies": sum(boundary is RallyBoundary.SPLIT for boundary, _ in boundaries),
        "missed_rallies": sum(boundary is RallyBoundary.MISSED for boundary, _ in boundaries),
        "one_to_one_rallies": sum(
            covered_per_span[span_id]
            for span_id, count in overlap_per_span.items()
            if count == 1
        ),
        "covered_rallies_in_non_one_to_one_spans": sum(
            covered_per_span[span_id]
            for span_id, count in overlap_per_span.items()
            if count > 1
        ),
        "spans_with_ground_truth_strokes": spans_with_strokes,
        "spans_without_ground_truth_strokes": n_spans - spans_with_strokes,
    }


def _span_overlap_counts(
    spans: Sequence[tuple[int, int]],
    rally_extents: Sequence[tuple[int, int]],
) -> Counter[int]:
    """Count every GT rally extent that overlaps each predicted span."""
    return Counter(
        {
            span_id: sum(
                span_start <= rally_end and rally_start < span_end
                for rally_start, rally_end in rally_extents
            )
            for span_id, (span_start, span_end) in enumerate(spans)
        }
    )


def build_scores(
    prediction_rows: Sequence[Mapping[str, str]], shots_master: pd.DataFrame
) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    """Build the primary one-to-one scoring population and boundary counts."""
    scored: list[dict[str, object]] = []
    boundary_metrics: dict[str, dict[str, int]] = {}
    for video_id, fps in VIDEO_FPS.items():
        video_name = f"sset_{video_id:02d}"
        predictions = [row for row in prediction_rows if row["video_id"] == video_name]
        spans = _prediction_spans(predictions)
        gt_rallies = load_gt_rallies(shots_master, video_id)
        boundaries = classify_all(spans, gt_rallies)
        covered_per_span = Counter(
            span_id
            for boundary, span_id in boundaries
            if boundary is RallyBoundary.COVERED and span_id is not None
        )
        overlap_per_span = _span_overlap_counts(
            spans, [rally.extent for rally in gt_rallies]
        )
        all_stroke_frames = {
            frame for rally in gt_rallies for frame in rally.stroke_frames
        }
        spans_with_strokes = sum(
            any(start <= frame < end for frame in all_stroke_frames)
            for start, end in spans
        )
        boundary_metrics[video_name] = _boundary_summary(
            boundaries,
            covered_per_span,
            overlap_per_span,
            spans_with_strokes,
        )
        prediction_by_span = {int(row["span_id"]): row for row in predictions}
        master_video = shots_master[shots_master["vid"] == video_id]
        for rally, (boundary, span_id) in zip(gt_rallies, boundaries, strict=True):
            if (
                boundary is not RallyBoundary.COVERED
                or span_id is None
                or overlap_per_span[span_id] != 1
            ):
                continue
            truth = master_video[
                (master_video["set_id"] == rally.set_id)
                & (master_video["rally"] == rally.rally)
                & (master_video["ball_round"] == 1)
            ]
            if len(truth) != 1:
                raise ValueError(
                    f"{video_name} {rally.set_id}/r{rally.rally}: expected one first stroke"
                )
            scored.append(
                _score_row(
                    prediction_by_span[span_id],
                    video_id,
                    rally.set_id,
                    rally.rally,
                    rally.stroke_frames,
                    _normalise_side(truth.iloc[0]["player_side"]),
                )
            )
        if fps <= 0:
            raise ValueError(f"invalid frozen fps for {video_name}")
    return scored, boundary_metrics


def build_metrics(
    rows: Sequence[Mapping[str, object]],
    boundary_metrics: Mapping[str, Mapping[str, int]],
    freeze_sha256: str,
) -> dict[str, object]:
    """Summarise the frozen comparison without selecting a new rule."""
    by_video: dict[str, object] = {}
    for video_name in sorted(boundary_metrics):
        video_rows = [row for row in rows if row["video_id"] == video_name]
        by_video[video_name] = {
            "boundaries": dict(boundary_metrics[video_name]),
            "pr82": _rule_summary(video_rows, "pr82"),
            "pr88": _rule_summary(video_rows, "pr88"),
        }
    paired = Counter(str(row["paired_server_outcome"]) for row in rows)
    return {
        "schema": "serve-rule-held-out-evaluation/1",
        "prediction_freeze_sha256": freeze_sha256,
        "tolerance_base30_frames": TOLERANCE_BASE30_FRAMES,
        "scope": "Frozen PR #82 and PR #88 rules on one-to-one held-out rallies.",
        "by_video": by_video,
        "overall": {
            "pr82": _rule_summary(rows, "pr82"),
            "pr88": _rule_summary(rows, "pr88"),
            "changed_server_predictions": sum(
                row["changed_vs_pr82"] == "True" for row in rows
            ),
            "fixes_vs_pr82": paired["fix"],
            "damages_vs_pr82": paired["damage"],
            "paired_server_outcomes": dict(sorted(paired.items())),
            "exact_mcnemar_two_sided_p": _exact_mcnemar_p(
                paired["fix"], paired["damage"]
            ),
            "pr88_categories": dict(
                sorted(Counter(str(row["pr88_category"]) for row in rows).items())
            ),
            "pr88_branches": dict(
                sorted(Counter(str(row["pr88_branch"]) for row in rows).items())
            ),
        },
    }


def _write_gzip_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty score table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    with path.open("wb") as raw_handle, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw_handle, mtime=0
    ) as zipped:
        zipped.write(stream.getvalue().encode("utf-8"))


def _write_gzip_json(path: Path, value: object) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("wb") as raw_handle, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw_handle, mtime=0
    ) as zipped:
        zipped.write(raw)


def run(freeze_dir: Path, output_dir: Path) -> dict[str, object]:
    """Verify predictions first, then open labels and write scored outputs."""
    predictions = _read_freeze(freeze_dir)
    freeze_sha256 = hashlib.sha256((freeze_dir / "predictions.csv.gz").read_bytes()).hexdigest()
    shots_master, _homography, _courts, _resolution = load_gt_tables()
    rows, boundaries = build_scores(predictions, shots_master)
    metrics = build_metrics(rows, boundaries, freeze_sha256)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(output_dir / "scored_rallies.csv.gz", rows)
    _write_gzip_json(output_dir / "metrics.json.gz", metrics)
    print(json.dumps(metrics["overall"], indent=2, sort_keys=True))
    return metrics


def main() -> None:
    """Parse paths and score the immutable prediction freeze."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.freeze_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
