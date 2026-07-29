"""Measure the sticky-anchor candidate set as a second gate arm."""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BST_ROOT = HERE / 'src' / 'bst_x'
if str(BST_ROOT) not in sys.path:
    sys.path.insert(0, str(BST_ROOT))

import m_miss_junk_census as census  # noqa: E402

from annotator.types import ContactCandidate  # noqa: E402  — census's import set up the src path

pin = census.pin
harness = census.harness
stage8 = census.stage8
point_winner = census.point_winner
from preparing_data.heuristics import sticky_anchor  # noqa: E402
from preparing_data.heuristics.base import ClipContext, RawClip  # noqa: E402
from preparing_data.prepare_train_on_shuttleset import normalize_joints  # noqa: E402

OUT_DIR = HERE / 's27_sticky_gate_arm_outputs'
GATE_THRESHOLD = stage8.BODY_UNIT_WRIST_THRESHOLD


@dataclass(frozen=True)
class FramePick:
    picks: tuple[int, int]
    raw_slots: np.ndarray
    kps_f: np.ndarray
    bboxes_f: np.ndarray
    n_counted: int


@dataclass
class StickyRun:
    records: list[FramePick | None]
    kept_frames: set[int]
    gaps: dict[int, float]
    contacts: list[ContactCandidate]
    scoring: object


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _fmt(value: object) -> str:
    if value is None:
        return ''
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return '' if not np.isfinite(value) else f'{value:.6f}'
    return str(value)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_fmt(row[column]) for column in columns])
    print(f'CSV md5 {path.relative_to(HERE)} {_md5(path)}')


def _filtered_raw_slots(
    frame: int, bboxes: np.ndarray, scores: np.ndarray, ndet: np.ndarray,
    ctx: ClipContext, params: sticky_anchor.StickyAnchorParams,
) -> np.ndarray:
    n_detections = int(ndet[frame])
    score_mask = scores[frame, :n_detections] > params.score_filter
    score_slots = np.flatnonzero(score_mask)
    boxes = bboxes[frame, score_slots].astype(np.float64)
    projected = sticky_anchor._project_bbox_bottom_centre(boxes, ctx)  # noqa: SLF001
    valid = ~np.isnan(projected).any(axis=1)
    return score_slots[valid]


def _run_detected_spans(
    run: census.VideoRun, res_df, all_court_info: dict,
) -> tuple[list[FramePick | None], dict[int, float]]:
    """Copy the `_run_clip` sequential loop for detected spans.

    The loop body follows `src/bst_x/preparing_data/heuristics/sticky_anchor.py:282-316`.
    It is driven by the private picker at `sticky_anchor.py:132-256`.  The clip unit is one
    detected rally span here, so EMA state is reset at every span and frames outside spans are
    skipped.
    """
    params = sticky_anchor.StickyAnchorParams()
    raw = RawClip(
        kps=run.kps,
        bboxes=run.bboxes,
        scores=run.scores,
        kp_scores=np.zeros((*run.scores.shape, 17), dtype=np.float32),
        ndet=run.ndet,
    )
    ctx = ClipContext(run.cfg.vid, all_court_info, res_df)
    court_info = ctx.all_court_info[ctx.vid]
    halfcourt_centre = sticky_anchor._compute_halfcourt_centres(court_info)  # noqa: SLF001
    records: list[FramePick | None] = [None] * len(run.track)
    gaps: dict[int, float] = {}
    pos = np.zeros((len(run.track), 2, 2), dtype=np.float64)
    joints = np.zeros((len(run.track), 2, 17, 2), dtype=np.float64)
    failed = np.zeros(len(run.track), dtype=bool)
    overcount = np.zeros(len(run.track), dtype=bool)
    ema_history = np.zeros((len(run.track), 2, 2), dtype=np.float64)

    for start, end in run.spans:
        ema = halfcourt_centre.copy()
        for frame in range(start, end):
            result = sticky_anchor._pick_one_frame(  # noqa: SLF001
                raw, frame, ema, halfcourt_centre, ctx, params,
            )
            if not result:
                failed[frame] = True
                ema[:] = halfcourt_centre
                ema_history[frame] = ema
                continue

            # `_pick_one_frame` processes Bottom then Top, but `picks` is indexed in its
            # public slot order Top=0, Bottom=1.  The retained tuple preserves that output order.
            picks, court_base_pos, kps_f, bboxes_f, n_counted = result
            overcount[frame] = n_counted > 2
            frame_has_zero = False
            for slot in (sticky_anchor.SLOT_TOP, sticky_anchor.SLOT_BOTTOM):
                if picks[slot] < 0:
                    frame_has_zero = True
                    ema[slot] = halfcourt_centre[slot]
                    continue
                candidate_position = court_base_pos[picks[slot]]
                pos[frame, slot] = candidate_position
                joints[frame, slot] = normalize_joints(
                    arr=kps_f[picks[slot]][None, :, :],
                    bbox=bboxes_f[picks[slot]][None, :],
                    v_height=None,
                    center_align=True,
                )[0]
                if sticky_anchor._in_generous_court(  # noqa: SLF001
                    candidate_position, params.update_gate_eps,
                ):
                    ema[slot] = (
                        params.ema_alpha * candidate_position
                        + (1 - params.ema_alpha) * ema[slot]
                    )
            failed[frame] = frame_has_zero
            ema_history[frame] = ema
            records[frame] = FramePick(
                picks=(int(picks[0]), int(picks[1])),
                raw_slots=_filtered_raw_slots(frame, run.bboxes, run.scores, run.ndet, ctx, params),
                kps_f=kps_f,
                bboxes_f=bboxes_f,
                n_counted=n_counted,
            )

    return records, gaps


def _sticky_gap(
    frame: int, record: FramePick, run: census.VideoRun,
) -> float:
    picked_filtered = [pick for pick in record.picks if pick >= 0]
    if not picked_filtered:
        return float('nan')
    picked_slots = [int(record.raw_slots[pick]) for pick in picked_filtered]
    picked_boxes = run.bboxes[frame, picked_slots].astype(np.float64)
    x1, y1, x2, y2 = picked_boxes.T
    try:
        gaps = point_winner._body_unit_gaps(  # noqa: SLF001
            frame, x1, y1, x2, y2, picked_slots, run.bboxes, run.scores, run.kps,
            run.cfg.court_box, run.track, harness.RESOLUTION[0], harness.RESOLUTION[1],
        )
    except ValueError as exc:
        if not str(exc).startswith('body-unit gap:'):
            raise
        return float('nan')
    finite_gaps = gaps[np.isfinite(gaps)]
    return float(np.min(finite_gaps)) if len(finite_gaps) else float('nan')


def _build_sticky_chain(run: census.VideoRun, sticky_contacts):
    original_segment_video = stage8.segment_video

    def fixed_contacts(*_args, **_kwargs):
        return run.spans, sticky_contacts

    stage8.segment_video = fixed_contacts
    try:
        return pin.build_chain(
            run.cfg, run.track, run.bboxes, run.scores, run.kps, run.ndet, run.dead,
            _HOMO_DF, _ALL_COURT_INFO[run.cfg.vid],
        )
    finally:
        stage8.segment_video = original_segment_video


def _render_side_by_side(court_scores: dict[str, object], sticky_scores: dict[str, object]) -> None:
    for video in ('pilot', 'vid15'):
        left = harness.render_summary(court_scores[video])
        right = harness.render_summary(sticky_scores[video])
        if len(left) != len(right):
            raise AssertionError(f'{video}: harness summaries have different line counts')
        print(f'=== {video}: courtbox | sticky-anchor ===')
        for left_line, right_line in zip(left, right):
            print(f'{left_line:<100} | {right_line}')
    print('=== pooled convention: harness.render_pooled, one pooled list per arm ===')
    print('--- courtbox pooled ---')
    print('\n'.join(harness.render_pooled([court_scores['pilot'], court_scores['vid15']])))
    print('--- sticky-anchor pooled ---')
    print('\n'.join(harness.render_pooled([sticky_scores['pilot'], sticky_scores['vid15']])))


def main() -> None:
    global _HOMO_DF, _ALL_COURT_INFO
    runs = census._promotion_pass()
    master = harness.pd.read_csv(harness.retest.SHOTS_MASTER)
    _HOMO_DF = harness.pd.read_csv(harness.retest.HOMOGRAPHY_CSV).set_index('id')
    _ALL_COURT_INFO = harness.retest.load_all_court_info(harness.retest.HOMOGRAPHY_CSV)
    res_df = harness.pd.read_csv(harness.retest.RESOLUTION_CSV).set_index('id')

    court_scores = {video: runs[video].scoring for video in ('pilot', 'vid15')}
    sticky_scores: dict[str, object] = {}
    delta_rows: list[dict[str, object]] = []
    sticky_runs: dict[str, StickyRun] = {}

    for video in ('pilot', 'vid15'):
        run = runs[video]
        records, _unused_gaps = _run_detected_spans(run, res_df, _ALL_COURT_INFO)
        gaps = {
            flag.frame: _sticky_gap(flag.frame, records[flag.frame], run)
            for flag in run.flags
            if records[flag.frame] is not None
        }
        gate_pass_frames = {
            frame for frame, gap in gaps.items()
            if np.isfinite(gap) and gap <= GATE_THRESHOLD
        }
        kept_frames = set(stage8.suppress_contact_flags([
            (flag.frame, flag.impulse)
            for flag in run.flags if flag.frame in gate_pass_frames
        ]))
        sticky_contacts = [
            ContactCandidate(
                flag.rally_id, flag.frame, None, flag.frame in gate_pass_frames,
                flag.frame in gate_pass_frames and flag.frame not in kept_frames,
            )
            for flag in run.flags
        ]
        chain = _build_sticky_chain(run, sticky_contacts)
        scoring = harness.score_video(
            run.cfg, chain, master, _HOMO_DF, _ALL_COURT_INFO,
        )
        sticky_scores[video] = scoring
        sticky_runs[video] = StickyRun(records, kept_frames, gaps, sticky_contacts, scoring)

        court_final = set(run.final_frames)
        sticky_final = {
            contact.contact_frame for contact in sticky_contacts
            if contact.wrist_near is not False and contact.suppressed is not True
        }
        for arm, only_frames, own_final in (
            ('sticky_only', sorted(sticky_final - court_final), sticky_final),
            ('courtbox_only', sorted(court_final - sticky_final), court_final),
        ):
            gt_rallies = harness.retest.load_gt_rallies(master, run.cfg.vid)
            gt_frames = [frame for rally in gt_rallies for frame in rally.stroke_frames]
            own_matches = census._global_matches(gt_frames, sorted(own_final))
            matched_frames = {
                sorted(own_final)[candidate_index]
                for _gt_index, candidate_index in own_matches
            }
            for frame in only_frames:
                delta_rows.append({
                    'video': video,
                    'only_arm': arm,
                    'frame': frame,
                    'label_under_own_arm': 'matched-true' if frame in matched_frames else 'junk',
                })

        print(
            f'{video}: sticky gate-pass={len(gate_pass_frames)} sticky accepted={len(kept_frames)} '
            f'sticky-only={len(sticky_final - court_final)} '
            f'courtbox-only={len(court_final - sticky_final)}'
        )

    delta_counts = Counter((row['only_arm'], row['label_under_own_arm']) for row in delta_rows)
    for key in sorted(delta_counts):
        print(f'verdict delta {key[0]} {key[1]}={delta_counts[key]}')

    for arm, scores in (('courtbox', court_scores), ('sticky_anchor', sticky_scores)):
        for video in ('pilot', 'vid15'):
            output_path = OUT_DIR / arm / video / 'rallies.csv'
            harness.write_rallies_csv(scores[video].rows, output_path)
            print(f'CSV md5 {output_path.relative_to(HERE)} {_md5(output_path)}')
    _write_csv(
        OUT_DIR / 'verdict_delta_census.csv',
        ['video', 'only_arm', 'frame', 'label_under_own_arm'],
        delta_rows,
    )
    _render_side_by_side(court_scores, sticky_scores)


if __name__ == '__main__':
    main()
