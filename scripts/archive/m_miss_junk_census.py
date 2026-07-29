"""Census the promoted arm's missed strokes and unmatched surviving contacts.

The promotion pin is the first operation in ``main``.  The detector, gate,
suppression, scorer, and fixture loader stay bound to the two-tree setup used
by ``s27_promotion_pin.py``.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REFERENCE_DIR = Path(
    '/home/ariel/Documents/COSC594/badminton_stroke_classification/'
    'local_scratch/autograder_architecture'
)
OUT_DIR = HERE / 's27_census_outputs'
TOLERANCE = 10
FPS = 25
EXPECTED_MD5 = {
    'pilot': '2dde5c2ea444ac9b69cfecd9cbd03daa',
    'vid15': 'fade30a55ef774f279a459eb10dd73b8',
}

sys.path.insert(0, str(REFERENCE_DIR))

_PIN_SPEC = importlib.util.spec_from_file_location(
    'worktree_s27_promotion_pin', HERE / 's27_promotion_pin.py',
)
if _PIN_SPEC is None or _PIN_SPEC.loader is None:
    raise ImportError('could not load the worktree promotion pin')
pin = importlib.util.module_from_spec(_PIN_SPEC)
_PIN_SPEC.loader.exec_module(pin)

harness = pin.harness
stage8 = pin.stage8
point_winner = pin.point_winner
import h_gap_bridging  # noqa: E402
from j4_miss_kinematics import span_junctions  # noqa: E402


@dataclass(frozen=True)
class Flag:
    video: str
    rally_id: int
    frame: int
    impulse: float


@dataclass(frozen=True)
class GateMeasurement:
    gap: float
    subclass: str | None


@dataclass
class VideoRun:
    cfg: object
    track: np.ndarray
    bboxes: np.ndarray
    scores: np.ndarray
    kps: np.ndarray
    ndet: np.ndarray
    dead: np.ndarray
    spans: list[tuple[int, int]]
    flags: list[Flag]
    gated_frames: set[int]
    final_frames: list[int]
    gate_measurements: dict[int, GateMeasurement]
    impulse_ratio: dict[int, float]
    burst_ratio: dict[int, float]
    turn_angle: dict[int, float]
    visible_run: dict[int, int]
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


def _load_arrays(cfg: object) -> tuple[np.ndarray, ...]:
    track = np.load(cfg.track_path)
    bboxes = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_bboxes.npy')
    scores = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_scores.npy')
    kps = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_kps.npy')
    ndet = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_ndet.npy')
    dead = np.load(cfg.mask_path)
    if not (len(track) == len(bboxes) == len(scores) == len(kps) == len(ndet) == len(dead)):
        raise AssertionError(f'{cfg.name}: track/pose/mask lengths disagree')
    if kps.ndim != 4 or kps.shape[2:] != (17, 2):
        raise AssertionError(f'{cfg.name}: pose shape {kps.shape} is not (frames, detections, 17, 2)')
    if bboxes.shape[:2] != scores.shape or bboxes.shape[:2] != kps.shape[:2]:
        raise AssertionError(f'{cfg.name}: pose array shapes disagree')
    finite_scores = np.isfinite(scores[:])
    if not finite_scores.any():
        raise AssertionError(f'{cfg.name}: no finite pose scores')
    return track, bboxes, scores, kps, ndet, dead


def _promotion_pass() -> dict[str, VideoRun]:
    """Rebuild the promoted chain and abort before any census work on drift."""
    master = harness.pd.read_csv(harness.retest.SHOTS_MASTER)
    homo_df = harness.pd.read_csv(harness.retest.HOMOGRAPHY_CSV).set_index('id')
    all_court_info = harness.retest.load_all_court_info(harness.retest.HOMOGRAPHY_CSV)
    runs: dict[str, VideoRun] = {}
    md5s: dict[str, str] = {}

    for cfg in (harness.retest.PILOT, harness.retest.VID15):
        track, bboxes, scores, kps, ndet, dead = _load_arrays(cfg)
        chain = pin.build_chain(
            cfg, track, bboxes, scores, kps, ndet, dead, homo_df, all_court_info[cfg.vid],
        )
        scoring = harness.score_video(cfg, chain, master, homo_df, all_court_info)
        output_path = OUT_DIR / 'promotion_pin' / cfg.name / 'rallies.csv'
        harness.write_rallies_csv(scoring.rows, output_path)
        md5s[cfg.name] = _md5(output_path)
        if md5s[cfg.name] != EXPECTED_MD5[cfg.name]:
            raise AssertionError(
                f'{cfg.name} rallies.csv md5 {md5s[cfg.name]} != {EXPECTED_MD5[cfg.name]}'
            )

        masked_track = stage8.apply_replay_mask(track, dead)
        flags: list[Flag] = []
        impulse_ratio: dict[int, float] = {}
        burst_ratio: dict[int, float] = {}
        turn_angle: dict[int, float] = {}
        visible_run: dict[int, int] = {}
        for rally_id, (start, end) in enumerate(chain.spans):
            span_flags = stage8.detect_contact_flags(masked_track, start, end)
            span_junction = span_junctions(masked_track, start, end)
            span_impulse = stage8.span_impulses(masked_track, start, end)
            if span_junction is None or span_impulse is None:
                if span_flags:
                    raise AssertionError(f'{cfg.name}: flags in an unmeasurable span')
                continue
            angle_deg, speed_in, speed_out, _around_visible = span_junction
            floor = stage8.rolling_floor(
                span_impulse,
                (masked_track[start:end - 2, 2] == 1)
                & (masked_track[start + 1:end - 1, 2] == 1)
                & (masked_track[start + 2:end, 2] == 1),
            )
            for frame, impulse in span_flags:
                local_junction = frame - start - 1
                if not 0 <= local_junction < len(span_impulse):
                    raise AssertionError(f'{cfg.name}: flag {frame} has no junction')
                flags.append(Flag(cfg.name, rally_id, frame, float(impulse)))
                local_floor = float(floor[local_junction])
                impulse_ratio[frame] = (
                    float(impulse) / local_floor
                    if np.isfinite(local_floor) and local_floor != 0
                    else float('nan')
                )
                incoming = float(speed_in[local_junction])
                outgoing = float(speed_out[local_junction])
                burst_ratio[frame] = (
                    outgoing / incoming if np.isfinite(incoming) and incoming != 0
                    and np.isfinite(outgoing) else float('nan')
                )
                turn_angle[frame] = float(angle_deg[local_junction])
                run_length = 0
                next_frame = frame + 1
                while next_frame < end and not dead[next_frame] and track[next_frame, 2] == 1:
                    run_length += 1
                    next_frame += 1
                visible_run[frame] = run_length

        raw_frames = [flag.frame for flag in flags]
        final_frames = [contact.contact_frame for contact in chain.filtered_contacts]
        if raw_frames != [contact.contact_frame for contact in chain.contacts]:
            raise AssertionError(f'{cfg.name}: independent raw flag list drifted from chain')

        gate_measurements = {
            flag.frame: _court_gate_measurement(
                flag.frame, track, bboxes, scores, kps, cfg.court_box,
            )
            for flag in flags
        }
        gated_frames = {
            flag.frame for flag in flags
            if gate_measurements[flag.frame].subclass is None
            and np.isfinite(gate_measurements[flag.frame].gap)
            and gate_measurements[flag.frame].gap <= stage8.BODY_UNIT_WRIST_THRESHOLD
        }
        if not gated_frames.issuperset(final_frames):
            raise AssertionError(f'{cfg.name}: final contact is not a gate survivor')
        spans_count = len(chain.spans)
        print(
            f'{cfg.name}: spans={spans_count} raw_flags={len(flags)} '
            f'gated={len(gated_frames)} accepted={len(final_frames)} '
            f'rallies.csv md5={md5s[cfg.name]}'
        )
        runs[cfg.name] = VideoRun(
            cfg=cfg, track=track, bboxes=bboxes, scores=scores, kps=kps, ndet=ndet, dead=dead,
            spans=chain.spans, flags=flags, gated_frames=gated_frames, final_frames=final_frames,
            gate_measurements=gate_measurements, impulse_ratio=impulse_ratio,
            burst_ratio=burst_ratio, turn_angle=turn_angle, visible_run=visible_run,
            scoring=scoring,
        )

    if md5s != EXPECTED_MD5:
        raise AssertionError(f'promotion md5s {md5s} != {EXPECTED_MD5}')
    print('promotion tripwire: OK')
    return runs


def _court_gate_measurement(
    frame: int, track: np.ndarray, bboxes: np.ndarray, scores: np.ndarray,
    kps: np.ndarray, court_box: object,
) -> GateMeasurement:
    """Measure the promoted gate and classify its first applicable kill cause."""
    x1, y1, x2, y2, candidate_scores = stage8.court_scale_boxes(
        bboxes[frame], scores[frame], court_box,
    )
    if len(x1) == 0:
        return GateMeasurement(float('nan'), 'no_boxes')
    frame_scores = scores[frame]
    candidate_slots = [
        int(np.flatnonzero(frame_scores == score)[0]) for score in candidate_scores
    ]
    gaps = point_winner._body_unit_gaps(  # noqa: SLF001
        frame, x1, y1, x2, y2, candidate_slots, bboxes, scores, kps, court_box,
        track, harness.RESOLUTION[0], harness.RESOLUTION[1],
    )
    finite_gaps = gaps[np.isfinite(gaps)]
    if len(finite_gaps) == 0:
        return GateMeasurement(float('nan'), 'gate_nan')
    best_gap = float(np.min(finite_gaps))
    passes = np.isfinite(gaps) & (gaps <= stage8.BODY_UNIT_WRIST_THRESHOLD)
    if passes.any():
        return GateMeasurement(best_gap, None)

    valid = np.isfinite(scores[frame]) & np.isfinite(bboxes[frame]).all(axis=1)
    finite_slots = np.flatnonzero(valid)
    nearest_excluded = False
    if len(finite_slots) and len(candidate_slots):
        shuttle_px = track[frame, :2] * np.asarray(harness.RESOLUTION)
        all_boxes = bboxes[frame, finite_slots]
        all_centres = (all_boxes[:, :2] + all_boxes[:, 2:]) / 2.0
        nearest_slot = int(finite_slots[int(np.argmin(np.linalg.norm(all_centres - shuttle_px, axis=1)))])
        nearest_excluded = nearest_slot not in candidate_slots and len(candidate_slots) < len(finite_slots)
    if nearest_excluded:
        return GateMeasurement(best_gap, 'nearest_excluded')
    return GateMeasurement(best_gap, 'measured_far')


def _video_gt(runs: VideoRun, master: pd.DataFrame) -> list[object]:
    return harness.retest.load_gt_rallies(master, runs.cfg.vid)


def _global_matches(
    gt_frames: list[int], candidate_frames: list[int],
) -> list[tuple[int, int]]:
    return h_gap_bridging.greedy_match_own(gt_frames, candidate_frames, TOLERANCE)


def _allocate_miss_classes(
    run: VideoRun, gt_frames: list[int], matches: list[tuple[int, int]],
) -> list[dict[str, object]]:
    candidate_frames = run.final_frames
    matched_gt = {gt_index for gt_index, _candidate_index in matches}
    candidate_to_gt = {candidate_index: gt_index for gt_index, candidate_index in matches}
    missed_indices = [index for index in range(len(gt_frames)) if index not in matched_gt]
    available = set(range(len(run.flags)))
    final_flag_indices = {
        index for index, flag in enumerate(run.flags) if flag.frame in set(candidate_frames)
    }

    allocations: dict[int, tuple[str, int | None, str | None]] = {}

    def claim(kind: str, eligible: callable) -> None:
        pairs: list[tuple[int, int, int]] = []
        for gt_index in missed_indices:
            if gt_index in allocations:
                continue
            for flag_index in sorted(available):
                flag = run.flags[flag_index]
                distance = abs(gt_frames[gt_index] - flag.frame)
                if distance <= TOLERANCE and eligible(gt_index, flag_index):
                    pairs.append((distance, gt_index, flag_index))
        pairs.sort()
        claimed_strokes: set[int] = set()
        for _distance, gt_index, flag_index in pairs:
            if gt_index in allocations or gt_index in claimed_strokes or flag_index not in available:
                continue
            allocations[gt_index] = (kind, flag_index, None)
            claimed_strokes.add(gt_index)
            available.remove(flag_index)

    claim(
        'contention',
        lambda _gt_index, flag_index: flag_index in final_flag_indices
        and flag_index in candidate_to_gt
        and candidate_to_gt[flag_index] != _gt_index,
    )
    claim(
        'suppressed',
        lambda _gt_index, flag_index: flag_index not in final_flag_indices
        and run.flags[flag_index].frame in run.gated_frames,
    )
    claim(
        'gate_killed',
        lambda _gt_index, flag_index: flag_index not in final_flag_indices
        and run.flags[flag_index].frame not in run.gated_frames,
    )

    rows: list[dict[str, object]] = []
    for gt_index in missed_indices:
        if gt_index in allocations:
            klass, flag_index, _ = allocations[gt_index]
            flag = run.flags[flag_index]
            subclass = run.gate_measurements[flag.frame].subclass if klass == 'gate_killed' else None
            evidence_frame = flag.frame
        else:
            klass = 'no_candidate'
            subclass = None
            evidence_frame = None
        rows.append({
            'video': run.cfg.name,
            'gt_index': gt_index,
            'gt_frame': gt_frames[gt_index],
            'miss_class': klass,
            'gate_killed_subclass': subclass,
            'evidence_flag_frame': evidence_frame,
        })

    if len(rows) != len(missed_indices) or len({row['gt_index'] for row in rows}) != len(rows):
        raise AssertionError(f'{run.cfg.name}: miss allocation is not one row per missed stroke')
    if Counter(row['miss_class'] for row in rows).total() != len(rows):
        raise AssertionError(f'{run.cfg.name}: miss classes do not sum to miss total')
    gate_rows = [row for row in rows if row['miss_class'] == 'gate_killed']
    if Counter(row['gate_killed_subclass'] for row in gate_rows).total() != len(gate_rows):
        raise AssertionError(f'{run.cfg.name}: gate-killed subclasses do not sum to gate kills')
    return rows


def _shuttle_y_bin(track: np.ndarray, frame: int) -> str:
    if frame < 0 or frame >= len(track) or track[frame, 2] != 1:
        return 'unseen'
    y_px = float(track[frame, 1] * harness.RESOLUTION[1])
    if not np.isfinite(y_px):
        return 'unseen'
    if y_px < harness.RESOLUTION[1] / 3:
        return 'top'
    if y_px < 2 * harness.RESOLUTION[1] / 3:
        return 'middle'
    return 'bottom'


def _type_join(run: VideoRun, gt_rallies: list[object]) -> dict[tuple[str, int, int], str]:
    result: dict[tuple[str, int, int], str] = {}
    for set_id in ('set1', 'set2', 'set3'):
        set_df = pd.read_csv(run.cfg.set_dir / f'{set_id}.csv', usecols=['rally', 'ball_round', 'type'])
        for row in set_df.itertuples(index=False):
            # Rally numbers restart in each set.  Namespace the three-field `video` key by
            # set_id so the requested (video, rally, ball_round) uniqueness assertion is true.
            join_video = f'{run.cfg.name}:{set_id}'
            key = (join_video, int(row.rally), int(float(row.ball_round)))
            if key in result:
                raise AssertionError(f'{run.cfg.name}: duplicate set-CSV join key {key}')
            result[key] = str(row.type) if pd.notna(row.type) else 'unknown'
    for rally in gt_rallies:
        for ball_round in range(1, rally.n_strokes + 1):
            join_video = f'{run.cfg.name}:{rally.set_id}'
            result.setdefault((join_video, rally.rally, ball_round), 'unknown')
    return result


def _census_rows(
    run: VideoRun, master: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    gt_rallies = _video_gt(run, master)
    gt_frames = [frame for rally in gt_rallies for frame in rally.stroke_frames]
    matches = _global_matches(gt_frames, run.final_frames)
    matched_candidates = {candidate_index for _gt_index, candidate_index in matches}
    miss_rows = _allocate_miss_classes(run, gt_frames, matches)
    type_map = _type_join(run, gt_rallies)

    for row in miss_rows:
        gt_index = int(row['gt_index'])
        # The flattened GT index is converted deterministically below, without reading any
        # additional annotation field.  The loop is short, and the explicit offset is clearer.
        remaining = gt_index
        matched_rally = None
        for rally in gt_rallies:
            if remaining < rally.n_strokes:
                stroke_index = remaining
                matched_rally = rally
                break
            remaining -= rally.n_strokes
        else:
            raise AssertionError(f'{run.cfg.name}: GT flatten index {gt_index} is out of range')
        if matched_rally is None:
            raise AssertionError(f'{run.cfg.name}: GT flatten index {gt_index} has no rally')
        row['shuttle_y_bin'] = _shuttle_y_bin(run.track, gt_frames[gt_index])
        row['type'] = type_map.get(
            (f'{run.cfg.name}:{matched_rally.set_id}', matched_rally.rally, stroke_index + 1),
            'unknown',
        )
        row['gt_set_id'] = matched_rally.set_id

    miss_total = len(gt_frames) - len(matches)
    if len(miss_rows) != miss_total:
        raise AssertionError(f'{run.cfg.name}: miss rows {len(miss_rows)} != {miss_total}')
    subclass_counts = Counter(
        row['gate_killed_subclass'] for row in miss_rows if row['miss_class'] == 'gate_killed'
    )
    if sum(subclass_counts.values()) != sum(row['miss_class'] == 'gate_killed' for row in miss_rows):
        raise AssertionError(f'{run.cfg.name}: subclass total mismatch')

    junk_rows: list[dict[str, object]] = []
    final_flag_lookup = {flag.frame: flag for flag in run.flags}
    final_rally_frames: dict[int, list[int]] = {}
    for flag in run.flags:
        if flag.frame in run.final_frames:
            final_rally_frames.setdefault(flag.rally_id, []).append(flag.frame)
    for candidate_index, frame in enumerate(run.final_frames):
        flag = final_flag_lookup[frame]
        is_matched = candidate_index in matched_candidates
        disagreement: int | None = None
        fitted_half = None
        attributed_half = point_winner.attribute_half(
            frame, run.track, run.bboxes, run.scores, run.kps,
            run.cfg.court_box, run.cfg.net_band, harness.RESOLUTION,
        )
        rally_frames = final_rally_frames[flag.rally_id]
        guesses = [
            point_winner.attribute_half(
                rally_frame, run.track, run.bboxes, run.scores, run.kps,
                run.cfg.court_box, run.cfg.net_band, harness.RESOLUTION,
            )
            for rally_frame in rally_frames
        ]
        fitted_half = point_winner.fit_alternation(guesses)
        stroke_index = rally_frames.index(frame)
        if fitted_half is not None:
            phase = point_winner._phase_assignment(fitted_half, len(rally_frames))  # noqa: SLF001
            disagreement = int(attributed_half is not None and attributed_half != phase[stroke_index])
        if not is_matched:
            junk_rows.append({
                'video': run.cfg.name,
                'frame': frame,
                'rally_id': flag.rally_id,
                'matched_true': False,
                'impulse_floor_ratio': run.impulse_ratio[frame],
                'burst_ratio_speed_out_in': run.burst_ratio[frame],
                'body_unit_gap': run.gate_measurements[frame].gap,
                'best_turn_angle_deg': run.turn_angle[frame],
                'post_flag_visible_run_frames': run.visible_run[frame],
                'alternation_disagreement': disagreement,
            })
        else:
            junk_rows.append({
                'video': run.cfg.name,
                'frame': frame,
                'rally_id': flag.rally_id,
                'matched_true': True,
                'impulse_floor_ratio': run.impulse_ratio[frame],
                'burst_ratio_speed_out_in': run.burst_ratio[frame],
                'body_unit_gap': run.gate_measurements[frame].gap,
                'best_turn_angle_deg': run.turn_angle[frame],
                'post_flag_visible_run_frames': run.visible_run[frame],
                'alternation_disagreement': disagreement,
            })
    if len(junk_rows) != len(run.final_frames):
        raise AssertionError(f'{run.cfg.name}: final flag partition mismatch')
    return miss_rows, junk_rows


def _quartile(values: list[object]) -> tuple[int, list[float]]:
    finite = np.asarray([float(value) for value in values if value is not None and np.isfinite(value)], dtype=float)
    if not len(finite):
        return 0, []
    return len(finite), [float(value) for value in np.percentile(finite, (25, 50, 75))]


def _print_quartiles(rows_by_video: dict[str, list[dict[str, object]]]) -> None:
    metrics = (
        'impulse_floor_ratio', 'burst_ratio_speed_out_in', 'body_unit_gap',
        'best_turn_angle_deg', 'post_flag_visible_run_frames',
    )
    for video, rows in rows_by_video.items():
        print(f'quartiles {video}: junk vs matched-true (numpy linear, 4dp)')
        for metric in metrics:
            junk = [row[metric] for row in rows if not row['matched_true']]
            true = [row[metric] for row in rows if row['matched_true']]
            junk_n, junk_q = _quartile(junk)
            true_n, true_q = _quartile(true)
            junk_text = 'n=0' if not junk_q else f'n={junk_n} ' + ' '.join(f'{value:.4f}' for value in junk_q)
            true_text = 'n=0' if not true_q else f'n={true_n} ' + ' '.join(f'{value:.4f}' for value in true_q)
            excluded = sum(value is None or not np.isfinite(value) for value in junk)
            print(f'  {metric}: junk [{junk_text}]  matched-true [{true_text}]  '
                  f'junk_denominator_excluded={excluded}')


def main() -> None:
    runs = _promotion_pass()
    master = harness.pd.read_csv(harness.retest.SHOTS_MASTER)
    all_misses: list[dict[str, object]] = []
    all_junk: list[dict[str, object]] = []
    junk_by_video: dict[str, list[dict[str, object]]] = {}
    for video in ('pilot', 'vid15'):
        misses, junk = _census_rows(runs[video], master)
        all_misses.extend(misses)
        all_junk.extend(junk)
        junk_by_video[video] = junk
        print(f'{video}: misses={len(misses)} junk={sum(not row["matched_true"] for row in junk)}')
        print(f'  miss classes={dict(sorted(Counter(row["miss_class"] for row in misses).items()))}')
        gate_killed = [row for row in misses if row['miss_class'] == 'gate_killed']
        subclass_counts = dict(sorted(
            Counter(row['gate_killed_subclass'] for row in gate_killed).items()
        ))
        print(f'  gate-killed subclasses={subclass_counts}')

    _write_csv(
        OUT_DIR / 'miss_census.csv',
        ['video', 'gt_index', 'gt_frame', 'miss_class', 'gate_killed_subclass',
         'evidence_flag_frame', 'gt_set_id', 'shuttle_y_bin', 'type'],
        all_misses,
    )
    _write_csv(
        OUT_DIR / 'junk_census.csv',
        ['video', 'frame', 'rally_id', 'matched_true', 'impulse_floor_ratio',
         'burst_ratio_speed_out_in', 'body_unit_gap', 'best_turn_angle_deg',
         'post_flag_visible_run_frames', 'alternation_disagreement'],
        all_junk,
    )
    _print_quartiles(junk_by_video)
    print('dedup losses are inside FIND by design: detect_contact_flags includes the stage-8 three-frame dedup')


if __name__ == '__main__':
    main()
