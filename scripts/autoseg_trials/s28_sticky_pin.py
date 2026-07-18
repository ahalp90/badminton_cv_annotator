"""Library-only acceptance pins for the sticky-anchor contact gate."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE_DIR = Path(
    '/home/ariel/Documents/COSC594/badminton_stroke_classification/'
    'local_scratch/autograder_architecture'
)
OUTPUT_DIR = HERE / 's28_sticky_pin_outputs'
EXPECTED_MD5 = {
    9: {'pilot': 'ddc4f60b058f03e85c326dd5f460924d',
        'vid15': 'e0fa89414b44e1b4453e3a6f00f80ac6'},
    7: {'pilot': 'c259e147f36d5e848ccabc27ca41ba0b',
        'vid15': 'c332cfb257285daf4346f59a314e7bfe'},
}


def _bind_worktree_package(name: str, package_dir: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, package_dir / '__init__.py', submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f'could not build a spec for {package_dir}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_imports() -> None:
    _bind_worktree_package('scraper', HERE / 'src' / 'scraper')
    _bind_worktree_package('scripts', HERE / 'scripts')
    sys.path.insert(0, str(REFERENCE_DIR))


_prepare_imports()
import h_end_to_end as harness  # noqa: E402

point_winner = importlib.import_module('scraper.point_winner')
stage8 = importlib.import_module('scraper.stage8_rally_segmentation')
LANDING_OPTS = point_winner.LandingFilterOptions(
    settle_win=7, settle_thr=0.004, settle_min=5, carry_win=7, carry_thr=0.75,
)


def _gate_context() -> tuple[object, object, dict[str, dict], object]:
    homo_df = harness.pd.read_csv(harness.retest.HOMOGRAPHY_CSV).set_index('id')
    all_court_info = harness.retest.load_all_court_info(harness.retest.HOMOGRAPHY_CSV)
    res_df = harness.pd.read_csv(harness.retest.RESOLUTION_CSV).set_index('id')
    gate_info = {str(video_id): info for video_id, info in all_court_info.items()}
    gate_res = res_df.copy()
    gate_res.index = gate_res.index.astype(str)
    return homo_df, all_court_info, gate_info, gate_res


def _build_chain(cfg, track, bboxes, scores, kps, ndet, dead, homo_df, all_court_info,
                 gate_info, gate_res, radius: int):
    spans, contacts = stage8.segment_video(
        track, positions=None, thresholds=None, span_open=stage8.SpanOpen.BACK_FILL,
        replay_mask=dead, pose_bboxes=bboxes, pose_scores=scores, pose_kps=kps,
        pose_ndet=ndet, court_box=cfg.court_box, gate_video_id=str(cfg.vid),
        gate_court_info=gate_info, gate_resolution_table=gate_res,
        resolution=harness.RESOLUTION, suppression_radius=radius,
    )
    filtered_contacts = [
        contact for contact in contacts
        if contact.wrist_near is not False and contact.suppressed is not True
    ]
    filtered_by_rally: dict[int, list[int]] = {}
    for contact in filtered_contacts:
        filtered_by_rally.setdefault(contact.rally_id, []).append(contact.contact_frame)

    striker_halves = []
    for rally_id in range(len(spans)):
        frames = filtered_by_rally.get(rally_id, [])
        guesses = [
            point_winner.attribute_half(
                frame, track, bboxes, scores, kps, cfg.court_box, cfg.net_band,
                harness.RESOLUTION,
            )
            for frame in frames
        ]
        striker_halves.append(point_winner.fit_alternation(guesses))
    n_strokes_list = [len(filtered_by_rally.get(rally_id, [])) for rally_id in range(len(spans))]
    next_servers = point_winner.next_server_half(striker_halves, n_strokes_list)
    fitted_first_all = [
        harness._first_stroke_half(half, count) if half is not None else None
        for half, count in zip(striker_halves, n_strokes_list)
    ]

    kin = point_winner.build_landing_kinematics(
        track, bboxes, scores, kps, cfg.court_box, harness.RESOLUTION,
    )
    band_m = point_winner.corner_error_band_m(
        cfg.vid, homo_df, all_court_info[cfg.vid], harness.REF_ERR_PX,
    )
    verdict_rows: dict[int, object] = {}
    landings: dict[int, object | None] = {}
    for rally_id in range(len(spans)):
        striker = striker_halves[rally_id]
        if striker is None:
            continue
        frames = filtered_by_rally[rally_id]
        final_contact = frames[-1]
        next_start = spans[rally_id + 1][0] if rally_id + 1 < len(spans) else len(track)
        landing = point_winner.pick_landing(
            final_contact, next_start, track, dead, kin, LANDING_OPTS, striker,
            cfg.net_band, harness.RESOLUTION, all_court_info[cfg.vid],
        )
        verdict_rows[rally_id] = point_winner.rally_verdict(
            rally_id, striker, next_servers[rally_id], landing, band_m,
        )
        landings[rally_id] = landing

    hit_height_by_frame: dict[int, int] = {}
    hit_height_failures: list[tuple[int, int, int, str]] = []
    for rally_id in range(len(spans)):
        for stroke_index, contact_frame in enumerate(filtered_by_rally.get(rally_id, [])):
            try:
                rows = point_winner.build_hit_height_rows(
                    [(rally_id, stroke_index, contact_frame)], track,
                    cfg.net_band, harness.RESOLUTION,
                )
            except ValueError as exc:
                hit_height_failures.append((rally_id, stroke_index, contact_frame, str(exc)))
                continue
            hit_height_by_frame[contact_frame] = rows[0].hit_height

    return harness.DetectedChain(
        spans=spans, contacts=contacts, filtered_contacts=filtered_contacts,
        filtered_by_rally=filtered_by_rally, striker_halves=striker_halves,
        n_strokes_list=n_strokes_list, next_servers=next_servers,
        fitted_first_all=fitted_first_all, verdict_rows=verdict_rows, landings=landings,
        hit_height_by_frame=hit_height_by_frame, hit_height_failures=hit_height_failures,
    )


def run_video(cfg, radius: int, output_root: Path = OUTPUT_DIR, score_output: bool = True):
    homo_df, all_court_info, gate_info, gate_res = _gate_context()
    track = np.load(cfg.track_path)
    bboxes = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_bboxes.npy')
    scores = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_scores.npy')
    kps = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_kps.npy')
    ndet = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_ndet.npy')
    dead = np.load(cfg.mask_path)
    chain = _build_chain(
        cfg, track, bboxes, scores, kps, ndet, dead, homo_df, all_court_info,
        gate_info, gate_res, radius,
    )
    digest = None
    if score_output:
        scoring = harness.score_video(
            cfg, chain, harness.pd.read_csv(harness.retest.SHOTS_MASTER), homo_df, all_court_info,
        )
        output_path = output_root / f'r{radius}' / cfg.name / 'rallies.csv'
        harness.write_rallies_csv(scoring.rows, output_path)
        digest = hashlib.md5(output_path.read_bytes()).hexdigest()
        print(f'library pin radius {radius} {cfg.name} rallies.csv md5 {digest}')
    return chain, digest


def run_radius(radius: int, output_root: Path = OUTPUT_DIR) -> dict[str, object]:
    results = {}
    for cfg in (harness.retest.PILOT, harness.retest.VID15):
        chain, digest = run_video(cfg, radius, output_root)
        expected = EXPECTED_MD5[radius][cfg.name]
        if digest != expected:
            raise AssertionError(f'{cfg.name} radius {radius}: {digest} != {expected}')
        results[cfg.name] = (chain, digest)
    print(f'library sticky pin radius {radius}: OK')
    return results


def main() -> None:
    # The audit requires the shipped-radius pin before the alternate radius.
    run_radius(9)
    run_radius(7)


if __name__ == '__main__':
    main()
