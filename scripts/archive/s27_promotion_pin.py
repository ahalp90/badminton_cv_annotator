"""Acceptance pin for the promoted contact chain.

The fixture loader and scorer come from the read-only architecture harness. All
contact finding, gating, suppression, and downstream chain work comes from this
worktree. Ground truth enters only when the harness scores the completed chain.
"""
from __future__ import annotations

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
EXPECTED_MD5 = {
    'pilot': '2dde5c2ea444ac9b69cfecd9cbd03daa',
    'vid15': 'fade30a55ef774f279a459eb10dd73b8',
}
OUTPUT_DIR = HERE / 's27_promotion_pin_outputs'


def _bind_worktree_package(name: str, package_dir: Path) -> ModuleType:
    """Bind a package name to this worktree before loading the reference harness."""
    spec = importlib.util.spec_from_file_location(
        name, package_dir / '__init__.py', submodule_search_locations=[str(package_dir)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f'could not build a spec for {package_dir}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_imports() -> None:
    """Use worktree scraper code and reference-only fixture/scoring code."""
    _bind_worktree_package('scraper', HERE / 'src' / 'scraper')
    _bind_worktree_package('scripts', HERE / 'scripts')
    stage8 = importlib.import_module('scraper.stage8_rally_segmentation')
    stage8._court_scale_boxes = stage8.court_scale_boxes  # noqa: SLF001
    sys.path.insert(0, str(REFERENCE_DIR))


_prepare_imports()
import h_end_to_end as harness  # noqa: E402

point_winner = importlib.import_module('scraper.point_winner')
stage8 = importlib.import_module('scraper.stage8_rally_segmentation')

LANDING_OPTS = point_winner.LandingFilterOptions(
    settle_win=7, settle_thr=0.004, settle_min=5, carry_win=7, carry_thr=0.75,
)


def build_chain(cfg, track, bboxes, scores, kps, ndet, dead, homo_df, court_info):
    """Build the GT-free promoted chain with the harness's downstream semantics."""
    del ndet
    spans, contacts = stage8.segment_video(
        track,
        positions=None,
        span_open=stage8.SpanOpen.BACK_FILL,
        replay_mask=dead,
        pose_bboxes=bboxes,
        pose_scores=scores,
        pose_kps=kps,
        court_box=cfg.court_box,
        resolution=harness.RESOLUTION,
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
        cfg.vid, homo_df, court_info, harness.REF_ERR_PX,
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
            cfg.net_band, harness.RESOLUTION, court_info,
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
                    [(rally_id, stroke_index, contact_frame)],
                    track, cfg.net_band, harness.RESOLUTION,
                )
            except ValueError as exc:
                hit_height_failures.append((rally_id, stroke_index, contact_frame, str(exc)))
                continue
            hit_height_by_frame[contact_frame] = rows[0].hit_height

    return harness.DetectedChain(
        spans=spans,
        contacts=contacts,
        filtered_contacts=filtered_contacts,
        filtered_by_rally=filtered_by_rally,
        striker_halves=striker_halves,
        n_strokes_list=n_strokes_list,
        next_servers=next_servers,
        fitted_first_all=fitted_first_all,
        verdict_rows=verdict_rows,
        landings=landings,
        hit_height_by_frame=hit_height_by_frame,
        hit_height_failures=hit_height_failures,
    )


def main() -> None:
    master = harness.pd.read_csv(harness.retest.SHOTS_MASTER)
    homo_df = harness.pd.read_csv(harness.retest.HOMOGRAPHY_CSV).set_index('id')
    all_court_info = harness.retest.load_all_court_info(harness.retest.HOMOGRAPHY_CSV)

    md5s: dict[str, str] = {}
    for cfg in (harness.retest.PILOT, harness.retest.VID15):
        track = np.load(cfg.track_path)
        bboxes = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_bboxes.npy')
        scores = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_scores.npy')
        kps = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_kps.npy')
        ndet = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_ndet.npy')
        dead = np.load(cfg.mask_path)
        chain = build_chain(
            cfg, track, bboxes, scores, kps, ndet, dead, homo_df, all_court_info[cfg.vid],
        )
        scoring = harness.score_video(cfg, chain, master, homo_df, all_court_info)
        output_path = OUTPUT_DIR / cfg.name / 'rallies.csv'
        harness.write_rallies_csv(scoring.rows, output_path)
        digest = harness._md5(output_path)  # noqa: SLF001
        md5s[cfg.name] = digest
        print(f'{cfg.name} rallies.csv md5 {digest}')

    for name, expected in EXPECTED_MD5.items():
        if md5s[name] != expected:
            raise AssertionError(f'{name} md5 {md5s[name]} != expected {expected}')
    print('s27 promotion pin: OK')


if __name__ == '__main__':
    main()
