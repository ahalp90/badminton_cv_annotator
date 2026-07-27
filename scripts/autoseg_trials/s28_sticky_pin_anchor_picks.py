"""Library-only acceptance pins for the sticky-anchor contact gate.

This successor re-earns the four suppression-radius pins against the current chain, where
every player detection comes from the sticky tracker's picks. The historical
``s28_sticky_pin_r30.py`` file remains byte-frozen as the pre-anchor-picks record.

Import order is load-bearing. This script binds ``annotator`` and ``shared`` to its own
checkout BEFORE importing the scoring harness (measurements/h_end_to_end.py under the
reference dir). The harness then binds the standing wt_annotator worktree's ``scraper``
and ``scripts`` packages; the ``scraper`` modules are import shims that re-export whatever
``annotator`` already resolves to, so the measured chain stays this checkout's. The
``scripts`` scoring helpers still execute from the standing worktree. Run from the standing
worktree itself, the two trees are the same tree; run from any other checkout, the script
prints a loud warning so nobody trusts digests earned across silently diverged trees.
"""
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
    "/home/ariel/Documents/COSC594/badminton_stroke_classification/"
    "local_scratch/autograder_architecture"
)
OUTPUT_DIR = HERE / "s28_sticky_pin_anchor_picks_outputs"
EXPECTED_MD5 = {
    9: {"pilot": "380cb095c627711a1beae1bf26a77eab", "vid15": "65b87c9a3a4646a7bce059979ae5fa1c"},
    7: {"pilot": "ee7eb23f322c3f709ad5131555cfafc0", "vid15": "43983c930266111625b9483088482c41"},
}


def _bind_worktree_package(name: str, package_dir: Path) -> ModuleType:
    """Bind a package to the checkout containing this script."""
    spec = importlib.util.spec_from_file_location(
        name, package_dir / "__init__.py", submodule_search_locations=[str(package_dir)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build a spec for {package_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_imports() -> None:
    """Bind this checkout's packages before importing the read-only harness.

    The pre-binds are load-bearing: the harness re-binds the standing worktree's packages
    at import, and Python's module cache is what keeps the measured chain (``annotator``,
    and the ``shared.court`` geometry it imports) on THIS checkout. ``scripts`` stays with
    the harness's own binding; the mismatch warning below covers it.
    """
    worktree_root = HERE.parents[1]
    sys.path.insert(0, str(worktree_root / "src"))
    _bind_worktree_package("annotator", worktree_root / "src" / "annotator")
    _bind_worktree_package("shared", worktree_root / "src" / "shared")
    _bind_worktree_package("scripts", worktree_root / "scripts")
    sys.path.insert(0, str(REFERENCE_DIR / "measurements"))


_prepare_imports()
import h_end_to_end as harness  # noqa: E402

if HERE.parents[1] != harness.WORKTREE_ROOT:
    print(
        f"WARNING: this run's checkout {HERE.parents[1]} differs from the harness's "
        f"standing worktree {harness.WORKTREE_ROOT}. The harness's scoring helpers "
        "(scripts/, and the scraper shim files) execute from the standing worktree; "
        "digests from this run are trustworthy only if the two trees match in those files."
    )

point_winner = importlib.import_module("annotator.point_winner")
rally_segmentation = importlib.import_module("annotator.rally_segmentation")
replay_mask = importlib.import_module("annotator.replay_mask")
fps_constants = importlib.import_module("annotator.fps_constants")
LANDING_OPTS = point_winner.LandingFilterOptions(
    settle_win=7, settle_thr=0.004, settle_min=5, carry_win=7, carry_thr=0.75
)


def _gate_context() -> tuple[object, object, dict[str, dict], object]:
    """Load the calibration tables needed by the scene-gated sticky result."""
    homo_df = harness.pd.read_csv(harness.retest.HOMOGRAPHY_CSV).set_index("id")
    all_court_info = harness.retest.load_all_court_info(harness.retest.HOMOGRAPHY_CSV)
    res_df = harness.pd.read_csv(harness.retest.RESOLUTION_CSV).set_index("id")
    gate_info = {str(video_id): info for video_id, info in all_court_info.items()}
    gate_res = res_df.copy()
    gate_res.index = gate_res.index.astype(str)
    return homo_df, all_court_info, gate_info, gate_res


def _build_chain(cfg, track, bboxes, scores, kps, ndet, dead, homo_df, all_court_info,
                 gate_info, gate_res, radius: int):
    """Build the current sticky-anchor chain for one video and radius."""
    fixture_root = harness.HERE
    court_present_path = {
        "pilot": fixture_root / "pilot_results/homography_smoothing/raw_keep_hard_any_m0p10.npy",
        "vid15": fixture_root / "vid15_results/composition_mask/vid15_keep_hard_any_m0p10.npy",
    }[cfg.name]
    scene_rows_path = fixture_root / f"{cfg.name}_results/scene_rows_content27_refcorners.csv"
    court_present = np.load(court_present_path)
    with scene_rows_path.open(newline="", encoding="utf-8") as handle:
        import csv

        homography_rows = [row for row in csv.DictReader(handle) if row.get("video_id") == str(cfg.vid)]
    segments = rally_segmentation.tracker_segments(homography_rows, court_present, len(track))
    sticky = rally_segmentation.build_sticky_result(
        track, segments, bboxes, scores, kps, ndet, str(cfg.vid),
        {str(cfg.vid): all_court_info[cfg.vid]}, gate_res.loc[[str(cfg.vid)]],
        cfg.court_box, harness.RESOLUTION, half_window=harness.ATTRIBUTION_HALF_WINDOW,
    )
    spans, contacts = rally_segmentation.segment_video(
        track, positions=None, thresholds=None, span_open=rally_segmentation.SpanOpen.BACK_FILL,
        replay_mask=dead, sticky_distances=sticky.distances, suppression_radius=radius,
        resolution=harness.RESOLUTION,
    )
    filtered_contacts = [
        contact for contact in contacts
        if (
            contact.wrist_near is not False
            and contact.suppressed is not True
            and not dead[contact.contact_frame]
        )
    ]
    filtered_by_rally: dict[int, list[int]] = {}
    for contact in filtered_contacts:
        filtered_by_rally.setdefault(contact.rally_id, []).append(contact.contact_frame)

    striker_halves = []
    for rally_id in range(len(spans)):
        frames = filtered_by_rally.get(rally_id, [])
        guesses = [point_winner.attribute_half(frame, track, sticky, bboxes, cfg.net_band)
                   for frame in frames]
        striker_halves.append(point_winner.fit_alternation(guesses))
    n_strokes_list = [len(filtered_by_rally.get(rally_id, [])) for rally_id in range(len(spans))]
    next_servers = point_winner.next_server_half(striker_halves, n_strokes_list)
    fitted_first_all = [
        harness._first_stroke_half(half, count) if half is not None else None
        for half, count in zip(striker_halves, n_strokes_list)
    ]
    kin = point_winner.build_landing_kinematics(track, sticky, kps, harness.RESOLUTION)
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
        next_start = spans[rally_id + 1][0] if rally_id + 1 < len(spans) else len(track)
        landing = point_winner.pick_landing(
            frames[-1], next_start, track, dead, kin, LANDING_OPTS, striker,
            cfg.net_band, harness.RESOLUTION, all_court_info[cfg.vid],
            harness.FPS_CONSTANTS, harness.CHAIN_FPS,
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
                    [(rally_id, stroke_index, contact_frame)], track, cfg.net_band, harness.RESOLUTION,
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
    bboxes = np.load(cfg.pose_dir / f"{cfg.pose_prefix}_bboxes.npy")
    scores = np.load(cfg.pose_dir / f"{cfg.pose_prefix}_scores.npy")
    kps = np.load(cfg.pose_dir / f"{cfg.pose_prefix}_kps.npy")
    ndet = np.load(cfg.pose_dir / f"{cfg.pose_prefix}_ndet.npy")
    dead = np.load(cfg.mask_path)
    if dead.all():
        raise ValueError('mask is all True: no live frame to anchor a frozen position to')
    dead = replay_mask.believe_raw_mask(
        dead, fps_constants.scale_for_fps(harness.CHAIN_FPS).replay_mask_min_frames,
    )
    chain = _build_chain(
        cfg, track, bboxes, scores, kps, ndet, dead, homo_df, all_court_info,
        gate_info, gate_res, radius,
    )
    digest = None
    if score_output:
        scoring = harness.score_video(
            cfg, chain, harness.pd.read_csv(harness.retest.SHOTS_MASTER), homo_df, all_court_info,
        )
        output_path = output_root / f"r{radius}" / cfg.name / "rallies.csv"
        harness.write_rallies_csv(scoring.rows, output_path)
        digest = hashlib.md5(output_path.read_bytes()).hexdigest()
        print(f"library pin radius {radius} {cfg.name} rallies.csv md5 {digest}")
    return chain, digest


def run_radius(radius: int, output_root: Path = OUTPUT_DIR) -> dict[str, object]:
    results = {}
    for cfg in (harness.retest.PILOT, harness.retest.VID15):
        chain, digest = run_video(cfg, radius, output_root)
        expected = EXPECTED_MD5[radius][cfg.name]
        if digest != expected:
            raise AssertionError(f"{cfg.name} radius {radius}: {digest} != {expected}")
        results[cfg.name] = (chain, digest)
    print(f"library sticky pin radius {radius}: OK")
    return results


def main() -> None:
    # The audit requires radius 9 before radius 7.
    run_radius(9)
    run_radius(7)


if __name__ == "__main__":
    main()
