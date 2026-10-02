"""The courts each direction pair builds before full scoring: horizon, player size and line-guess average.

Runs only the court search, on the laptop, with full scoring switched off (combined_ranking='axis').
The courts a pair builds do not depend on scoring, so each pair builds the same courts as in a full
run. For each pair this records how many courts would go on to full scoring and two sanity measures:

- The horizon's distance from the image centre, in image widths (the longer side). A camera tilted
  t degrees from straight down, with focal length f, puts the horizon f / tan(t) from the principal
  point, taken as the image centre. With any lens up to 90 degrees wide (f at least half a width), a
  camera within 10 degrees of straight down puts it at least 2.84 widths away, and within 20 degrees
  at least 1.37. So counts past those distances are upper bounds on near-straight-down courts
- The players' width in metres. For each person box in the view's frame whose foot (bottom centre)
  the court puts on the court or within 1 m of it, the floor distance between the box's bottom
  corners, each mapped onto the floor through the court. Per court, the median over those people;
  none gives no value. Every person box counts, sitting or standing

For every G0 or G1 court in the run's overall shortlists, and behind the net choice's chosen court and
top five, it records the court's rank in its pair by the average of its two line-guess scores, with
ties broken by build order. A check compares each such court's corners and average with the run's,
since the laptop could build a court slightly differently.

With --without-upright-filter the search runs as before the upright-camera filter: it keeps pairs whose
horizon tilts past 45 degrees and courts above their pair's horizon. Build positions then differ from the
run's, so only the per-pair counts are recorded.

Usage, from the worktree root, one view per process:
  PYTHONPATH=.:src python local_built_courts.py ARTEFACT_DIR VIEW OUT_DIR [--without-upright-filter]
  ARTEFACT_DIR: a joined-detector run's artefacts (the final 26 September Carmack run's blend_default)
"""

import os

# Before numpy loads: one thread per process, as run_views.py does.
for thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from scratch.court_det_fix.court_detector import search
from scratch.court_det_fix.court_detector.detect import (
    DIRECTION_BUDGET,
    MAX_HORIZON_TILT_DEG,
    ROOT,
    CourtDetector,
    Switches,
    freeze_arrays,
    source_record,
)
from scratch.court_det_fix.court_detector.inputs import ViewInputs
from scratch.court_det_fix.court_detector.run_views import (
    VIEWS,
    pack_sources,
    read_json_gz,
)

COURT_SIZE_M = np.array([6.1, 13.4])
COURT_CORNERS_M = np.array([[0.0, 0.0], [6.1, 0.0], [6.1, 13.4], [0.0, 13.4]])
NEAR_COURT_M = 1.0
# Horizon distance, in image widths, that a camera within this many degrees of straight down reaches
# at least, with a lens up to 90 degrees wide.
STEEP_HORIZON_WIDTHS = {limit: 0.5 / np.tan(np.radians(limit)) for limit in (10, 20)}
# Player-width histogram bins, metres: 20 per tenfold step, from 1 cm to 100 m; outside values clip to the ends.
WIDTH_BINS_M = np.geomspace(0.01, 100.0, 81)


def horizon_widths(homographies: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Distance from the image centre to each court's horizon, in image widths; inf when it has none."""
    # The floor's two directions vanish at the homography's first two columns; the horizon joins them.
    horizons = np.cross(homographies[:, :, 0], homographies[:, :, 1])
    width, height = size
    offsets = np.abs(horizons @ np.array([width / 2, height / 2, 1.0]))
    with np.errstate(divide="ignore"):
        return offsets / np.hypot(horizons[:, 0], horizons[:, 1]) / max(width, height)


def player_widths_m(homographies: np.ndarray, boxes_px: np.ndarray) -> np.ndarray:
    """Per court, the median width in metres of the people it puts on or near the court; NaN if none."""
    x1, _y1, x2, y2 = boxes_px.T
    inverse = np.linalg.inv(homographies)
    # A point above the horizon maps to the floor behind the camera, with the opposite sign to the court's centre.
    centre_sign = np.sign(homographies[:, 2] @ np.array([*COURT_SIZE_M / 2, 1.0]))

    def to_floor(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The image points (xs, y2) on each court's floor, (courts, people, 2), and whether each is in front."""
        floor = np.einsum("cij,pj->cpi", inverse, np.column_stack((xs, y2, np.ones(len(boxes_px)))))
        return floor[..., :2] / floor[..., 2:], np.sign(floor[..., 2]) == centre_sign[:, None]

    (left, left_in_front), (foot, foot_in_front), (right, right_in_front) = (
        to_floor(xs) for xs in (x1, (x1 + x2) / 2, x2))
    on_or_near_court = (left_in_front & foot_in_front & right_in_front & (foot >= -NEAR_COURT_M).all(-1)
                        & (foot <= COURT_SIZE_M + NEAR_COURT_M).all(-1))
    # The distance between the box's bottom corners on the floor.
    widths = np.where(on_or_near_court, np.hypot(right[..., 0] - left[..., 0], right[..., 1] - left[..., 1]), np.nan)
    medians = np.full(len(homographies), np.nan)
    counted = on_or_near_court.any(axis=1)
    medians[counted] = np.nanmedian(widths[counted], axis=1)
    return medians


def referenced_courts(artefact: dict) -> dict[tuple[str, int, int], dict[str, set[str]]]:
    """(search, pair, build position) of each court that matters: role -> the ids of the courts it stands for.

    A chosen or top-five court can be built by several pairs, and needs only one of those copies to
    survive, so its copies share its id (the net choice's parent key). Each shortlist entry is its own id.
    """
    roles: dict[tuple[str, int, int], dict[str, set[str]]] = {}

    def add(source: str, candidate_id: str, role: str, court_id: str) -> None:
        pair_id, position = map(int, candidate_id.split(":"))
        roles.setdefault((source, pair_id, position), {}).setdefault(role, set()).add(court_id)

    for name in ("G0", "G1"):
        for entry in artefact["populations"][name]:
            add(name, entry["candidate_id"], "shortlist", f"{name} {entry['candidate_id']}")
    chosen = artefact["net_choice"]["chosen"]
    if chosen is not None:
        record = artefact["w5"]["record"]
        parents = {parent["origin_key"]: parent for parent in record["parents"]}
        parent_of = {child["origin_key"]: child["parent_origin_key"] for child in record["valid_children"]}
        rows = sorted(artefact["net_choice"]["rows"], key=lambda row: -row["combined_score"])
        for role, keys in (("chosen", [chosen]), ("top five", [row["origin_key"] for row in rows[:5]])):
            for key in keys:
                parent_key = parent_of.get(key, key)
                for occurrence in parents[parent_key]["source_occurrences"]:
                    if occurrence["source"] in ("G0", "G1"):
                        add(occurrence["source"], occurrence["candidate_id"], role, parent_key)
    return roles


def main() -> None:
    artefact_dir, view_id, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    upright_filter = sys.argv[4:] != ["--without-upright-filter"]
    if sys.argv[4:] not in ([], ["--without-upright-filter"]):
        raise SystemExit(__doc__)
    started = perf_counter()
    artefact = read_json_gz(artefact_dir / f"{view_id}.json.gz")
    detector = CourtDetector(Switches(self_checks=False))
    live = detector.live
    sources, provenances, frame_paths = pack_sources(live.verifier.CASE_PACKS)
    anchors = {view["case_id"]: view["anchor"] for view in json.loads(VIEWS.read_text())}
    source = sources[view_id]
    frame_path = frame_paths.get(view_id) or live.verifier.frame_path(ROOT, source, provenances[view_id])
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(frame_path)
    view = ViewInputs(
        view_id=view_id, frame=frame, frame_index=anchors[view_id],
        scene_frames=(anchors[view_id], anchors[view_id]),
        segments_px=np.asarray(source["segments_px"], dtype=float).reshape(-1, 4),
        person_boxes_px=np.asarray(source["bbox_px"], dtype=float).reshape(-1, 4),
        provenance=provenances[view_id],
    )
    native_frame = view.frame.view()
    native_frame.flags.writeable = False
    detector_source = source_record(view, artefact["feet"]["all_feet_px"])
    context = live.verifier.view_context(view_id, detector_source, view.provenance, native_frame, view_id)
    freeze_arrays(context)

    # Each call builds one matched pair's courts; calls come in pair order.
    calls: list[dict] = []
    build_courts = live.run_automatic.propose_role

    def build_without_full_scoring(*args, **kwargs):
        proposed = build_courts(*args, **{**kwargs, "combined_ranking": "axis"})
        corners = np.asarray([candidate.corners_px for candidate in proposed.candidates], dtype=float)
        if len(corners):
            homographies = np.asarray(proposed.homographies, dtype=float)
            # The measures assume floor metres to working pixels; the court's corners prove it. Checked
            # on the floor, because a corner near the horizon moves many pixels for a tiny rounding change.
            floor = np.einsum("cij,ckj->cki", np.linalg.inv(homographies),
                              np.concatenate((corners, np.ones((*corners.shape[:2], 1))), axis=2))
            if np.abs(floor[..., :2] / floor[..., 2:] - COURT_CORNERS_M).max() > 0.01:
                raise RuntimeError(f"{view_id}: homographies do not map the court's corners to its corners")
            horizons, widths = horizon_widths(homographies, context.size), player_widths_m(homographies, boxes_px)
        else:
            horizons, widths = np.empty(0), np.empty(0)
        calls.append({
            "usable": len(corners),
            "steep_horizon": {limit: int((horizons >= distance).sum())
                              for limit, distance in STEEP_HORIZON_WIDTHS.items()},
            "no_horizon": int(np.isinf(horizons).sum()),
            "no_player": int(np.isnan(widths).sum()),
            "width_counts": np.histogram(np.clip(widths[~np.isnan(widths)], WIDTH_BINS_M[0], WIDTH_BINS_M[-1]),
                                         WIDTH_BINS_M)[0].tolist(),
            "horizons": horizons, "widths": widths,
            "axis_scores": np.asarray(proposed.axis_scores, dtype=float),
            "corners": corners,
        })
        return proposed

    dimensions = detector_source["dimensions"]
    scale = np.asarray([dimensions["width"], dimensions["height"]], dtype=float) / np.asarray(context.size,
                                                                                            dtype=float)
    # The view's frame's person boxes, in working pixels like the homographies.
    boxes_px = np.asarray(detector_source["bbox_px"], dtype=float).reshape(-1, 4) / np.tile(scale, 2)
    live.run_automatic.propose_role = build_without_full_scoring
    roles = referenced_courts(artefact) if upright_filter else {}
    direction = live.generation.direction_record(context, search.DIRECTION_SETTINGS, live.vp_pruning)
    filtered = search.filtered_source(detector_source, search.paint_mask(detector_source, native_frame, scale))
    pairs, courts = [], []
    for name, population_source in (("G0", detector_source), ("G1", filtered)):
        calls.clear()
        record = live.automatic_generation.generate(
            population_source, direction, live.runtime["zone"], ROOT, live.run_automatic, DIRECTION_BUDGET,
            legacy_evidence=False, max_horizon_tilt_deg=MAX_HORIZON_TILT_DEG if upright_filter else None,
        )
        matched = [pair for pair in record["pairs"] if pair["status"] == "matched"]
        if len(matched) != len(calls):
            raise RuntimeError(f"{view_id} {name}: {len(matched)} matched pairs but {len(calls)} builds")
        for pair, call in zip(matched, calls, strict=True):
            if pair["raw_parent_count"] != call["usable"]:
                raise RuntimeError(f"{view_id} {name} pair {pair['pair_id']}: court counts differ")
            pairs.append({"search": name, "pair_id": pair["pair_id"], "usable": call["usable"],
                          **{f"horizon_within_{limit}_deg_bound": count for limit, count in call["steep_horizon"].items()},
                          "no_horizon": call["no_horizon"], "no_player": call["no_player"],
                          "width_counts": call["width_counts"]})
            # Rank 1 is the best average; equal averages keep build order.
            order = np.lexsort((np.arange(call["usable"]), -call["axis_scores"]))
            average_rank = np.empty(call["usable"], dtype=int)
            average_rank[order] = np.arange(1, call["usable"] + 1)
            for (search_name, pair_id, position), court_roles in roles.items():
                if search_name != name or pair_id != pair["pair_id"]:
                    continue
                courts.append({"search": name, "pair_id": pair_id, "build_position": position + 1,
                               "roles": {role: sorted(ids) for role, ids in court_roles.items()},
                               "average_rank": int(average_rank[position]),
                               "average": float(call["axis_scores"][position]),
                               "horizon_widths": float(call["horizons"][position]),
                               "player_width_m": float(call["widths"][position]),
                               "corners_native_px": (call["corners"][position] * scale).tolist()})
    # Each shortlisted court's corners and average, against the run's, as the check on the rebuild.
    # Far corners can sit 10^5 px off-frame, so the corner gap is relative to the corner's distance.
    run_entries = {(name, entry["candidate_id"]): entry for name in ("G0", "G1")
                   for entry in artefact["populations"][name]}
    for court in courts:
        entry = run_entries.get((court["search"], f"{court['pair_id']}:{court['build_position'] - 1}"))
        if entry is not None:
            run_corners = np.asarray(entry["corners_px"])
            corner_gaps = np.linalg.norm(np.asarray(court["corners_native_px"]) - run_corners, axis=1)
            court["relative_corner_gap"] = float((corner_gaps / np.maximum(1.0, np.linalg.norm(run_corners,
                                                                                               axis=1))).max())
            court["average_gap"] = abs(court["average"] - entry["axis_score"])
    found = {(court["search"], court["pair_id"], court["build_position"] - 1) for court in courts}
    missing = [f"{key[0]} {key[1]}:{key[2]}" for key in roles if key not in found]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{view_id}.json").write_text(json.dumps({
        "view_id": view_id, "seconds": perf_counter() - started, "pairs": pairs, "courts": courts,
        "missing": missing}))
    print(f"{view_id}: {len(pairs)} pairs, {sum(pair['usable'] for pair in pairs)} courts built, "
          f"{len(courts)} referenced, {len(missing)} missing, {perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
