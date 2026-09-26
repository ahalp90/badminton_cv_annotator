"""Choose among the gated W5 courts, rewarding net posts that line fragments support.

Each gated court implies a net: two tape halves and two posts
(experiments/annotator/independent_court/net_geometry.py). A post counts as
supported when a line fragment covers one of its six lowest samples and no
covering fragment reaches more than overrun_px working pixels below its base.
The winner has the highest W5 evidence plus weight x post reward (0, 0.5 or 1);
the first row in W5 rank order wins exact ties. The evidence is W5's ranking score
(normally the paint score), optionally blended with W5's geometry score.

Research scripts import these back, so this module stays a leaf: it imports only
numpy and the experiments package, and never edits sys.path.
"""

from __future__ import annotations

import numpy as np

from . import net_geometry

SAMPLES_PER_PIECE = 24
PERPENDICULAR_TOLERANCE_WORKING_PX = 4.0
DIRECTION_TOLERANCE_DEG = 8.0
EXTENT_MARGIN_WORKING_PX = 2.0
LOWER_SAMPLE_COUNT = 6


def project_pieces(corners_native: list, context) -> dict:
    """Project the net from native corners, then convert its pieces to working pixels once."""
    try:
        projection = net_geometry.project_net(corners_native, context.native_size)
    except ValueError as error:
        return {"state": "projection_failed", "reason": str(error)}
    native_per_working = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    return {
        "state": "measured",
        "camera_error": projection.camera_error,
        "focal_widths": projection.focal_widths,
        "pieces_native_px": projection.segments_px,  # (piece, start/end, x/y)
        "pieces_working_px": projection.segments_px / native_per_working,
    }


def post_features(piece: np.ndarray, segments: np.ndarray, size: tuple[int, int]) -> dict:
    """Match the first six old samples to aligned DeepLSD segments in working pixels."""
    base, top = piece
    direction = (top - base) / np.linalg.norm(top - base)
    fractions = np.linspace(0, 1, SAMPLES_PER_PIECE)[:LOWER_SAMPLE_COUNT]
    samples = base + fractions[:, None] * (top - base)
    sample_x, sample_y = samples.T
    width, height = size
    in_frame = (sample_x >= 0) & (sample_x < width) & (sample_y >= 0) & (sample_y < height)
    covering = np.zeros((LOWER_SAMPLE_COUNT, len(segments)), dtype=bool)
    if in_frame[0] and len(segments):
        starts = segments[:, :2]
        vectors = segments[:, 2:] - starts
        lengths = np.linalg.norm(vectors, axis=1)
        unit = vectors / lengths[:, None]
        aligned = np.abs(unit @ direction) >= np.cos(np.radians(DIRECTION_TOLERANCE_DEG))
        offset = samples[:, None, :] - starts[None, :, :]
        along = np.sum(offset * unit[None, :, :], axis=2)
        perpendicular = np.abs(offset[:, :, 0] * unit[None, :, 1] - offset[:, :, 1] * unit[None, :, 0])
        covering = (in_frame[:, None] & aligned[None, :]
                    & (along >= -EXTENT_MARGIN_WORKING_PX)
                    & (along <= lengths[None, :] + EXTENT_MARGIN_WORKING_PX)
                    & (perpendicular <= PERPENDICULAR_TOLERANCE_WORKING_PX))
    ids = np.flatnonzero(covering.any(axis=0))
    lowest = None
    if len(ids):
        endpoints = segments[ids].reshape(-1, 2)
        lowest = float(np.min((endpoints - base) @ direction))
    return {
        "samples_working_px": samples.tolist(),
        "sample_in_frame": in_frame.tolist(),
        "sample_covered": covering.any(axis=1).tolist(),
        "visible_count": int(in_frame.sum()),
        "covered_count": int(covering.any(axis=1).sum()),
        "covering_ids": ids.tolist(),
        "lowest_endpoint_offset_working_px": lowest,
    }


def supported(feature: dict, overrun_px: float) -> bool:
    offset = feature["lowest_endpoint_offset_working_px"]
    return offset is not None and offset >= -overrun_px


def reward(features: dict, overrun_px: float) -> float:
    return (int(supported(features["post_left"], overrun_px))
            + int(supported(features["post_right"], overrun_px))) / 2


def net_reward(net_state: str, posts: dict, overrun_px: float) -> float:
    return reward(posts, overrun_px) if net_state == "measured" else 0.0


def evidence_score(row: dict, geometry_weight: float) -> float:
    """W5's ranking score, with geometry_weight of W5's geometry score blended in.

    The geometry score counts fragment support alone; the paint score also needs paint at
    each spot. Rows from the research trials carry no geometry score, so they need weight 0.
    """
    if not geometry_weight:
        return row["paint_score"]
    return (1 - geometry_weight) * row["paint_score"] + geometry_weight * row["geometry_score"]


def choose(rows: list[dict], weight: float, overrun_px: float,
           geometry_weight: float = 0.0) -> tuple[str | None, list[dict]]:
    """Keep the first camera-ranked row on exact combined-score ties."""
    scored = []
    best_key = None
    best_score = -float("inf")
    for row in rows:
        if not row["historical_fullcourt"]:
            continue
        reward_value = net_reward(row["net_state"], row["posts"], overrun_px)
        bonus = weight * reward_value
        assert 0 <= bonus <= weight
        score = evidence_score(row, geometry_weight) + bonus
        scored.append({"origin_key": row["origin_key"], "full_court_rank": row["full_court_rank"],
                       "paint_score": row["paint_score"], "net_reward": reward_value,
                       "bonus": bonus, "combined_score": score})
        if score > best_score:
            best_key, best_score = row["origin_key"], score
    return best_key, scored


def net_posts(corners_native: list, context) -> tuple[str, dict]:
    """Project a court's net and match its two posts to line fragments: (projection state, posts)."""
    projection = project_pieces(corners_native, context)
    posts = {}
    if projection["state"] == "measured":
        for name, piece_index in (("post_left", 2), ("post_right", 3)):
            posts[name] = post_features(
                np.asarray(projection["pieces_working_px"][piece_index]), context.segments, context.size,
            )
    return projection["state"], posts


def net_rows(record: dict, context) -> list[dict]:
    """Rows for choose, built as net_recovery/bounded_trial.measure_case builds them.

    :param record: W5 case record, as read back from its JSON file.
    :param context: The view's verifier.ViewContext (working-size fragments and size).
    """
    candidates = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
    ranking = record["rankings"]["C"]["provisional_rank"]
    criterion = record["rankings"]["C"]["r2_criterion"]
    rows = []
    full_rank = 0
    for rank, key in enumerate(ranking, start=1):
        candidate = candidates[key]
        if not candidate["historical"]["historical_fullcourt"]:
            continue
        full_rank += 1
        net_state, posts = net_posts(candidate["corners_px"], context)
        rows.append({
            "origin_key": key, "candidate_id": candidate["candidate_id"], "source": candidate["source"],
            "original_rank": rank, "full_court_rank": full_rank, "historical_fullcourt": True,
            "camera_eligible": candidate["camera_eligible"], "gate_camera_error": candidate["gates"]["camera_error"],
            "paint_score": candidate["evidence"][criterion],
            "geometry_score": candidate["evidence"]["q_geom_span_weighted"], "net_state": net_state, "posts": posts,
        })
    return rows
