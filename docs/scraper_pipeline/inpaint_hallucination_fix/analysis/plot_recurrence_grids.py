"""Plot locations and grouped 16-frame sequence families from audit views.

The input masks come from ``audit_tracks.py``. Each panel uses the 512x288
TrackNet image plane, so a reviewer can compare locations and paths with the
source video. The plots are evidence views, not a new detector.

The existing and event-union views are available:

``uncaught``
    Frames flagged by the exploratory RANSAC lens but assigned guard code 0.
    Sequence windows contain at least one such frame; every frame in the
    window remains valid and guard-clean.
``inpaint``
    Frames selected by the producer's stride-8 inpaint sidecar. Location
    counts use valid coordinates; sequence windows require all 16 frames to be
    sidecar-selected and valid.
``union``
    The union of the uncaught and sidecar-selected masks. Location counts use
    valid coordinates; sequence windows require all 16 frames to be selected
    by this union.
``uncaught_inpaint_impulse``
    The union of the existing uncaught-plus-sidecar-inpaint evidence view and
    raw contact impulse events from the resolved per-video default detector
    path. Location counts use valid coordinates; sequence windows contain at
    least one selected frame and do not require guard-clean frames.
``uncaught_impulse_tp_rally_end``
    The union of uncaught candidates, raw contact impulse events and inductive
    TP rally-ender span-close events. A TP rally-ender means shuttle events
    that have closed a valid rally, did not overlap with another valid GT
    rally, and are valid within our rally-ending ruleset. ShuttleSet's GT does
    not actually record the rally's final event, so we only ever know it
    inductively. The GT dataset only ever records the final contact, so this is
    an audit proxy rather than direct GT truth.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import fclusterdata
from sklearn.metrics import silhouette_score

from compressed_io import read_npy_xz, write_json_gz


FRAME_WIDTH = 512
FRAME_HEIGHT = 288
WINDOW = 16
FIXTURE_NAMES = ("sset_01", "sset_15", "sset_21")
SEQUENCE_CLUSTER_SAMPLE_LIMIT = 256
SILHOUETTE_TARGET = 0.5
SILHOUETTE_T_VALUES = (
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    12.0,
    16.0,
    24.0,
    32.0,
    48.0,
    64.0,
    96.0,
    128.0,
)
LOCATION_LABELS = {
    "uncaught": "uncaught RANSAC candidates",
    "inpaint": "sidecar-selected inpaint frames",
    "union": "union of uncaught candidates and sidecar-selected inpaint frames",
    "uncaught_inpaint_impulse": (
        "uncaught-plus-sidecar-inpaint frames ∪ raw contact impulse events"
    ),
    "uncaught_impulse_tp_rally_end": (
        "uncaught candidates ∪ raw impulse events ∪ inductive TP rally-enders"
    ),
}
SEQUENCE_LABELS = {
    "uncaught": (
        "16-frame windows containing at least one uncaught RANSAC candidate; "
        "all frames valid and guard-clean"
    ),
    "inpaint": "16-frame windows fully selected by the inpaint sidecar and valid",
    "union": (
        "16-frame windows fully selected by the union of uncaught candidates "
        "and sidecar-selected inpaint frames"
    ),
    "uncaught_inpaint_impulse": (
        "valid 16-frame windows containing at least one uncaught, sidecar or "
        "impulse frame; guard-clean status is not required"
    ),
    "uncaught_impulse_tp_rally_end": (
        "valid 16-frame windows containing at least one uncaught, impulse or "
        "inductive TP rally-ender frame; guard-clean status is not required"
    ),
}
EVENT_UNION_VIEWS = (
    "uncaught_inpaint_impulse",
    "uncaught_impulse_tp_rally_end",
)
ALL_VIEWS = ("uncaught", "inpaint", "union", *EVENT_UNION_VIEWS)
OUTPUT_NAMES = {
    "uncaught": (
        "top_locations.csv.gz",
        "top_sequences.json.gz",
        "top_uncaught_locations.png",
        "top_uncaught_sequences.png",
    ),
    "inpaint": (
        "top_inpaint_locations.csv.gz",
        "top_inpaint_sequences.json.gz",
        "top_inpaint_locations.png",
        "top_inpaint_sequences.png",
    ),
    "union": (
        "top_unfiltered_inpaint_locations.csv.gz",
        "top_unfiltered_inpaint_sequences.json.gz",
        "top_unfiltered_inpaint_locations.png",
        "top_unfiltered_inpaint_sequences.png",
    ),
    "uncaught_inpaint_impulse": (
        "top_uncaught_inpaint_impulse_locations.csv.gz",
        "top_uncaught_inpaint_impulse_sequences.json.gz",
        "top_uncaught_inpaint_impulse_locations.png",
        "top_uncaught_inpaint_impulse_sequences.png",
    ),
    "uncaught_impulse_tp_rally_end": (
        "top_uncaught_impulse_tp_rally_end_locations.csv.gz",
        "top_uncaught_impulse_tp_rally_end_sequences.json.gz",
        "top_uncaught_impulse_tp_rally_end_locations.png",
        "top_uncaught_impulse_tp_rally_end_sequences.png",
    ),
}


def pixel_points(track: np.ndarray) -> np.ndarray:
    return track[:, :2] * np.array([FRAME_WIDTH, FRAME_HEIGHT], dtype=np.float64)


def image_axes(axis: plt.Axes) -> None:
    axis.set_xlim(0, FRAME_WIDTH)
    axis.set_ylim(FRAME_HEIGHT, 0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x pixel")
    axis.set_ylabel("y pixel")
    axis.set_xticks(np.arange(0, FRAME_WIDTH + 1, 64))
    axis.set_yticks(np.arange(0, FRAME_HEIGHT + 1, 48))
    axis.grid(color="#d0d0d0", linewidth=0.4, alpha=0.6)


def rounded_locations(
    points: np.ndarray,
    frame_mask: np.ndarray,
    top_n: int,
) -> list[dict[str, int]]:
    integer_points = np.rint(points[frame_mask]).astype(np.int64)
    integer_points[:, 0] = np.clip(integer_points[:, 0], 0, FRAME_WIDTH - 1)
    integer_points[:, 1] = np.clip(integer_points[:, 1], 0, FRAME_HEIGHT - 1)
    counts = Counter(map(tuple, integer_points.tolist()))
    return [
        {
            "rank": rank,
            "x_px": x,
            "y_px": y,
            "instance_count": count,
        }
        for rank, ((x, y), count) in enumerate(counts.most_common(top_n), start=1)
    ]


def sequence_starts(
    track: np.ndarray,
    guard_codes: np.ndarray,
    frame_mask: np.ndarray,
    view: str,
) -> np.ndarray:
    valid = ~np.all(track[:, :2] == 0, axis=1)
    valid_windows = np.lib.stride_tricks.sliding_window_view(valid, WINDOW).all(axis=1)
    masked_windows = np.lib.stride_tricks.sliding_window_view(frame_mask, WINDOW)
    if view == "uncaught":
        clean_windows = np.lib.stride_tricks.sliding_window_view(
            guard_codes == 0, WINDOW
        ).all(axis=1)
        selected_windows = masked_windows.any(axis=1)
        return np.flatnonzero(valid_windows & clean_windows & selected_windows)
    if view in {"inpaint", "union"}:
        selected_windows = masked_windows.all(axis=1)
        return np.flatnonzero(valid_windows & selected_windows)
    if view in EVENT_UNION_VIEWS:
        selected_windows = masked_windows.any(axis=1)
        return np.flatnonzero(valid_windows & selected_windows)
    raise ValueError(f"unknown plot view: {view}")


def exact_sequences(
    track: np.ndarray,
    guard_codes: np.ndarray,
    frame_mask: np.ndarray,
    view: str,
) -> list[dict[str, object]]:
    starts = sequence_starts(track, guard_codes, frame_mask, view)
    sequence_starts_by_key: defaultdict[bytes, list[int]] = defaultdict(list)
    for start in starts.tolist():
        key = np.ascontiguousarray(track[start : start + WINDOW, :2]).tobytes()
        sequence_starts_by_key[key].append(start)

    rows: list[dict[str, object]] = []
    ranked_sequences = sorted(
        sequence_starts_by_key.items(),
        key=lambda item: (-len(item[1]), item[1][0]),
    )[:SEQUENCE_CLUSTER_SAMPLE_LIMIT]
    for rank, (key, starts_for_sequence) in enumerate(ranked_sequences, start=1):
        sequence = np.frombuffer(key, dtype=track.dtype).reshape(WINDOW, 2)
        sequence_px = sequence * np.array([FRAME_WIDTH, FRAME_HEIGHT])
        rows.append({
            "rank": rank,
            "count": len(starts_for_sequence),
            "starts": starts_for_sequence,
            "points_px": sequence_px.tolist(),
        })
    return rows


def cluster_sequence_families(
    sequences: list[dict[str, object]],
    top_n: int,
) -> dict[str, object]:
    if not sequences:
        return {
            "exact_sequences_considered": [],
            "clusters": [],
            "clustering": {
                "sample_size": 0,
                "distance_units": "sequence RMS over 32 scalar x/y values, in pixels",
                "distance_formula": "sqrt(sum(dx_px**2 + dy_px**2) / 32)",
                "silhouette_target": SILHOUETTE_TARGET,
                "selected_t_rms_px": None,
                "selected_silhouette_score": None,
                "threshold_evaluations": [],
            },
        }

    features = np.asarray(
        [sequence["points_px"] for sequence in sequences],
        dtype=np.float64,
    ).reshape(len(sequences), -1)
    features /= math.sqrt(WINDOW * 2)
    evaluations: list[dict[str, object]] = []
    labels_by_threshold: dict[float, np.ndarray] = {}
    if len(sequences) == 1:
        labels_by_threshold[SILHOUETTE_T_VALUES[0]] = np.ones(1, dtype=np.int64)
        evaluations.append({
            "t_rms_px": SILHOUETTE_T_VALUES[0],
            "cluster_count": 1,
            "silhouette_score": None,
        })
    else:
        for threshold in SILHOUETTE_T_VALUES:
            labels = fclusterdata(
                features,
                t=threshold,
                criterion="distance",
                method="complete",
                metric="euclidean",
            )
            labels_by_threshold[threshold] = labels
            cluster_count = len(np.unique(labels))
            score = (
                float(silhouette_score(features, labels, metric="euclidean"))
                if 1 < cluster_count < len(sequences)
                else None
            )
            evaluations.append({
                "t_rms_px": threshold,
                "cluster_count": cluster_count,
                "silhouette_score": score,
            })

    valid_evaluations = [
        evaluation
        for evaluation in evaluations
        if evaluation["silhouette_score"] is not None
    ]
    qualified = [
        evaluation
        for evaluation in valid_evaluations
        if float(evaluation["silhouette_score"]) >= SILHOUETTE_TARGET
    ]
    if qualified:
        selected = min(qualified, key=lambda evaluation: float(evaluation["t_rms_px"]))
        selection_rule = "smallest tested t meeting the 0.5 silhouette target"
    elif valid_evaluations:
        selected = max(
            valid_evaluations,
            key=lambda evaluation: (
                float(evaluation["silhouette_score"]),
                -float(evaluation["t_rms_px"]),
            ),
        )
        selection_rule = "highest tested silhouette score because no t met the target"
    else:
        selected = evaluations[0]
        selection_rule = "first tested t because silhouette is undefined for one cluster"

    selected_threshold = float(selected["t_rms_px"])
    labels = labels_by_threshold[selected_threshold]
    members_by_label: defaultdict[int, list[int]] = defaultdict(list)
    for sequence_index, label in enumerate(labels.tolist()):
        members_by_label[int(label)].append(sequence_index)

    clusters: list[dict[str, object]] = []
    for member_indices in members_by_label.values():
        representative_index = max(
            member_indices,
            key=lambda index: (
                int(sequences[index]["count"]),
                -int(sequences[index]["rank"]),
            ),
        )
        clusters.append({
            "member_count": len(member_indices),
            "exact_start_count": sum(
                int(sequences[index]["count"]) for index in member_indices
            ),
            "member_ranks": [int(sequences[index]["rank"]) for index in member_indices],
            "representative_rank": int(sequences[representative_index]["rank"]),
            "representative_points_px": sequences[representative_index]["points_px"],
        })

    clusters.sort(
        key=lambda cluster: (
            -int(cluster["exact_start_count"]),
            -int(cluster["member_count"]),
            int(cluster["representative_rank"]),
        )
    )
    for rank, cluster in enumerate(clusters[:top_n], start=1):
        cluster["rank"] = rank

    return {
        "exact_sequences_considered": sequences,
        "clusters": clusters[:top_n],
        "clustering": {
            "sample_size": len(sequences),
            "sample_limit": SEQUENCE_CLUSTER_SAMPLE_LIMIT,
            "feature_scaling": "flattened 16-frame x/y pixels divided by sqrt(32)",
            "distance_units": "sequence RMS over 32 scalar x/y values, in pixels",
            "distance_formula": "sqrt(sum(dx_px**2 + dy_px**2) / 32)",
            "method": "scipy.cluster.hierarchy.fclusterdata complete linkage",
            "silhouette_target": SILHOUETTE_TARGET,
            "selection_rule": selection_rule,
            "selected_t_rms_px": selected_threshold,
            "selected_silhouette_score": selected["silhouette_score"],
            "threshold_evaluations": evaluations,
        },
    }


def write_location_csv(
    path: Path,
    view: str,
    locations_by_fixture: dict[str, list[dict[str, int]]],
) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=9) as target:
        writer = csv.writer(target)
        writer.writerow(["view", "fixture", "rank", "x_px", "y_px", "instance_count"])
        for fixture, locations in locations_by_fixture.items():
            for location in locations:
                writer.writerow([
                    view,
                    fixture,
                    location["rank"],
                    location["x_px"],
                    location["y_px"],
                    location["instance_count"],
                ])


def plot_locations(
    points_by_fixture: dict[str, np.ndarray],
    masks_by_fixture: dict[str, np.ndarray],
    locations_by_fixture: dict[str, list[dict[str, int]]],
    view: str,
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        len(FIXTURE_NAMES),
        figsize=(19, 5.5),
        constrained_layout=True,
    )
    for axis, fixture in zip(np.atleast_1d(axes), FIXTURE_NAMES, strict=True):
        frame_mask = masks_by_fixture[fixture]
        histogram = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.int64)
        integer_points = np.rint(points_by_fixture[fixture][frame_mask]).astype(np.int64)
        integer_points[:, 0] = np.clip(integer_points[:, 0], 0, FRAME_WIDTH - 1)
        integer_points[:, 1] = np.clip(integer_points[:, 1], 0, FRAME_HEIGHT - 1)
        if len(integer_points):
            np.add.at(histogram, (integer_points[:, 1], integer_points[:, 0]), 1)
        axis.imshow(
            np.log1p(histogram),
            origin="upper",
            extent=(0, FRAME_WIDTH, FRAME_HEIGHT, 0),
            cmap="viridis",
            interpolation="nearest",
        )
        handles: list[Line2D] = []
        for location in locations_by_fixture[fixture]:
            rank = location["rank"]
            x = location["x_px"]
            y = location["y_px"]
            count = location["instance_count"]
            axis.scatter(
                [x + 0.5],
                [y + 0.5],
                s=24,
                facecolors="none",
                edgecolors="#e8d5a3",
            )
            axis.text(
                x + 3,
                y - 3,
                f"#{rank}",
                color="#ffffff",
                fontsize=8,
                weight="bold",
            )
            handles.append(Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markeredgecolor="#e8d5a3",
                label=f"#{rank}: ({x}, {y}), {count} frames",
            ))
        image_axes(axis)
        axis.set_title(
            f"{fixture}: {int(frame_mask.sum())} selected frames\n"
            "heatmap = log1p(frames at each rounded pixel)"
        )
        if handles:
            axis.legend(
                handles=handles,
                title="#rank = frequency order; count = frames at pixel",
                loc="upper left",
                fontsize=6,
                title_fontsize=6.5,
                framealpha=0.9,
            )
    figure.suptitle(
        f"Top-n {LOCATION_LABELS[view]} in the 512x288 image plane\n"
        "Labels are frequency ranks of rounded integer pixels, not detector grades"
    )
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_sequence_families(
    sequence_summaries_by_fixture: dict[str, dict[str, object]],
    top_n: int,
    view: str,
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(FIXTURE_NAMES),
        top_n,
        figsize=(3.2 * top_n, 3.2 * len(FIXTURE_NAMES)),
        squeeze=False,
        constrained_layout=True,
    )
    colour_map = plt.get_cmap("viridis")
    selected_thresholds: list[float] = []
    selected_scores: list[float] = []
    for row_index, fixture in enumerate(FIXTURE_NAMES):
        summary = sequence_summaries_by_fixture[fixture]
        families = summary["clusters"]
        clustering = summary["clustering"]
        if clustering["selected_t_rms_px"] is not None:
            selected_thresholds.append(float(clustering["selected_t_rms_px"]))
        if clustering["selected_silhouette_score"] is not None:
            selected_scores.append(float(clustering["selected_silhouette_score"]))
        for column_index in range(top_n):
            axis = axes[row_index, column_index]
            if column_index >= len(families):
                axis.axis("off")
                continue
            family = families[column_index]
            points = np.asarray(family["representative_points_px"], dtype=np.float64)
            colours = colour_map(np.linspace(0.15, 0.9, len(points)))
            for point_index in range(len(points) - 1):
                axis.plot(
                    points[point_index : point_index + 2, 0],
                    points[point_index : point_index + 2, 1],
                    color=colours[point_index],
                    linewidth=1.5,
                )
            axis.scatter(
                points[:, 0],
                points[:, 1],
                c=np.arange(len(points)),
                cmap="viridis",
                s=8,
            )
            image_axes(axis)
            axis.set_title(
                f"{fixture} family {family['rank']}\n"
                f"{family['member_count']} unique / "
                f"{family['exact_start_count']} exact starts"
            )
    threshold_text = ", ".join(f"{value:g}" for value in sorted(set(selected_thresholds)))
    score_text = ", ".join(f"{value:.3f}" for value in selected_scores)
    figure.suptitle(
        f"Top-n clustered exact 16-frame sequence families from top-{SEQUENCE_CLUSTER_SAMPLE_LIMIT} "
        f"exact sequences\n{SEQUENCE_LABELS[view]}\n"
        "distance = sequence RMS over 32 scalar x/y values; "
        "windows scanned at every frame start; "
        f"selected t values = {threshold_text or 'none'}; silhouette target = {SILHOUETTE_TARGET:g}; "
        f"selected scores = {score_text or 'undefined'}"
    )
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workset",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="analysis workset directory (default: this script's parent workset)",
    )
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument(
        "--view",
        choices=(
            "uncaught",
            "inpaint",
            "union",
            "uncaught_inpaint_impulse",
            "uncaught_impulse_tp_rally_end",
            "both",
            "events",
            "all",
        ),
        default="both",
        help="which evidence view set to write",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workset = args.workset.resolve()
    analysis_dir = workset / "analysis"
    plot_dir = workset / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if args.top_n < 1:
        raise ValueError("--top-n must be positive")

    if args.view == "both":
        views = ("uncaught", "inpaint", "union")
    elif args.view == "events":
        views = EVENT_UNION_VIEWS
    elif args.view == "all":
        views = ALL_VIEWS
    else:
        views = (args.view,)
    points_by_fixture: dict[str, np.ndarray] = {}
    masks_by_view: dict[str, dict[str, np.ndarray]] = {view: {} for view in ALL_VIEWS}
    locations_by_view: dict[str, dict[str, list[dict[str, int]]]] = {
        view: {} for view in views
    }
    sequences_by_view: dict[str, dict[str, dict[str, object]]] = {
        view: {} for view in views
    }

    for fixture in FIXTURE_NAMES:
        track = read_npy_xz(workset / "raw" / f"{fixture}_track.npy.xz")
        guard_codes = read_npy_xz(analysis_dir / f"{fixture}_guard_codes.npy.xz")
        uncaught = read_npy_xz(analysis_dir / f"{fixture}_uncaught_mask.npy.xz")
        inpaint = read_npy_xz(analysis_dir / f"{fixture}_sidecar_inpaint_mask.npy.xz")
        impulse = read_npy_xz(analysis_dir / f"{fixture}_impulse_event_mask.npy.xz")
        tp_rally_ender = read_npy_xz(
            analysis_dir / f"{fixture}_tp_rally_ender_mask.npy.xz"
        )
        points = pixel_points(track)
        valid = ~np.all(track[:, :2] == 0, axis=1)
        if not all(
            len(values) == len(track)
            for values in (guard_codes, uncaught, inpaint, impulse, tp_rally_ender)
        ):
            raise ValueError(f"{fixture}: derived arrays do not match track length")
        points_by_fixture[fixture] = points
        uncaught_view = uncaught & valid
        inpaint_view = inpaint & valid
        masks_by_view["uncaught"][fixture] = uncaught_view
        masks_by_view["inpaint"][fixture] = inpaint_view
        masks_by_view["union"][fixture] = uncaught_view | inpaint_view
        impulse_view = impulse & valid
        tp_rally_ender_view = tp_rally_ender & valid
        masks_by_view["uncaught_inpaint_impulse"][fixture] = (
            uncaught_view | inpaint_view | impulse_view
        )
        masks_by_view["uncaught_impulse_tp_rally_end"][fixture] = (
            uncaught_view | impulse_view | tp_rally_ender_view
        )
        for view in views:
            frame_mask = masks_by_view[view][fixture]
            locations_by_view[view][fixture] = rounded_locations(points, frame_mask, args.top_n)
            exact = exact_sequences(track, guard_codes, frame_mask, view)
            sequences_by_view[view][fixture] = cluster_sequence_families(exact, args.top_n)

    for view in views:
        (
            locations_name,
            sequences_name,
            locations_plot_name,
            sequences_plot_name,
        ) = OUTPUT_NAMES[view]
        write_location_csv(analysis_dir / locations_name, view, locations_by_view[view])
        write_json_gz(
            analysis_dir / sequences_name,
            {
                "view": view,
                "description": SEQUENCE_LABELS[view],
                "fixtures": sequences_by_view[view],
            },
        )
        plot_locations(
            points_by_fixture,
            masks_by_view[view],
            locations_by_view[view],
            view,
            plot_dir / locations_plot_name,
        )
        plot_sequence_families(
            sequences_by_view[view],
            args.top_n,
            view,
            plot_dir / sequences_plot_name,
        )
    print(f"wrote {', '.join(views)} views to {workset}")


if __name__ == "__main__":
    main()
