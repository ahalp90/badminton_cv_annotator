"""Compare every defined RANSAC-adjacent mask in the issue-95 evidence set."""

from __future__ import annotations

import csv
import gzip
import json
import lzma
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
WORKSET = Path(__file__).resolve().parents[1]
ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from annotator import inpaint_guard  # noqa: E402


FIXTURES = {"sset_01": 1, "sset_15": 15, "sset_21": 21}


def read_npy_xz(path: Path) -> np.ndarray:
    """Load an XZ-wrapped NumPy array.

    :param path: Compressed array path.
    :return: Loaded array with pickle disabled.
    """
    with lzma.open(path, "rb") as source:
        return np.load(source, allow_pickle=False)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Expand true frames by an absolute radius.

    :param mask: One-dimensional boolean source mask.
    :param radius: Number of neighbouring frames to include on each side.
    :return: Expanded boolean mask.
    """
    expanded = np.zeros(len(mask), dtype=bool)
    for offset in range(-radius, radius + 1):
        source_start = max(0, -offset)
        source_stop = min(len(mask), len(mask) - offset)
        target_start = source_start + offset
        target_stop = source_stop + offset
        expanded[target_start:target_stop] |= mask[source_start:source_stop]
    return expanded


def retain_runs(mask: np.ndarray, minimum_length: int) -> np.ndarray:
    """Keep frames in contiguous true runs meeting a length floor.

    :param mask: One-dimensional boolean source mask.
    :param minimum_length: Smallest retained run length.
    :return: Run-filtered boolean mask.
    """
    result = np.zeros(len(mask), dtype=bool)
    changes = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    for start, stop in zip(starts, stops, strict=True):
        if stop - start >= minimum_length:
            result[start:stop] = True
    return result


def load_frame_fields(fixture: str) -> dict[str, np.ndarray]:
    """Load the RANSAC vote and residual fields from the tracked frame audit.

    :param fixture: Fixture name.
    :return: Numeric arrays keyed by field name.
    """
    path = ANALYSIS / f"{fixture}_frame_audit.csv.gz"
    fields: dict[str, list[float]] = {
        "eligible": [],
        "votes": [],
        "fraction": [],
        "residual": [],
    }
    with gzip.open(path, "rt", newline="") as source:
        for row in csv.DictReader(source):
            fields["eligible"].append(float(row["ransac_eligible_windows"]))
            fields["votes"].append(float(row["ransac_outlier_votes"]))
            fields["fraction"].append(float(row["ransac_outlier_fraction"]))
            fields["residual"].append(float(row["ransac_max_residual_px"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def load_contact_frames() -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Load unique contact and final-contact frames from ShuttleSet labels.

    :return: Contact and final-contact frame sets keyed by video id.
    """
    path = REPO / "training/data/shuttleset/annotations/shots_master.csv"
    contacts: dict[int, set[int]] = defaultdict(set)
    rallies: dict[tuple[int, str, int], list[tuple[int, int]]] = defaultdict(list)
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            video_id = int(row["vid"])
            if video_id not in FIXTURES.values():
                continue
            frame = int(row["frame_num"])
            ball_round = int(row["ball_round"])
            contacts[video_id].add(frame)
            rallies[(video_id, row["set_id"], int(row["rally"]))].append(
                (ball_round, frame)
            )

    final_contacts: dict[int, set[int]] = defaultdict(set)
    for (video_id, _set_id, _rally), rows in rallies.items():
        final_contacts[video_id].add(max(rows)[1])
    return contacts, final_contacts


def load_visual_spans(fixture: str) -> list[tuple[int, int]]:
    """Load the 18 positive-only human-labelled spans for one fixture.

    :param fixture: Fixture name.
    :return: Half-open frame ranges.
    """
    path = ANALYSIS / f"{fixture}_visual_hallucination_audit.csv.gz"
    spans: list[tuple[int, int]] = []
    with gzip.open(path, "rt", newline="") as source:
        for row in csv.DictReader(source):
            if row["human_visual_label"] != "hallucination":
                raise ValueError(f"{fixture}: unexpected visual label")
            spans.append((int(row["start_frame"]), int(row["stop_frame_exclusive"])))
    return spans


def split_recurrence_components(
    track: np.ndarray, codes: np.ndarray
) -> dict[str, np.ndarray]:
    """Reproduce the current recurrence core, halo, and global-hit masks.

    :param track: Canonical fixture track.
    :param codes: Current public ``grade_track`` output.
    :return: Rejection policies for the documented recurrence decomposition.
    """
    window = inpaint_guard.DEFAULT_WINDOW
    halo_frames = inpaint_guard.DEFAULT_HALO_FRAMES
    # Grade 3 does not expose its two components. This analysis-only
    # reconstruction follows the current internals and asserts exact agreement.
    _, _, varying, flat, _ = inpaint_guard._candidate_attractors(
        track, window, halo_frames
    )
    proven = inpaint_guard._cover(len(track), varying, window)
    suspect = inpaint_guard._cover(len(track), flat, window)
    core = proven | suspect

    halo = np.zeros(len(track), dtype=bool)
    edges = np.diff(np.concatenate(([False], core, [False])).astype(np.int8))
    for start in np.flatnonzero(edges == 1):
        halo[max(0, int(start) - halo_frames) : int(start)] = True
    for stop in np.flatnonzero(edges == -1):
        stop = int(stop)
        halo[stop : min(len(track), stop + halo_frames)] = True

    positions: set[tuple[Any, Any]] = set()
    for key in (*varying, *flat):
        points = np.frombuffer(key, dtype=track.dtype).reshape(window, 2)
        for point in points:
            positions.add((point[0], point[1]))
    global_hit = np.zeros(len(track), dtype=bool)
    for pos_x, pos_y in positions:
        global_hit |= (track[:, 0] == pos_x) & (track[:, 1] == pos_y)

    coordinate_valid = ~np.all(track[:, :2] == 0, axis=1)
    degraded = (halo | global_hit) & ~core & coordinate_valid
    if not np.array_equal(degraded, codes == inpaint_guard.DEGRADED):
        raise ValueError("reproduced grade-3 components disagree with grade_track")
    return {
        "recurrence_grade1_core": codes == inpaint_guard.FABRICATED,
        "recurrence_core_plus_halo": (core | halo) & coordinate_valid,
        "recurrence_core_plus_global_hit": (core | global_hit) & coordinate_valid,
        "recurrence_current_all_nonzero": codes != inpaint_guard.NO_FLAG,
    }


def fully_selected_blocks(
    sidecar: np.ndarray, valid: np.ndarray, window: int = 16
) -> np.ndarray:
    """Mark fully selected non-overlap producer blocks tiled from frame zero.

    :param sidecar: Producer fill provenance mask.
    :param valid: Coordinate-valid mask.
    :param window: InpaintNet producer width.
    :return: Frame mask for aligned blocks with no TrackNet pass-through.
    """
    result = np.zeros(len(sidecar), dtype=bool)
    for start in range(0, len(sidecar) - window + 1, window):
        stop = start + window
        if sidecar[start:stop].all() and valid[start:stop].all():
            result[start:stop] = True
    return result


def metric(
    mask: np.ndarray, contacts: set[int], finals: set[int], spans: list[tuple[int, int]]
) -> dict[str, int]:
    """Calculate fixed evidence checks for one candidate mask.

    :param mask: Candidate or contextual frame mask.
    :param contacts: Labelled contact frames.
    :param finals: Labelled final-contact frames.
    :param spans: Positive-only human-labelled spans.
    :return: Counts used in the issue-95 decision.
    """
    contact_mask = np.zeros(len(mask), dtype=bool)
    final_mask = np.zeros(len(mask), dtype=bool)
    contact_mask[list(contacts)] = True
    final_mask[list(finals)] = True
    visual_mask = np.zeros(len(mask), dtype=bool)
    span_hits = 0
    for start, stop in spans:
        visual_mask[start:stop] = True
        span_hits += int(mask[start:stop].any())
    contacts_within_radius = 0
    for frame in contacts:
        contacts_within_radius += int(
            mask[max(0, frame - 3) : min(len(mask), frame + 4)].any()
        )
    finals_within_radius = 0
    for frame in finals:
        finals_within_radius += int(
            mask[max(0, frame - 3) : min(len(mask), frame + 4)].any()
        )
    return {
        "selected_frames": int(mask.sum()),
        "visual_positive_frames": int((mask & visual_mask).sum()),
        "visual_positive_spans_hit": span_hits,
        "exact_contacts": int((mask & contact_mask).sum()),
        "contacts_within_radius3": contacts_within_radius,
        "exact_final_contacts": int((mask & final_mask).sum()),
        "final_contacts_within_radius3": finals_within_radius,
    }


def add_metrics(
    totals: dict[str, dict[str, dict[str, int]]],
    category: str,
    name: str,
    values: dict[str, int],
) -> None:
    """Accumulate one fixture's metrics.

    :param totals: Nested output accumulator.
    :param category: Proposal category.
    :param name: Proposal name.
    :param values: Fixture metrics.
    """
    target = totals.setdefault(category, {}).setdefault(name, defaultdict(int))
    for key, value in values.items():
        target[key] += value


def main() -> None:
    """Run the complete finite-proposal comparison and print JSON evidence."""
    contacts_by_video, finals_by_video = load_contact_frames()
    totals: dict[str, dict[str, dict[str, int]]] = {}
    fixture_checks: dict[str, dict[str, int]] = {}

    for fixture, video_id in FIXTURES.items():
        track = read_npy_xz(WORKSET / "raw" / f"{fixture}_track.npy.xz")
        candidate = read_npy_xz(ANALYSIS / f"{fixture}_ransac_candidate.npy.xz").astype(
            bool
        )
        sidecar = read_npy_xz(
            ANALYSIS / f"{fixture}_sidecar_inpaint_mask.npy.xz"
        ).astype(bool)
        impulse = read_npy_xz(ANALYSIS / f"{fixture}_impulse_event_mask.npy.xz").astype(
            bool
        )
        tp_ender = read_npy_xz(
            ANALYSIS / f"{fixture}_tp_rally_ender_mask.npy.xz"
        ).astype(bool)
        valid = ~np.all(track[:, :2] == 0, axis=1)
        codes, info = inpaint_guard.grade_track(track)
        if info["detector_version"] != 4 or info["halo_frames"] != 3:
            raise ValueError(f"{fixture}: current recurrence baseline changed")
        fields = load_frame_fields(fixture)
        if not np.array_equal(
            candidate, (fields["eligible"] > 0) & (fields["fraction"] >= 0.5) & valid
        ):
            raise ValueError(
                f"{fixture}: frame audit does not reproduce stored candidate mask"
            )

        contacts = contacts_by_video[video_id]
        finals = finals_by_video[video_id]
        spans = load_visual_spans(fixture)
        guard_clean = candidate & (codes == inpaint_guard.NO_FLAG)
        long_sidecar_runs = retain_runs(sidecar & valid, 15)
        selected_blocks = fully_selected_blocks(sidecar, valid)

        ransac_masks = {
            "raw_candidate": candidate,
            "current_guard_clean": guard_clean,
            "guard_clean_sidecar_positive": guard_clean & sidecar,
            "guard_clean_sidecar_negative": guard_clean & ~sidecar,
            "guard_clean_sidecar_long_run_ge15": guard_clean & long_sidecar_runs,
            "raw_candidate_fully_selected_aligned_block": candidate & selected_blocks,
            "guard_clean_fully_selected_aligned_block": guard_clean & selected_blocks,
            "guard_clean_impulse_veto_radius3": guard_clean & ~dilate(impulse, 3),
        }
        for threshold in (50, 100, 200, 250, 400):
            ransac_masks[f"guard_clean_max_residual_ge_{threshold}"] = guard_clean & (
                fields["residual"] >= threshold
            )
        for fraction in (0.75, 1.0):
            suffix = str(fraction).replace(".", "_")
            ransac_masks[f"guard_clean_vote_fraction_ge_{suffix}"] = (
                (codes == inpaint_guard.NO_FLAG)
                & valid
                & (fields["eligible"] > 0)
                & (fields["fraction"] >= fraction)
            )
        for eligible in (2, 3, 4):
            ransac_masks[f"guard_clean_min_eligible_windows_{eligible}"] = (
                guard_clean & (fields["eligible"] >= eligible)
            )
        for run_length in (2, 4, 8):
            ransac_masks[f"guard_clean_min_run_{run_length}"] = retain_runs(
                guard_clean, run_length
            )

        recurrence_masks = split_recurrence_components(track, codes)
        context_masks = {
            "uncaught_or_sidecar": guard_clean | (sidecar & valid),
            "union1_uncaught_sidecar_impulse": guard_clean
            | (sidecar & valid)
            | (impulse & valid),
            "union2_uncaught_impulse_tp_ender": guard_clean
            | (impulse & valid)
            | (tp_ender & valid),
            "current_guard_or_union2": (codes != 0)
            | guard_clean
            | (impulse & valid)
            | (tp_ender & valid),
        }
        source_masks = {
            "sidecar_long_run_ge15": long_sidecar_runs,
            "fully_selected_aligned_block": selected_blocks,
        }

        for category, masks in (
            ("ransac_candidates", ransac_masks),
            ("recurrence_policies", recurrence_masks),
            ("context_only_unions", context_masks),
            ("source_aware_proposal", source_masks),
        ):
            for name, mask in masks.items():
                add_metrics(
                    totals, category, name, metric(mask, contacts, finals, spans)
                )

        fixture_checks[fixture] = {
            "frames": len(track),
            "contacts": len(contacts),
            "final_contacts": len(finals),
            "visual_positive_spans": len(spans),
        }

    serializable = {
        "scope": {
            "fixtures": list(FIXTURES),
            "guard_baseline": "detector v4, halo 3",
            "visual_labels": "18 positive-only issue-31 spans",
            "warning": "Context unions are review views, not proposed rejection masks.",
            "unscored_proposals": [
                "Aligned support fields other than the all-selected block have no proposed rejection cutoff.",
                "Producer-phase and boundary diagnostics do not define a mask; phase-shift nulls were not run.",
                "Early versus late application requires a selected candidate mask and fixed-clip counterfactual.",
                "Isolation Forest has no fitted model, held-out score, or production cutoff.",
                "LOF, One-Class SVM, and Elliptic Envelope are deferred or rejected behind Isolation Forest.",
                "Base-TrackNet heatmap morphology lacks saved inputs and a defined rule or cutoff.",
                "Centred and scale-normalised path shape was proposed for plotting, not as a rejection rule.",
                "Circular-shift, block-preserving, and span-length nulls test association rather than precision.",
            ],
        },
        "fixture_checks": fixture_checks,
        "totals": totals,
    }
    print(json.dumps(serializable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
