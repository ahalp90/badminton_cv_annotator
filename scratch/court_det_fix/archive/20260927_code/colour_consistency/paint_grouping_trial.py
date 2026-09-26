"""Compare fixed observed-paint decisions with raw-colour and hue-only arms."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from scratch.court_det_fix.colour_consistency import observed_colour as observed

SOURCE = Path(__file__).with_name("fragment_trial.json.gz")
OUTPUT = Path(__file__).with_suffix(".json.gz")
NEUTRAL = np.array([128.0, 128.0])
STRONG_CHROMA = 20.0
GROUP_RADIUS = 20.0
HUES = (15.0, 20.0, 25.0)
FLOORS = (15.0, 20.0, 25.0)


def colour_group(colours: dict[str, np.ndarray], anchor: str, mode: str, hue_limit: float) -> frozenset[str]:
    centre = colours[anchor]
    chroma = float(np.linalg.norm(centre - NEUTRAL))
    members = set()
    for name, colour in colours.items():
        other_chroma = float(np.linalg.norm(colour - NEUTRAL))
        if mode == "raw":
            matching = np.linalg.norm(colour - centre) <= GROUP_RADIUS
        elif chroma >= STRONG_CHROMA and other_chroma >= STRONG_CHROMA:
            angle = np.arctan2(*(centre - NEUTRAL)[::-1])
            other_angle = np.arctan2(*(colour - NEUTRAL)[::-1])
            difference = np.angle(np.exp(1j * (angle - other_angle)))
            matching = abs(np.degrees(difference)) <= hue_limit
        else:
            matching = chroma < STRONG_CHROMA and other_chroma < STRONG_CHROMA
            matching &= np.linalg.norm(colour - centre) <= GROUP_RADIUS
        if matching:
            members.add(name)
    return frozenset(members)


def choose_group(colours: dict[str, np.ndarray], mode: str, hue_limit: float) -> tuple[list[str], str | None]:
    if len(colours) < 2:
        return [], "insufficient_distinct_markings"
    groups = [colour_group(colours, name, mode, hue_limit) for name in colours]
    maximum = max(map(len, groups))
    winners = {group for group in groups if len(group) == maximum}
    if maximum < 2 or maximum <= len(colours) / 2 or len(winners) != 1:
        return [], "ambiguous_reference"
    return sorted(winners.pop()), None


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, percentile * cumulative[-1], side="left")])


def hue_degrees(raw_ab: np.ndarray) -> float:
    offset = raw_ab - NEUTRAL
    return float(np.degrees(np.arctan2(offset[1], offset[0])))


def angular_distance(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def decide_hue_only(signatures: dict, fragments: list[dict], chromatic: bool,
                    hue_limit: float = 20.0) -> dict:
    targets = {}
    fragment_lookup = {fragment["raw_fragment_id"]: fragment for fragment in fragments}
    for target in sorted(observed.OUTER_MARKINGS):
        row = signatures.get(target)
        if row is None:
            targets[target] = {"decision": "no_decision", "reason": "target_absent"}
            continue
        if not chromatic:
            targets[target] = {"decision": "no_decision", "reason": "greyscale"}
            continue
        target_ab = np.asarray(row["raw_ab"], dtype=float)
        if np.linalg.norm(target_ab - NEUTRAL) < STRONG_CHROMA:
            targets[target] = {"decision": "no_decision", "reason": "low_chroma_target"}
            continue
        others = {name: value for name, value in signatures.items() if name != target}
        supported = {name: np.asarray(value["raw_ab"], dtype=float) for name, value in others.items()}
        reliable = {name: colour for name, colour in supported.items()
                    if np.linalg.norm(colour - NEUTRAL) >= STRONG_CHROMA}
        if len(supported) < 2:
            failure = "insufficient_distinct_markings"
        elif len(reliable) < 2:
            failure = "low_chroma_reference"
        else:
            references, failure = choose_group(reliable, "hue", hue_limit)
            if not failure and len(references) <= len(supported) / 2:
                failure = "low_chroma_reference" if len(reliable) < len(supported) else "ambiguous_reference"
        if failure:
            targets[target] = {"decision": "no_decision", "reason": failure,
                               "other_supported_markings": sorted(others), "reference_markings": []}
            continue
        reference_angles = np.radians([hue_degrees(reliable[name]) for name in references])
        direction = float(np.degrees(np.angle(np.mean(np.exp(1j * reference_angles)))))
        dispersions = []
        weights = []
        for name in references:
            members = [fragment_lookup[fragment_id] for fragment_id in others[name]["fragment_ids"]]
            chromatic_members = [np.asarray(member["median_raw_ab"], dtype=float) for member in members]
            chromatic_members = [colour for colour in chromatic_members
                                 if np.linalg.norm(colour - NEUTRAL) >= STRONG_CHROMA]
            if not chromatic_members:
                failure = "low_chroma_reference"
                break
            for colour in chromatic_members:
                dispersions.append(angular_distance(hue_degrees(colour), direction))
                weights.append(1.0 / len(chromatic_members))
        if failure:
            targets[target] = {"decision": "no_decision", "reason": failure,
                               "other_supported_markings": sorted(others), "reference_markings": []}
            continue
        scale = weighted_percentile(np.asarray(dispersions), np.asarray(weights), 0.9)
        distance = angular_distance(hue_degrees(target_ab), direction)
        threshold = max(hue_limit, 3 * scale)
        rejected = distance > threshold
        targets[target] = {"decision": "abstain" if rejected else "keep",
                           "reason": "observed_hue_contradiction" if rejected else "no_hue_contradiction",
                           "reference_markings": references, "other_supported_markings": sorted(others),
                           "reference_hue_degrees": direction, "reference_dispersion_degrees": scale,
                           "distance_degrees": distance, "threshold_degrees": threshold}
    decisions = [row["decision"] for row in targets.values()]
    after = "abstain" if "abstain" in decisions else "keep" if "keep" in decisions else "no_decision"
    return {"after": after, "targets": targets}


def decide(signatures: dict, fragments: list[dict], chromatic: bool, mode: str,
           hue_limit: float = 20.0, floor: float = 20.0) -> dict:
    targets = {}
    for target in sorted(observed.OUTER_MARKINGS):
        row = signatures.get(target)
        if row is None:
            targets[target] = {"decision": "no_decision", "reason": "target_absent"}
            continue
        if not chromatic:
            targets[target] = {"decision": "no_decision", "reason": "greyscale"}
            continue
        others = {name: value for name, value in signatures.items() if name != target}
        colours = {name: np.asarray(value["raw_ab"], dtype=float) for name, value in others.items()}
        references, failure = choose_group(colours, mode, hue_limit)
        if failure:
            targets[target] = {"decision": "no_decision", "reason": failure,
                               "other_supported_markings": sorted(others), "reference_markings": references}
            continue
        centre = np.median([colours[name] for name in references], axis=0)
        fragment_lookup = {fragment["raw_fragment_id"]: fragment for fragment in fragments}
        distances = []
        weights = []
        for name in references:
            members = [fragment_lookup[fragment_id] for fragment_id in others[name]["fragment_ids"]]
            for member in members:
                distances.append(float(np.linalg.norm(np.asarray(member["median_raw_ab"]) - centre)))
                weights.append(1.0 / len(members))
        scale = weighted_percentile(np.asarray(distances), np.asarray(weights), 0.9)
        distance = float(np.linalg.norm(np.asarray(row["raw_ab"]) - centre))
        threshold = max(floor, 3 * scale)
        rejected = distance > threshold
        targets[target] = {"decision": "abstain" if rejected else "keep",
                           "reason": "observed_paint_contradiction" if rejected else "no_colour_contradiction",
                           "reference_markings": references, "other_supported_markings": sorted(others),
                           "reference_raw_ab": centre.tolist(), "reference_scale_ab": scale,
                           "distance_ab": distance, "threshold_ab": threshold}
    decisions = [row["decision"] for row in targets.values()]
    after = "abstain" if "abstain" in decisions else "keep" if "keep" in decisions else "no_decision"
    return {"after": after, "targets": targets}


def run() -> dict:
    baseline = json.loads(gzip.decompress(SOURCE.read_bytes()))
    cases = []
    for case in baseline["cases"]:
        if case["status"] != "measured":
            cases.append({**case, "balanced_raw": None, "balanced_hue": None,
                          "grouping_sensitivity": None, "hue_only": None, "hue_only_sensitivity": None})
            continue
        signatures = case["markings"]
        fragments = case["observed"]["fragments"]
        chromatic = case["chromatic"]
        raw = decide(signatures, fragments, chromatic, "raw")
        hue = decide(signatures, fragments, chromatic, "hue")
        hue_only = decide_hue_only(signatures, fragments, chromatic)
        hue_only_sensitivity = {f"hue{int(limit)}": decide_hue_only(signatures, fragments, chromatic, limit)
                                for limit in HUES}
        sensitivity = {}
        for limit in HUES:
            for floor in FLOORS:
                sensitivity[f"hue{int(limit)}_floor{int(floor)}"] = decide(
                    signatures, fragments, chromatic, "hue", limit, floor)
        cases.append({**case, "balanced_raw": raw, "balanced_hue": hue,
                      "grouping_sensitivity": sensitivity, "hue_only": hue_only,
                      "hue_only_sensitivity": hue_only_sensitivity})
        print(case["case_id"], case["primary"]["after"], raw["after"], hue["after"], hue_only["after"])
    return {"schema": "balanced-paint-grouping-trial/2", "source_schema": baseline["schema"],
            "settings": {"neutral_ab": NEUTRAL.tolist(), "strong_chroma": STRONG_CHROMA,
                         "raw_group_radius_ab": GROUP_RADIUS, "neutral_group_radius_ab": GROUP_RADIUS,
                         "hue_limits_degrees": HUES, "target_floors_ab": FLOORS,
                         "balanced_hue_description": "hue-grouped raw-colour mismatch",
                         "hue_only_description": "chromatic reference consensus and circular target hue distance",
                         "reference_scale": "weighted nearest-rank 90th percentile; each marking has weight one"},
            "cases": cases}


if __name__ == "__main__":
    OUTPUT.write_bytes(gzip.compress(json.dumps(run(), allow_nan=False, separators=(",", ":")).encode(), mtime=0))
