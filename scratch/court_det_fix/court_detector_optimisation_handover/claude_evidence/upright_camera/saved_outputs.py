"""What the upright-camera tests would remove from the 26 September saved search outputs.

A court's horizon is the line through its direction pair's two vanishing points; its tilt is
the camera roll the court needs. The pair test skips pairs tilted more than 45 degrees. The
court test drops courts above their horizon, which need an upside-down camera. Both pass a
horizon more than 10 image diagonals away, as run_given.horizon does.

Prints the pairs and pair time the pair test removes, the share of kept pairs' shortlisted
courts the court test removes, and for each view the implied roll of the chosen court and
the search rank of its parent among the saved shortlist courts that pass both tests. Last, per
view, how many places in each search's overall shortlist went to courts from pairs the pair
test removes.

Usage: python saved_outputs.py <populations dir> <artefact dir>
"""

import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

MAX_TILT_DEG = 45.
FAR_HORIZON_DIAGONALS = 10.
populations_dir, artefact_dir = Path(sys.argv[1]), Path(sys.argv[2])
# Court corners in metres (width, length), in the detector's corner order.
COURT_CORNERS = np.array([[0., 0., 1.], [6.1, 0., 1.], [6.1, 13.4, 1.], [0., 13.4, 1.]])


def read_gz(path: Path) -> dict:
    with gzip.open(path) as handle:
        return json.load(handle)


def horizon(points: np.ndarray, size: list[int]) -> np.ndarray | None:
    line = np.cross(points[0], points[1])
    normal_length = np.hypot(line[0], line[1])
    if normal_length == 0:
        return None
    centre_distance = abs(line @ [size[0] / 2, size[1] / 2, 1.]) / normal_length
    return None if centre_distance > FAR_HORIZON_DIAGONALS * np.hypot(*size) else line


def pair_horizons(population: dict) -> dict[int, np.ndarray | None]:
    points = np.asarray(population["estimator"]["points_working"])
    return {pair["pair_id"]: horizon(points[pair["pencils"]], population["working_size"])
            for pair in population["pairs"] if pair["status"] == "matched"}


def steep(line: np.ndarray | None) -> bool:
    return line is not None and np.degrees(np.arctan2(abs(line[0]), abs(line[1]))) > MAX_TILT_DEG


def above(line: np.ndarray | None, homography: list) -> bool:
    if line is None:
        return False
    projected = COURT_CORNERS @ np.asarray(homography).T
    corners = projected[:, :2] / projected[:, 2:]
    return bool((np.sign(line[1]) * (corners @ line[:2] + line[2]) <= 0).any())


def passes(entry: dict, horizons: dict) -> bool:
    line = horizons[entry["pair_id"]]
    return not steep(line) and not above(line, entry["homography_working"])


def roll_deg(corners: list) -> float:
    """Signed roll the court needs, from its corners: 0 level, beyond 90 upside down."""
    homography = cv2.getPerspectiveTransform(COURT_CORNERS[:, :2].astype(np.float32),
                                             np.asarray(corners, np.float32)).astype(float)
    homography *= np.sign((homography @ COURT_CORNERS.mean(axis=0))[2])
    line = np.linalg.inv(homography).T @ [0., 0., 1.]
    return float(np.degrees(np.arctan2(line[0], line[1])))


for source in ("G0", "G1"):
    pairs = steep_pairs = 0
    seconds = {"all": 0., "steep": 0.}
    kept_courts = upside_down = 0
    for path in sorted((populations_dir / source).glob("*.json.gz")):
        population = read_gz(path)
        horizons = pair_horizons(population)
        for pair in population["pairs"]:
            if pair["status"] != "matched":
                continue
            pairs += 1
            seconds["all"] += pair["elapsed_s"]
            if steep(horizons[pair["pair_id"]]):
                steep_pairs += 1
                seconds["steep"] += pair["elapsed_s"]
                continue
            kept_courts += len(pair["shortlist"])
            upside_down += sum(above(horizons[pair["pair_id"]], entry["homography_working"])
                               for entry in pair["shortlist"])
    print(f"{source}: pair test removes {steep_pairs}/{pairs} matched pairs, {seconds['steep'] / seconds['all']:.1%} "
          f"of pair time; court test removes {upside_down}/{kept_courts} ({upside_down / kept_courts:.1%}) "
          "of the kept pairs' shortlisted courts")

print("\nview\tchosen roll deg\tchosen parent: saved rank -> rank among courts passing both tests")
for path in sorted(artefact_dir.glob("*.json.gz")):
    view = path.name.removesuffix(".json.gz")
    artefact = read_gz(path)
    chosen = artefact["net_choice"]["chosen"]
    if chosen is None:
        print(f"{view}\t-\tno court")
        continue
    record = artefact["w5"]["record"]
    candidates = {candidate["origin_key"]: candidate for candidate in record["parents"] + record["valid_children"]}
    parent_key = candidates[chosen].get("parent_origin_key") or chosen
    moves = []
    for occurrence in candidates[parent_key]["source_occurrences"]:
        if occurrence["source"] not in ("G0", "G1"):
            moves.append(occurrence["source"])
            continue
        population = read_gz(populations_dir / occurrence["source"] / f"{view}.json.gz")
        horizons = pair_horizons(population)
        index = occurrence["origin_index"]
        entry = population["entries"][index]
        assert entry["candidate_id"] == occurrence["candidate_id"]
        if not passes(entry, horizons):
            test = "pair" if steep(horizons[entry["pair_id"]]) else "court"
            moves.append(f"{occurrence['source']} #{index + 1} -> removed by the {test} test")
            continue
        rank = sum(passes(earlier, horizons) for earlier in population["entries"][:index]) + 1
        moves.append(f"{occurrence['source']} #{index + 1} -> #{rank}")
    print(f"{view}\t{roll_deg(candidates[chosen]['corners_px']):.1f}\t" + "; ".join(moves))

print("\nview\tG0 overall places held by steep pairs\tG1 overall places held by steep pairs")
for path in sorted(artefact_dir.glob("*.json.gz")):
    view = path.name.removesuffix(".json.gz")
    cells = []
    for source in ("G0", "G1"):
        population = read_gz(populations_dir / source / f"{view}.json.gz")
        horizons = pair_horizons(population)
        steep_places = sum(steep(horizons[entry["pair_id"]]) for entry in population["entries"])
        cells.append(f"{steep_places}/{len(population['entries'])}")
    print(f"{view}\t" + "\t".join(cells))
