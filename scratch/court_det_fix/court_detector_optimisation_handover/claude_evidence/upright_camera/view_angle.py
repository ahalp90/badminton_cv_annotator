"""How far from straight down each court says the camera looks, at the court's centre.

The court's homography maps floor metres to image pixels. Near the court's centre it stretches the
floor evenly in every direction when the camera looks straight down at that point, and squashes it
along the viewing direction by cos(angle) otherwise. So the ratio of the local map's two singular
values gives the angle between the viewing ray and straight down, with no focal length needed.
Square pixels are assumed.

For each view: the chosen court's angle; how many courts in G0's and G1's overall shortlists imply
a camera within 10 and 20 degrees of straight down; and the 5th to 95th percentile of those courts'
angles.
Usage: python view_angle.py <artefact dir>
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

COURT_CENTRE_M = np.array([3.05, 6.7])


def view_angle_deg(homography: list) -> float:
    matrix = np.asarray(homography, dtype=float)
    x, y, w = matrix @ np.array([*COURT_CENTRE_M, 1.0])
    x, y = x / w, y / w
    jacobian = (matrix[:2, :2] - np.outer([x, y], matrix[2, :2])) / w
    smaller, larger = np.linalg.svd(jacobian, compute_uv=False)[::-1]
    return float(np.degrees(np.arccos(np.clip(smaller / larger, 0.0, 1.0))))


print("view\tchosen court (deg)\tG0 within 10 / 20 deg\tG1 within 10 / 20 deg\tshortlists 5th-95th percentile (deg)")
for path in sorted(Path(sys.argv[1]).glob("*.json.gz")):
    with gzip.open(path) as handle:
        artefact = json.load(handle)
    record = artefact["w5"]["record"]
    courts = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
    chosen = artefact["net_choice"]["chosen"]
    chosen_angle = "-" if chosen is None else f"{view_angle_deg(courts[chosen]['homography_working']):.0f}"
    cells = [chosen_angle]
    shortlist_angles = []
    for name in ("G0", "G1"):
        angles = [view_angle_deg(entry["homography_working"]) for entry in artefact["populations"][name]]
        shortlist_angles += angles
        cells.append(f"{sum(angle < 10 for angle in angles)} / {sum(angle < 20 for angle in angles)} of {len(angles)}")
    low, high = np.percentile(shortlist_angles, [5, 95]) if shortlist_angles else (np.nan, np.nan)
    cells.append("-" if not shortlist_angles else f"{low:.0f}-{high:.0f}")
    print(path.name.removesuffix(".json.gz") + "\t" + "\t".join(cells))
