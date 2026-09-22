"""Test for lens distortion (line bowing) on the GX footage, with amateur and
broadcast footage as controls.

Read-only on every input. Writes only under OUT.

For each case, takes every merged line row with at least four member
fragments whose endpoints span at least 40 percent of the image width or
height. For each such row, fits a straight line to the member endpoints by
total least squares, then fits a quadratic to the perpendicular residuals
along the line to get a bow sagitta. Reports median/max absolute sagitta
(normalised by image width), the fraction of rows bowing away from the
image centre, and the Spearman correlation between absolute sagitta and
distance from the image centre.
"""

import gzip
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[6]
OUT_DIR = Path(__file__).resolve().parent

MIN_MEMBERS = 4
MIN_SPAN_FRACTION = 0.4

MERGED_ROWS_DIR = (
    REPO_ROOT
    / "scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/e0"
)

# Each pack entry: pack file, list of case ids, group label, frame image directory (or None).
PACKS = [
    {
        "group": "gx",
        "pack": REPO_ROOT
        / "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/"
        "20260909/gx_extension/inputs.json.gz",
        "case_ids": ["gxBQ_window_00_frame_0", "gxBQ_window_00_frame_5"],
        "image_dir": REPO_ROOT
        / "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/"
        "20260909/gx_extension/people/images",
    },
    {
        "group": "amateur",
        "pack": REPO_ROOT
        / "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/"
        "20260908/marking_refit/marking_inputs.json.gz",
        "case_ids": [
            "am2_window_00_frame_150",
            "am2_window_01_frame_28019",
            "am3_window_00_frame_0",
        ],
        "image_dir": REPO_ROOT
        / "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/"
        "20260908/marking_refit/images",
    },
    {
        "group": "broadcast",
        "pack": REPO_ROOT
        / "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/"
        "20260909/broadcast_extension/inputs.json.gz",
        "case_ids": [
            "shuttleset_03_scene_0016",
            "shuttleset_03_scene_0019",
            "shuttleset_21_scene_0020",
        ],
        "image_dir": REPO_ROOT
        / "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/"
        "20260909/broadcast_extension/images",
    },
]


def load_gz_json(path: Path) -> dict:
    with gzip.open(path) as f:
        return json.load(f)


def total_least_squares_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a straight line to points by total least squares (PCA).

    Returns (centroid, direction_unit_vector, normal_unit_vector).
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    cov = centred.T @ centred
    _eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh sorts ascending: last column is the major axis (line direction).
    direction = eigvecs[:, 1]
    normal = eigvecs[:, 0]
    return centroid, direction, normal


def analyse_row(
    points: np.ndarray, image_centre: np.ndarray
) -> dict | None:
    """Fit the row's member endpoints and compute its bow sagitta.

    Returns None if fewer than two distinct positions along the line exist
    (degenerate fit).
    """
    centroid, direction, normal = total_least_squares_line(points)

    # Orient the normal so a positive residual points away from the image
    # centre, per the brief's sign convention for sagitta.
    centre_to_centroid = centroid - image_centre
    if np.dot(centre_to_centroid, normal) < 0:
        normal = -normal

    centred = points - centroid
    s = centred @ direction  # position along the line
    r = centred @ normal  # signed perpendicular residual

    rms_residual = float(np.sqrt(np.mean(r**2)))

    order = np.argsort(s)
    s_sorted = s[order]
    span_length = float(s_sorted[-1] - s_sorted[0])
    if span_length <= 0:
        return None

    # Quadratic fit r(s) = a*s^2 + b*s + c. The coefficient 'a' is invariant
    # to where s=0 sits, so the sagitta formula below holds regardless of the
    # centroid not being the span midpoint.
    a, _b, _c = np.polyfit(s, r, 2)
    sagitta = float(-a * (span_length / 2.0) ** 2)

    mean_distance_from_centre = float(
        np.linalg.norm(points - image_centre, axis=1).mean()
    )

    return {
        "rms_residual_px": rms_residual,
        "sagitta_px": sagitta,
        "span_length_px": span_length,
        "mean_distance_from_centre_px": mean_distance_from_centre,
        "centroid": centroid,
        "normal": normal,
    }


def process_case(case_id: str, pack: dict, merged: dict) -> tuple[list[dict], np.ndarray, int]:
    """Return (tested rows with results, endpoint array by row, width) for one case."""
    segments = np.asarray(pack["segments_px"], dtype=float)
    width = pack["dimensions"]["width"]
    height = pack["dimensions"]["height"]
    image_centre = np.array([width / 2.0, height / 2.0])

    tested_rows = []
    for row in merged["membership"]:
        member_ids = row["member_raw_fragment_ids"]
        if len(member_ids) < MIN_MEMBERS:
            continue

        member_segments = segments[member_ids]
        points = np.vstack(
            [member_segments[:, 0:2], member_segments[:, 2:4]]
        )

        x_span = points[:, 0].max() - points[:, 0].min()
        y_span = points[:, 1].max() - points[:, 1].min()
        if x_span < MIN_SPAN_FRACTION * width and y_span < MIN_SPAN_FRACTION * height:
            continue

        result = analyse_row(points, image_centre)
        if result is None:
            continue
        result["line_id"] = row["line_id"]
        result["points"] = points
        tested_rows.append(result)

    return tested_rows, width, height


def summarise_case(case_id: str, group: str, tested_rows: list[dict], width: int) -> dict:
    n = len(tested_rows)
    if n == 0:
        return {
            "case_id": case_id,
            "group": group,
            "n_rows_tested": 0,
            "median_abs_sagitta_norm": float("nan"),
            "max_abs_sagitta_norm": float("nan"),
            "max_abs_sagitta_px": float("nan"),
            "frac_positive_sagitta": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
        }

    sagittas = np.array([r["sagitta_px"] for r in tested_rows])
    distances = np.array([r["mean_distance_from_centre_px"] for r in tested_rows])
    abs_sagitta_norm = np.abs(sagittas) / width

    frac_positive = float(np.mean(sagittas > 0))

    if n >= 3 and np.ptp(np.abs(sagittas)) > 0 and np.ptp(distances) > 0:
        rho, pval = spearmanr(np.abs(sagittas), distances)
    else:
        rho, pval = float("nan"), float("nan")

    idx_max = int(np.argmax(np.abs(sagittas)))

    return {
        "case_id": case_id,
        "group": group,
        "n_rows_tested": n,
        "median_abs_sagitta_norm": float(np.median(abs_sagitta_norm)),
        "max_abs_sagitta_norm": float(np.max(abs_sagitta_norm)),
        "max_abs_sagitta_px": float(np.abs(sagittas[idx_max])),
        "frac_positive_sagitta": frac_positive,
        "spearman_r": float(rho) if not np.isnan(rho) else float("nan"),
        "spearman_p": float(pval) if not np.isnan(pval) else float("nan"),
    }


def find_frame_image(image_dir: Path, case_id: str) -> Path | None:
    if not image_dir.exists():
        return None
    # GX images are named e.g. gxBQ_window_00_frame_00000000.png (8-digit
    # zero-padded frame number); case ids use gxBQ_window_00_frame_0.
    parts = case_id.rsplit("_frame_", 1)
    if len(parts) == 2:
        prefix, frame_num = parts
        padded = f"{prefix}_frame_{int(frame_num):08d}.png"
        candidate = image_dir / padded
        if candidate.exists():
            return candidate
    candidate = image_dir / f"{case_id}.png"
    if candidate.exists():
        return candidate
    return None


def draw_figure(case_id: str, group: str, tested_rows: list[dict], width: int, height: int, image_path: Path | None) -> None:
    fig, ax = plt.subplots(figsize=(width / 150, height / 150), dpi=150)

    if image_path is not None:
        img = cv2.imread(str(image_path))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb, extent=[0, width, height, 0])
    else:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_facecolor("white")

    for row in tested_rows:
        pts = row["points"]
        colour = "tab:red" if row["sagitta_px"] > 0 else "tab:blue"
        ax.scatter(pts[:, 0], pts[:, 1], s=8, color=colour, alpha=0.8)

    ax.set_title(f"{case_id} ({group}): red = bows away from centre, blue = bows toward centre")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{case_id}.png")
    plt.close(fig)


def main() -> None:
    all_summaries = []
    files_used = []

    for pack_info in PACKS:
        pack_data = load_gz_json(pack_info["pack"])
        files_used.append(str(pack_info["pack"].relative_to(REPO_ROOT)))
        cases_by_id = {c["id"]: c for c in pack_data["cases"]}

        for case_id in pack_info["case_ids"]:
            if case_id not in cases_by_id:
                raise RuntimeError(
                    f"case {case_id} not found in pack {pack_info['pack']}"
                )
            pack_case = cases_by_id[case_id]

            merged_path = MERGED_ROWS_DIR / f"{case_id}.json.gz"
            if not merged_path.exists():
                raise RuntimeError(f"merged-rows file missing: {merged_path}")
            merged = load_gz_json(merged_path)
            files_used.append(str(merged_path.relative_to(REPO_ROOT)))

            tested_rows, width, height = process_case(case_id, pack_case, merged)
            summary = summarise_case(case_id, pack_info["group"], tested_rows, width)
            all_summaries.append(summary)

            image_path = find_frame_image(pack_info["image_dir"], case_id)
            if image_path is not None:
                draw_figure(case_id, pack_info["group"], tested_rows, width, height, image_path)
                files_used.append(
                    f"(picture background) {image_path.relative_to(REPO_ROOT)}"
                )
            else:
                print(f"skipping picture for {case_id}: no frame image found")

    write_table(all_summaries)
    write_note(all_summaries)
    print("\nFiles referenced as inputs:")
    for f in files_used:
        print(" ", f)


def write_table(summaries: list[dict]) -> None:
    columns = [
        "case_id",
        "group",
        "n_rows_tested",
        "median_abs_sagitta_norm",
        "max_abs_sagitta_norm",
        "max_abs_sagitta_px",
        "frac_positive_sagitta",
        "spearman_r",
        "spearman_p",
    ]
    lines = [",".join(columns)]
    for s in summaries:
        lines.append(",".join(str(s[c]) for c in columns))
    (OUT_DIR / "table.csv").write_text("\n".join(lines) + "\n")


def write_note(summaries: list[dict]) -> None:
    by_group = {}
    for s in summaries:
        by_group.setdefault(s["group"], []).append(s)

    gx_max_px = max(
        (s["max_abs_sagitta_px"] for s in by_group.get("gx", []) if s["max_abs_sagitta_px"] == s["max_abs_sagitta_px"]),
        default=float("nan"),
    )
    gx_median_norm = np.median(
        [s["median_abs_sagitta_norm"] for s in by_group.get("gx", [])]
    )
    broadcast_median_norm = np.median(
        [s["median_abs_sagitta_norm"] for s in by_group.get("broadcast", [])]
    )
    amateur_median_norm = np.median(
        [s["median_abs_sagitta_norm"] for s in by_group.get("amateur", [])]
    )

    gx_frac_pos = np.mean(
        [s["frac_positive_sagitta"] for s in by_group.get("gx", [])]
    )
    amateur_frac_pos = np.mean(
        [s["frac_positive_sagitta"] for s in by_group.get("amateur", [])]
    )

    gx_corrs = [s["spearman_r"] for s in by_group.get("gx", []) if s["spearman_r"] == s["spearman_r"]]
    gx_corr_txt = f"{np.mean(gx_corrs):.2f}" if gx_corrs else "not available (too few rows or no spread)"

    control_note = (
        "The broadcast control shows a similar or larger median absolute sagitta than GX, "
        "so this test does not isolate lens distortion on the GX footage: any bow is not "
        "clearly distinguishable from the same fitting noise seen in the undistorted control."
        if broadcast_median_norm >= gx_median_norm * 0.5
        else "The broadcast control's median absolute sagitta is small compared with GX's, "
        "consistent with the GX bow being lens distortion rather than fitting noise."
    )

    table_md_lines = ["| case_id | group | n_rows | median |sagitta| (norm) | max |sagitta| (norm) | max |sagitta| (px) | frac positive | spearman r |",
                       "|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        spearman_text = f"{s['spearman_r']:.2f}" if not np.isnan(s["spearman_r"]) else "nan"
        table_md_lines.append(
            f"| {s['case_id']} | {s['group']} | {s['n_rows_tested']} | "
            f"{s['median_abs_sagitta_norm']:.5f} | {s['max_abs_sagitta_norm']:.5f} | "
            f"{s['max_abs_sagitta_px']:.2f} | {s['frac_positive_sagitta']:.2f} | "
            f"{spearman_text} |"
        )

    note = []
    note.append("# Lens distortion check on the GX footage")
    note.append("")
    note.append("## Per-case table")
    note.append("")
    note.append("See `table.csv` for the full machine-readable table. Summary:")
    note.append("")
    note.extend(table_md_lines)
    note.append("")
    note.append("## Gate: broadcast as control")
    note.append("")
    note.append(control_note)
    note.append("")
    note.append("## Answer")
    note.append("")
    note.append(
        f"GX's median normalised absolute sagitta is {gx_median_norm:.5f} of image width, "
        f"against {broadcast_median_norm:.5f} for broadcast and {amateur_median_norm:.5f} for amateur. "
        f"GX rows bow away from the image centre in {gx_frac_pos:.0%} of tested rows, "
        f"against {amateur_frac_pos:.0%} for amateur. "
        f"The Spearman correlation between absolute sagitta and distance from centre in GX averages {gx_corr_txt} across its cases. "
        f"The largest GX sagitta measured was {gx_max_px:.2f} native pixels. No fix is proposed here."
    )

    (OUT_DIR / "note.md").write_text("\n".join(note) + "\n")


if __name__ == "__main__":
    main()
