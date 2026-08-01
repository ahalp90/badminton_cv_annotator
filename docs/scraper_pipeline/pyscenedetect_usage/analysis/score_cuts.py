"""Score disposable PySceneDetect cut frames against ShuttleSet rally extents."""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

from annotator.calibration.scoring import _scale_base30_frames, load_gt_rallies


EXPECTED_RALLIES = {1: 113, 15: 104, 21: 75}
METRICS = ("start_gap", "end_gap", "nearest_start_dist", "nearest_end_dist")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--cuts", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()

    cuts_table = pd.read_csv(args.cuts)
    n_frames_values = cuts_table["n_frames"].dropna().unique()
    assert len(n_frames_values) == 1
    n_frames = int(n_frames_values[0])
    cut_frames = sorted(int(frame) for frame in cuts_table["cut_frame"])
    assert cut_frames == sorted(set(cut_frames))

    shots_master_path = Path(__file__).resolve().parents[4] / "training/data/shuttleset/annotations/shots_master.csv"
    gt_rallies = load_gt_rallies(pd.read_csv(shots_master_path), args.video_id)
    assert len(gt_rallies) == EXPECTED_RALLIES[args.video_id]
    assert all(0 <= frame < n_frames for rally in gt_rallies for frame in rally.stroke_frames)

    rows = []
    for rally in gt_rallies:
        first_stroke, last_stroke = rally.extent
        prev_index = bisect_right(cut_frames, first_stroke) - 1
        next_index = bisect_left(cut_frames, last_stroke)
        prev_cut = cut_frames[prev_index] if prev_index >= 0 else -1
        next_cut = cut_frames[next_index] if next_index < len(cut_frames) else -1
        rows.append({
            "set_id": rally.set_id,
            "rally": rally.rally,
            "first_stroke": first_stroke,
            "last_stroke": last_stroke,
            "prev_cut": prev_cut,
            "start_gap": first_stroke - prev_cut if prev_cut != -1 else None,
            "next_cut": next_cut,
            "end_gap": next_cut - last_stroke if next_cut != -1 else None,
            "nearest_start_dist": min(abs(cut - first_stroke) for cut in cut_frames),
            "nearest_end_dist": min(abs(cut - last_stroke) for cut in cut_frames),
            "n_cuts_inside": sum(first_stroke < cut < last_stroke for cut in cut_frames),
        })

    rally_table = pd.DataFrame(rows)
    # .gz suffix makes pandas write gzip-compressed CSV (and read it back transparently)
    rally_path = Path(f"{args.out_prefix}_rallies.csv.gz")
    summary_path = Path(f"{args.out_prefix}_summary.csv.gz")
    rally_table.to_csv(rally_path, index=False)

    summary = {"n_rallies": len(rally_table), "n_cuts": len(cut_frames)}
    for metric in METRICS:
        values = rally_table[metric].dropna().to_numpy(dtype=float)
        summary[f"{metric}_median"] = float(np.percentile(values, 50))
        summary[f"{metric}_p75"] = float(np.percentile(values, 75))
        summary[f"{metric}_p90"] = float(np.percentile(values, 90))
        summary[f"{metric}_max"] = int(values.max())

    perfect_frames = _scale_base30_frames(5, args.fps)
    ok_frames = _scale_base30_frames(10, args.fps)
    for metric in ("start_gap", "end_gap"):
        values = rally_table[metric]
        summary[f"{metric}_frac_perfect"] = float((values.notna() & (values <= perfect_frames)).mean())
        summary[f"{metric}_frac_ok"] = float((values.notna() & (values <= ok_frames)).mean())

    inside = rally_table["n_cuts_inside"]
    summary["frac_rallies_with_inside_cuts"] = float((inside > 0).mean())
    positive_inside = inside[inside > 0]
    summary["median_n_cuts_inside"] = positive_inside.median()
    pd.DataFrame([summary]).to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
