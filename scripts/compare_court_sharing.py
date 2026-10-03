"""Compare original and patched court sharing over all released videos.

Run from the repository root with ``PYTHONPATH=.:src`` and the project Python.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import evaluate_courts_fast_robust as evaluator

COHORT_SIZE = 86
BOOTSTRAP_SEED = 20261003
METHODS = ("original_extraction", "court_sharing_patched")
SCENE_KEY = ["video_id", "scene_index", "start_frame", "end_frame", "frame_index"]
RALLY_KEY = ["video_id", "set", "rally", "start_frame", "end_frame"]
BEFORE_POOL = ["before_pool_error_mean_px", "before_pool_error_max_px", "before_pool_iou"]
METRICS = {
    "representative_error_mean_px": ("px", True),
    "representative_iou": ("IoU", True),
    "representative_within_10px_pct": ("%", True),
    "fixed_main_final_within_10px_pct": ("%", True),
    "fixed_main_before_sharing_within_20px_pct": ("%", True),
    "rally_time_coverage_pct": ("%", True),
    "rally_view_court_pct": ("%", True),
    "rally_view_within_10px_pct": ("%", True),
}
POOLED = {
    "representative_within_10px_pct": ("representative_within_10px", "representative_videos", "videos"),
    "fixed_main_final_within_10px_pct": (
        "fixed_main_final_within_10px", "fixed_main_scenes", "fixed-main scenes",
    ),
    "fixed_main_before_sharing_within_20px_pct": (
        "fixed_main_before_sharing_within_20px", "fixed_main_scenes", "fixed-main scenes",
    ),
    "rally_time_coverage_pct": ("detected_rally_frames", "rally_frames", "rally frames"),
    "rally_view_court_pct": ("rally_view_courts", "rallies", "rallies"),
    "rally_view_within_10px_pct": ("rally_view_within_10px", "rallies", "rallies"),
}


def read_tables(root: Path) -> dict[str, pd.DataFrame]:
    """Read evaluator summaries from one output folder."""
    return {
        "videos": pd.read_csv(root / "per_video.csv.gz", dtype={"source_id": "string"}),
        "scenes": pd.read_csv(root / "per_scene.csv.gz"),
        "rallies": pd.read_csv(root / "rally_views" / "per_rally_view.csv.gz"),
    }


def validate_inputs(
    original: dict[str, pd.DataFrame], patched: dict[str, pd.DataFrame]
) -> list[str]:
    """Require the exact released cohort and matching scene, rally and source inputs."""
    released = {(dataset, str(source)) for dataset, source in evaluator.released_sources()}
    if len(released) != COHORT_SIZE:
        raise ValueError(f"release manifests contain {len(released)} sources, expected {COHORT_SIZE}")
    required = {
        "videos": {"video_id", "dataset", "source_id", "representative_error_mean_px", "representative_iou",
                   "rally_frames", "detected_rally_frames", "rallies", "time_coverage"},
        "scenes": {*SCENE_KEY, "is_dominant_group", "error_max_px", *BEFORE_POOL},
        "rallies": {*RALLY_KEY, "rally_frames", "court_detected", "error_mean_px"},
    }
    for label, tables in (("original", original), ("patched", patched)):
        for name, columns in required.items():
            missing = columns - set(tables[name].columns)
            if missing:
                raise ValueError(f"{label} {name} table is missing columns: {sorted(missing)}")
        videos = tables["videos"]
        sources = set(zip(videos["dataset"].astype(str), videos["source_id"].astype(str)))
        if len(videos) != COHORT_SIZE or videos["video_id"].isna().any() or videos["video_id"].duplicated().any():
            raise ValueError(f"{label} per_video must contain {COHORT_SIZE} unique video IDs")
        if sources != released:
            raise ValueError(f"{label} per_video differs from the release manifests")

    video_ids = sorted(original["videos"]["video_id"].tolist())
    if video_ids != sorted(patched["videos"]["video_id"].tolist()):
        raise ValueError("original and patched video IDs differ")
    for column in ("dataset", "source_id"):
        before = original["videos"].set_index("video_id")[column].sort_index()
        after = patched["videos"].set_index("video_id")[column].sort_index()
        pd.testing.assert_series_equal(before, after, check_dtype=False)

    for name, key in (("scenes", SCENE_KEY), ("rallies", RALLY_KEY)):
        before, after = original[name], patched[name]
        if before.duplicated(key).any() or after.duplicated(key).any():
            raise ValueError(f"{name} tables contain duplicate comparison keys")
        before = before[key].sort_values(key).reset_index(drop=True)
        after = after[key].sort_values(key).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(before, after, check_dtype=False)
        except AssertionError as error:
            raise ValueError(f"original and patched {name} intervals differ") from error

    fixed_main = original["scenes"]["is_dominant_group"]
    if fixed_main.isna().any() or not pd.api.types.is_bool_dtype(fixed_main.dtype):
        raise ValueError("original is_dominant_group must be a complete boolean column")
    before = original["scenes"].sort_values(SCENE_KEY)
    after = patched["scenes"].sort_values(SCENE_KEY)
    for column in BEFORE_POOL:
        if not np.allclose(before[column], after[column], equal_nan=True):
            raise ValueError(f"saved individual fits differ in {column}")
    return video_ids


def build_video_metrics(
    original: dict[str, pd.DataFrame], patched: dict[str, pd.DataFrame], video_ids: list[str]
) -> pd.DataFrame:
    """Build one per-video row for each method using the original fixed-main mask."""
    fixed = original["scenes"][SCENE_KEY + ["is_dominant_group"]].rename(
        columns={"is_dominant_group": "fixed_main"}
    )
    method_rows = []
    for method, tables in zip(METHODS, (original, patched), strict=True):
        videos = tables["videos"].set_index("video_id").loc[video_ids].copy()
        scenes = fixed.merge(tables["scenes"], on=SCENE_KEY, validate="one_to_one")
        main = scenes[scenes["fixed_main"]].copy()
        if main.empty:
            raise ValueError("original dominant-group population is empty")
        main["fixed_main_final_within_10px"] = main["error_max_px"] <= 10
        main["fixed_main_before_sharing_within_20px"] = main["before_pool_error_max_px"] <= 20
        scene_metrics = main.groupby("video_id").agg(
            fixed_main_scenes=("fixed_main", "size"),
            fixed_main_final_within_10px=("fixed_main_final_within_10px", "sum"),
            fixed_main_before_sharing_within_20px=("fixed_main_before_sharing_within_20px", "sum"),
        )

        rallies = tables["rallies"].copy()
        detected = rallies["court_detected"].astype(bool)
        errors = rallies["error_mean_px"]
        if not np.array_equal(detected.to_numpy(), errors.notna().to_numpy()):
            raise ValueError("rally-view court flags disagree with error availability")
        rallies["rally_view_within_10px"] = errors <= 10
        rally_metrics = rallies.groupby("video_id").agg(
            rallies=("rally", "size"),
            rally_frames=("rally_frames", "sum"),
            rally_view_courts=("court_detected", "sum"),
            rally_view_within_10px=("rally_view_within_10px", "sum"),
        )
        if not rally_metrics.reindex(video_ids)["rallies"].eq(videos["rallies"]).all():
            raise ValueError("rally-view row counts differ from per_video.rallies")
        if not rally_metrics.reindex(video_ids)["rally_frames"].eq(videos["rally_frames"]).all():
            raise ValueError("rally-view intervals differ from per_video.rally_frames")
        rally_metrics = rally_metrics[["rally_view_courts", "rally_view_within_10px"]]
        if (videos["rally_frames"] <= 0).any() or not np.allclose(
            videos["time_coverage"], videos["detected_rally_frames"] / videos["rally_frames"]
        ):
            raise ValueError("per-video rally-time coverage disagrees with its counts")

        videos = videos.join(scene_metrics).join(rally_metrics)
        videos["representative_within_10px"] = videos["representative_error_mean_px"].le(10).astype(int)
        videos["representative_videos"] = 1
        videos["fixed_main_final_within_10px_pct"] = (
            100 * videos["fixed_main_final_within_10px"] / videos["fixed_main_scenes"]
        )
        videos["fixed_main_before_sharing_within_20px_pct"] = (
            100 * videos["fixed_main_before_sharing_within_20px"] / videos["fixed_main_scenes"]
        )
        videos["representative_within_10px_pct"] = 100 * videos["representative_within_10px"]
        videos["rally_time_coverage_pct"] = 100 * videos["detected_rally_frames"] / videos["rally_frames"]
        videos["rally_view_court_pct"] = 100 * videos["rally_view_courts"] / videos["rallies"]
        videos["rally_view_within_10px_pct"] = 100 * videos["rally_view_within_10px"] / videos["rallies"]
        videos["scope"], videos["method"] = "full86", method
        method_rows.append(videos.reset_index())
    return pd.concat(method_rows, ignore_index=True)


def build_summary(metrics: pd.DataFrame, draws: np.ndarray) -> pd.DataFrame:
    """Summarise equally weighted per-video metrics and pooled counts."""
    scopes = [("full86", metrics), *[(str(name), rows) for name, rows in metrics.groupby("dataset", sort=True)]]
    rows = []
    for scope, scope_metrics in scopes:
        for method in METHODS:
            method_metrics = scope_metrics[scope_metrics["method"].eq(method)]
            for metric, (unit, interval) in METRICS.items():
                values = method_metrics[metric]
                row = {
                    "scope": scope, "method": method, "metric": metric, "unit": unit,
                    "videos": len(method_metrics), "mean": values.mean(),
                    **evaluator.tail_spread(values, 90),
                }
                if scope == "full86" and interval:
                    row |= evaluator.bootstrap_mean(values, draws)
                if metric in POOLED:
                    count_column, denominator_column, pooled_of = POOLED[metric]
                    count, denominator = int(method_metrics[count_column].sum()), int(method_metrics[denominator_column].sum())
                    row |= {
                        "pooled_count": count, "pooled_denominator": denominator,
                        "pooled_of": pooled_of, "pooled_pct": 100 * count / denominator,
                    }
                rows.append(row)
    return pd.DataFrame(rows)


def build_effects(metrics: pd.DataFrame, draws: np.ndarray) -> pd.DataFrame:
    """Summarise paired patched-minus-original differences on the shared draws."""
    before, after = [metrics[metrics["method"].eq(method)].set_index("video_id") for method in METHODS]
    rows = []
    for metric, (unit, has_interval) in METRICS.items():
        if not has_interval:
            continue
        difference = after.loc[before.index, metric] - before[metric]
        complete = difference.dropna()
        # Reading saved CSVs can introduce rounding noise in otherwise equal errors.
        equal = np.isclose(complete, 0.0, rtol=0, atol=1e-12)
        interval = evaluator.bootstrap_mean(difference, draws)
        row = {
            "before": METHODS[0], "after": METHODS[1], "metric": metric,
            "statistic": "mean_paired_difference",
            "unit": "percentage points" if unit == "%" else unit,
            "videos": len(difference), "paired_videos_with_values": len(complete),
            "estimate": interval["mean"], "ci95_low": interval["ci95_low"], "ci95_high": interval["ci95_high"],
            "video_min": complete.min(), "video_max": complete.max(),
            "videos_higher": int(((complete > 0) & ~equal).sum()),
            "videos_lower": int(((complete < 0) & ~equal).sum()),
            "videos_equal": int(equal.sum()),
        }
        if metric in POOLED:
            count_column, denominator_column, pooled_of = POOLED[metric]
            denominator = int(before[denominator_column].sum())
            if denominator != int(after[denominator_column].sum()):
                raise ValueError(f"{metric}: paired methods have different denominators")
            row |= {
                "pooled_before": int(before[count_column].sum()), "pooled_after": int(after[count_column].sum()),
                "pooled_denominator": denominator, "pooled_of": pooled_of,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True, help="original evaluator output folder")
    parser.add_argument("--patched", type=Path, required=True, help="patched evaluator output folder")
    parser.add_argument("--output", type=Path, help="output folder; defaults to --patched")
    args = parser.parse_args()
    original, patched = read_tables(args.original), read_tables(args.patched)
    video_ids = validate_inputs(original, patched)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(video_ids), size=(evaluator.BOOTSTRAP_REPLICATES, len(video_ids))
    )
    metrics = build_video_metrics(original, patched, video_ids)
    summary, effects = build_summary(metrics, draws), build_effects(metrics, draws)
    output = args.output or args.patched
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "comparison_video_metrics.csv.gz", index=False)
    summary.to_csv(output / "comparison_summary.csv.gz", index=False)
    effects.to_csv(output / "comparison_effects.csv.gz", index=False)
    counts = original["videos"]["dataset"].value_counts().sort_index().to_dict()
    print(f"Validated {len(video_ids)} released videos: {counts}")
    print(summary[summary["scope"].eq("full86")].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
