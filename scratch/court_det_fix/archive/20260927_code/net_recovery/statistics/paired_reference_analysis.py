"""Compare saved bounded net choices with independent court references."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
BOUNDED = REPO / "local_scratch/net_recovery/20260923/bounded/bounded_trial.json.gz"
MANIFEST = (
    REPO / "scratch/court_det_fix/wider_evaluation/runs/20260922/manifest.json.gz"
)
SETTINGS = ("zero", "weight_02", "primary", "weight_08", "overrun_02", "overrun_08")
THRESHOLDS_WORKING_PX = (1.0, 2.0, 4.0)
BOOTSTRAP_SEED = 20260923
BOOTSTRAP_SAMPLES = 10_000


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def quantiles(values: list[float]) -> dict:
    data = np.asarray(values, dtype=float)
    return {
        "count": len(data),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "q75": float(np.quantile(data, 0.75)),
        "q90": float(np.quantile(data, 0.9)),
        "max": float(data.max()),
    }


def project(
    homography: list, court_metres: list, native_size: list, working_size: list
) -> np.ndarray:
    points = np.asarray(court_metres, dtype=float)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    transformed = homogeneous @ np.asarray(homography, dtype=float).T
    return (
        transformed[:, :2]
        / transformed[:, 2, None]
        * np.asarray(native_size)
        / np.asarray(working_size)
    )


def reference_errors(
    case: dict, geometry: dict, reference: dict
) -> tuple[str, list[float]]:
    scale = np.asarray(case["native_size_wh"], dtype=float) / np.asarray(
        case["working_size_wh"], dtype=float
    )
    landmarks = reference.get("landmarks", [])
    if landmarks:
        prediction = project(
            geometry["homography_working"],
            [landmark["court_m"] for landmark in landmarks],
            case["native_size_wh"],
            case["working_size_wh"],
        )
        observed = np.asarray(
            [landmark["image_px"] for landmark in landmarks], dtype=float
        )
        return "visible_landmarks", np.linalg.norm(
            (prediction - observed) / scale, axis=1
        ).tolist()

    prediction = np.asarray(geometry["corners_native_px"], dtype=float)
    observed = np.asarray(reference["corners_px"], dtype=float)
    direct = np.linalg.norm((prediction - observed) / scale, axis=1)
    rotated = np.linalg.norm((prediction[[2, 3, 0, 1]] - observed) / scale, axis=1)
    distances = rotated if rotated.max() < direct.max() else direct
    visible = reference.get("corner_visible")
    if visible:
        indices = [index for index, flag in enumerate(visible) if str(flag) == "1"]
        return "visible_corners", distances[indices].tolist()
    return "all_corners", distances.tolist()


def load_references(manifest: dict) -> dict:
    references = {}
    for source_pack in sorted({item["source_pack"] for item in manifest["cases"]}):
        path = REPO / source_pack
        if not path.exists():
            path = REPO / "scratch/court_det_fix" / source_pack
        references.update(read(path).get("references", {}))
    return references


def summarise(rows: list[dict]) -> dict:
    measured = [row for row in rows if row["reference_metric"] is not None]
    result = {"frames": len(rows), "reference_frames": len(measured), "metrics": {}}
    for setting in SETTINGS:
        eligible = [row for row in measured if row["errors"][setting] is not None]
        errors = [row["errors"][setting] for row in eligible]
        result["metrics"][setting] = {
            "frames": len(eligible),
            "frame_median_working_px": quantiles([error["median"] for error in errors])
            if errors
            else None,
            "frame_max_working_px": quantiles([error["max"] for error in errors])
            if errors
            else None,
        }
        if setting == "zero":
            continue
        paired = [row for row in eligible if row["errors"]["zero"] is not None]
        deltas = [
            row["errors"][setting]["median"] - row["errors"]["zero"]["median"]
            for row in paired
        ]
        max_deltas = [
            row["errors"][setting]["max"] - row["errors"]["zero"]["max"]
            for row in paired
        ]
        result["metrics"][setting]["paired_vs_zero"] = {
            "median_error_delta_working_px": quantiles(deltas) if deltas else None,
            "max_error_delta_working_px": quantiles(max_deltas) if max_deltas else None,
            "choice_changed": sum(
                row["choices"][setting] != row["choices"]["zero"] for row in paired
            ),
            "thresholds": {
                str(threshold): {
                    "better": sum(delta < -threshold for delta in deltas),
                    "within": sum(abs(delta) <= threshold for delta in deltas),
                    "worse": sum(delta > threshold for delta in deltas),
                }
                for threshold in THRESHOLDS_WORKING_PX
            },
        }
    return result


def main() -> None:
    bounded = read(BOUNDED)
    manifest = {item["case_id"]: item for item in read(MANIFEST)["cases"]}
    references = load_references({"cases": list(manifest.values())})
    cases = bounded["cases"]
    assert len(cases) == 72 and len({case["case_id"] for case in cases}) == 71
    assert set(SETTINGS) == set(cases[0]["choices"])

    rows = []
    for case in cases:
        source = manifest[case["case_id"]]
        reference = references.get(case["case_id"], {})
        status = reference.get("reference_status", case.get("reference_status"))
        measurable = (
            bool(reference.get("landmarks") or reference.get("corners_px"))
            and status != "view_unverified"
        )
        errors = {}
        metric = None
        count = None
        for setting in SETTINGS:
            key = case["choices"][setting]
            if not measurable or key is None:
                errors[setting] = None
                continue
            metric, distances = reference_errors(
                case, case["selected_geometry"][key], reference
            )
            count = len(distances)
            errors[setting] = {
                "median": float(np.median(distances)),
                "max": float(max(distances)),
            }
        rows.append(
            {
                "case_id": case["case_id"],
                "pool": case["source_label"],
                "cohort": case["cohort"],
                "group": case["group"],
                "video": source["video"],
                "arm": case["arm"],
                "reference_status": status,
                "reference_metric": metric,
                "reference_count": count,
                "choices": case["choices"],
                "errors": errors,
            }
        )

    primary = [
        row
        for row in rows
        if row["pool"] == "am1_seeded_pool"
        or row["case_id"] != "am1_window_00_frame_54"
    ]
    frozen = [row for row in rows if row["pool"] == "frozen71"]
    assert len(primary) == len(frozen) == 71
    subsets = {
        "primary_replaced71": primary,
        "original_frozen71": frozen,
        "seeded_am1": [row for row in rows if row["pool"] == "am1_seeded_pool"],
        "primary_development": [
            row for row in primary if row["cohort"] == "development"
        ],
        "primary_withheld": [row for row in primary if row["cohort"] != "development"],
        "primary_landmarks": [
            row for row in primary if row["reference_metric"] == "visible_landmarks"
        ],
        "primary_corners": [
            row
            for row in primary
            if row["reference_metric"] in ("visible_corners", "all_corners")
        ],
    }
    for video in sorted({row["video"] for row in primary}):
        subsets[f"video:{video}"] = [row for row in primary if row["video"] == video]

    comparisons = {}
    for left, right in (
        ("weight_02", "primary"),
        ("primary", "weight_08"),
        ("weight_02", "weight_08"),
        ("overrun_02", "primary"),
        ("primary", "overrun_08"),
    ):
        differing = [
            row for row in primary if row["choices"][left] != row["choices"][right]
        ]
        measured = [
            row for row in differing if row["errors"][left] and row["errors"][right]
        ]
        deltas = [
            row["errors"][right]["median"] - row["errors"][left]["median"]
            for row in measured
        ]
        comparisons[f"{left}_to_{right}"] = {
            "different_choices": len(differing),
            "measured_different_choices": len(measured),
            "median_error_delta_on_differing": quantiles(deltas) if deltas else None,
            "case_ids": [row["case_id"] for row in differing],
            "thresholds": {
                str(threshold): {
                    "right_better": sum(delta < -threshold for delta in deltas),
                    "within": sum(abs(delta) <= threshold for delta in deltas),
                    "right_worse": sum(delta > threshold for delta in deltas),
                }
                for threshold in THRESHOLDS_WORKING_PX
            },
        }

    differing = [
        row for row in primary if len({row["choices"][arm] for arm in SETTINGS[:4]}) > 1
    ]
    missing_reference = [row for row in differing if row["reference_metric"] is None]
    measured_differences = [
        row for row in differing if row["reference_metric"] is not None
    ]
    measured_differences.sort(
        key=lambda row: max(
            row["errors"][arm]["max"] - row["errors"]["zero"]["max"]
            for arm in ("weight_02", "primary", "weight_08")
        ),
        reverse=True,
    )
    review = missing_reference + measured_differences
    source_balanced = {}
    for setting in SETTINGS[1:]:
        source_means = {}
        for name, members in subsets.items():
            if not name.startswith("video:"):
                continue
            summary = summarise(members)
            paired = summary["metrics"][setting]["paired_vs_zero"][
                "median_error_delta_working_px"
            ]
            if paired is not None:
                source_means[name.removeprefix("video:")] = paired["mean"]
        source_balanced[setting] = {
            "source_mean_paired_delta_working_px": source_means,
            "equal_source_mean_working_px": float(np.mean(list(source_means.values()))),
            "equal_source_mean_excluding_am1_working_px": float(
                np.mean(
                    [value for video, value in source_means.items() if video != "am1"]
                )
            ),
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    source_names = sorted(
        source_balanced["primary"]["source_mean_paired_delta_working_px"]
    )
    sampled_indices = rng.integers(
        len(source_names), size=(BOOTSTRAP_SAMPLES, len(source_names))
    )
    source_bootstrap = {}
    for left, right in (("weight_02", "primary"), ("primary", "weight_08")):
        differences = np.asarray(
            [
                source_balanced[right]["source_mean_paired_delta_working_px"][name]
                - source_balanced[left]["source_mean_paired_delta_working_px"][name]
                for name in source_names
            ]
        )
        interval = np.quantile(
            differences[sampled_indices].mean(axis=1), [0.025, 0.975]
        )
        source_bootstrap[f"{left}_to_{right}"] = {
            "equal_source_mean_difference_working_px": float(differences.mean()),
            "percentile_95_interval_working_px": interval.tolist(),
            "source_count": len(source_names),
        }
    report = {
        "schema": "bounded-paired-reference-analysis/1",
        "inputs": [str(BOUNDED.relative_to(REPO)), str(MANIFEST.relative_to(REPO))],
        "units": "Euclidean pixels on the saved 960-wide working image; per-frame median and maximum of reference-point errors",
        "thresholds_working_px": THRESHOLDS_WORKING_PX,
        "coverage": {
            "primary_reference_metric": dict(
                Counter(row["reference_metric"] or "excluded" for row in primary)
            ),
            "primary_exclusion_reason": dict(
                Counter(
                    "unverified_view"
                    if row["reference_status"] == "view_unverified"
                    else "no_reference"
                    if row["reference_metric"] is None
                    else "included"
                    for row in primary
                )
            ),
            "primary_abstentions": sum(
                row["choices"]["zero"] is None for row in primary
            ),
            "non_court": {
                setting: {
                    "count": sum(
                        row["reference_status"] == "non_court" for row in primary
                    ),
                    "accepted": sum(
                        row["reference_status"] == "non_court"
                        and row["choices"][setting] is not None
                        for row in primary
                    ),
                }
                for setting in SETTINGS
            },
        },
        "subsets": {name: summarise(members) for name, members in subsets.items()},
        "source_balanced": source_balanced,
        "source_cluster_bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_SAMPLES,
            "comparisons": source_bootstrap,
        },
        "comparisons": comparisons,
        "review_candidates": [row["case_id"] for row in review[:5]],
        "frames": rows,
    }
    with gzip.open(
        OUTPUT / "paired_reference_results.json.gz", "wt", encoding="utf-8"
    ) as stream:
        json.dump(report, stream, allow_nan=False)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("coverage", "comparisons", "review_candidates")
            },
            indent=2,
        )
    )
    for name in (
        "primary_replaced71",
        "primary_development",
        "primary_withheld",
        "primary_landmarks",
        "primary_corners",
        "original_frozen71",
        "seeded_am1",
    ):
        summary = report["subsets"][name]
        print(name, summary["frames"], summary["reference_frames"])
        for setting in SETTINGS:
            metric = summary["metrics"][setting]
            print(
                " ",
                setting,
                metric["frame_median_working_px"],
                metric.get("paired_vs_zero", {}).get("thresholds", {}).get("2.0"),
            )


if __name__ == "__main__":
    main()
