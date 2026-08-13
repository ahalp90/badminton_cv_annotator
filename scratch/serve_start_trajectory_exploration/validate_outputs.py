"""Independently recalculate the corrected serve-trajectory result bundle."""

from __future__ import annotations

import gc
import gzip
import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = RUN_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_PATH = RUN_DIR / "report.md"

PRIMARY_POPULATION = "primary_239_one_to_one"
COVERED_POPULATION = "covered_249_merge_sensitivity"
ALL_POPULATION = "all_292_end_to_end"
EXPECTED_PLOTS = {
    "anchor_alignment.png",
    "motion_evidence_and_inpaint.png",
    "server_attribution.png",
    "trend_and_jitter_diagnostics.png",
    "trend_rule_errors.png",
    "unmatched_anchor_followup.png",
}

RALLY_KEY = ["fixture", "video_id", "set_id", "rally"]
SPAN_KEY = ["fixture", "span_id"]
POINT_KEY = [*RALLY_KEY, "path_definition", "sample_index"]
PATH_KEY = [*RALLY_KEY, "path_definition"]
PATH_VARIANTS = ("recurrence_clean", "producer_original")
CONTACT_TOLERANCES_BASE30 = (5, 10, 30)

EXPECTED_FIXTURE_RALLIES = {"sset_01": 113, "sset_15": 104, "sset_21": 75}
EXPECTED_FIXTURE_SPANS = {"sset_01": 113, "sset_15": 145, "sset_21": 86}
EXPECTED_PRIMARY_RALLIES = {"sset_01": 104, "sset_15": 84, "sset_21": 51}
FIXTURE_RESOLUTION = {"sset_01": (1920.0, 1080.0), "sset_15": (1920.0, 1080.0), "sset_21": (1920.0, 1080.0)}

MIN_PATH_FRAMES = 5
MAX_FRAMES_TO_CONTACT_BASE30 = 2
MAX_LARGEST_STEP_RATIO = 4.0
MIN_TOTAL_MOVEMENT_BH = 0.25
MIN_NET_CLOSURE_BH = 0.25
MIN_CLOSING_FRACTION = 0.55
MIN_FITTED_DECREASE_BH = 0.05

METHOD_COLUMNS = {
    "old alternating fit": "baseline_server",
    "anchor player": "assume_first_contact_is_serve",
    "historical rule, recurrence mask": "recurrence_clean_historical_server",
    "0.05-BH trend rule, recurrence mask": "recurrence_clean_robust_trend_server",
    "historical rule, recurrence plus producer mask": "producer_original_historical_server",
    "0.05-BH trend rule, recurrence plus producer mask": "producer_original_robust_trend_server",
    "0.05-BH trend evidence only": "evidence_only_server",
    "0.05-BH trend then prepend unknown player": "missing_contact_refit_server",
    "0.05-BH trend then like-for-like refit": "anchor_fallback_refit_server",
    "0.05-BH trend then prepend other player": "inferred_player_refit_server",
}

CENTRAL_SERVER_PLOT_RESULTS = {
    "old alternating fit": (124, 217),
    "anchor player": (152, 239),
    "0.05-BH trend rule, recurrence mask": (163, 239),
}


@dataclass(frozen=True)
class ResultTables:
    """The six saved outputs consumed by the validator."""

    rallies: pd.DataFrame
    spans: pd.DataFrame
    path_points: pd.DataFrame
    fixed_rules: pd.DataFrame
    trend_diagnostics: pd.DataFrame
    metrics: dict[str, object]


@dataclass(frozen=True)
class RefitComparison:
    """Independently rebuilt primary-population comparison of the two prepend fallbacks."""

    total: int
    non_triggered: int
    triggered: int
    direct_correct: int
    like_for_like_correct: int
    like_for_like_known: int
    old_fit_fallback_correct: int
    old_fit_fallback_known: int
    non_triggered_anchor_correct: int
    non_triggered_old_fit_correct: int
    triggered_direct_correct: int
    triggered_refit_correct: int
    disagreements: int
    later_vote_overrides: int
    tied_refits: int
    disagreement_rows: tuple[tuple[str, ...], ...]


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate member names."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"metrics.json.gz contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject JavaScript numeric constants that are invalid JSON."""
    raise ValueError(f"metrics.json.gz contains invalid JSON constant {value}")


def _load_tables() -> ResultTables:
    """Load the complete compressed result bundle with strict JSON parsing."""
    paths = {
        name: OUTPUT_DIR / f"{name}.csv.gz"
        for name in ("rallies", "spans", "path_points", "fixed_rules", "trend_diagnostics")
    }
    metrics_path = OUTPUT_DIR / "metrics.json.gz"
    for path in (*paths.values(), metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"required output is missing: {path}")

    with gzip.open(metrics_path, "rt", encoding="utf-8") as handle:
        metrics = json.load(handle, parse_constant=_reject_json_constant, object_pairs_hook=_strict_object)
    if not isinstance(metrics, dict):
        raise TypeError("metrics.json.gz must contain one JSON object")
    return ResultTables(
        rallies=pd.read_csv(paths["rallies"]),
        spans=pd.read_csv(paths["spans"]),
        path_points=pd.read_csv(paths["path_points"]),
        fixed_rules=pd.read_csv(paths["fixed_rules"]),
        trend_diagnostics=pd.read_csv(paths["trend_diagnostics"]),
        metrics=metrics,
    )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], table: str) -> None:
    """Require the columns used for an independent calculation."""
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise AssertionError(f"{table} is missing required columns: {', '.join(missing)}")


def _assert_value(actual: object, expected: object, label: str) -> None:
    """Compare one scalar with explicit missing-value and floating-point handling."""
    actual_missing = pd.isna(actual)
    expected_missing = pd.isna(expected)
    scalar_missing = isinstance(actual_missing, (bool, np.bool_)) and isinstance(expected_missing, (bool, np.bool_))
    if scalar_missing and (actual_missing or expected_missing):
        if actual_missing and expected_missing:
            return
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")
    if isinstance(actual, (float, np.floating)) or isinstance(expected, (float, np.floating)):
        if not np.isclose(float(actual), float(expected), rtol=1e-9, atol=1e-12):
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def _assert_mapping(actual: object, expected: object, label: str) -> None:
    """Compare a nested metrics object, allowing only harmless float rounding."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise TypeError(f"{label}: expected an object, got {type(actual).__name__}")
        if actual.keys() != expected.keys():
            missing = sorted(expected.keys() - actual.keys())
            extra = sorted(actual.keys() - expected.keys())
            raise AssertionError(f"{label}: key mismatch; missing={missing}, extra={extra}")
        for key, value in expected.items():
            _assert_mapping(actual[key], value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            actual_length = len(actual) if isinstance(actual, list) else type(actual).__name__
            raise AssertionError(f"{label}: expected list length {len(expected)}, got {actual_length}")
        for index, value in enumerate(expected):
            _assert_mapping(actual[index], value, f"{label}[{index}]")
        return
    _assert_value(actual, expected, label)


def _assert_frame_equal(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str], label: str) -> None:
    """Compare two tables after deterministic key ordering."""
    if list(actual.columns) != list(expected.columns):
        raise AssertionError(f"{label} columns differ from the independent calculation")
    actual_sorted = actual.sort_values(keys).reset_index(drop=True)
    expected_sorted = expected.sort_values(keys).reset_index(drop=True)
    if len(actual_sorted) != len(expected_sorted):
        raise AssertionError(f"{label} has {len(actual_sorted)} rows; expected {len(expected_sorted)}")
    for column in actual.columns:
        actual_values = actual_sorted[column]
        expected_values = expected_sorted[column]
        if pd.api.types.is_numeric_dtype(actual_values) and pd.api.types.is_numeric_dtype(expected_values):
            if not np.allclose(actual_values, expected_values, rtol=1e-9, atol=1e-12, equal_nan=True):
                mismatch = ~np.isclose(actual_values, expected_values, rtol=1e-9, atol=1e-12, equal_nan=True)
                row_index = int(np.flatnonzero(mismatch)[0])
                raise AssertionError(
                    f"{label}.{column} differs at sorted row {row_index}: "
                    f"{actual_values.iloc[row_index]!r} != {expected_values.iloc[row_index]!r}"
                )
        elif not actual_values.fillna("<missing>").equals(expected_values.fillna("<missing>")):
            raise AssertionError(f"{label}.{column} differs from the independent calculation")


def _parse_frame_list(value: object, label: str, *, allow_empty: bool) -> tuple[int, ...]:
    """Parse one compact frame list and enforce strict ordering."""
    try:
        parsed = json.loads(str(value), parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(parsed, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in parsed):
        raise TypeError(f"{label} must be a JSON array of integers")
    if not allow_empty and not parsed:
        raise ValueError(f"{label} must not be empty")
    if any(current <= previous for previous, current in pairwise(parsed)):
        raise ValueError(f"{label} must be strictly increasing")
    return tuple(parsed)


def _optional_int(value: object, label: str) -> int | None:
    """Convert a nullable table value to an integer without truncation."""
    if pd.isna(value):
        return None
    integer = int(value)
    if float(value) != integer:
        raise ValueError(f"{label} must be an integer or missing")
    return integer


def _scaled_frame_count(value_base30: int, fps: float) -> int:
    """Scale a base-30 frame count with the producer's half-up convention."""
    return max(1, math.floor(value_base30 * fps / 30.0 + 0.5))


def _other_player(player: object) -> str | None:
    """Return the opposite court half, preserving a missing player."""
    if pd.isna(player):
        return None
    if player == "Top":
        return "Bot"
    if player == "Bot":
        return "Top"
    raise ValueError(f"unknown player label {player!r}")


def _source_half(value: object) -> str | None:
    """Return the CSV spelling of a source player label."""
    if value is None:
        return None
    text = str(getattr(value, "value", value))
    if text.lower() == "top":
        return "Top"
    if text.lower() in ("bot", "bottom"):
        return "Bot"
    raise ValueError(f"frozen source contains unknown player label {value!r}")


def _validate_source_spans(fixture_rows: pd.DataFrame, data: Any) -> None:
    """Compare the saved span table with one frozen fixture."""
    fixture = data.fixture
    if len(fixture_rows) != len(data.spans):
        raise AssertionError(
            f"{fixture.name} spans.csv.gz has {len(fixture_rows)} rows; frozen source has {len(data.spans)}"
        )
    by_span_id = fixture_rows.set_index("span_id", verify_integrity=True)
    if set(by_span_id.index) != set(range(len(data.spans))):
        raise AssertionError(f"{fixture.name} saved span ids do not match the frozen source")
    for span_id, (start_frame, end_frame) in enumerate(data.spans):
        row = by_span_id.loc[span_id]
        _assert_value(row["video_id"], fixture.video_id, f"{fixture.name} span {span_id} video id")
        _assert_value(row["start_frame"], start_frame, f"{fixture.name} span {span_id} start")
        _assert_value(row["end_frame"], end_frame, f"{fixture.name} span {span_id} end")


def _validate_source_contact_fields(row: pd.Series, data: Any, span_id: int | None, point_winner: Any) -> None:
    """Compare saved contact and baseline fields with one frozen span."""
    set_id, rally_number = row.name
    identity = f"{row['fixture']} {set_id} rally {rally_number}"
    if span_id is None:
        accepted: list[int] = []
        raw_contacts: list[Any] = []
        baseline_server = None
        n_strokes = None
    else:
        accepted = sorted(data.accepted_by_span.get(span_id, []))
        raw_contacts = data.raw_contacts_by_span.get(span_id, [])
        baseline_server = _source_half(data.annotations["fitted_first_all"][span_id])
        n_strokes = data.annotations["n_strokes_list"][span_id]

    saved_accepted = _parse_frame_list(
        row["accepted_contact_frames_json"],
        f"{identity} accepted contacts",
        allow_empty=True,
    )
    _assert_value(saved_accepted, tuple(accepted), f"{identity} accepted contact frames")
    _assert_value(row["accepted_contact_count"], len(accepted), f"{identity} accepted contact count")
    _assert_value(row["n_strokes_list"], n_strokes, f"{identity} frozen stroke count")
    _assert_value(row["baseline_server"], baseline_server, f"{identity} frozen baseline server")
    if span_id is None:
        baseline_missing = True
        baseline_wrong = baseline_correct = frozen_failure = False
    else:
        baseline_missing = baseline_server is None
        baseline_wrong = baseline_server is not None and baseline_server != row["gt_server"]
        baseline_correct = baseline_server == row["gt_server"]
        frozen_failure = baseline_server != row["gt_server"]
    _assert_value(row["baseline_missing"], baseline_missing, f"{identity} baseline missing flag")
    _assert_value(row["baseline_wrong"], baseline_wrong, f"{identity} baseline wrong flag")
    _assert_value(row["baseline_correct"], baseline_correct, f"{identity} baseline correct flag")
    _assert_value(row["frozen_server_failure"], frozen_failure, f"{identity} frozen failure flag")
    _assert_value(row["raw_candidate_count"], len(raw_contacts), f"{identity} raw candidate count")

    if not accepted:
        _assert_value(row["anchor_frame"], None, f"{identity} source anchor")
        _assert_value(row["anchor_player"], None, f"{identity} source anchor player")
        _assert_value(row["direct_contact_guesses"], None, f"{identity} direct contact guesses")
        return

    guesses = [
        point_winner.attribute_half(
            frame,
            data.track,
            data.sticky,
            data.bboxes,
            data.fixture.net_band,
        )
        for frame in accepted
    ]
    guess_text = "|".join(_source_half(guess) or "Unknown" for guess in guesses)
    anchor = accepted[0]
    _assert_value(row["anchor_frame"], anchor, f"{identity} source anchor")
    _assert_value(row["anchor_player"], _source_half(guesses[0]), f"{identity} source anchor player")
    _assert_value(row["direct_contact_guesses"], guess_text, f"{identity} direct contact guesses")
    earlier_raw = [contact for contact in raw_contacts if contact.contact_frame < anchor]
    _assert_value(row["earlier_raw_candidates"], len(earlier_raw), f"{identity} earlier raw candidates")
    _assert_value(
        row["earlier_wrist_rejections"],
        sum(contact.wrist_near is False for contact in earlier_raw),
        f"{identity} earlier wrist rejections",
    )
    _assert_value(
        row["earlier_suppressed_candidates"],
        sum(contact.suppressed is True for contact in earlier_raw),
        f"{identity} earlier suppressed candidates",
    )
    _assert_value(
        row["earlier_definitive_exclusions"],
        sum(contact.definitive_exclusion for contact in earlier_raw),
        f"{identity} earlier definitive exclusions",
    )


def _validate_source_rallies(fixture_rows: pd.DataFrame, data: Any, point_winner: Any) -> None:
    """Compare saved GT, mapping, player and frozen prediction inputs."""
    fixture = data.fixture
    by_rally = fixture_rows.set_index(["set_id", "rally"], verify_integrity=True)
    source_keys = {(rally.set_id, rally.rally) for rally in data.gt_rallies}
    if set(by_rally.index) != source_keys:
        raise AssertionError(f"{fixture.name} saved rally identities do not match the frozen source")

    for gt_rally, (boundary, span_id) in zip(data.gt_rallies, data.boundaries, strict=True):
        row = by_rally.loc[(gt_rally.set_id, gt_rally.rally)]
        identity = f"{fixture.name} {gt_rally.set_id} rally {gt_rally.rally}"
        truth = data.truth_first_second[(gt_rally.set_id, gt_rally.rally)]
        _assert_value(row["fixture"], fixture.name, f"{identity} fixture")
        _assert_value(row["video_id"], fixture.video_id, f"{identity} video id")
        _assert_value(row["fps"], fixture.fps, f"{identity} fps")
        saved_gt_frames = _parse_frame_list(row["gt_stroke_frames_json"], f"{identity} GT strokes", allow_empty=False)
        _assert_value(saved_gt_frames, gt_rally.stroke_frames, f"{identity} full GT stroke sequence")
        _assert_value(row["gt_stroke_count"], len(gt_rally.stroke_frames), f"{identity} GT stroke count")
        _assert_value(row["gt_first_frame"], truth["gt_first_frame"], f"{identity} semantic first frame")
        _assert_value(row["gt_second_frame"], truth["gt_second_frame"], f"{identity} semantic second frame")
        _assert_value(row["gt_server"], _source_half(truth["gt_server"]), f"{identity} GT server")
        _assert_value(row["gt_receiver"], _source_half(truth["gt_receiver"]), f"{identity} GT receiver")
        _assert_value(row["boundary"], boundary.value, f"{identity} frozen boundary")
        _assert_value(row["span_id"], span_id, f"{identity} frozen span id")
        _validate_source_contact_fields(row, data, span_id, point_winner)


def _validate_frozen_sources(rallies: pd.DataFrame, spans: pd.DataFrame) -> None:
    """Reload each fixture and bind the saved audit rows to frozen inputs."""
    source_root = RUN_DIR.parents[1] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    import experiment_data

    from annotator import point_winner

    shared_gt_tables = experiment_data.load_gt_tables()
    expected_fixtures = {fixture.name for fixture in experiment_data.FIXTURES}
    if set(rallies["fixture"]) != expected_fixtures or set(spans["fixture"]) != expected_fixtures:
        raise AssertionError("saved fixture identities do not match the frozen fixture set")
    for fixture in experiment_data.FIXTURES:
        data = experiment_data.load_video_data(fixture, shared_gt_tables)
        _validate_source_spans(spans[spans["fixture"].eq(fixture.name)], data)
        _validate_source_rallies(rallies[rallies["fixture"].eq(fixture.name)], data, point_winner)
        del data
        gc.collect()


def _validate_identity_and_boundaries(rallies: pd.DataFrame, spans: pd.DataFrame) -> None:
    """Rebuild the half-open rally-to-span mapping and approved populations."""
    rally_columns = [
        *RALLY_KEY,
        "fps",
        "boundary",
        "span_id",
        "predicted_span_key",
        "predicted_span_start",
        "predicted_span_end",
        "span_multiplicity",
        "primary_one_to_one",
        "population_detail",
        "gt_stroke_frames_json",
        "gt_stroke_count",
        "gt_first_frame",
        "gt_second_frame",
    ]
    _require_columns(rallies, rally_columns, "rallies.csv.gz")
    _require_columns(spans, [*SPAN_KEY, "video_id", "start_frame", "end_frame"], "spans.csv.gz")
    if (
        len(rallies) != 292
        or rallies[RALLY_KEY].duplicated().any()
        or rallies[["video_id", "set_id", "rally"]].duplicated().any()
    ):
        raise AssertionError("rallies.csv.gz must contain 292 unique rally identities")
    fixture_counts = rallies.groupby("fixture", sort=True).size().to_dict()
    if fixture_counts != EXPECTED_FIXTURE_RALLIES:
        raise AssertionError(f"rally fixture counts differ from the approved structure: {fixture_counts}")
    if spans[SPAN_KEY].duplicated().any():
        raise AssertionError("spans.csv.gz contains duplicate (fixture, span_id) keys")
    span_counts = spans.groupby("fixture", sort=True).size().to_dict()
    if span_counts != EXPECTED_FIXTURE_SPANS:
        raise AssertionError(f"predicted span counts differ from the frozen releases: {span_counts}")
    if (spans["start_frame"] >= spans["end_frame"]).any():
        raise AssertionError("every predicted span must have a non-empty half-open interval")

    span_lookup: dict[str, list[tuple[int, int, int]]] = {}
    span_video_ids: dict[str, int] = {}
    for fixture, fixture_spans in spans.groupby("fixture", sort=True):
        video_ids = fixture_spans["video_id"].unique()
        if len(video_ids) != 1:
            raise AssertionError(f"{fixture} spans have inconsistent video ids")
        span_video_ids[str(fixture)] = int(video_ids[0])
        span_lookup[str(fixture)] = [
            (int(row.span_id), int(row.start_frame), int(row.end_frame))
            for row in fixture_spans.itertuples(index=False)
        ]

    expected_boundaries: list[str] = []
    expected_span_ids: list[int | None] = []
    for row in rallies.itertuples(index=False):
        fixture = str(row.fixture)
        if fixture not in span_lookup:
            raise AssertionError(f"rally fixture has no span table: {fixture}")
        _assert_value(row.video_id, span_video_ids[fixture], f"{fixture} rally {row.rally} video id")
        frames = _parse_frame_list(row.gt_stroke_frames_json, f"{fixture} {row.set_id} rally {row.rally} GT", allow_empty=False)
        _assert_value(row.gt_stroke_count, len(frames), f"{fixture} rally {row.rally} GT stroke count")
        _assert_value(row.gt_first_frame, frames[0], f"{fixture} rally {row.rally} first GT frame")

        containing_per_stroke = [
            [span_id for span_id, start, end in span_lookup[fixture] if start <= frame < end]
            for frame in frames
        ]
        containing_ids = {span_id for containing in containing_per_stroke for span_id in containing}
        if not containing_ids:
            boundary, span_id = "missed", None
        elif len(containing_ids) == 1 and all(len(containing) == 1 for containing in containing_per_stroke):
            boundary, span_id = "covered", next(iter(containing_ids))
        else:
            boundary, span_id = "split", None
        expected_boundaries.append(boundary)
        expected_span_ids.append(span_id)
        _assert_value(row.boundary, boundary, f"{fixture} rally {row.rally} boundary")
        _assert_value(row.span_id, span_id, f"{fixture} rally {row.rally} span id")

        if span_id is None:
            expected_key = expected_start = expected_end = None
        else:
            _, expected_start, expected_end = next(span for span in span_lookup[fixture] if span[0] == span_id)
            expected_key = f"{fixture}:{span_id}"
        _assert_value(row.predicted_span_key, expected_key, f"{fixture} rally {row.rally} predicted span key")
        _assert_value(row.predicted_span_start, expected_start, f"{fixture} rally {row.rally} predicted span start")
        _assert_value(row.predicted_span_end, expected_end, f"{fixture} rally {row.rally} predicted span end")

    rebuilt = rallies.copy()
    rebuilt["_expected_boundary"] = expected_boundaries
    rebuilt["_expected_span_id"] = expected_span_ids
    covered = rebuilt[rebuilt["_expected_boundary"].eq("covered")]
    multiplicities = covered.groupby(["fixture", "_expected_span_id"]).size()
    if (
        len(covered) != 249
        or len(multiplicities) != 244
        or int((multiplicities == 1).sum()) != 239
        or int((multiplicities == 2).sum()) != 5
        or int((multiplicities > 2).sum()) != 0
    ):
        raise AssertionError("rebuilt mapping differs from the approved 292/249/244/239/five-merged structure")

    expected_multiplicity = covered.groupby(["fixture", "_expected_span_id"])["rally"].transform("size")
    rebuilt.loc[covered.index, "_expected_multiplicity"] = expected_multiplicity.to_numpy()
    rebuilt["_expected_multiplicity"] = rebuilt["_expected_multiplicity"].fillna(0).astype(int)
    expected_primary = rebuilt["_expected_boundary"].eq("covered") & rebuilt["_expected_multiplicity"].eq(1)
    primary_counts = rebuilt.loc[expected_primary].groupby("fixture").size().to_dict()
    if primary_counts != EXPECTED_PRIMARY_RALLIES:
        raise AssertionError(f"primary fixture counts differ from the approved structure: {primary_counts}")
    expected_detail = np.select(
        [expected_primary, rebuilt["_expected_boundary"].eq("covered"), rebuilt["_expected_boundary"].eq("split")],
        ["primary_239", "covered_merged_sensitivity", "end_to_end_split"],
        default="end_to_end_missed",
    )
    for index, row in rebuilt.iterrows():
        identity = f"{row['fixture']} {row['set_id']} rally {row['rally']}"
        _assert_value(row["span_multiplicity"], row["_expected_multiplicity"], f"{identity} span multiplicity")
        _assert_value(row["primary_one_to_one"], bool(expected_primary.loc[index]), f"{identity} primary flag")
        _assert_value(row["population_detail"], expected_detail[index], f"{identity} population detail")


def _alignment_label(nearest_ordinal: int, in_window_count: int) -> str:
    """Name a nearest GT stroke when the tolerance contains any stroke."""
    if in_window_count == 0:
        return "unmatched"
    if nearest_ordinal == 1:
        return "contact_1"
    if nearest_ordinal == 2:
        return "contact_2"
    return "later"


def _validate_alignments_and_sequences(rallies: pd.DataFrame) -> None:
    """Rebuild every anchor alignment and unmatched accepted-contact summary."""
    columns = [
        "accepted_contact_count",
        "accepted_contact_frames_json",
        "n_strokes_list",
        "anchor_frame",
        "anchor_gt_match",
    ]
    for tolerance in CONTACT_TOLERANCES_BASE30:
        prefix = f"anchor_tolerance_{tolerance}"
        columns.extend(
            [
                f"{prefix}_nearest_gt_ordinal",
                f"{prefix}_signed_offset_base30",
                f"{prefix}_absolute_offset_base30",
                f"{prefix}_in_window_count",
                f"{prefix}_multiple",
                f"{prefix}_label",
            ]
        )
    sequence_columns = [
        "unmatched_sequence_checked",
        "later_contacts_checked",
        "later_serve_within_tolerance",
        "later_first_return_within_tolerance",
        "first_gt_match_rank",
        "first_gt_match_ordinal",
        "first_gt_match_multiple",
        "reused_gt_ordinal",
    ]
    _require_columns(rallies, [*columns, *sequence_columns], "rallies.csv.gz")

    for row in rallies.itertuples(index=False):
        identity = f"{row.fixture} {row.set_id} rally {row.rally}"
        gt_frames = np.asarray(_parse_frame_list(row.gt_stroke_frames_json, f"{identity} GT", allow_empty=False), dtype=int)
        accepted = _parse_frame_list(row.accepted_contact_frames_json, f"{identity} accepted", allow_empty=True)
        _assert_value(row.accepted_contact_count, len(accepted), f"{identity} accepted contact count")
        if row.boundary == "covered":
            _assert_value(row.n_strokes_list, len(accepted), f"{identity} frozen stroke count")
            span_start = _optional_int(row.predicted_span_start, f"{identity} span start")
            span_end = _optional_int(row.predicted_span_end, f"{identity} span end")
            if span_start is None or span_end is None:
                raise AssertionError(f"{identity} covered row has no predicted span bounds")
            if any(not span_start <= frame < span_end for frame in accepted):
                raise AssertionError(f"{identity} accepted contact lies outside its predicted span")
        else:
            _assert_value(row.n_strokes_list, None, f"{identity} non-covered stroke count")
            if accepted:
                raise AssertionError(f"{identity} non-covered row must not reuse accepted contacts")
        anchor = _optional_int(row.anchor_frame, f"{identity} anchor")
        _assert_value(anchor, accepted[0] if accepted else None, f"{identity} earliest accepted anchor")

        if anchor is None:
            for tolerance in CONTACT_TOLERANCES_BASE30:
                prefix = f"anchor_tolerance_{tolerance}"
                _assert_value(getattr(row, f"{prefix}_nearest_gt_ordinal"), None, f"{identity} {tolerance} nearest ordinal")
                _assert_value(getattr(row, f"{prefix}_signed_offset_base30"), None, f"{identity} {tolerance} signed offset")
                _assert_value(getattr(row, f"{prefix}_absolute_offset_base30"), None, f"{identity} {tolerance} absolute offset")
                _assert_value(getattr(row, f"{prefix}_in_window_count"), 0, f"{identity} {tolerance} in-window count")
                _assert_value(getattr(row, f"{prefix}_multiple"), False, f"{identity} {tolerance} multiple flag")
                _assert_value(getattr(row, f"{prefix}_label"), "no_anchor", f"{identity} {tolerance} label")
            expected_anchor_label = "no_anchor"
        else:
            for tolerance in CONTACT_TOLERANCES_BASE30:
                prefix = f"anchor_tolerance_{tolerance}"
                offsets = anchor - gt_frames
                absolute_offsets = np.abs(offsets)
                nearest_index = int(np.argmin(absolute_offsets))
                source_tolerance = _scaled_frame_count(tolerance, float(row.fps))
                in_window_count = int(np.count_nonzero(absolute_offsets <= source_tolerance))
                signed_offset_base30 = float(offsets[nearest_index] * 30.0 / float(row.fps))
                expected = {
                    f"{prefix}_nearest_gt_ordinal": nearest_index + 1,
                    f"{prefix}_signed_offset_base30": signed_offset_base30,
                    f"{prefix}_absolute_offset_base30": abs(signed_offset_base30),
                    f"{prefix}_in_window_count": in_window_count,
                    f"{prefix}_multiple": in_window_count > 1,
                    f"{prefix}_label": _alignment_label(nearest_index + 1, in_window_count),
                }
                for column, value in expected.items():
                    _assert_value(getattr(row, column), value, f"{identity} {column}")
            expected_anchor_label = _alignment_label(
                int(np.argmin(np.abs(anchor - gt_frames))) + 1,
                int(np.count_nonzero(np.abs(anchor - gt_frames) <= _scaled_frame_count(10, float(row.fps)))),
            )
        _assert_value(row.anchor_gt_match, expected_anchor_label, f"{identity} anchor GT label")

        if anchor is not None and expected_anchor_label == "unmatched":
            tolerance = _scaled_frame_count(10, float(row.fps))
            matches_per_ordinal = np.zeros(len(gt_frames), dtype=int)
            serve_match = first_return_match = first_multiple = False
            first_rank: int | None = None
            first_ordinal: int | None = None
            for accepted_rank, contact_frame in enumerate(accepted[1:], start=2):
                absolute_offsets = np.abs(gt_frames - contact_frame)
                matching = np.flatnonzero(absolute_offsets <= tolerance)
                if len(matching) == 0:
                    continue
                matches_per_ordinal[matching] += 1
                serve_match |= bool(np.any(matching == 0))
                first_return_match |= bool(np.any(matching == 1))
                if first_rank is None:
                    first_rank = accepted_rank
                    first_ordinal = int(np.argmin(absolute_offsets)) + 1
                    first_multiple = len(matching) > 1
            expected_sequence: dict[str, object] = {
                "unmatched_sequence_checked": True,
                "later_contacts_checked": len(accepted) - 1,
                "later_serve_within_tolerance": serve_match,
                "later_first_return_within_tolerance": first_return_match,
                "first_gt_match_rank": first_rank,
                "first_gt_match_ordinal": first_ordinal,
                "first_gt_match_multiple": first_multiple,
                "reused_gt_ordinal": bool(np.any(matches_per_ordinal > 1)),
            }
        else:
            expected_sequence = {
                "unmatched_sequence_checked": False,
                "later_contacts_checked": 0,
                "later_serve_within_tolerance": False,
                "later_first_return_within_tolerance": False,
                "first_gt_match_rank": None,
                "first_gt_match_ordinal": None,
                "first_gt_match_multiple": False,
                "reused_gt_ordinal": False,
            }
        for column, value in expected_sequence.items():
            _assert_value(getattr(row, column), value, f"{identity} {column}")


def _fit_robust_trend(distances: np.ndarray) -> dict[str, float]:
    """Fit the pairwise-median distance trend over normalised path time."""
    sample_time = np.linspace(0.0, 1.0, len(distances))
    slopes = [
        (distances[end] - distances[start]) / (sample_time[end] - sample_time[start])
        for start in range(len(distances) - 1)
        for end in range(start + 1, len(distances))
    ]
    slope = float(np.median(slopes))
    intercept = float(np.median(distances - slope * sample_time))
    residuals = distances - (intercept + slope * sample_time)
    residual_rms = float(np.sqrt(np.mean(residuals**2)))
    fitted_decrease = -slope
    if residual_rms == 0.0:
        trend_to_jitter = math.copysign(math.inf, fitted_decrease) if fitted_decrease else 0.0
    else:
        trend_to_jitter = fitted_decrease / residual_rms
    return {
        "robust_slope_bh_per_path": slope,
        "robust_intercept_bh": intercept,
        "fitted_decrease_bh": fitted_decrease,
        "residual_rms_bh": residual_rms,
        "trend_to_jitter": trend_to_jitter,
    }


def _fit_path(shuttle_xy: np.ndarray) -> dict[str, float]:
    """Fit linear and quadratic path residuals from the saved samples."""
    frame_numbers = np.arange(len(shuttle_xy), dtype=float)
    linear_design = np.column_stack((frame_numbers, np.ones(len(shuttle_xy))))
    linear_coefficients, *_ = np.linalg.lstsq(linear_design, shuttle_xy, rcond=None)
    linear_residual = shuttle_xy - linear_design @ linear_coefficients
    linear_rmse = float(np.sqrt(np.mean(np.sum(linear_residual**2, axis=1))))
    if len(shuttle_xy) < 5:
        return {"linear_rmse": linear_rmse, "quadratic_rmse": math.nan, "quadratic_improvement": math.nan}
    quadratic_design = np.column_stack((frame_numbers**2, frame_numbers, np.ones(len(shuttle_xy))))
    quadratic_coefficients, *_ = np.linalg.lstsq(quadratic_design, shuttle_xy, rcond=None)
    quadratic_residual = shuttle_xy - quadratic_design @ quadratic_coefficients
    quadratic_rmse = float(np.sqrt(np.mean(np.sum(quadratic_residual**2, axis=1))))
    improvement = 0.0 if linear_rmse == 0.0 else 1.0 - quadratic_rmse / linear_rmse
    return {"linear_rmse": linear_rmse, "quadratic_rmse": quadratic_rmse, "quadratic_improvement": improvement}


def _path_measurements(points: pd.DataFrame, resolution: tuple[float, float]) -> dict[str, float]:
    """Recalculate motion and fit measurements from one saved point run."""
    distances = points["distance_bh"].to_numpy(dtype=float)
    shuttle_xy = points[["shuttle_x", "shuttle_y"]].to_numpy(dtype=float)
    heights = points["bbox_height_px"].to_numpy(dtype=float)
    distance_changes = np.diff(distances)
    step_pixels = np.linalg.norm(np.diff(shuttle_xy, axis=0) * np.asarray(resolution), axis=1)
    step_bh = step_pixels / heights[1:]
    non_zero_steps = step_bh[step_bh > 0]
    largest_step_ratio = 0.0 if len(non_zero_steps) == 0 else float(np.max(step_bh) / np.median(non_zero_steps))
    measurements = {
        "start_distance_bh": float(distances[0]),
        "end_distance_bh": float(distances[-1]),
        "net_closure_bh": float(distances[0] - distances[-1]),
        "movements_towards_player": float(np.mean(distance_changes < 0)),
        "total_movement_bh": float(np.sum(step_bh)),
        "largest_step_ratio": largest_step_ratio,
    }
    measurements.update(_fit_robust_trend(distances))
    measurements.update(_fit_path(shuttle_xy))
    return measurements


def _validate_path_points(rallies: pd.DataFrame, path_points: pd.DataFrame) -> None:
    """Validate point identities, complete sample runs, measurements and rules."""
    point_columns = [*POINT_KEY, "source_frame", "distance_bh", "shuttle_x", "shuttle_y", "bbox_height_px"]
    _require_columns(path_points, point_columns, "path_points.csv.gz")
    if path_points[POINT_KEY].duplicated().any():
        raise AssertionError("path_points.csv.gz contains duplicate identity/sample keys")
    if path_points[[*PATH_KEY, "source_frame"]].duplicated().any():
        raise AssertionError("path_points.csv.gz contains duplicate source frames within a path")
    if not path_points["path_definition"].isin(PATH_VARIANTS).all():
        raise AssertionError("path_points.csv.gz contains an unknown path definition")
    numeric = path_points[["distance_bh", "shuttle_x", "shuttle_y", "bbox_height_px"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (path_points["bbox_height_px"] <= 0).any():
        raise AssertionError("path_points.csv.gz measurements must be finite with positive bbox heights")

    rally_identities = set(map(tuple, rallies[RALLY_KEY].itertuples(index=False, name=None)))
    point_identities = set(map(tuple, path_points[RALLY_KEY].drop_duplicates().itertuples(index=False, name=None)))
    unknown_identities = point_identities.difference(rally_identities)
    if unknown_identities:
        raise AssertionError(f"path_points.csv.gz contains unknown rally identities: {sorted(unknown_identities)[:3]}")

    grouped_points = {
        key: group.sort_values("sample_index").reset_index(drop=True)
        for key, group in path_points.groupby(PATH_KEY, sort=False)
    }
    used_groups: set[tuple[object, ...]] = set()
    for row in rallies.itertuples(index=False):
        for variant in PATH_VARIANTS:
            identity = (row.fixture, row.video_id, row.set_id, row.rally, variant)
            label = f"{row.fixture} {row.set_id} rally {row.rally} {variant}"
            selected = bool(getattr(row, f"{variant}_selected_path"))
            point_group = grouped_points.get(identity)
            path_frames = int(getattr(row, f"{variant}_path_frames"))
            if not selected:
                if point_group is not None:
                    raise AssertionError(f"{label} has point samples despite an unselected path")
                _assert_value(path_frames, 0, f"{label} path frame count")
                for suffix in (
                    "path_start",
                    "path_end",
                    "frames_to_contact",
                    "start_distance_bh",
                    "end_distance_bh",
                    "net_closure_bh",
                    "movements_towards_player",
                    "total_movement_bh",
                    "largest_step_ratio",
                    "linear_rmse",
                    "quadratic_rmse",
                    "quadratic_improvement",
                    "robust_slope_bh_per_path",
                    "robust_intercept_bh",
                    "fitted_decrease_bh",
                    "residual_rms_bh",
                    "trend_to_jitter",
                ):
                    _assert_value(getattr(row, f"{variant}_{suffix}"), None, f"{label} {suffix}")
                for suffix in (
                    "path_available",
                    "path_quality_pass",
                    "common_path_eligible",
                    "historical_path_eligible",
                    "historical_incoming",
                    "robust_trend_incoming",
                ):
                    _assert_value(getattr(row, f"{variant}_{suffix}"), False, f"{label} {suffix}")
                continue

            if point_group is None:
                raise AssertionError(f"{label} is missing all point samples")
            used_groups.add(identity)
            start = _optional_int(getattr(row, f"{variant}_path_start"), f"{label} path start")
            end = _optional_int(getattr(row, f"{variant}_path_end"), f"{label} path end")
            if start is None or end is None or end <= start:
                raise AssertionError(f"{label} has an invalid half-open point interval")
            expected_count = end - start
            if path_frames != expected_count or len(point_group) != expected_count:
                raise AssertionError(
                    f"{label} has incomplete samples: interval={expected_count}, path_frames={path_frames}, rows={len(point_group)}"
                )
            if point_group["sample_index"].tolist() != list(range(expected_count)):
                raise AssertionError(f"{label} sample indexes are not contiguous from zero")
            if point_group["source_frame"].tolist() != list(range(start, end)):
                raise AssertionError(f"{label} source frames do not exactly cover [path_start, path_end)")
            frames_to_contact = _optional_int(getattr(row, f"{variant}_frames_to_contact"), f"{label} contact gap")
            anchor = _optional_int(row.anchor_frame, f"{label} anchor")
            if anchor is None or frames_to_contact != anchor - (end - 1):
                raise AssertionError(f"{label} frames-to-contact does not match its anchor and final sample")

            if expected_count < 2:
                for suffix in (
                    "start_distance_bh",
                    "end_distance_bh",
                    "net_closure_bh",
                    "movements_towards_player",
                    "total_movement_bh",
                    "largest_step_ratio",
                    "linear_rmse",
                    "quadratic_rmse",
                    "quadratic_improvement",
                    "robust_slope_bh_per_path",
                    "robust_intercept_bh",
                    "fitted_decrease_bh",
                    "residual_rms_bh",
                    "trend_to_jitter",
                ):
                    _assert_value(getattr(row, f"{variant}_{suffix}"), None, f"{label} {suffix}")
                expected_common = expected_historical = expected_historical_call = expected_robust_call = False
                expected_available = False
            else:
                measurements = _path_measurements(point_group, FIXTURE_RESOLUTION[str(row.fixture)])
                for suffix, value in measurements.items():
                    if suffix == "trend_to_jitter":
                        continue
                    _assert_value(getattr(row, f"{variant}_{suffix}"), value, f"{label} {suffix}")
                saved_decrease = float(getattr(row, f"{variant}_fitted_decrease_bh"))
                saved_residual_rms = float(getattr(row, f"{variant}_residual_rms_bh"))
                if saved_residual_rms == 0.0:
                    expected_ratio = math.copysign(math.inf, saved_decrease) if saved_decrease else 0.0
                else:
                    expected_ratio = saved_decrease / saved_residual_rms
                _assert_value(
                    getattr(row, f"{variant}_trend_to_jitter"),
                    expected_ratio,
                    f"{label} trend_to_jitter",
                )
                maximum_gap = _scaled_frame_count(MAX_FRAMES_TO_CONTACT_BASE30, float(row.fps))
                expected_available = expected_count >= MIN_PATH_FRAMES and frames_to_contact <= maximum_gap
                expected_common = expected_available and measurements["largest_step_ratio"] <= MAX_LARGEST_STEP_RATIO
                expected_historical = expected_common and measurements["total_movement_bh"] >= MIN_TOTAL_MOVEMENT_BH
                expected_historical_call = (
                    expected_historical
                    and measurements["net_closure_bh"] >= MIN_NET_CLOSURE_BH
                    and measurements["movements_towards_player"] >= MIN_CLOSING_FRACTION
                )
                expected_robust_call = expected_common and measurements["fitted_decrease_bh"] >= MIN_FITTED_DECREASE_BH
            decisions = {
                "path_available": expected_available,
                "path_quality_pass": expected_historical,
                "common_path_eligible": expected_common,
                "historical_path_eligible": expected_historical,
                "historical_incoming": expected_historical_call,
                "robust_trend_incoming": expected_robust_call,
            }
            for suffix, value in decisions.items():
                _assert_value(getattr(row, f"{variant}_{suffix}"), value, f"{label} {suffix}")

    unused_groups = set(grouped_points).difference(used_groups)
    if unused_groups:
        raise AssertionError(f"path_points.csv.gz has unclaimed point groups: {sorted(unused_groups)[:3]}")


def _fit_alternation(guesses: list[str | None]) -> tuple[str | None, int, int]:
    """Fit the two alternating phases using plain parity arithmetic."""
    scores: dict[str, int] = {}
    final_index = len(guesses) - 1
    for final_player in ("Top", "Bot"):
        score = 0
        for contact_index, guess in enumerate(guesses):
            assigned = final_player if (final_index - contact_index) % 2 == 0 else _other_player(final_player)
            if guess is not None and guess == assigned:
                score += 1
        scores[final_player] = score
    final = None if scores["Top"] == scores["Bot"] else max(scores, key=scores.get)
    return final, scores["Top"], scores["Bot"]


def _first_player(final_player: str | None, count: int) -> str | None:
    """Return the first player implied by a fitted final player and count."""
    if final_player is None:
        return None
    return final_player if (count - 1) % 2 == 0 else _other_player(final_player)


def _parse_guesses(value: object, label: str) -> list[str | None]:
    """Parse the compact direct-contact player sequence."""
    if pd.isna(value) or value == "":
        return []
    guesses: list[str | None] = []
    for item in str(value).split("|"):
        if item not in ("Top", "Bot", "Unknown"):
            raise ValueError(f"{label} contains unknown direct-contact label {item!r}")
        guesses.append(None if item == "Unknown" else item)
    return guesses


def _validate_server_columns(rallies: pd.DataFrame) -> RefitComparison:
    """Rebuild fixed-rule server answers and both prepend fallback policies."""
    required = [
        "anchor_player",
        "gt_server",
        "baseline_correct",
        "baseline_missing",
        "baseline_wrong",
        "frozen_server_failure",
        "direct_contact_guesses",
        "direct_fit_final",
        "direct_fit_first",
    ]
    required.extend(METHOD_COLUMNS.values())
    required.extend(f"{column}_correct" for column in METHOD_COLUMNS.values() if column != "baseline_server")
    _require_columns(rallies, required, "rallies.csv.gz")
    comparison_counts = {
        "total": 0,
        "non_triggered": 0,
        "triggered": 0,
        "direct_correct": 0,
        "like_for_like_correct": 0,
        "like_for_like_known": 0,
        "old_fit_fallback_correct": 0,
        "old_fit_fallback_known": 0,
        "non_triggered_anchor_correct": 0,
        "non_triggered_old_fit_correct": 0,
        "triggered_direct_correct": 0,
        "triggered_refit_correct": 0,
        "disagreements": 0,
        "later_vote_overrides": 0,
        "tied_refits": 0,
    }
    disagreement_rows: list[tuple[str, ...]] = []
    for row in rallies.itertuples(index=False):
        identity = f"{row.fixture} {row.set_id} rally {row.rally}"
        anchor_player = None if pd.isna(row.anchor_player) else str(row.anchor_player)
        _other_player(anchor_player)
        gt_server = str(row.gt_server)
        baseline_server = None if pd.isna(row.baseline_server) else str(row.baseline_server)
        _assert_value(row.baseline_correct, baseline_server == gt_server, f"{identity} baseline correct")
        _assert_value(row.baseline_missing, baseline_server is None, f"{identity} baseline missing")
        _assert_value(
            row.baseline_wrong,
            baseline_server is not None and baseline_server != gt_server,
            f"{identity} baseline wrong",
        )
        _assert_value(
            row.frozen_server_failure,
            row.boundary == "covered" and baseline_server != gt_server,
            f"{identity} frozen server failure",
        )
        guesses = _parse_guesses(row.direct_contact_guesses, identity)
        natural_final, top_score, bot_score = _fit_alternation(guesses)
        natural_first = _first_player(natural_final, len(guesses)) if guesses else None
        _assert_value(row.direct_fit_final, natural_final, f"{identity} direct final fit")
        _assert_value(row.direct_fit_first, natural_first, f"{identity} direct first fit")
        _assert_value(row.direct_fit_top_score, top_score, f"{identity} direct Top score")
        _assert_value(row.direct_fit_bot_score, bot_score, f"{identity} direct Bot score")
        _assert_value(row.direct_fit_margin, abs(top_score - bot_score), f"{identity} direct fit margin")

        expected_predictions: dict[str, str | None] = {"assume_first_contact_is_serve": anchor_player}
        for variant in PATH_VARIANTS:
            for rule in ("historical", "robust_trend"):
                incoming = bool(getattr(row, f"{variant}_{rule}_incoming"))
                expected_predictions[f"{variant}_{rule}_server"] = (
                    _other_player(anchor_player) if incoming else anchor_player
                )
        main_incoming = bool(row.recurrence_clean_robust_trend_incoming)
        main_server = _other_player(anchor_player) if main_incoming else anchor_player
        expected_predictions["motion_rule_server"] = main_server
        expected_predictions["evidence_only_server"] = (
            main_server if bool(row.recurrence_clean_common_path_eligible) else None
        )
        if main_incoming and anchor_player is not None:
            parity_final, _, _ = _fit_alternation([None, *guesses])
            labelled_final, _, _ = _fit_alternation([_other_player(anchor_player), *guesses])
            parity_first = _first_player(parity_final, len(guesses) + 1)
            labelled_first = _first_player(labelled_final, len(guesses) + 1)
            anchor_fallback_first = anchor_player if labelled_first is None else labelled_first
            final_changed = labelled_final != natural_final
        else:
            parity_first = labelled_first = natural_first
            anchor_fallback_first = anchor_player
            labelled_final = None
            final_changed = False
        expected_predictions["missing_contact_refit_server"] = parity_first
        expected_predictions["anchor_fallback_refit_server"] = anchor_fallback_first
        expected_predictions["inferred_player_refit_server"] = labelled_first
        _assert_value(row.incoming_motion_found, main_incoming, f"{identity} main incoming flag")
        _assert_value(row.inferred_player_vote_changed_final_fit, final_changed, f"{identity} changed-final flag")
        for column, prediction in expected_predictions.items():
            _assert_value(getattr(row, column), prediction, f"{identity} {column}")
            _assert_value(getattr(row, f"{column}_correct"), prediction == gt_server, f"{identity} {column} correctness")

        if not bool(row.primary_one_to_one):
            continue
        comparison_counts["total"] += 1
        comparison_counts["direct_correct"] += int(main_server == gt_server)
        comparison_counts["like_for_like_correct"] += int(anchor_fallback_first == gt_server)
        comparison_counts["like_for_like_known"] += int(anchor_fallback_first is not None)
        comparison_counts["old_fit_fallback_correct"] += int(labelled_first == gt_server)
        comparison_counts["old_fit_fallback_known"] += int(labelled_first is not None)
        if not main_incoming:
            comparison_counts["non_triggered"] += 1
            comparison_counts["non_triggered_anchor_correct"] += int(anchor_player == gt_server)
            comparison_counts["non_triggered_old_fit_correct"] += int(natural_first == gt_server)
            continue

        comparison_counts["triggered"] += 1
        comparison_counts["triggered_direct_correct"] += int(main_server == gt_server)
        comparison_counts["triggered_refit_correct"] += int(anchor_fallback_first == gt_server)
        if main_server == anchor_fallback_first:
            continue
        comparison_counts["disagreements"] += 1
        if main_server != gt_server or anchor_fallback_first == gt_server:
            raise AssertionError(f"{identity} has an unexpected direct-versus-refit disagreement outcome")
        if labelled_final is None:
            comparison_counts["tied_refits"] += 1
            behaviour = "Ties; retain earliest-contact fallback"
        else:
            comparison_counts["later_vote_overrides"] += 1
            behaviour = f"Later votes override to {labelled_first}"
        disagreement_rows.append(
            (
                f"{row.fixture} {row.set_id} rally {int(row.rally)}",
                behaviour,
                str(main_server),
                str(anchor_fallback_first),
                gt_server,
            )
        )

    comparison = RefitComparison(**comparison_counts, disagreement_rows=tuple(disagreement_rows))
    checked_counts = (
        comparison.total,
        comparison.non_triggered,
        comparison.triggered,
        comparison.direct_correct,
        comparison.like_for_like_correct,
        comparison.like_for_like_known,
        comparison.old_fit_fallback_correct,
        comparison.old_fit_fallback_known,
        comparison.disagreements,
        comparison.later_vote_overrides,
        comparison.tied_refits,
    )
    if checked_counts != (239, 224, 15, 163, 159, 239, 127, 217, 4, 2, 2):
        raise AssertionError(
            "independent like-for-like refit rebuild differs from the checked "
            "239/224/15 population, 163/159/127 score comparison, answer counts or 4/2/2 disagreement outcome"
        )
    if comparison.non_triggered_anchor_correct - comparison.non_triggered_old_fit_correct != 32:
        raise AssertionError("non-triggered anchor fallback does not explain 32 additional correct calls")
    if comparison.triggered_direct_correct - comparison.triggered_refit_correct != 4:
        raise AssertionError("triggered direct inference does not explain four additional correct calls")
    if comparison.direct_correct - comparison.old_fit_fallback_correct != 32 + 4:
        raise AssertionError("163-versus-127 gap does not decompose into 32 fallback and four refit calls")
    return comparison


def _binary_rule_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, int | float]:
    """Return confusion counts and scores for one incoming-motion rule."""
    true_positive = int(np.count_nonzero(predicted & truth))
    false_positive = int(np.count_nonzero(predicted & ~truth))
    false_negative = int(np.count_nonzero(~predicted & truth))
    true_negative = int(np.count_nonzero(~predicted & ~truth))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _build_fixed_rules(rallies: pd.DataFrame) -> pd.DataFrame:
    """Rebuild all sixteen global and per-fixture fixed-rule rows."""
    truth_rows = rallies[
        rallies["primary_one_to_one"].astype(bool)
        & rallies["anchor_tolerance_10_label"].isin(["contact_1", "contact_2"])
        & rallies["anchor_tolerance_10_in_window_count"].eq(1)
    ]
    scopes: list[tuple[str, pd.DataFrame]] = [("global", truth_rows)]
    scopes.extend((str(fixture), group) for fixture, group in truth_rows.groupby("fixture", sort=True))
    rows: list[dict[str, object]] = []
    for scope, frame in scopes:
        truth = frame["anchor_tolerance_10_label"].eq("contact_2").to_numpy()
        for variant in PATH_VARIANTS:
            for rule in ("historical", "robust_trend"):
                predicted = frame[f"{variant}_{rule}_incoming"].astype(bool).to_numpy()
                eligibility = "historical_path_eligible" if rule == "historical" else "common_path_eligible"
                rows.append(
                    {
                        "scope": scope,
                        "population": "primary_239_unique_tolerance_10_truth",
                        "path_definition": variant,
                        "rule": rule,
                        "n_truth": len(frame),
                        "gt_serves": int(np.count_nonzero(~truth)),
                        "gt_first_returns": int(np.count_nonzero(truth)),
                        "common_paths_eligible": int(frame[f"{variant}_common_path_eligible"].astype(bool).sum()),
                        "rule_paths_eligible": int(frame[f"{variant}_{eligibility}"].astype(bool).sum()),
                        "incoming_calls": int(np.count_nonzero(predicted)),
                        **_binary_rule_metrics(truth, predicted),
                    }
                )
    return pd.DataFrame(rows)


def _build_trend_diagnostics(rallies: pd.DataFrame) -> pd.DataFrame:
    """Rebuild every continuous trend-diagnostic row."""
    truth = rallies[
        rallies["primary_one_to_one"].astype(bool)
        & rallies["anchor_tolerance_10_label"].isin(["contact_1", "contact_2"])
        & rallies["anchor_tolerance_10_in_window_count"].eq(1)
    ]
    rows: list[dict[str, object]] = []
    for rally in truth.itertuples(index=False):
        is_first_return = rally.anchor_tolerance_10_label == "contact_2"
        for variant in PATH_VARIANTS:
            incoming = bool(getattr(rally, f"{variant}_robust_trend_incoming"))
            rows.append(
                {
                    "fixture": rally.fixture,
                    "video_id": int(rally.video_id),
                    "set_id": rally.set_id,
                    "rally": int(rally.rally),
                    "path_definition": variant,
                    "gt_anchor_identity": "first_return" if is_first_return else "serve",
                    "selected_path": bool(getattr(rally, f"{variant}_selected_path")),
                    "common_path_eligible": bool(getattr(rally, f"{variant}_common_path_eligible")),
                    "path_frames": int(getattr(rally, f"{variant}_path_frames")),
                    "fitted_decrease_bh": getattr(rally, f"{variant}_fitted_decrease_bh"),
                    "residual_rms_bh": getattr(rally, f"{variant}_residual_rms_bh"),
                    "trend_to_jitter": getattr(rally, f"{variant}_trend_to_jitter"),
                    "incoming_call": incoming,
                    "call_correct": incoming == is_first_return,
                }
            )
    return pd.DataFrame(rows)


def _global_and_by_fixture(frame: pd.DataFrame, summarise: Callable[[pd.DataFrame], object]) -> dict[str, object]:
    """Apply one metric summary globally and to each fixture."""
    by_fixture = {str(fixture): summarise(group) for fixture, group in frame.groupby("fixture", sort=True)}
    return {"global": summarise(frame), "by_fixture": by_fixture}


def _alignment_counts(frame: pd.DataFrame) -> dict[str, object]:
    """Count saved alignment states after their row values have been rebuilt."""
    result: dict[str, object] = {}
    for tolerance in CONTACT_TOLERANCES_BASE30:
        prefix = f"anchor_tolerance_{tolerance}"
        labels = frame[f"{prefix}_label"].value_counts(dropna=False)
        result[str(tolerance)] = {
            "n": len(frame),
            "labels": {str(label): int(count) for label, count in labels.items()},
            "multiple": int(frame[f"{prefix}_multiple"].astype(bool).sum()),
            "unique_contact_1": int((frame[f"{prefix}_label"].eq("contact_1") & frame[f"{prefix}_in_window_count"].eq(1)).sum()),
            "unique_contact_2": int((frame[f"{prefix}_label"].eq("contact_2") & frame[f"{prefix}_in_window_count"].eq(1)).sum()),
        }
    return result


def _path_counts(frame: pd.DataFrame) -> dict[str, object]:
    """Count saved path and rule states after their row values have been rebuilt."""
    result: dict[str, object] = {}
    for variant in PATH_VARIANTS:
        result[variant] = {
            "n": len(frame),
            "anchors": int(frame["anchor_frame"].notna().sum()),
            "anchors_with_player": int(frame["anchor_player"].notna().sum()),
            "selected_paths": int(frame[f"{variant}_selected_path"].astype(bool).sum()),
            "path_available": int(frame[f"{variant}_path_available"].astype(bool).sum()),
            "common_path_eligible": int(frame[f"{variant}_common_path_eligible"].astype(bool).sum()),
            "historical_path_eligible": int(frame[f"{variant}_historical_path_eligible"].astype(bool).sum()),
            "historical_incoming": int(frame[f"{variant}_historical_incoming"].astype(bool).sum()),
            "robust_trend_incoming": int(frame[f"{variant}_robust_trend_incoming"].astype(bool).sum()),
        }
    return result


def _sequence_counts(frame: pd.DataFrame) -> dict[str, object]:
    """Count independently rebuilt later-contact outcomes."""
    unmatched = frame[frame["anchor_frame"].notna() & frame["anchor_tolerance_10_label"].eq("unmatched")]
    serve = unmatched["later_serve_within_tolerance"].astype(bool)
    first_return = unmatched["later_first_return_within_tolerance"].astype(bool)
    any_match = unmatched["first_gt_match_rank"].notna()
    rank_counts = unmatched["first_gt_match_rank"].dropna().astype(int).value_counts().sort_index()
    return {
        "anchors_unmatched_at_tolerance_10": len(unmatched),
        "sequence_checked": int(unmatched["unmatched_sequence_checked"].astype(bool).sum()),
        "later_serve_match": int(serve.sum()),
        "no_later_serve_but_first_return_match": int((~serve & first_return).sum()),
        "other_later_gt_match": int((~serve & ~first_return & any_match).sum()),
        "no_later_gt_match": int((~any_match).sum()),
        "first_gt_match_rank": {str(rank): int(count) for rank, count in rank_counts.items()},
        "first_match_multiple": int(unmatched["first_gt_match_multiple"].astype(bool).sum()),
        "reused_gt_ordinal": int(unmatched["reused_gt_ordinal"].astype(bool).sum()),
    }


def _class_metrics(truth: np.ndarray, predictions: np.ndarray, label: str) -> dict[str, int | float]:
    """Calculate precision, recall and F1 for one server class."""
    true_positive = int(np.count_nonzero((predictions == label) & (truth == label)))
    false_positive = int(np.count_nonzero((predictions == label) & (truth != label)))
    false_negative = int(np.count_nonzero((predictions != label) & (truth == label)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "support": int(np.count_nonzero(truth == label))}


def _server_scores(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Recalculate every server method, including per-class scores."""
    truth = frame["gt_server"].astype(str).to_numpy()
    result: dict[str, dict[str, object]] = {}
    for method, column in METHOD_COLUMNS.items():
        predictions = frame[column].fillna("Unknown").astype(str).to_numpy()
        top = _class_metrics(truth, predictions, "Top")
        bot = _class_metrics(truth, predictions, "Bot")
        result[method] = {
            "n": len(frame),
            "known": int(np.count_nonzero(predictions != "Unknown")),
            "correct": int(np.count_nonzero(predictions == truth)),
            "accuracy": float(np.mean(predictions == truth)),
            "macro_f1": float((top["f1"] + bot["f1"]) / 2.0),
            "top": top,
            "bot": bot,
        }
    return result


def _build_metrics(rallies: pd.DataFrame, fixed_rules: pd.DataFrame) -> dict[str, object]:
    """Rebuild every population, alignment, path, sequence and server metric."""
    populations = {
        "all_292_end_to_end": rallies,
        "covered_249_merge_sensitivity": rallies[rallies["boundary"].eq("covered")],
        "primary_239_one_to_one": rallies[rallies["primary_one_to_one"].astype(bool)],
    }
    population_counts = {
        name: {
            "global": len(frame),
            "by_fixture": {str(fixture): len(group) for fixture, group in frame.groupby("fixture", sort=True)},
        }
        for name, frame in populations.items()
    }
    fixed_rule_records = json.loads(fixed_rules.to_json(orient="records"))
    return {
        "question": (
            "Does the shuttle show a clear approach towards the contact player beyond ordinary "
            "track wobble, and what does that imply for anchor and server attribution?"
        ),
        "population_counts": population_counts,
        "rules": {
            "historical": {
                "minimum_path_frames": MIN_PATH_FRAMES,
                "maximum_frames_to_contact_base30": MAX_FRAMES_TO_CONTACT_BASE30,
                "maximum_largest_step_ratio": MAX_LARGEST_STEP_RATIO,
                "minimum_total_movement_bh": MIN_TOTAL_MOVEMENT_BH,
                "minimum_net_closure_bh": MIN_NET_CLOSURE_BH,
                "minimum_closing_fraction": MIN_CLOSING_FRACTION,
                "provenance": "introduced and selected within the historical analysis",
            },
            "robust_trend": {
                "minimum_path_frames": MIN_PATH_FRAMES,
                "maximum_frames_to_contact_base30": MAX_FRAMES_TO_CONTACT_BASE30,
                "maximum_largest_step_ratio": MAX_LARGEST_STEP_RATIO,
                "minimum_fitted_decrease_bh": MIN_FITTED_DECREASE_BH,
                "provenance": "engineering judgement fixed before corrected scoring",
                "residual_rms_and_trend_to_jitter_are_diagnostic_only": True,
            },
        },
        "alignment": {name: _global_and_by_fixture(frame, _alignment_counts) for name, frame in populations.items()},
        "path_funnel": {name: _global_and_by_fixture(frame, _path_counts) for name, frame in populations.items()},
        "unmatched_anchor_sequences": _global_and_by_fixture(populations["primary_239_one_to_one"], _sequence_counts),
        "fixed_rule_results": fixed_rule_records,
        "server_scores": {
            name: _global_and_by_fixture(frame, _server_scores) for name, frame in populations.items()
        },
    }


def _parse_markdown_row(line: str) -> tuple[str, ...]:
    """Return stripped cells from one pipe-delimited Markdown row."""
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def _parse_markdown_tables(report: str) -> dict[tuple[str, ...], list[list[tuple[str, ...]]]]:
    """Extract ordinary pipe tables, retaining repeated table headings."""
    lines = report.splitlines()
    tables: dict[tuple[str, ...], list[list[tuple[str, ...]]]] = {}
    line_index = 0
    while line_index + 1 < len(lines):
        if not lines[line_index].startswith("|") or not lines[line_index + 1].startswith("|"):
            line_index += 1
            continue
        header = _parse_markdown_row(lines[line_index])
        separator = _parse_markdown_row(lines[line_index + 1])
        valid_separator = len(separator) == len(header)
        for cell in separator:
            if "-" not in cell or not set(cell).issubset({"-", ":"}):
                valid_separator = False
                break
        if not valid_separator:
            line_index += 1
            continue
        rows: list[tuple[str, ...]] = []
        line_index += 2
        while line_index < len(lines) and lines[line_index].startswith("|"):
            row = _parse_markdown_row(lines[line_index])
            if len(row) != len(header):
                raise AssertionError(f"report table {header[0]!r} contains a malformed row")
            rows.append(row)
            line_index += 1
        tables.setdefault(header, []).append(rows)
    return tables


def _report_table(
    tables: dict[tuple[str, ...], list[list[tuple[str, ...]]]],
    header: tuple[str, ...],
    *,
    occurrence: int = 0,
) -> list[tuple[str, ...]]:
    """Return one named report table with a useful missing-table error."""
    matches = tables.get(header, [])
    if occurrence >= len(matches):
        raise AssertionError(f"report is missing table {header[0]!r} occurrence {occurrence + 1}")
    return matches[occurrence]


def _validate_report_populations(
    tables: dict[tuple[str, ...], list[list[tuple[str, ...]]]],
    metrics: dict[str, object],
) -> None:
    """Bind the three population meanings and fixture counts to rebuilt metrics."""
    header = ("Rally group", "All videos", "sset_01", "sset_15", "sset_21", "What it is used for")
    populations: Any = metrics["population_counts"]
    definitions = (
        (ALL_POPULATION, "All ground-truth (GT) rallies", "End-to-end view, including segmentation failures"),
        (
            COVERED_POPULATION,
            "Covered rallies",
            "Check how results change under the current COVERED definition, including merged rallies",
        ),
        (PRIMARY_POPULATION, "One-to-one rallies", "Analyses that need one predicted rally for each GT rally"),
    )
    expected: list[tuple[str, ...]] = []
    for population, label, meaning in definitions:
        values = populations[population]
        by_fixture = values["by_fixture"]
        expected.append(
            (
                label,
                str(values["global"]),
                str(by_fixture["sset_01"]),
                str(by_fixture["sset_15"]),
                str(by_fixture["sset_21"]),
                meaning,
            )
        )
    _assert_value(_report_table(tables, header), expected, "report population table")


def _validate_report_alignment_and_sequences(
    report: str,
    tables: dict[tuple[str, ...], list[list[tuple[str, ...]]]],
    metrics: dict[str, object],
) -> None:
    """Bind primary alignment, ambiguity and later-contact outcomes."""
    alignment_header = (
        "Tolerance",
        "GT serve",
        "GT first return",
        "Later GT stroke",
        "No GT stroke in window",
        "More than one GT stroke in window",
    )
    alignment: Any = metrics["alignment"][PRIMARY_POPULATION]["global"]
    alignment_rows: list[tuple[str, ...]] = []
    for tolerance in ("5", "10", "30"):
        values = alignment[tolerance]
        labels = values["labels"]
        alignment_rows.append(
            (
                f"±{tolerance}",
                str(labels.get("contact_1", 0)),
                str(labels.get("contact_2", 0)),
                str(labels.get("later", 0)),
                str(labels.get("unmatched", 0)),
                str(values["multiple"]),
            )
        )
    _assert_value(_report_table(tables, alignment_header), alignment_rows, "report primary alignment table")

    sequence_header = (
        "Video",
        "Unmatched anchors",
        "Later contact matches serve",
        "No serve match, but return matches",
        "First match is another GT stroke",
        "No later GT match",
    )
    sequence: Any = metrics["unmatched_anchor_sequences"]
    sequence_rows: list[tuple[str, ...]] = []
    for label, values in [("All", sequence["global"]), *sequence["by_fixture"].items()]:
        sequence_rows.append(
            (
                label,
                str(values["anchors_unmatched_at_tolerance_10"]),
                str(values["later_serve_match"]),
                str(values["no_later_serve_but_first_return_match"]),
                str(values["other_later_gt_match"]),
                str(values["no_later_gt_match"]),
            )
        )
    _assert_value(_report_table(tables, sequence_header), sequence_rows, "report unmatched-sequence table")
    rank_counts = sequence["global"]["first_gt_match_rank"]
    later_rank = sum(count for rank, count in rank_counts.items() if int(rank) >= 5)
    rank_statement = (
        f"The first later match appears at contact rank 2 in {rank_counts.get('2', 0)} rallies, "
        f"rank 3 in {rank_counts.get('3', 0)}, rank 4 in {rank_counts.get('4', 0)}, "
        f"and rank 5 or later in {later_rank}."
    )
    if rank_statement not in report:
        raise AssertionError("report later-contact section does not match the rebuilt first-match ranks")


def _validate_report_paths_and_rules(
    tables: dict[tuple[str, ...], list[list[tuple[str, ...]]]],
    metrics: dict[str, object],
    fixed_rules: pd.DataFrame,
) -> None:
    """Bind primary evidence availability and all four fixed comparisons."""
    path_header = (
        "Track source check",
        "Rallies",
        "Continuous run selected",
        "At least 5 points and close enough to contact",
        "Passes the shared jump check",
        "0.05-BH incoming calls",
    )
    funnel: Any = metrics["path_funnel"][PRIMARY_POPULATION]["global"]
    labels = {
        "recurrence_clean": "Exclude recurrence-flagged points",
        "producer_original": "Also exclude producer-marked inpainted points",
    }
    path_rows: list[tuple[str, ...]] = []
    for variant, label in labels.items():
        path_rows.append(
            (
                label,
                str(funnel[variant]["n"]),
                str(funnel[variant]["selected_paths"]),
                str(funnel[variant]["path_available"]),
                str(funnel[variant]["common_path_eligible"]),
                str(funnel[variant]["robust_trend_incoming"]),
            )
        )
    _assert_value(_report_table(tables, path_header), path_rows, "report primary path table")

    inpaint_header = (
        "Track source check",
        "Labelled paths with usable motion",
        "Correct return calls",
        "False return calls",
        "Returns missed",
    )
    global_rules = fixed_rules[fixed_rules["scope"].eq("global")]
    inpaint_rows: list[tuple[str, ...]] = []
    for variant, label in labels.items():
        row = global_rules[
            global_rules["path_definition"].eq(variant)
            & global_rules["rule"].eq("robust_trend")
        ].iloc[0]
        inpaint_rows.append(
            (
                label,
                f"{int(row['rule_paths_eligible'])}/{int(row['n_truth'])}",
                f"{int(row['tp'])}/{int(row['gt_first_returns'])}",
                f"{int(row['fp'])}/{int(row['gt_serves'])}",
                f"{int(row['fn'])}/{int(row['gt_first_returns'])}",
            )
        )
    _assert_value(_report_table(tables, inpaint_header), inpaint_rows, "report fixed 0.05-BH inpaint table")

    rule_header = (
        "Fixed comparison",
        "Paths eligible for this rule",
        "Correct return calls",
        "False return calls",
        "Returns missed",
        "Precision",
        "Recall",
    )
    rule_labels = {
        ("recurrence_clean", "historical"): "Historical absolute-closure rule; recurrence check",
        ("recurrence_clean", "robust_trend"): "0.05-BH trend rule; recurrence check",
        ("producer_original", "historical"): "Historical rule; recurrence plus producer mask",
        ("producer_original", "robust_trend"): "0.05-BH trend rule; recurrence plus producer mask",
    }
    rule_rows: list[tuple[str, ...]] = []
    for (variant, rule), label in rule_labels.items():
        row = global_rules[
            global_rules["path_definition"].eq(variant) & global_rules["rule"].eq(rule)
        ].iloc[0]
        rule_rows.append(
            (
                label,
                str(int(row["rule_paths_eligible"])),
                str(int(row["tp"])),
                str(int(row["fp"])),
                str(int(row["fn"])),
                f"{row['precision']:.1%}",
                f"{row['recall']:.1%}",
            )
        )
    _assert_value(_report_table(tables, rule_header), rule_rows, "report fixed-rule table")


def _count_word(value: int) -> str:
    """Spell the small counts used in the missed-return explanation."""
    words = (
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    )
    if not 0 <= value < len(words):
        return str(value)
    return words[value]


def _validate_report_reconciliations(
    report: str,
    metrics: dict[str, object],
    fixed_rules: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    """Bind the three explanations that reconcile different evidence counts."""
    global_rules = fixed_rules[fixed_rules["scope"].eq("global")].set_index(["path_definition", "rule"])
    historical_recurrence = global_rules.loc[("recurrence_clean", "historical")]
    trend_recurrence = global_rules.loc[("recurrence_clean", "robust_trend")]
    historical_producer = global_rules.loc[("producer_original", "historical")]
    trend_producer = global_rules.loc[("producer_original", "robust_trend")]
    historical_definition = (
        "The older rule requires all three of the following: at least 0.25 BH of total shuttle movement, at "
        "least 0.25 BH of net movement towards the player, and at least 55% of steps moving towards the player."
    )
    if historical_definition not in report:
        raise AssertionError("report does not retain the historical and trend rule definitions")
    historical_provenance = (
        "The 0.25-BH values come from the older analysis. The 55% step threshold was chosen using the older "
        "±5/249 scoring setup. The 0.05-BH value was set in advance as an engineering judgement."
    )
    if historical_provenance not in report:
        raise AssertionError("report does not retain the fixed rules' investigation-time provenance")
    shared_checks_statement = (
        "Both rules use the same checks for sample count, distance from the final path point to the contact, "
        "recurrence flags, valid measurements and large jumps."
    )
    if shared_checks_statement not in report:
        raise AssertionError("report does not state the fixed rules' shared eligibility checks")
    eligibility_statement = (
        "Because the older rule also requires 0.25 BH of total movement, it leaves "
        f"{int(historical_recurrence['rule_paths_eligible'])} eligible paths instead of "
        f"{int(trend_recurrence['rule_paths_eligible'])} with the recurrence check, and "
        f"{int(historical_producer['rule_paths_eligible'])} instead of "
        f"{int(trend_producer['rule_paths_eligible'])} with the producer mask added."
    )
    if eligibility_statement not in report:
        raise AssertionError("report does not explain the historical rule's extra movement-floor exclusions")

    primary_recurrence: Any = metrics["path_funnel"][PRIMARY_POPULATION]["global"]["recurrence_clean"]
    all_usable = int(primary_recurrence["common_path_eligible"])
    all_incoming = int(primary_recurrence["robust_trend_incoming"])
    truth_usable = int(trend_recurrence["rule_paths_eligible"])
    local_evidence_statements = (
        (
            f"We can judge the motion rule against **{int(trend_recurrence['n_truth'])} earliest contacts** "
            "where the ±10 ground truth identifies either serve or first return without ambiguity: "
            f"{int(trend_recurrence['gt_serves'])} serves and "
            f"{int(trend_recurrence['gt_first_returns'])} first returns."
        ),
        (
            f"{_count_word(truth_usable).capitalize()} of those {int(trend_recurrence['n_truth'])} have usable "
            "paths under the recurrence check."
        ),
        (
            f"Across all {int(primary_recurrence['n'])} rallies, there are {all_usable} usable paths and "
            f"{all_incoming} incoming calls. {_count_word(all_usable - truth_usable).capitalize()} of those "
            "usable paths belong to contacts that are unmatched or match a later stroke, so they cannot be "
            "included in the "
            f"{int(trend_recurrence['n_truth'])}-rally serve-versus-return scoring set."
        ),
    )
    if any(statement not in report for statement in local_evidence_statements):
        raise AssertionError("report does not reconcile local truth and all-primary evidence counts")

    missed_breakdown: dict[str, tuple[int, int]] = {}
    for variant in PATH_VARIANTS:
        returns = diagnostics[
            diagnostics["path_definition"].eq(variant)
            & diagnostics["gt_anchor_identity"].eq("first_return")
        ]
        usable = returns["common_path_eligible"].astype(bool)
        incoming = returns["incoming_call"].astype(bool)
        missed_breakdown[variant] = (
            int((usable & ~incoming).sum()),
            int((~usable).sum()),
        )
    recurrence_usable, recurrence_missing = missed_breakdown["recurrence_clean"]
    producer_usable, producer_missing = missed_breakdown["producer_original"]
    breakdown_statements = (
        (
            f"Looking specifically at the {int(trend_recurrence['gt_first_returns'])} ground-truth first "
            f"returns: {int(trend_recurrence['tp'])} are correctly called incoming, "
            f"{recurrence_usable} have usable paths but stay below the 0.05-BH threshold, and "
            f"{recurrence_missing} do not have a usable path at all."
        ),
        (
            f"With the stricter source check, {_count_word(producer_usable)} missed return has usable motion "
            f"but stays below 0.05 BH, while {_count_word(producer_missing)} have no usable path."
        ),
    )
    if any(statement not in report for statement in breakdown_statements):
        raise AssertionError("report does not separate usable-negative and no-evidence return misses under both masks")


def _finite_median(series: pd.Series) -> float:
    """Return a diagnostic median after excluding non-finite ratios."""
    values = series.astype(float).to_numpy()
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else math.nan


def _validate_report_diagnostics(
    tables: dict[tuple[str, ...], list[list[tuple[str, ...]]]],
    diagnostics: pd.DataFrame,
) -> None:
    """Bind every displayed continuous diagnostic group and median."""
    eligible = diagnostics[
        diagnostics["path_definition"].eq("recurrence_clean")
        & diagnostics["common_path_eligible"].astype(bool)
    ].copy()
    group_header = (
        "Group",
        "Paths",
        "Median fitted decrease (BH)",
        "Median residual scatter (BH)",
        "Median trend-to-jitter",
    )
    group_specs = (
        ("gt_anchor_identity", "serve", "GT serves"),
        ("gt_anchor_identity", "first_return", "GT first returns"),
        ("call_correct", True, "Correct calls"),
        ("call_correct", False, "Incorrect calls"),
    )
    expected_groups: list[tuple[str, ...]] = []
    for column, value, label in group_specs:
        group = eligible[eligible[column].eq(value)]
        expected_groups.append(
            (
                label,
                str(len(group)),
                f"{_finite_median(group['fitted_decrease_bh']):.3f}",
                f"{_finite_median(group['residual_rms_bh']):.3f}",
                f"{_finite_median(group['trend_to_jitter']):.3f}",
            )
        )
    actual_groups = [*_report_table(tables, group_header), *_report_table(tables, group_header, occurrence=1)]
    _assert_value(actual_groups, expected_groups, "report continuous diagnostic group tables")

    length_header = (
        "Observed path length",
        "Paths",
        "Median fitted decrease (BH)",
        "Median residual scatter (BH)",
    )
    eligible["length_group"] = pd.cut(
        eligible["path_frames"],
        bins=[4, 5, 9, np.inf],
        labels=["5 points", "6-9 points", "10+ points"],
    )
    length_rows: list[tuple[str, ...]] = []
    for label in ("5 points", "6-9 points", "10+ points"):
        group = eligible[eligible["length_group"].eq(label)]
        length_rows.append(
            (
                label,
                str(len(group)),
                f"{_finite_median(group['fitted_decrease_bh']):.3f}",
                f"{_finite_median(group['residual_rms_bh']):.3f}",
            )
        )
    _assert_value(_report_table(tables, length_header), length_rows, "report path-length diagnostic table")


def _validate_report_servers_and_errors(
    report: str,
    tables: dict[tuple[str, ...], list[list[tuple[str, ...]]]],
    metrics: dict[str, object],
    diagnostics: pd.DataFrame,
    refit: RefitComparison,
) -> None:
    """Bind population server counts and the complete usable-path error list."""
    primary_scores: Any = metrics["server_scores"][PRIMARY_POPULATION]["global"]
    primary_denominator = primary_scores["old alternating fit"]["n"]
    method_header = (
        "Server method",
        "Correct",
        "Answers made",
        f"Overall accuracy (n={primary_denominator})",
    )
    method_labels = {
        "old alternating fit": "Released alternating fit",
        "anchor player": "Assume the earliest contact player served",
        "historical rule, recurrence mask": "Flip player when the historical rule says incoming",
        "0.05-BH trend rule, recurrence mask": (
            "Use earliest-contact player; flip when the 0.05-BH trend says incoming"
        ),
        "0.05-BH trend then like-for-like refit": (
            "Earliest-contact fallback; prepend inferred server and refit on incoming triggers"
        ),
        "0.05-BH trend rule, recurrence plus producer mask": (
            "Same fallback and 0.05-BH flip; also mask producer inpaint"
        ),
        "0.05-BH trend evidence only": "Motion answer only; abstain without usable evidence",
        "0.05-BH trend then prepend unknown player": "Prepend one unknown contact before alternating fit",
    }
    method_rows: list[tuple[str, ...]] = []
    for method, label in method_labels.items():
        values = primary_scores[method]
        method_rows.append(
            (
                label,
                f"{values['correct']}/{values['n']}",
                f"{values['known']}/{values['n']}",
                f"{values['accuracy']:.1%}",
            )
        )
    _assert_value(_report_table(tables, method_header), method_rows, "report primary server table")

    comparison_header = (
        "Motion group",
        "Rallies",
        "Earliest-contact baseline",
        "Direct correction",
        "Prepend/refit",
    )
    anchor_scores = primary_scores["anchor player"]
    direct_scores = primary_scores["0.05-BH trend rule, recurrence mask"]
    like_scores = primary_scores["0.05-BH trend then like-for-like refit"]
    comparison_rows = [
        (
            "No incoming-motion trigger",
            str(refit.non_triggered),
            f"{refit.non_triggered_anchor_correct} correct; {refit.non_triggered} answers",
            f"{refit.non_triggered_anchor_correct} correct; {refit.non_triggered} answers",
            f"{refit.non_triggered_anchor_correct} correct; {refit.non_triggered} answers",
        ),
        (
            "Incoming-motion trigger",
            str(refit.triggered),
            (
                f"{int(anchor_scores['correct']) - refit.non_triggered_anchor_correct} correct; "
                f"{refit.triggered} answers"
            ),
            f"{refit.triggered_direct_correct} correct; {refit.triggered} answers",
            f"{refit.triggered_refit_correct} correct; {refit.triggered} answers",
        ),
        (
            "All primary rallies",
            str(refit.total),
            f"{anchor_scores['correct']} correct; {anchor_scores['known']} answers",
            f"{direct_scores['correct']} correct; {direct_scores['known']} answers",
            f"{like_scores['correct']} correct; {like_scores['known']} answers",
        ),
    ]
    _assert_value(_report_table(tables, comparison_header), comparison_rows, "report prepend/refit comparison table")

    disagreement_header = (
        "Rally",
        "What the augmented fit does",
        "Direct inference",
        "Prepend/refit result",
        "GT server",
    )
    _assert_value(
        _report_table(tables, disagreement_header),
        list(refit.disagreement_rows),
        "report prepend/refit disagreement table",
    )

    fallback_gain = refit.non_triggered_anchor_correct - refit.non_triggered_old_fit_correct
    triggered_gain = refit.triggered_direct_correct - refit.triggered_refit_correct
    try:
        provenance = report.split("## Note about an earlier exploratory comparison", maxsplit=1)[1].split(
            "## Limits", maxsplit=1
        )[0]
    except IndexError as error:
        raise AssertionError("report is missing its methodological-provenance boundaries") from error
    provenance_statement = (
        "That result is not a like-for-like comparison with the direct method because the fallback is different. "
        f"Of the {refit.direct_correct - refit.old_fit_fallback_correct}-correct-answer gap between that "
        f"calculation and the direct method, {fallback_gain} answers come from the different fallback and only "
        f"{_count_word(triggered_gain)} come from the refitting on triggered rallies."
    )
    if provenance_statement not in provenance:
        raise AssertionError("report provenance does not give the rebuilt 127 and 32-plus-four decomposition")
    old_score_text = f"{refit.old_fit_fallback_correct}/{refit.total}"
    if report.count(old_score_text) != 1:
        raise AssertionError(f"report must contain {old_score_text} exactly once, in methodological provenance")
    disagreement_statement = (
        f"For the {refit.triggered} triggered rallies, direct inference and prepend/refit agree in "
        f"{refit.triggered - refit.disagreements} cases. In {_count_word(refit.later_vote_overrides)} cases, "
        "later contact votes overturn a correct direct guess. In "
        f"{_count_word(refit.tied_refits)} more, the alternating fit ties."
    )
    if disagreement_statement not in report:
        raise AssertionError("report does not give the rebuilt two-override and two-tie outcomes")

    server_header = (
        "Rally group",
        "Released fit",
        "Earliest-contact player",
        "Earliest-contact fallback plus 0.05-BH flip",
    )
    population_labels = {
        PRIMARY_POPULATION: "239 one-to-one",
        COVERED_POPULATION: "249 covered, including merges",
        ALL_POPULATION: "292 end-to-end, including segmentation failures",
    }
    server_rows: list[tuple[str, ...]] = []
    for population, label in population_labels.items():
        scores: Any = metrics["server_scores"][population]["global"]
        cells = []
        for method in ("old alternating fit", "anchor player", "0.05-BH trend rule, recurrence mask"):
            values = scores[method]
            cells.append(f"{values['correct']}/{values['n']} ({values['accuracy']:.1%})")
        server_rows.append((label, *cells))
    _assert_value(_report_table(tables, server_header), server_rows, "report population server table")

    fixture_header = (
        "Video",
        "Rallies",
        "Released fit",
        "Earliest-contact player",
        "Direct motion correction",
        "Prepend/refit, same fallback",
    )
    fixture_rows: list[tuple[str, ...]] = []
    fixture_scores: Any = metrics["server_scores"][PRIMARY_POPULATION]["by_fixture"]
    for fixture, scores in fixture_scores.items():
        released_score = scores["old alternating fit"]
        anchor_score = scores["anchor player"]
        direct_score = scores["0.05-BH trend rule, recurrence mask"]
        refit_score = scores["0.05-BH trend then like-for-like refit"]
        fixture_rows.append(
            (
                fixture,
                str(direct_score["n"]),
                str(released_score["correct"]),
                str(anchor_score["correct"]),
                str(direct_score["correct"]),
                str(refit_score["correct"]),
            )
        )
    _assert_value(_report_table(tables, fixture_header), fixture_rows, "report primary server-by-video table")

    evidence_only = primary_scores["0.05-BH trend evidence only"]
    incoming_calls: Any = metrics["path_funnel"][PRIMARY_POPULATION]["global"]["recurrence_clean"][
        "robust_trend_incoming"
    ]
    fallback_statements = (
        "When motion does not trigger, both methods choose the player at the earliest accepted contact.",
        "When motion does trigger, the direct method chooses the other player as server.",
        (
            "Prepend/refit adds that inferred server to the contact sequence, reruns the alternating fit, and "
            "falls back to the earliest-contact player if the fit ties."
        ),
    )
    if any(statement not in report for statement in fallback_statements):
        raise AssertionError("report does not state both shared-fallback method semantics")
    if int(incoming_calls) != refit.triggered:
        raise AssertionError("path funnel incoming calls do not match the rebuilt triggered-refit population")
    if (int(evidence_only["known"]), int(evidence_only["n"])) != (24, 239):
        raise AssertionError("evidence-only server answers do not match the approved 24/239 motion coverage")

    errors = diagnostics[
        diagnostics["path_definition"].eq("recurrence_clean")
        & diagnostics["common_path_eligible"].astype(bool)
        & ~diagnostics["call_correct"].astype(bool)
    ].copy()
    errors["error_type"] = np.where(
        errors["incoming_call"].astype(bool),
        "False return call on a GT serve",
        "Missed GT first return",
    )
    errors = errors.sort_values(["error_type", "residual_rms_bh"])
    error_cases: list[str] = []
    for row in errors.itertuples(index=False):
        error_cases.append(f"{row.fixture} {row.set_id} rally {int(row.rally)}")
    if len(error_cases) != 8:
        raise AssertionError(f"rebuilt usable-path error set has {len(error_cases)} cases; expected 8")
    false_return_calls = int(errors["incoming_call"].astype(bool).sum())
    missed_returns = len(errors) - false_return_calls
    error_statements = (
        (
            f"There are eight errors among the usable paths: {_count_word(false_return_calls)} false return "
            f"calls on ground-truth serves and {_count_word(missed_returns)} missed ground-truth returns."
        ),
        f"The cases are {', '.join(error_cases)}.",
    )
    if any(statement not in report for statement in error_statements):
        raise AssertionError("report error-case statement does not match all eight rebuilt cases")


def _validate_motion_plot_contract(metrics: dict[str, object]) -> None:
    """Bind the motion plot's two evidence-availability bars."""
    funnel: Any = metrics["path_funnel"][PRIMARY_POPULATION]["global"]
    recurrence_coverage = int(funnel["recurrence_clean"]["common_path_eligible"])
    producer_coverage = int(funnel["producer_original"]["common_path_eligible"])
    denominator = int(funnel["recurrence_clean"]["n"])
    if (recurrence_coverage, producer_coverage, denominator) != (24, 14, 239):
        raise AssertionError(
            "motion plot contract differs from the approved 24/239 recurrence and 14/239 producer counts"
        )

    scores: Any = metrics["server_scores"][PRIMARY_POPULATION]["global"]
    evidence_only = scores["0.05-BH trend evidence only"]
    recurrence_method = scores["0.05-BH trend rule, recurrence mask"]
    producer_method = scores["0.05-BH trend rule, recurrence plus producer mask"]
    if int(evidence_only["known"]) != recurrence_coverage:
        raise AssertionError("recurrence evidence-only answers do not match the motion plot's usable count")
    if int(recurrence_method["known"]) != denominator or int(producer_method["known"]) != denominator:
        raise AssertionError("both server methods must fall back to the contact player outside usable motion evidence")


def _validate_server_plot_contract(metrics: dict[str, object], refit: RefitComparison) -> None:
    """Bind all four central server methods and their correct/known counts."""
    scores: Any = metrics["server_scores"][PRIMARY_POPULATION]["global"]
    expected_results = {
        **CENTRAL_SERVER_PLOT_RESULTS,
        "0.05-BH trend then like-for-like refit": (refit.like_for_like_correct, refit.like_for_like_known),
    }
    for method, (expected_correct, expected_known) in expected_results.items():
        values = scores[method]
        actual = (int(values["correct"]), int(values["known"]), int(values["n"]))
        expected = (expected_correct, expected_known, 239)
        if actual != expected:
            raise AssertionError(
                f"central server plot method {method!r} has correct/known/n {actual}; expected {expected}"
            )


def _validate_report_prose(report: str, metrics: dict[str, object], refit: RefitComparison) -> None:
    """Bind the durable report structure and main narrative to rebuilt values."""
    required_headings = (
        "## Main takeaway",
        "## Why getting the first contact right matters",
        "## What should we try next?",
        "## Contents",
        "## Why do we use 292, 249 and 239 rallies in different places?",
        "## Is the first accepted contact the serve?",
        "## What happens when the first contact does not match?",
        "## How does the motion correction work?",
        "## How often do we have usable motion?",
        "## Does excluding producer-marked interpolation help?",
        "## Does prepend/refit improve the motion-based guess?",
        "## Extra diagnostics and rule comparisons (optional)",
        "### Do the diagnostics point to a better rule?",
        "### How does the older 0.25-BH rule compare?",
        "### Which usable paths give the wrong answer?",
        "## Extra breakdowns (optional)",
        "## Note about an earlier exploratory comparison",
        "## Limits",
        "## Output files",
    )
    heading_positions = [report.find(heading) for heading in required_headings]
    if any(position < 0 for position in heading_positions) or heading_positions != sorted(heading_positions):
        raise AssertionError("report is missing a required durable-framing heading or places one out of order")
    if "## Extended summary" in report:
        raise AssertionError("report must not contain the superseded Extended summary section")

    try:
        main_takeaway = report.split("## Main takeaway", maxsplit=1)[1].split(
            "## Why getting the first contact right matters", maxsplit=1
        )[0]
        anchor_evidence = report.split("## Why getting the first contact right matters", maxsplit=1)[1].split(
            "## What should we try next?", maxsplit=1
        )[0]
    except IndexError as error:
        raise AssertionError("report is missing the Main takeaway or first-contact boundary") from error

    main_takeaway_word_count = len(main_takeaway.split())
    if not 90 <= main_takeaway_word_count <= 180:
        raise AssertionError(
            f"report Main takeaway has {main_takeaway_word_count} words; expected 90 to 180"
        )

    alignment: Any = metrics["alignment"][PRIMARY_POPULATION]["global"]["10"]
    alignment_labels = alignment["labels"]
    sequence: Any = metrics["unmatched_anchor_sequences"]["global"]
    paths: Any = metrics["path_funnel"][PRIMARY_POPULATION]["global"]["recurrence_clean"]
    servers: Any = metrics["server_scores"][PRIMARY_POPULATION]["global"]
    released = servers["old alternating fit"]
    anchor = servers["anchor player"]
    direct = servers["0.05-BH trend rule, recurrence mask"]
    like_refit = servers["0.05-BH trend then like-for-like refit"]

    main_takeaway_fragments = (
        f"All server results here use the same {released['n']} one-to-one rallies.",
        f"released alternating-fit method gets **{released['correct']}** right",
        f"that rises to **{anchor['correct']}**.",
        f"Only **{paths['common_path_eligible']}/{paths['n']}** rallies have usable shuttle motion",
        f"Applying the motion correction directly raises the result to **{direct['correct']}** correct.",
        (
            f"The prepend/refit method uses the same earliest-contact fallback and the same {refit.triggered} "
            f"motion triggers, but gets **{like_refit['correct']}** correct."
        ),
        (
            f"On those {refit.triggered} triggered rallies, the direct method gets "
            f"{refit.triggered_direct_correct} right and prepend/refit gets {refit.triggered_refit_correct}."
        ),
        "Rerunning the full fit after adding the motion-based guess does not help in this sample.",
        "making sure we start from the right contact and increasing the number of rallies with usable motion paths",
    )
    for fragment in main_takeaway_fragments:
        if fragment not in main_takeaway:
            raise AssertionError(f"report Main takeaway is missing rebuilt claim {fragment!r}")
    deferred_method_sentences = (
        "The direct method changes the baseline only when the shuttle clearly approaches the contact player.",
        "Prepend/refit instead injects the inferred server and reruns the full alternating fit.",
    )
    if any(sentence in main_takeaway for sentence in deferred_method_sentences):
        raise AssertionError("report Main takeaway retains method-definition detail meant for the main narrative")

    anchor_evidence_fragments = (
        f"**{alignment_labels.get('unmatched', 0)} of {paths['n']}** earliest accepted contacts do not match",
        (
            f"recovers the serve in {sequence['later_serve_match']} of those rallies and the first return in another "
            f"{sequence['no_later_serve_but_first_return_match']}"
        ),
        "Many errors happen before we try to classify the shuttle motion.",
    )
    for fragment in anchor_evidence_fragments:
        if fragment not in anchor_evidence:
            raise AssertionError(f"report anchor-selection section is missing rebuilt claim {fragment!r}")

    largest_category_statement = (
        f"The serve is the largest single category, at **{alignment_labels.get('contact_1', 0)} of "
        f"{paths['n']}** rallies, but **{alignment_labels.get('unmatched', 0)} of {paths['n']}** earliest contacts "
        "do not match"
    )
    ordinary_detector_statement = (
        "The earliest accepted contact is the first output accepted by the released contact detector; it is not "
        "produced by a dedicated serve detector. The detector begins with shuttle impulses and player proximity, "
        "then applies wrist, suppression and exclusion checks. For this analysis, we independently measure which "
        "player is nearest at the accepted frame rather than relying on the released alternating fit."
    )
    trend_statement = (
        "For the trend measure, we calculate the slope between every pair of shuttle-to-player distance samples "
        "and take the median slope. Time is scaled from zero to one across the path. We then use the negative slope "
        "as the fitted decrease in distance. If that decrease is at least **0.05 apparent player body heights "
        "(BH)**, we call the path incoming."
    )
    path_quality_statement = (
        "To build the motion path, we look back by at most 30 frames on a 30-fps base timeline, staying within the "
        "same court scene. We choose the continuous run closest to the contact. A path is usable only if it has at "
        "least five samples, ends close enough to the contact, has recurrence guard `NO_FLAG`, has valid "
        "player-distance and body-height measurements, and contains no extremely large single-step jump."
    )
    required_statements = {
        "serve is the largest ±10 anchor category": largest_category_statement,
        "anchor is an ordinary contact-detector output": ordinary_detector_statement,
        "robust trend uses the pairwise-median slope": trend_statement,
        "path-quality gates are explicit": path_quality_statement,
        "±10 is the practical alignment": "We use ±10 as the main tolerance.",
        "multiple windows keep nearest identity": (
            "We still use whichever stroke is closest to the accepted contact. This is rare at ±10 (5 rallies), "
            "but happens in 117 rallies at ±30"
        ),
        "diagnostics do not make calls": "they do not decide whether a path is usable and they are not separate classifiers",
        "no visual false-contact claim": "does not mean we manually inspected them and proved they were false contacts",
        "reported thresholds precede scoring": (
            "The thresholds reported here were fixed before this scoring, but these results are not an "
            "independent external validation."
        ),
    }
    for label, statement in required_statements.items():
        if statement not in report:
            raise AssertionError(f"report lacks the plain statement that {label}")


def _validate_final_files(report: str) -> None:
    """Require only the corrected report plots and no historical threshold artefacts."""
    thresholds_path = OUTPUT_DIR / "thresholds.csv.gz"
    if thresholds_path.exists():
        raise AssertionError(f"historical threshold output must be absent: {thresholds_path}")
    if not PLOT_DIR.is_dir():
        raise FileNotFoundError(f"corrected plot directory is missing: {PLOT_DIR}")
    actual_plots: set[str] = set()
    empty_plots: list[str] = []
    for path in PLOT_DIR.glob("*.png"):
        if not path.is_file():
            continue
        actual_plots.add(path.name)
        if path.stat().st_size == 0:
            empty_plots.append(path.name)
    if actual_plots != EXPECTED_PLOTS:
        missing = sorted(EXPECTED_PLOTS.difference(actual_plots))
        extra = sorted(actual_plots.difference(EXPECTED_PLOTS))
        raise AssertionError(f"corrected top-level PNG set differs; missing={missing}, extra={extra}")
    if empty_plots:
        raise AssertionError(f"corrected plots are empty: {sorted(empty_plots)}")
    if (PLOT_DIR / "cases").exists():
        raise AssertionError("historical per-case plot directory must be absent")
    for plot_name in EXPECTED_PLOTS:
        if f"outputs/plots/{plot_name}" not in report:
            raise AssertionError(f"report does not reference corrected plot {plot_name}")


def _validate_report_and_plots(
    metrics: dict[str, object],
    fixed_rules: pd.DataFrame,
    diagnostics: pd.DataFrame,
    refit: RefitComparison,
) -> None:
    """Bind the final Markdown and PNG inventory to independently rebuilt results."""
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"corrected report is missing: {REPORT_PATH}")
    report = REPORT_PATH.read_text(encoding="utf-8")
    tables = _parse_markdown_tables(report)
    _validate_final_files(report)
    _validate_report_prose(report, metrics, refit)
    _validate_report_populations(tables, metrics)
    _validate_report_alignment_and_sequences(report, tables, metrics)
    _validate_report_paths_and_rules(tables, metrics, fixed_rules)
    _validate_report_reconciliations(report, metrics, fixed_rules, diagnostics)
    _validate_report_diagnostics(tables, diagnostics)
    _validate_report_servers_and_errors(report, tables, metrics, diagnostics, refit)
    _validate_motion_plot_contract(metrics)
    _validate_server_plot_contract(metrics, refit)


def validate() -> None:
    """Validate every saved output against independent arithmetic."""
    tables = _load_tables()
    _validate_frozen_sources(tables.rallies, tables.spans)
    _validate_identity_and_boundaries(tables.rallies, tables.spans)
    _validate_alignments_and_sequences(tables.rallies)
    _validate_path_points(tables.rallies, tables.path_points)
    refit_comparison = _validate_server_columns(tables.rallies)

    recalculated_rules = _build_fixed_rules(tables.rallies)
    if len(tables.fixed_rules) != 16 or tables.fixed_rules[["scope", "path_definition", "rule"]].duplicated().any():
        raise AssertionError("fixed_rules.csv.gz must contain 16 unique global/per-fixture rule rows")
    _assert_frame_equal(
        tables.fixed_rules,
        recalculated_rules,
        ["scope", "path_definition", "rule"],
        "fixed_rules.csv.gz",
    )

    recalculated_diagnostics = _build_trend_diagnostics(tables.rallies)
    if tables.trend_diagnostics[[*PATH_KEY]].duplicated().any():
        raise AssertionError("trend_diagnostics.csv.gz contains duplicate rally/path identities")
    _assert_frame_equal(
        tables.trend_diagnostics,
        recalculated_diagnostics,
        PATH_KEY,
        "trend_diagnostics.csv.gz",
    )

    recalculated_metrics = _build_metrics(tables.rallies, recalculated_rules)
    _assert_mapping(tables.metrics, recalculated_metrics, "metrics.json.gz")
    _validate_report_and_plots(
        recalculated_metrics,
        recalculated_rules,
        recalculated_diagnostics,
        refit_comparison,
    )
    print(
        "validated 292 rallies, 344 spans, "
        f"{len(tables.path_points)} path points, 16 fixed-rule rows, all metrics, report and six plots"
    )


if __name__ == "__main__":
    validate()
