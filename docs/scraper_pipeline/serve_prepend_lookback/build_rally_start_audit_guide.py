"""Build deterministic rally-start visibility audit guides for issue 32.

Run from the repository root::

    PYTHONPATH=src python \
        docs/scraper_pipeline/serve_prepend_lookback/build_rally_start_audit_guide.py

The generated rows include annotation proposals, pending event records, and
reviewed pilot records joined from separate human decisions. They do not change
the canonical broadcast timelines or production annotator output.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

from annotator.broadcast_timeline_labels import (
    LabelInterval,
    SceneTruth,
    VideoMetadata,
    interval_index_at,
    read_label_csv,
    validate_partition,
)
from annotator.calibration.fixtures import FIXTURES, SHARED_FILES, Fixture
from annotator.rally_start_events import (
    CONTRACT_PATH,
    DECISION_COLUMNS as DECISION_COLUMNS,
    DECISION_VALUE_COLUMNS,
    TARGET_COLUMNS,
    RallyStartDecision,
    RallyStartKey,
    RallyStartTarget,
    ReviewStatus,
    ServeVisibility,
    build_decision_seed,
    decision_from_row,
    decision_to_row,
    read_decision_csv,
    target_from_row,
    validate_decision,
    write_decision_csv,
)


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data/rally_start_visibility_audit_20260809"
REVIEW_DECISIONS_RELATIVE_PATH = Path(
    "docs/scraper_pipeline/serve_prepend_lookback/data/"
    "rally_start_visibility_review_20260809/pilot_decisions.csv.gz"
)
COMMITTED_DIR = SCRIPT_DIR / "data/serve_prepend_lookback_189c5af_20260808"
TIMELINE_DIR = SCRIPT_DIR.parent / "broadcast_nonstandard_camera_id/data"
MASTER_RELATIVE_PATH = Path("training/data/shuttleset/annotations/shots_master.csv")

TARGET_STATUS = "serve_missed_later_strokes_matched"
LIVE_TRUTHS = {SceneTruth.LIVE, SceneTruth.LIVE_NON_STANDARD}
REVIEW_CONTEXT_SECONDS = 10.0
DEFAULT_CONTROLS_PER_VIDEO = 2

EXPECTED_TARGET_COUNTS = {"sset_01": 63, "sset_15": 39, "sset_21": 34}
EXPECTED_FLAW_COUNTS = {"sset_01": 2, "sset_15": 0, "sset_21": 24}
EXPECTED_UNKNOWN_COUNTS = {"sset_01": 1, "sset_15": 0, "sset_21": 24}

SERVE_VISIBILITIES = tuple(visibility.value for visibility in ServeVisibility)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_review_decisions(path: Path) -> list[dict[str, str]]:
    """Read the compressed primary decision table and validate its header."""
    return [decision_to_row(decision) for decision in read_decision_csv(path)]


def _parse_int(value: object, field: str, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source}: {field} must be an integer, got {value!r}")
    try:
        parsed = float(str(value))
    except ValueError as exc:
        raise ValueError(f"{source}: {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(f"{source}: {field} must be an integer, got {value!r}")
    return int(parsed)


def select_ball_round_one(rows: Sequence[dict[str, str]], source: str) -> dict[str, str]:
    """Return the sole ball-round-1 row, independent of tied frame ordering."""
    selected = [row for row in rows if _parse_int(row.get("ball_round"), "ball_round", source) == 1]
    if len(selected) != 1:
        raise ValueError(f"{source}: expected one ball_round=1 row, found {len(selected)}")
    return selected[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _validate_tracked_pin(path: Path, repo_root: Path) -> None:
    relative = path.resolve().relative_to(repo_root.resolve())
    pins = {pin.path: pin.md5 for pin in SHARED_FILES if pin.root == "repo"}
    expected = pins.get(relative)
    if expected is None:
        raise ValueError(f"tracked input has no fixture pin: {relative}")
    actual = _md5(path)
    if actual != expected:
        raise ValueError(f"tracked input MD5 differs for {relative}: {actual} != {expected}")


def _raw_rows_by_rally(fixture: Fixture, repo_root: Path) -> tuple[dict[tuple[str, int], list[dict[str, str]]], list[Path]]:
    rows_by_rally: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    paths = sorted((repo_root / fixture.gt_set_dir).glob("set*.csv"))
    if not paths:
        raise ValueError(f"{fixture.name}: no raw set CSVs under {fixture.gt_set_dir}")
    for path in paths:
        _validate_tracked_pin(path, repo_root)
        for row_number, row in enumerate(_read_csv(path), start=2):
            source = f"{_relative(path, repo_root)} row {row_number}"
            rally = _parse_int(row.get("rally"), "rally", source)
            row = dict(row)
            row["_source"] = source
            rows_by_rally[(path.stem, rally)].append(row)
    return dict(rows_by_rally), paths


def _master_rows(repo_root: Path) -> tuple[dict[tuple[int, str, int, int], dict[str, str]], Path]:
    path = repo_root / MASTER_RELATIVE_PATH
    _validate_tracked_pin(path, repo_root)
    by_key: dict[tuple[int, str, int, int], dict[str, str]] = {}
    for row_number, row in enumerate(_read_csv(path), start=2):
        source = f"{MASTER_RELATIVE_PATH.as_posix()} row {row_number}"
        key = (
            _parse_int(row.get("vid"), "vid", source),
            str(row.get("set_id", "")),
            _parse_int(row.get("rally"), "rally", source),
            _parse_int(row.get("ball_round"), "ball_round", source),
        )
        if key in by_key:
            raise ValueError(f"{source}: duplicate shots_master key {key}")
        by_key[key] = row
    return by_key, path


def _committed_path(fixture: Fixture, repo_root: Path) -> Path:
    relative = COMMITTED_DIR.relative_to(DEFAULT_REPO_ROOT)
    return repo_root / relative / f"{fixture.name}_committed_rallies.csv.gz"


def _timeline_path(fixture: Fixture, repo_root: Path) -> Path:
    relative = TIMELINE_DIR.relative_to(DEFAULT_REPO_ROOT)
    return repo_root / relative / f"{fixture.name}_broadcast_timeline_labels.csv.gz"


def _target_committed_rows(fixture: Fixture, repo_root: Path) -> tuple[list[dict[str, str]], Path]:
    path = _committed_path(fixture, repo_root)
    rows = _read_csv(path)
    if len(rows) != fixture.n_rallies:
        raise ValueError(f"{fixture.name}: committed rows {len(rows)} != expected {fixture.n_rallies}")
    targets = [row for row in rows if row.get("status") == TARGET_STATUS]
    expected = EXPECTED_TARGET_COUNTS[fixture.name]
    if len(targets) != expected:
        raise ValueError(f"{fixture.name}: target rows {len(targets)} != expected {expected}")
    return targets, path


def _timeline_context(
    fixture: Fixture,
    repo_root: Path,
) -> tuple[list[LabelInterval], VideoMetadata, Path]:
    path = _timeline_path(fixture, repo_root)
    intervals = read_label_csv(path)
    metadata = validate_partition(intervals)
    if metadata.video_id != fixture.name or not math.isclose(metadata.fps, fixture.fps):
        raise ValueError(f"{fixture.name}: timeline metadata differs: {metadata!r}")
    return intervals, metadata, path


def _build_fixture_rows(
    fixture: Fixture,
    repo_root: Path,
    master_by_key: dict[tuple[int, str, int, int], dict[str, str]],
) -> tuple[list[dict[str, object]], list[Path]]:
    raw_by_rally, raw_paths = _raw_rows_by_rally(fixture, repo_root)
    committed_rows, committed_path = _target_committed_rows(fixture, repo_root)
    intervals, metadata, timeline_path = _timeline_context(fixture, repo_root)
    context_frames = int(round(REVIEW_CONTEXT_SECONDS * fixture.fps))
    rows: list[dict[str, object]] = []

    for committed in committed_rows:
        set_id = str(committed["set_id"])
        rally = _parse_int(committed.get("rally_number"), "rally_number", fixture.name)
        key = (set_id, rally)
        source_rows = raw_by_rally.get(key)
        if source_rows is None:
            raise ValueError(f"{fixture.name}: missing raw rows for {key}")
        first = select_ball_round_one(source_rows, f"{fixture.name}/{set_id}/rally {rally}")
        first_frame = _parse_int(first.get("frame_num"), "frame_num", first["_source"])
        committed_frame = _parse_int(committed.get("gt_serve_frame"), "gt_serve_frame", fixture.name)
        if first_frame != committed_frame:
            raise ValueError(
                f"{fixture.name}/{set_id}/rally {rally}: ball-round-1 frame {first_frame} "
                f"!= committed GT frame {committed_frame}"
            )
        master_key = (fixture.video_id, set_id, rally, 1)
        master = master_by_key.get(master_key)
        if master is None:
            raise ValueError(f"{fixture.name}: missing shots_master row {master_key}")
        master_frame = _parse_int(master.get("frame_num"), "frame_num", str(master_key))
        if master_frame != first_frame:
            raise ValueError(f"{fixture.name}: shots_master frame {master_frame} != raw frame {first_frame}")

        interval_index = interval_index_at(intervals, first_frame)
        if interval_index is None:
            raise ValueError(f"{fixture.name}: no timeline interval contains frame {first_frame}")
        interval = intervals[interval_index]
        if interval.truth not in LIVE_TRUTHS:
            raise ValueError(f"{fixture.name}: target frame {first_frame} has truth {interval.truth.value}")
        preceding_truth = intervals[interval_index - 1].truth.value if interval_index else "boundary"
        flaw = bool(str(first.get("flaw", "")).strip())
        row = {
            "video_id": fixture.name,
            "fps": metadata.fps,
            "frame_count": metadata.frame_count,
            "set_id": set_id,
            "rally": rally,
            "gt_first_frame": first_frame,
            "gt_first_ball_round": 1,
            "gt_first_type_raw": str(first.get("type", "")),
            "gt_first_type_en": str(master.get("raw_type_en", "")),
            "gt_first_flaw": flaw,
            "gt_first_server": _parse_int(first.get("server"), "server", first["_source"]),
            "committed_status": str(committed["status"]),
            "later_strokes_matched": _parse_int(
                committed.get("later_strokes_matched"), "later_strokes_matched", fixture.name,
            ),
            "timeline_truth": interval.truth.value,
            "timeline_interval_start": interval.start_frame,
            "timeline_interval_end": interval.end_frame,
            "preceding_truth": preceding_truth,
            "live_transition_frame": interval.start_frame,
            "frames_from_live_transition": first_frame - interval.start_frame,
            "review_start_frame": max(
                0,
                min(first_frame - context_frames, interval.start_frame),
            ),
            "review_end_frame": min(metadata.frame_count, first_frame + context_frames + 1),
            "pilot_stratum": "quality-audit" if flaw else "full-audit-only",
            "note": "",
            "review_status": "pending",
            "serve_visibility": "",
            "visible_serve_frame": "",
            "first_visible_rally_frame": "",
            "broadcast_return_frame": "",
            "confidence": "",
            "review_note": "",
        }
        if row["gt_first_server"] != 1:
            raise ValueError(f"{fixture.name}/{set_id}/rally {rally}: first row server is not 1")
        rows.append(row)

    rows.sort(key=lambda row: (str(row["set_id"]), int(row["rally"])))
    return rows, [*raw_paths, committed_path, timeline_path]


def _set_pilot_strata(rows: Sequence[dict[str, object]], controls_per_video: int) -> list[dict[str, object]]:
    if controls_per_video < 1:
        raise ValueError("controls_per_video must be positive")
    updated = [dict(row) for row in rows]
    by_video: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in updated:
        by_video[str(row["video_id"])].append(row)
    for video_id, video_rows in by_video.items():
        controls = [row for row in video_rows if not bool(row["gt_first_flaw"])]
        controls.sort(
            key=lambda row: (
                int(row["frames_from_live_transition"]),
                str(row["set_id"]),
                int(row["rally"]),
            )
        )
        if len(controls) < controls_per_video:
            raise ValueError(f"{video_id}: only {len(controls)} unflagged control candidates")
        for row in controls[:controls_per_video]:
            row["pilot_stratum"] = "transition-control"
    for row in updated:
        row["note"] = (
            f"{row['set_id']} rally {row['rally']}; GT {row['gt_first_frame']}; "
            f"type {row['gt_first_type_en']}; flaw {str(bool(row['gt_first_flaw'])).lower()}; "
            f"current {row['timeline_truth']} after {row['preceding_truth']}; "
            f"stratum {row['pilot_stratum']}"
        )
    return updated


def build_audit_rows(repo_root: Path, controls_per_video: int = DEFAULT_CONTROLS_PER_VIDEO) -> tuple[list[dict[str, object]], list[Path]]:
    """Build and validate every target row without writing outputs.

    :param repo_root: Repository root containing the pinned inputs.
    :param controls_per_video: Number of unflagged transition controls per video.
    :return: Validated rows and every source path read.
    """
    repo_root = repo_root.resolve()
    master_by_key, master_path = _master_rows(repo_root)
    rows: list[dict[str, object]] = []
    input_paths: list[Path] = [master_path, SCRIPT_PATH, CONTRACT_PATH]
    for fixture in FIXTURES:
        fixture_rows, fixture_paths = _build_fixture_rows(fixture, repo_root, master_by_key)
        rows.extend(fixture_rows)
        input_paths.extend(fixture_paths)
    rows = _set_pilot_strata(rows, controls_per_video)
    _validate_rows(rows, controls_per_video)
    unique_paths = sorted(set(path.resolve() for path in input_paths))
    return rows, unique_paths


def _validate_rows(rows: Sequence[dict[str, object]], controls_per_video: int) -> None:
    if len(rows) != sum(EXPECTED_TARGET_COUNTS.values()):
        raise ValueError(f"pooled target rows {len(rows)} != 136")
    keys = [(row["video_id"], row["set_id"], row["rally"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("target rows contain duplicate (video_id, set_id, rally) keys")
    for row_number, row in enumerate(rows, start=2):
        if set(row) != set(TARGET_COLUMNS):
            raise ValueError(f"target row {row_number} does not match the schema")
        frame_count = int(row["frame_count"])
        start = int(row["review_start_frame"])
        end = int(row["review_end_frame"])
        frame = int(row["gt_first_frame"])
        transition = int(row["live_transition_frame"])
        if not 0 <= start <= frame < end <= frame_count:
            raise ValueError(f"target row {row_number} has invalid review bounds")
        if not start <= transition <= frame:
            raise ValueError(f"target row {row_number} excludes its live transition")
        if row["review_status"] != "pending" or any(
            row[column] != ""
            for column in (
                "serve_visibility",
                "visible_serve_frame",
                "first_visible_rally_frame",
                "broadcast_return_frame",
                "confidence",
                "review_note",
            )
        ):
            raise ValueError(f"target row {row_number} is not a pending event template")

    counts = Counter(str(row["video_id"]) for row in rows)
    flaws = Counter(str(row["video_id"]) for row in rows if bool(row["gt_first_flaw"]))
    unknown = Counter(
        str(row["video_id"]) for row in rows if str(row["gt_first_type_en"]) == "unknown"
    )
    controls = Counter(
        str(row["video_id"]) for row in rows if row["pilot_stratum"] == "transition-control"
    )
    if dict(counts) != EXPECTED_TARGET_COUNTS:
        raise ValueError(f"target count drift: {dict(counts)}")
    if {name: flaws[name] for name in EXPECTED_FLAW_COUNTS} != EXPECTED_FLAW_COUNTS:
        raise ValueError(f"flaw count drift: {dict(flaws)}")
    if {name: unknown[name] for name in EXPECTED_UNKNOWN_COUNTS} != EXPECTED_UNKNOWN_COUNTS:
        raise ValueError(f"unknown count drift: {dict(unknown)}")
    if any(controls[name] != controls_per_video for name in EXPECTED_TARGET_COUNTS):
        raise ValueError(f"control count drift: {dict(controls)}")


def apply_review_decisions(
    rows: Sequence[dict[str, object]],
    decisions: Sequence[dict[str, str]],
    source: str,
) -> list[dict[str, object]]:
    """Join human decisions to fresh source-derived pilot rows by exact key."""
    pilot_rows = [row for row in rows if row["pilot_stratum"] != "full-audit-only"]
    pilot_by_key = {
        target_from_row(row, "generated pilot").key: row
        for row in pilot_rows
    }
    decisions_by_key: dict[RallyStartKey, RallyStartDecision] = {}
    for row_number, decision in enumerate(decisions, start=2):
        row_source = f"{source} row {row_number}"
        parsed = decision_from_row(decision, row_source)
        validate_decision(parsed, source=row_source)
        if parsed.key in decisions_by_key:
            raise ValueError(f"{row_source}: duplicate decision key {parsed.key}")
        decisions_by_key[parsed.key] = parsed

    expected_keys = set(pilot_by_key)
    decision_keys = set(decisions_by_key)
    if decision_keys != expected_keys:
        missing = sorted(expected_keys - decision_keys)
        extra = sorted(decision_keys - expected_keys)
        raise ValueError(f"{source}: decision key mismatch; missing={missing}, extra={extra}")

    reviewed_rows: list[dict[str, object]] = []
    for row_number, pilot_row in enumerate(pilot_rows, start=2):
        target = target_from_row(pilot_row, "generated pilot")
        reviewed = dict(pilot_row)
        decision = decisions_by_key[target.key]
        decision_row = decision_to_row(decision)
        for column in DECISION_VALUE_COLUMNS:
            reviewed[column] = decision_row[column]
        validate_decision(decision, target, source=f"reviewed pilot row {row_number}")
        reviewed_rows.append(reviewed)
    return reviewed_rows


def _serialise(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _write_csv_gz(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = [{column: _serialise(row[column]) for column in TARGET_COLUMNS} for row in rows]
    temporary = path.with_name(f".tmp.{path.name}")
    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_handle:
                    writer = csv.DictWriter(text_handle, fieldnames=TARGET_COLUMNS, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(serialised)
        reloaded = _read_csv(temporary)
        if reloaded != serialised:
            raise RuntimeError(f"gzip CSV reload changed values: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_gz(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
                compressed.write(payload)
        with gzip.open(temporary, "rt", encoding="utf-8") as handle:
            if json.load(handle) != value:
                raise RuntimeError(f"gzip JSON reload changed value: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_records(paths: Iterable[Path], repo_root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(set(path.resolve() for path in paths)):
        records.append({
            "path": _relative(path, repo_root),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return records


def _write_fixture_package(
    fixture: Fixture,
    rows: Sequence[dict[str, object]],
    targets: Sequence[RallyStartTarget],
    reviewed_rows: Sequence[dict[str, object]],
    reviewed_decisions: Sequence[RallyStartDecision],
    output_dir: Path,
) -> tuple[list[Path], dict[str, object], list[RallyStartDecision]]:
    fixture_rows = [row for row in rows if row["video_id"] == fixture.name]
    pilot_rows = [row for row in fixture_rows if row["pilot_stratum"] != "full-audit-only"]
    fixture_reviewed = [row for row in reviewed_rows if row["video_id"] == fixture.name]
    fixture_targets = [target for target in targets if target.key.video_id == fixture.name]
    fixture_pilot_decisions = [
        decision
        for decision in reviewed_decisions
        if decision.key.video_id == fixture.name
    ]
    fixture_seed = build_decision_seed(fixture_targets, fixture_pilot_decisions)
    target_path = output_dir / f"{fixture.name}_rally_start_targets.csv.gz"
    pilot_path = output_dir / f"{fixture.name}_rally_start_pilot.csv.gz"
    reviewed_path = output_dir / f"{fixture.name}_rally_start_reviewed.csv.gz"
    seed_path = output_dir / f"{fixture.name}_rally_start_decision_seed.csv.gz"
    _write_csv_gz(target_path, fixture_rows)
    _write_csv_gz(pilot_path, pilot_rows)
    _write_csv_gz(reviewed_path, fixture_reviewed)
    write_decision_csv(seed_path, fixture_seed, fixture_targets)

    visibility_counts = Counter(str(row["serve_visibility"]) for row in fixture_reviewed)
    counts: dict[str, object] = {
        "targets": len(fixture_rows),
        "quality_audit": sum(row["pilot_stratum"] == "quality-audit" for row in fixture_rows),
        "transition_controls": sum(
            row["pilot_stratum"] == "transition-control" for row in fixture_rows
        ),
        "pilot_rows": len(pilot_rows),
        "reviewed_pilot_rows": len(fixture_reviewed),
        "full_audit_seed_rows": len(fixture_seed),
        "pending_full_audit_rows": sum(
            decision.review_status is ReviewStatus.PENDING
            for decision in fixture_seed
        ),
        "reviewed_visibility": {
            visibility: visibility_counts[visibility]
            for visibility in SERVE_VISIBILITIES
        },
        "unknown_first_type": sum(
            row["gt_first_type_en"] == "unknown" for row in fixture_rows
        ),
    }
    return [target_path, pilot_path, reviewed_path, seed_path], counts, fixture_seed


def write_audit_package(
    repo_root: Path,
    output_dir: Path,
    controls_per_video: int = DEFAULT_CONTROLS_PER_VIDEO,
    decisions_path: Path | None = None,
) -> dict[str, object]:
    """Write deterministic targets, pilot results, and full-audit seeds.

    :param repo_root: Repository root containing the pinned inputs.
    :param output_dir: Destination for generated gzip artifacts.
    :param controls_per_video: Number of unflagged transition controls per video.
    :param decisions_path: Compressed primary human-decision table.
    :return: Reload-checked summary written with the package.
    """
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if decisions_path is None:
        decisions_path = repo_root / REVIEW_DECISIONS_RELATIVE_PATH
    decisions_path = decisions_path.resolve()
    rows, input_paths = build_audit_rows(repo_root, controls_per_video)
    reviewed_decisions = read_decision_csv(decisions_path)
    decision_rows = [decision_to_row(decision) for decision in reviewed_decisions]
    reviewed_rows = apply_review_decisions(
        rows,
        decision_rows,
        _relative(decisions_path, repo_root),
    )
    targets = [target_from_row(row, "generated full target") for row in rows]
    input_paths.append(decisions_path)
    output_paths: list[Path] = []
    full_seed: list[RallyStartDecision] = []
    per_video: dict[str, dict[str, object]] = {}
    for fixture in FIXTURES:
        fixture_paths, fixture_counts, fixture_seed = _write_fixture_package(
            fixture,
            rows,
            targets,
            reviewed_rows,
            reviewed_decisions,
            output_dir,
        )
        output_paths.extend(fixture_paths)
        full_seed.extend(fixture_seed)
        per_video[fixture.name] = fixture_counts
    pooled_visibility = Counter(str(row["serve_visibility"]) for row in reviewed_rows)
    summary: dict[str, object] = {
        "schema_version": 3,
        "scope": "issue-28 unmatched first ShuttleSet stroke with a later matched stroke",
        "review_context_seconds_each_side": REVIEW_CONTEXT_SECONDS,
        "controls_per_video": controls_per_video,
        "counts": {
            "per_video": per_video,
            "pooled_targets": len(rows),
            "pooled_quality_audit": sum(row["pilot_stratum"] == "quality-audit" for row in rows),
            "pooled_transition_controls": sum(
                row["pilot_stratum"] == "transition-control" for row in rows
            ),
            "pooled_pilot_rows": sum(row["pilot_stratum"] != "full-audit-only" for row in rows),
            "pooled_reviewed_pilot_rows": len(reviewed_rows),
            "pooled_full_audit_seed_rows": len(full_seed),
            "pooled_pending_full_audit_rows": sum(
                decision.review_status is ReviewStatus.PENDING
                for decision in full_seed
            ),
            "pooled_reviewed_visibility": {
                visibility: pooled_visibility[visibility]
                for visibility in SERVE_VISIBILITIES
            },
        },
        "inputs": _file_records(input_paths, repo_root),
        "outputs": _file_records(output_paths, output_dir),
        "guardrails": [
            "flaw rows are a source-quality stratum, not an omission-prevalence sample",
            "transition controls are deterministic workflow controls, not a prevalence sample",
            "off-frame is a definite camera-boundary outcome, not uncertainty or omission",
            "human event truth remains separate from canonical timelines",
            "full-audit seeds preserve reviewed pilot decisions and leave all other targets pending",
            "the exact previously observed sset_15 omitted-start row is not pinned",
        ],
    }
    _write_json_gz(output_dir / "summary.json.gz", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--controls-per-video", type=int, default=DEFAULT_CONTROLS_PER_VIDEO)
    parser.add_argument("--decisions", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = write_audit_package(
        args.repo_root,
        args.out,
        args.controls_per_video,
        args.decisions,
    )
    counts = summary["counts"]
    assert isinstance(counts, dict)
    print(
        f"wrote {counts['pooled_targets']} target rows and "
        f"{counts['pooled_reviewed_pilot_rows']} reviewed pilot rows to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
