"""Score strict rally outputs from the fixed HGB contact event stream.

The evaluator has one deliberately narrow job. It verifies the held-out
candidate-score sidecar, keeps the already-retained HGB events unchanged,
replays the shipped Top/Bottom rule at those event frames, and then measures
which frozen rally spans can be kept as complete rallies. Labels are loaded
only after those three prediction inputs are fixed.

A span is kept at a timing-confidence requirement when every event in the span
has at least that score and every event has an answered side. A kept span is
fully correct only when it maps to one real rally, all contacts match
one-to-one at the requested tolerance, no extra event remains, and every side
answer is correct. A confidence failure abstains on the whole span. It never
removes one weak event and scores the remainder.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
CONTACT_DET_ROOT = MODULE_ROOT.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import freeze_contact_evidence as evidence_freezer
import score_contact_evidence as evidence_scorer
import score_tree_contact_detector as tree_scorer

RESULTS_SCHEMA = "contact-rally-score/1"
PRIMARY_TOLERANCE_BASE30 = 10
SENSITIVITY_TOLERANCE_BASE30 = 5
RETAINED_DECISION = tree_scorer.CANDIDATE_RETAINED
DEFAULT_CONFIDENCE_REQUIREMENTS = tuple(round(value, 2) for value in np.linspace(0.0, 1.0, 21))
PHYSICS_WITHOUT_RAW_MOTION_VARIANT = tree_scorer.PHYSICS_WITHOUT_RAW_MOTION_VARIANT
PHYSICS_BASE30_MOTION_VARIANT = tree_scorer.PHYSICS_BASE30_MOTION_VARIANT
TRIAL_VARIANT_RESULT_PATHS = {
    tree_scorer.CANDIDATE_VARIANT: ("histogram_boosting", "physics"),
    tree_scorer.PHYSICS_WITHOUT_RAW_MOTION_VARIANT: ("histogram_boosting", "physics"),
    tree_scorer.PHYSICS_BASE30_MOTION_VARIANT: ("histogram_boosting", "physics"),
}

REASON_NO_EVENTS = "no_events"
REASON_NO_RALLY = "no_rally"
REASON_MULTIPLE_RALLIES = "multiple_rallies"
REASON_MISSING_CONTACT = "missing_contact"
REASON_EXTRA_EVENT = "extra_event"
REASON_TIMING_MISMATCH = "timing_mismatch"
REASON_LOW_TIMING_CONFIDENCE = "low_timing_confidence"
REASON_SIDE_UNANSWERED = "side_unanswered"
REASON_SIDE_INCORRECT = "side_incorrect"


@dataclass(frozen=True)
class FixedEvent:
    """One retained HGB event and its replayed player-side answer."""

    fixture: str
    frame: int
    timing_score: float
    predicted_side: str | None


@dataclass(frozen=True)
class FixedSpan:
    """One frozen half-open contact-evidence span."""

    fixture: str
    span_id: int
    start_frame: int
    end_frame: int
    events: tuple[FixedEvent, ...]


@dataclass(frozen=True)
class RallyReference:
    """A ground-truth rally with a stable human-readable identity."""

    fixture: str
    rally_index: int
    rally_id: str
    frames: tuple[int, ...]


@dataclass(frozen=True)
class SpanScore:
    """Strict and confidence results for one span at one confidence cut-off."""

    fixture: str
    span_id: int
    start_frame: int
    end_frame: int
    rally_id: str | None
    event_count: int
    ground_truth_contacts: int
    timing_matches: int
    timing_confidence: float | None
    confidence_requirement: float
    confidence_pass: bool
    side_answerable: bool
    correct_side_answers: int
    kept: bool
    fully_correct: bool
    rejection_reasons: tuple[str, ...]


def _normalise_half(value: object, name: str) -> str | None:
    """Normalise the two side spellings used by project data."""
    if value is None:
        return None
    if value == "Top":
        return "Top"
    if value in {"Bot", "Bottom"}:
        return "Bot"
    raise ValueError(f"{name}: expected Top, Bot, Bottom, or null; found {value!r}")


def _rally_frames(rally: object) -> tuple[int, ...]:
    """Read stroke frames from a ShuttleSet rally or a small test object."""
    if hasattr(rally, "stroke_frames"):
        values = getattr(rally, "stroke_frames")  # noqa: B009
    elif isinstance(rally, Mapping):
        values = rally.get("stroke_frames")
    else:
        values = None
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("GT rally does not expose stroke_frames")
    frames = tuple(int(frame) for frame in values)
    if not frames:
        raise ValueError("GT rally has no stroke frames")
    if tuple(sorted(frames)) != frames or len(set(frames)) != len(frames):
        raise ValueError("GT rally stroke frames must be strictly increasing")
    return frames


def _rally_identity(rally: object, fixture: str, rally_index: int) -> str:
    """Return the stable ShuttleSet set/rally identity when it is available."""
    set_id = getattr(rally, "set_id", None)
    rally_number = getattr(rally, "rally", None)
    if isinstance(rally, Mapping):
        set_id = rally.get("set_id", set_id)
        rally_number = rally.get("rally", rally_number)
    if set_id is not None and rally_number is not None:
        return f"{set_id}:{rally_number}"
    return f"{fixture}:rally_{rally_index}"


def normalise_rallies(
    rallies_by_fixture: Mapping[str, Sequence[object]],
) -> dict[str, tuple[RallyReference, ...]]:
    """Convert loader objects into stable rally references for pure scoring."""
    output: dict[str, tuple[RallyReference, ...]] = {}
    for fixture, rallies in rallies_by_fixture.items():
        references: list[RallyReference] = []
        for index, rally in enumerate(rallies):
            references.append(
                RallyReference(
                    fixture=str(fixture),
                    rally_index=index,
                    rally_id=_rally_identity(rally, str(fixture), index),
                    frames=_rally_frames(rally),
                )
            )
        output[str(fixture)] = tuple(references)
    return output


def _decode_rows_fixture(rows: np.ndarray) -> np.ndarray:
    values = rows["fixture"]
    if values.dtype.kind == "S":
        return np.char.decode(values, "ascii")
    if values.dtype.kind == "U":
        return values.astype(str)
    raise TypeError("candidate rows fixture field must be fixed-width text")


def retained_events_from_scores(
    rows: np.ndarray,
    attribution: Mapping[tuple[str, int], str | None],
) -> dict[str, tuple[FixedEvent, ...]]:
    """Build the fixed event stream from decision-2 candidate rows.

    The candidate verifier owns sidecar provenance and row-order checks. This
    helper adds the event-level invariants needed by the rally scorer and keeps
    the stable ``(fixture, frame)`` identity when the side map is applied.
    """
    required = {"fixture", "frame", "timing_score", "decision"}
    if rows.dtype.names is None or not required.issubset(rows.dtype.names):
        raise ValueError(f"candidate rows are missing fields: {sorted(required)}")
    fixture_names = _decode_rows_fixture(rows)
    retained = rows[rows["decision"] == RETAINED_DECISION]
    retained_names = fixture_names[rows["decision"] == RETAINED_DECISION]
    output: dict[str, list[FixedEvent]] = {}
    identities: set[tuple[str, int]] = set()
    for fixture, row in zip(retained_names, retained, strict=True):
        fixture_name = str(fixture)
        frame = int(row["frame"])
        score = float(row["timing_score"])
        if frame < 0 or not math.isfinite(score):
            raise ValueError(f"{fixture_name}/{frame}: retained candidate identity or score is malformed")
        identity = (fixture_name, frame)
        if identity in identities:
            raise ValueError(f"{fixture_name}/{frame}: retained candidate identity is duplicated")
        identities.add(identity)
        predicted_side = _normalise_half(attribution.get(identity), f"{fixture_name}/{frame} attribution")
        output.setdefault(fixture_name, []).append(
            FixedEvent(fixture_name, frame, score, predicted_side)
        )
    return {
        fixture: tuple(sorted(events, key=lambda event: event.frame))
        for fixture, events in output.items()
    }


def fixed_spans_from_evidence(
    evidence: Mapping[str, Any],
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
) -> tuple[FixedSpan, ...]:
    """Read current spans and assign events using half-open boundaries."""
    fixtures = evidence.get("fixtures")
    if not isinstance(fixtures, Sequence) or isinstance(fixtures, (str, bytes)):
        raise TypeError("evidence fixtures must be a list")
    spans: list[FixedSpan] = []
    for fixture_row in fixtures:
        if not isinstance(fixture_row, Mapping):
            raise TypeError("evidence fixture row must be an object")
        fixture = str(fixture_row["fixture"])
        raw_spans = fixture_row.get("spans")
        if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, (str, bytes)):
            raise TypeError(f"{fixture}: evidence spans must be a list")
        fixture_events = tuple(events_by_fixture.get(fixture, ()))
        for raw_span in raw_spans:
            if not isinstance(raw_span, Mapping):
                raise TypeError(f"{fixture}: evidence span must be an object")
            span_id = int(raw_span["span_id"])
            start = int(raw_span["start_frame"])
            end = int(raw_span["end_frame"])
            if not 0 <= start < end:
                raise ValueError(f"{fixture} span {span_id}: invalid half-open bounds")
            events = tuple(event for event in fixture_events if start <= event.frame < end)
            spans.append(FixedSpan(fixture, span_id, start, end, events))
    for fixture, fixture_events in events_by_fixture.items():
        for event in fixture_events:
            containing = [
                span
                for span in spans
                if span.fixture == fixture and span.start_frame <= event.frame < span.end_frame
            ]
            if len(containing) > 1:
                raise ValueError(f"{fixture}/{event.frame}: event belongs to overlapping spans")
    return tuple(spans)


def unassigned_events(
    spans: Sequence[FixedSpan],
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
) -> tuple[FixedEvent, ...]:
    """Return retained events outside every current half-open span."""
    assigned = {
        (span.fixture, event.frame)
        for span in spans
        for event in span.events
    }
    return tuple(
        event
        for fixture in sorted(events_by_fixture)
        for event in sorted(events_by_fixture[fixture], key=lambda item: item.frame)
        if (event.fixture, event.frame) not in assigned
    )


def _greedy_matches(
    gt_frames: Sequence[int], event_frames: Sequence[int], tolerance: int
) -> tuple[tuple[int, int], ...]:
    """Match closest GT/event pairs with the production tie ordering."""
    if isinstance(tolerance, bool) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative integer")
    matches = tree_scorer._greedy_matches(
        np.asarray(gt_frames, dtype=np.int32),
        np.asarray(event_frames, dtype=np.int32),
        tolerance,
    )
    return tuple((gt_index, event_index) for gt_index, event_index, _offset in matches)


def _span_rally_candidates(
    span: FixedSpan,
    rallies: Sequence[RallyReference],
) -> tuple[RallyReference, ...]:
    return tuple(
        rally
        for rally in rallies
        if any(span.start_frame <= frame < span.end_frame for frame in rally.frames)
    )


def evaluate_span(
    span: FixedSpan,
    rallies: Sequence[RallyReference],
    target_sides: Mapping[tuple[str, int], str],
    tolerance_frames: int,
    confidence_requirement: float,
) -> SpanScore:
    """Evaluate one span without mutating its fixed event list."""
    if not 0.0 <= float(confidence_requirement) <= 1.0:
        raise ValueError("confidence requirement must lie between 0 and 1")
    events = span.events
    event_frames = tuple(event.frame for event in events)
    timing_confidence = min((event.timing_score for event in events), default=None)
    confidence_pass = timing_confidence is not None and timing_confidence >= confidence_requirement
    side_answerable = bool(events) and all(event.predicted_side is not None for event in events)
    candidates = _span_rally_candidates(span, rallies)
    rally = candidates[0] if len(candidates) == 1 else None
    matches: tuple[tuple[int, int], ...] = ()
    ground_truth_contacts = 0 if rally is None else len(rally.frames)
    if rally is not None:
        matches = _greedy_matches(rally.frames, event_frames, tolerance_frames)

    reasons: list[str] = []
    if not events:
        reasons.append(REASON_NO_EVENTS)
    if not candidates:
        reasons.append(REASON_NO_RALLY)
    elif len(candidates) > 1:
        reasons.append(REASON_MULTIPLE_RALLIES)
    if rally is not None:
        if len(matches) < len(rally.frames):
            reasons.append(REASON_MISSING_CONTACT)
        if len(matches) < len(events):
            reasons.append(REASON_EXTRA_EVENT)
        if len(matches) != len(rally.frames) or len(matches) != len(events):
            reasons.append(REASON_TIMING_MISMATCH)
    if events and not confidence_pass:
        reasons.append(REASON_LOW_TIMING_CONFIDENCE)
    if events and not side_answerable:
        reasons.append(REASON_SIDE_UNANSWERED)

    correct_side_answers = 0
    side_mismatch = False
    if rally is not None:
        for gt_index, event_index in matches:
            gt_frame = rally.frames[gt_index]
            target_side = target_sides[(span.fixture, gt_frame)]
            if events[event_index].predicted_side is None:
                # The answerability gate reports this as one whole-span
                # abstention. It is not also a wrong side answer.
                continue
            elif events[event_index].predicted_side == target_side:
                correct_side_answers += 1
            else:
                side_mismatch = True
    if side_mismatch:
        reasons.append(REASON_SIDE_INCORRECT)

    # A span can only be fully correct if it has exactly one rally candidate,
    # exact one-to-one timing, and correct side answers for every contact.
    timing_correct = (
        rally is not None
        and len(matches) == len(rally.frames)
        and len(matches) == len(events)
    )
    side_correct = timing_correct and correct_side_answers == len(events)
    kept = confidence_pass and side_answerable
    fully_correct = kept and timing_correct and side_correct and not side_mismatch and len(candidates) == 1
    return SpanScore(
        fixture=span.fixture,
        span_id=span.span_id,
        start_frame=span.start_frame,
        end_frame=span.end_frame,
        rally_id=None if rally is None else rally.rally_id,
        event_count=len(events),
        ground_truth_contacts=ground_truth_contacts,
        timing_matches=len(matches),
        timing_confidence=timing_confidence,
        confidence_requirement=float(confidence_requirement),
        confidence_pass=confidence_pass,
        side_answerable=side_answerable,
        correct_side_answers=correct_side_answers,
        kept=kept,
        fully_correct=fully_correct,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
    )


def _validate_requirements(requirements: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in requirements)
    if not values:
        raise ValueError("at least one confidence requirement is needed")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("confidence requirements must be finite values between 0 and 1")
    if tuple(sorted(set(values))) != values:
        raise ValueError("confidence requirements must be sorted and unique")
    return values


def confidence_curve(
    spans: Sequence[FixedSpan],
    rallies_by_fixture: Mapping[str, Sequence[RallyReference]],
    target_sides: Mapping[tuple[str, int], str],
    tolerance_frames_by_fixture: Mapping[str, int],
    requirements: Sequence[float],
) -> tuple[dict[str, int | float | None], ...]:
    """Return kept-yield and fully-correct counts as confidence rises."""
    values = _validate_requirements(requirements)
    curve: list[dict[str, int | float | None]] = []
    for requirement in values:
        scores = tuple(
            evaluate_span(
                span,
                rallies_by_fixture.get(span.fixture, ()),
                target_sides,
                tolerance_frames_by_fixture[span.fixture],
                requirement,
            )
            for span in spans
        )
        kept = sum(score.kept for score in scores)
        fully_correct = sum(score.fully_correct for score in scores)
        side_answerable = sum(score.side_answerable for score in scores)
        timing_confident = sum(score.confidence_pass for score in scores)
        curve.append(
            {
                "confidence_requirement": requirement,
                "rallies_kept": kept,
                "fully_correct_kept_rallies": fully_correct,
                "kept_rally_accuracy": fully_correct / kept if kept else None,
                "side_answerable_rallies": side_answerable,
                "timing_confident_rallies": timing_confident,
            }
        )
    return tuple(curve)


def score_strict_rallies(
    spans: Sequence[FixedSpan],
    rallies_by_fixture: Mapping[str, Sequence[RallyReference]],
    target_sides: Mapping[tuple[str, int], str],
    fps_by_fixture: Mapping[str, float],
    *,
    tolerance_base30: int = PRIMARY_TOLERANCE_BASE30,
    requirements: Sequence[float] = DEFAULT_CONFIDENCE_REQUIREMENTS,
    detail_requirement: float = 0.0,
) -> dict[str, Any]:
    """Score one timing tolerance and return its confidence-versus-yield curve."""
    tolerance_frames = {
        fixture: evidence_scorer.scale_base30_frames(tolerance_base30, fps)
        for fixture, fps in fps_by_fixture.items()
    }
    detail_scores = tuple(
        evaluate_span(
            span,
            rallies_by_fixture.get(span.fixture, ()),
            target_sides,
            tolerance_frames[span.fixture],
            detail_requirement,
        )
        for span in spans
    )
    curve = confidence_curve(
        spans,
        rallies_by_fixture,
        target_sides,
        tolerance_frames,
        requirements,
    )
    return {
        "tolerance_base30": tolerance_base30,
        "tolerance_frames": tolerance_frames,
        "confidence_curve": list(curve),
        "detail_requirement": detail_requirement,
        "spans": [asdict(score) for score in detail_scores],
    }


def _candidate_variant(verified_candidates: object) -> str:
    """Return the experiment variant bound to the verified score sidecar."""
    manifest = getattr(verified_candidates, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise TypeError("verified candidate scores do not expose a manifest")
    manifest_variant = manifest.get("variant")
    if manifest_variant not in TRIAL_VARIANT_RESULT_PATHS:
        raise ValueError("verified candidate manifest does not select an allowed variant")
    tree_result = getattr(verified_candidates, "tree_result", None)
    if not isinstance(tree_result, Mapping):
        raise TypeError("verified candidate scores do not expose a tree result")
    if tree_scorer._result_variant(tree_result) != manifest_variant:
        raise ValueError("tree result variant differs from the candidate manifest variant")
    return str(manifest_variant)


def _variant_model_result(result: Mapping[str, Any], variant_name: str) -> Mapping[str, Any]:
    """Select the retained result branch for one verified experiment variant."""
    result_path = TRIAL_VARIANT_RESULT_PATHS.get(variant_name)
    if result_path is None:
        raise ValueError(f"candidate variant is not an allowed rally-scoring variant: {variant_name!r}")
    model_name, feature_set = result_path
    try:
        models = result["models"]
        model = models[model_name]
        selected = model[feature_set]
    except (KeyError, TypeError) as error:
        raise ValueError(f"tree result does not contain candidate variant {variant_name!r}") from error
    if not isinstance(selected, Mapping):
        raise TypeError(f"tree result variant {variant_name!r} is malformed")
    return selected


def _prediction_frames_from_result(
    result: Mapping[str, Any],
    variant_name: str = tree_scorer.CANDIDATE_VARIANT,
) -> dict[str, np.ndarray]:
    """Read retained event frames for one verified model variant."""
    variant = _variant_model_result(result, variant_name)
    folds = variant.get("folds")
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)):
        raise TypeError(f"tree result variant {variant_name!r} does not contain folds")
    output: dict[str, np.ndarray] = {}
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise TypeError(f"tree result variant {variant_name!r} contains a malformed fold")
        fixture = str(fold["test_fixture"])
        frames = np.asarray(fold["prediction_frames"], dtype=np.int32)
        if fold["prediction_count"] != len(frames) or not np.array_equal(frames, np.unique(frames)):
            raise ValueError(f"{variant_name}/{fixture}: retained prediction frames are malformed")
        if fixture in output:
            raise ValueError(f"{variant_name}/{fixture}: retained prediction fixture is duplicated")
        output[fixture] = frames
    if set(output) != set(evidence_freezer.FIXTURE_SPECS):
        raise ValueError(f"retained {variant_name} prediction fixture set differs")
    return output


def _load_timing_rallies() -> dict[str, tuple[RallyReference, ...]]:
    """Load rally timing labels after prediction inputs are frozen."""
    import pandas as pd

    from annotator.calibration.fixtures import REPO_ROOT as CALIBRATION_ROOT
    from annotator.calibration.fixtures import SHARED_FILES, verify_file
    from annotator.calibration.scoring import load_gt_rallies

    master_pin = next(pin for pin in SHARED_FILES if pin.path.name == "shots_master.csv")
    verify_file(master_pin)
    # Keep player_side out of this table. The side labels have a separate
    # load below, after the event frames, timing scores and side predictions
    # are all fixed.
    timing_table = pd.read_csv(
        CALIBRATION_ROOT / master_pin.path,
        usecols=["vid", "set_id", "rally", "frame_num"],
    )
    raw: dict[str, Sequence[object]] = {
        fixture: load_gt_rallies(timing_table, video_id)
        for fixture, (video_id, _fps) in evidence_freezer.FIXTURE_SPECS.items()
    }
    rallies = normalise_rallies(raw)
    if sum(len(values) for values in rallies.values()) != 292:
        raise ValueError("expected 292 ShuttleSet rallies")
    if sum(len(rally.frames) for values in rallies.values() for rally in values) != 3128:
        raise ValueError("expected 3,128 ShuttleSet contacts")
    return rallies


def _load_side_labels() -> dict[tuple[str, int], str]:
    """Load player-side labels only after all event and side predictions are fixed."""
    from annotator.calibration.gt_scoring import load_gt_tables

    master, _homography, _court_info, _resolution = load_gt_tables()
    sides: dict[tuple[str, int], str] = {}
    for fixture, (video_id, _fps) in evidence_freezer.FIXTURE_SPECS.items():
        rows = master[master["vid"] == video_id]
        if "player_side" not in rows:
            raise ValueError("ShuttleSet shots_master is missing player_side")
        for frame, value in zip(rows["frame_num"], rows["player_side"], strict=True):
            side = _normalise_half(value, f"{fixture}/{frame} player_side")
            key = (fixture, int(frame))
            if side is None or key in sides:
                raise ValueError(f"{fixture}/{frame}: player-side identity differs")
            sides[key] = side
    if len(sides) != 3128:
        raise ValueError(f"expected 3,128 player-side labels, found {len(sides)}")
    return sides


def _shipped_attribution(
    arguments: argparse.Namespace,
    retained_frames: Mapping[str, np.ndarray],
    variant_name: str,
) -> dict[tuple[str, int], str | None]:
    """Replay the existing attribution rule at the fixed retained frames."""
    import score_contact_player_attribution as attribution_scorer

    freeze_arguments = argparse.Namespace(
        region_v2_manifest=arguments.feature_manifest,
        region_v2_results=arguments.tree_results,
        region_v1_manifest=arguments.region_v1_manifest,
        region_v1_results=arguments.region_v1_results,
    )
    freezes = attribution_scorer._load_tree_freezes(freeze_arguments)
    variants = {variant_name: dict(retained_frames)}
    attribution = attribution_scorer._shipped_attribution_map(arguments.data_root, freezes, variants)
    expected = {
        (fixture, int(frame))
        for fixture, frames in retained_frames.items()
        for frame in frames
    }
    if set(attribution) != expected:
        raise ValueError("shipped attribution replay does not cover every retained event")
    return attribution


def score(arguments: argparse.Namespace) -> dict[str, Any]:
    """Verify fixed inputs, freeze predictions, replay sides, then read labels."""
    evidence = evidence_scorer.verify_freeze(arguments.evidence_manifest)
    verified_features = tree_scorer.verify_freeze(arguments.feature_manifest)
    verified_candidates = tree_scorer.verify_candidate_scores(
        arguments.candidate_manifest,
        verified_features,
        arguments.tree_results,
    )
    selected_variant = _candidate_variant(verified_candidates)
    retained_frames = _prediction_frames_from_result(verified_candidates.tree_result, selected_variant)

    # Keep this boundary label-blind: events, scores, spans and shipped side
    # predictions are all fixed before either timing or player-side labels load.
    attribution = _shipped_attribution(arguments, retained_frames, selected_variant)
    events_by_fixture = retained_events_from_scores(verified_candidates.rows, attribution)
    spans = fixed_spans_from_evidence(evidence.evidence, events_by_fixture)
    unassigned = unassigned_events(spans, events_by_fixture)

    timing_rallies = _load_timing_rallies()
    side_labels = _load_side_labels()
    fps_by_fixture = {fixture: fps for fixture, (_video_id, fps) in evidence_freezer.FIXTURE_SPECS.items()}
    requirements = tuple(float(value) for value in arguments.confidence_requirements)
    primary = score_strict_rallies(
        spans,
        timing_rallies,
        side_labels,
        fps_by_fixture,
        tolerance_base30=PRIMARY_TOLERANCE_BASE30,
        requirements=requirements,
        detail_requirement=0.0,
    )
    sensitivity = score_strict_rallies(
        spans,
        timing_rallies,
        side_labels,
        fps_by_fixture,
        tolerance_base30=SENSITIVITY_TOLERANCE_BASE30,
        requirements=requirements,
        detail_requirement=0.0,
    )
    return {
        "schema": RESULTS_SCHEMA,
        "fixture_set": list(evidence_freezer.FIXTURE_SPECS),
        "selected_variant": selected_variant,
        "primary_tolerance_base30": PRIMARY_TOLERANCE_BASE30,
        "sensitivity_tolerance_base30": SENSITIVITY_TOLERANCE_BASE30,
        "labels_read_after_predictions_fixed": True,
        "inputs": {
            "contact_evidence_sha256": evidence.manifest["evidence_sha256"],
            "feature_sha256": verified_features.manifest["feature_sha256"],
            "candidate_scores_sha256": verified_candidates.manifest["candidate_sha256"],
            "tree_result_sha256": verified_candidates.manifest["tree_result_sha256"],
        },
        "unassigned_events": [asdict(event) for event in unassigned],
        "primary": primary,
        "sensitivity": sensitivity,
    }


def write_results(path: Path, payload: Mapping[str, object]) -> None:
    """Write deterministic JSON or gzip JSON."""
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.name.endswith(".gz"):
        with destination.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as zipped:
            zipped.write(encoded)
    else:
        destination.write_bytes(encoded)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    raw_root = CONTACT_DET_ROOT / "raw"
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=raw_root / "region_v2" / "run_a" / "tree_contact_features_manifest.json",
    )
    parser.add_argument(
        "--tree-results",
        type=Path,
        default=raw_root / "region_v2" / "tree_contact_results.json.gz",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        default=raw_root / "contact_evidence_manifest.json",
    )
    parser.add_argument(
        "--region-v1-manifest",
        type=Path,
        default=raw_root / "tree_trial" / "tree_contact_features_manifest.json",
    )
    parser.add_argument(
        "--region-v1-results",
        type=Path,
        default=raw_root / "tree_trial" / "tree_contact_results_with_frames.json.gz",
    )
    parser.add_argument("--data-root", type=Path, default=raw_root / "region_v2_inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confidence-requirement",
        dest="confidence_requirements",
        type=float,
        action="append",
        default=None,
        help="timing-score cut-off; repeat to select a custom confidence curve",
    )
    arguments = parser.parse_args(argv)
    if arguments.confidence_requirements is None:
        arguments.confidence_requirements = list(DEFAULT_CONFIDENCE_REQUIREMENTS)
    arguments.confidence_requirements = list(_validate_requirements(arguments.confidence_requirements))
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    payload = score(arguments)
    write_results(arguments.output, payload)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
