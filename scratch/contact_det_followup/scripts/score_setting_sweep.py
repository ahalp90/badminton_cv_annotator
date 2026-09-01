"""Choose contact cut-off and merge settings with strict rally timing."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    evaluate_span,
)
from scratch.contact_det_followup.scripts.prediction_io import (
    DEVELOPMENT_SPLIT,
    REPO_ROOT,
    load_development_predictions,
    read_json,
)
from scratch.contact_det_full_ds_fit.scripts.baseline_config import load_baseline_config
from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    VideoSpec,
    load_development_split,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    HumanLabels,
    scale_base30_frames,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    contact_counts,
    predictions_for_settings,
)
from scratch.contact_det_full_ds_fit.scripts.score_final_contact_groups import (
    _read_raw_scores,
    load_final_score_groups,
)

SCORE_PATH = (
    REPO_ROOT
    / "scratch/contact_det_full_ds_fit/raw/final_contact_scores/combined_repeat/combined_raw_contact_scores.npy.xz"
)
SCORE_RESULT_PATH = (
    REPO_ROOT
    / "scratch/contact_det_full_ds_fit/raw/final_contact_scores/combined_repeat/final_contact_setting_result.json"
)
GROUP_PATH = REPO_ROOT / "scratch/contact_det_full_ds_fit/records/final_video_score_groups.json"
CONFIG_PATH = REPO_ROOT / "scratch/contact_det_full_ds_fit/records/baseline_runs.json"
LABEL_PATH = REPO_ROOT / "training/data/shuttleset/annotations/shots_master.csv"
OUTPUT_PATH = REPO_ROOT / "scratch/contact_det_followup/results/setting_sweep.json"
GROUP_NAMES = ("A", "B", "C", "D", "V")
BASELINE_SETTING = (0.9, 6)
TOLERANCES_AT_30_FPS = (5, 10)
PRIMARY_TOLERANCE_AT_30_FPS = 5


@dataclass(frozen=True)
class SettingEvaluation:
    """One setting's timing-complete sections and contact metrics."""

    score_cutoff: float
    duplicate_distance_at_30_fps: int
    timing_complete_ids: frozenset[tuple[str, int]]
    timing_complete_by_group: Mapping[str, int]
    contact_by_group: Mapping[str, Mapping[str, int | float | None]]
    unassigned_by_group: Mapping[str, int]


@dataclass(frozen=True)
class _PreparedSetting:
    """One setting's predictions and span assignment, shared by tolerances."""

    score_cutoff: float
    duplicate_distance_at_30_fps: int
    predictions: Mapping[str, np.ndarray]
    spans: tuple[FixedSpan, ...]
    unassigned_by_group: Mapping[str, int]


def _repository_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _events_from_kept_rows(
    scores: np.ndarray,
    kept: np.ndarray,
) -> dict[str, tuple[FixedEvent, ...]]:
    selected = scores[kept]
    fixtures = np.char.decode(selected["fixture"], "ascii")
    events: dict[str, list[FixedEvent]] = {}
    for fixture, row in zip(fixtures, selected, strict=True):
        fixture_name = str(fixture)
        events.setdefault(fixture_name, []).append(
            FixedEvent(
                fixture=fixture_name,
                frame=int(row["frame"]),
                timing_score=float(row["contact_score"]),
                predicted_side=None,
            )
        )
    return {
        fixture: tuple(sorted(fixture_events, key=lambda event: event.frame))
        for fixture, fixture_events in events.items()
    }


def assign_events_to_spans(
    span_templates: Sequence[FixedSpan],
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
) -> tuple[tuple[FixedSpan, ...], int]:
    """Assign sorted events to fixed half-open spans and count the rest."""
    frames_by_fixture = {
        fixture: np.asarray([event.frame for event in events], dtype=np.int32)
        for fixture, events in events_by_fixture.items()
    }
    assigned: set[tuple[str, int]] = set()
    spans: list[FixedSpan] = []
    for template in span_templates:
        fixture_events = events_by_fixture.get(template.fixture, ())
        fixture_frames = frames_by_fixture.get(
            template.fixture,
            np.empty(0, dtype=np.int32),
        )
        start_index = int(np.searchsorted(fixture_frames, template.start_frame, side="left"))
        end_index = int(np.searchsorted(fixture_frames, template.end_frame, side="left"))
        span_events = tuple(fixture_events[start_index:end_index])
        for event in span_events:
            identity = (event.fixture, event.frame)
            if identity in assigned:
                raise ValueError(f"{event.fixture}/{event.frame}: event belongs to two spans")
            assigned.add(identity)
        spans.append(
            FixedSpan(
                fixture=template.fixture,
                span_id=template.span_id,
                start_frame=template.start_frame,
                end_frame=template.end_frame,
                events=span_events,
            )
        )
    event_count = sum(len(events) for events in events_by_fixture.values())
    return tuple(spans), event_count - len(assigned)


def timing_complete_ids(
    spans: Sequence[FixedSpan],
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    tolerance_at_30_fps: int = PRIMARY_TOLERANCE_AT_30_FPS,
) -> frozenset[tuple[str, int]]:
    """Return sections with exact one-rally timing, without judging player side."""
    complete: set[tuple[str, int]] = set()
    rally_counts: Counter[tuple[str, str]] = Counter()
    for span in spans:
        score = evaluate_span(
            span,
            labels.rallies[span.fixture],
            labels.target_sides,
            scale_base30_frames(tolerance_at_30_fps, fps_by_fixture[span.fixture]),
            confidence_requirement=0.0,
        )
        timing_complete = (
            score.rally_id is not None
            and score.event_count == score.ground_truth_contacts
            and score.timing_matches == score.event_count
        )
        if timing_complete:
            identity = (span.fixture, span.span_id)
            complete.add(identity)
            rally_counts[(span.fixture, score.rally_id)] += 1
    if any(count > 1 for count in rally_counts.values()):
        raise ValueError("One labelled rally is timing-complete in two sections")
    return frozenset(complete)


def _combined_contact_metrics(
    evaluation: SettingEvaluation,
    groups: Sequence[str],
) -> dict[str, int | float]:
    totals = {
        name: sum(int(evaluation.contact_by_group[group][name]) for group in groups)
        for name in (
            "contact_count",
            "prediction_count",
            "matched",
            "first_contact_count",
            "first_contact_matched",
        )
    }
    precision = totals["matched"] / totals["prediction_count"]
    recall = totals["matched"] / totals["contact_count"]
    return {
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
        "first_contact_recall": (
            totals["first_contact_matched"] / totals["first_contact_count"]
        ),
    }


def choose_setting(
    evaluations: Sequence[SettingEvaluation],
    training_groups: Sequence[str],
) -> SettingEvaluation:
    """Choose one setting using only the named development groups."""
    def key(evaluation: SettingEvaluation) -> tuple[float, ...]:
        contact = _combined_contact_metrics(evaluation, training_groups)
        timing_complete = sum(
            evaluation.timing_complete_by_group[group] for group in training_groups
        )
        return (
            float(timing_complete),
            float(contact["f1"]),
            float(contact["recall"]),
            float(contact["precision"]),
            float(evaluation.duplicate_distance_at_30_fps),
            evaluation.score_cutoff,
        )

    return max(evaluations, key=key)


def _verify_score_groups(
    scores: np.ndarray,
    group_by_fixture: Mapping[str, str],
) -> None:
    fixture_names = np.char.decode(scores["fixture"], "ascii")
    source_groups = np.char.decode(scores["source_group"], "ascii")
    observed: dict[str, set[str]] = {}
    for fixture, group in zip(fixture_names, source_groups, strict=True):
        observed.setdefault(str(fixture), set()).add(str(group))
    expected = set(group_by_fixture)
    if set(observed) != expected:
        raise ValueError("Raw score fixture coverage differs from the fixed groups")
    for fixture, groups in observed.items():
        if groups != {group_by_fixture[fixture]}:
            raise ValueError(f"{fixture}: raw score source group differs")


def _verify_source_record(source: Mapping[str, object], scores: np.ndarray) -> None:
    """Check the compact score array against its companion result record."""
    if (
        source.get("schema") != "final-contact-setting-result/1"
        or source.get("status") != "complete"
        or source.get("combined_raw_score_file") != SCORE_PATH.name
        or source.get("combined_raw_score_row_count") != len(scores)
        or source.get("selected_score_cutoff") != BASELINE_SETTING[0]
        or source.get("selected_duplicate_distance_at_30_fps")
        != BASELINE_SETTING[1]
    ):
        raise ValueError("Raw score source record differs")


def _prepare_setting(
    scores: np.ndarray,
    videos: Sequence[VideoSpec],
    group_by_fixture: Mapping[str, str],
    span_templates: Sequence[FixedSpan],
    score_cutoff: float,
    duplicate_distance_at_30_fps: int,
) -> _PreparedSetting:
    predictions, kept = predictions_for_settings(
        scores,
        videos,
        score_cutoff,
        duplicate_distance_at_30_fps,
    )
    events_by_fixture = _events_from_kept_rows(scores, kept)
    spans, unassigned_count = assign_events_to_spans(span_templates, events_by_fixture)
    if sum(len(events) for events in events_by_fixture.values()) != sum(
        len(frames) for frames in predictions.values()
    ):
        raise ValueError("Kept event count differs from setting predictions")
    assigned_by_group = Counter(
        group_by_fixture[span.fixture]
        for span in spans
        for _event in span.events
    )
    event_by_group: Counter[str] = Counter()
    for fixture, events in events_by_fixture.items():
        event_by_group[group_by_fixture[fixture]] += len(events)
    unassigned_by_group = {
        group: event_by_group[group] - assigned_by_group[group]
        for group in GROUP_NAMES
    }
    if sum(unassigned_by_group.values()) != unassigned_count:
        raise ValueError("Unassigned event group counts differ")
    return _PreparedSetting(
        score_cutoff=score_cutoff,
        duplicate_distance_at_30_fps=duplicate_distance_at_30_fps,
        predictions=predictions,
        spans=spans,
        unassigned_by_group=unassigned_by_group,
    )


def _evaluate_prepared_setting(
    prepared: _PreparedSetting,
    videos_by_group: Mapping[str, Sequence[VideoSpec]],
    group_by_fixture: Mapping[str, str],
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    tolerance_at_30_fps: int,
) -> SettingEvaluation:
    """Score one prepared setting at one timing tolerance."""
    complete_ids = timing_complete_ids(
        prepared.spans,
        labels,
        fps_by_fixture,
        tolerance_at_30_fps,
    )
    timing_by_group = Counter(group_by_fixture[fixture] for fixture, _span_id in complete_ids)
    contact_by_group = {
        group: contact_counts(
            labels.contact_labels,
            prepared.predictions,
            group_videos,
            tolerance_at_30_fps=tolerance_at_30_fps,
        )
        for group, group_videos in videos_by_group.items()
    }
    return SettingEvaluation(
        score_cutoff=prepared.score_cutoff,
        duplicate_distance_at_30_fps=prepared.duplicate_distance_at_30_fps,
        timing_complete_ids=complete_ids,
        timing_complete_by_group={group: timing_by_group[group] for group in GROUP_NAMES},
        contact_by_group=contact_by_group,
        unassigned_by_group=prepared.unassigned_by_group,
    )


def _evaluate_setting(
    scores: np.ndarray,
    videos: Sequence[VideoSpec],
    videos_by_group: Mapping[str, Sequence[VideoSpec]],
    group_by_fixture: Mapping[str, str],
    span_templates: Sequence[FixedSpan],
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    score_cutoff: float,
    duplicate_distance_at_30_fps: int,
    tolerance_at_30_fps: int = PRIMARY_TOLERANCE_AT_30_FPS,
) -> SettingEvaluation:
    """Prepare and score one setting, retaining the legacy helper interface."""
    prepared = _prepare_setting(
        scores,
        videos,
        group_by_fixture,
        span_templates,
        score_cutoff,
        duplicate_distance_at_30_fps,
    )
    return _evaluate_prepared_setting(
        prepared,
        videos_by_group,
        group_by_fixture,
        labels,
        fps_by_fixture,
        tolerance_at_30_fps,
    )


def _setting_summary(evaluation: SettingEvaluation) -> dict[str, object]:
    return {
        "score_cutoff": evaluation.score_cutoff,
        "duplicate_distance_at_30_fps": evaluation.duplicate_distance_at_30_fps,
        "timing_complete_sections": len(evaluation.timing_complete_ids),
        "contact_metrics": _combined_contact_metrics(evaluation, GROUP_NAMES),
        "unassigned_events": sum(evaluation.unassigned_by_group.values()),
        "by_group": [
            {
                "group": group,
                "timing_complete_sections": evaluation.timing_complete_by_group[group],
                "contact_metrics": dict(evaluation.contact_by_group[group]),
                "unassigned_events": evaluation.unassigned_by_group[group],
            }
            for group in GROUP_NAMES
        ],
    }


def _run_sweep_at_tolerance(
    evaluations: Sequence[SettingEvaluation],
    videos_by_group: Mapping[str, Sequence[VideoSpec]],
    tolerance_at_30_fps: int,
) -> dict[str, object]:
    """Score the fixed setting menu and sensitivity folds at one tolerance."""
    if len(evaluations) != 57:
        raise ValueError("Setting menu does not contain 57 choices")
    by_setting = {
        (evaluation.score_cutoff, evaluation.duplicate_distance_at_30_fps): evaluation
        for evaluation in evaluations
    }
    baseline = by_setting[BASELINE_SETTING]
    global_best = choose_setting(evaluations, GROUP_NAMES)
    global_repairs = global_best.timing_complete_ids - baseline.timing_complete_ids
    global_breaks = baseline.timing_complete_ids - global_best.timing_complete_ids

    sensitivity_folds: list[dict[str, object]] = []
    sensitivity_repairs: set[tuple[str, int]] = set()
    sensitivity_breaks: set[tuple[str, int]] = set()
    sensitivity_baseline = sensitivity_revised = 0
    for omitted_group in GROUP_NAMES:
        selection_groups = tuple(group for group in GROUP_NAMES if group != omitted_group)
        chosen = choose_setting(evaluations, selection_groups)
        omitted_fixtures = {
            video.fixture for video in videos_by_group[omitted_group]
        }
        baseline_ids = {
            identity
            for identity in baseline.timing_complete_ids
            if identity[0] in omitted_fixtures
        }
        chosen_ids = {
            identity
            for identity in chosen.timing_complete_ids
            if identity[0] in omitted_fixtures
        }
        repairs = chosen_ids - baseline_ids
        breaks = baseline_ids - chosen_ids
        sensitivity_repairs.update(repairs)
        sensitivity_breaks.update(breaks)
        sensitivity_baseline += len(baseline_ids)
        sensitivity_revised += len(chosen_ids)
        sensitivity_folds.append(
            {
                "omitted_group": omitted_group,
                "selection_groups": list(selection_groups),
                "score_cutoff": chosen.score_cutoff,
                "duplicate_distance_at_30_fps": chosen.duplicate_distance_at_30_fps,
                "baseline_timing_complete_sections": len(baseline_ids),
                "revised_timing_complete_sections": len(chosen_ids),
                "repaired_sections": len(repairs),
                "broken_sections": len(breaks),
                "net_sections": len(repairs) - len(breaks),
                "omitted_group_contact_metrics": dict(
                    chosen.contact_by_group[omitted_group]
                ),
            }
        )
    sensitivity_net = len(sensitivity_repairs) - len(sensitivity_breaks)
    global_net = len(global_repairs) - len(global_breaks)
    decision = "continue" if global_net >= 25 else "stop"
    return {
        "tolerance_at_30_fps": tolerance_at_30_fps,
        "setting_count": len(evaluations),
        "baseline": _setting_summary(baseline),
        "global_descriptive_best": {
            **_setting_summary(global_best),
            "repaired_sections": len(global_repairs),
            "broken_sections": len(global_breaks),
            "net_sections": len(global_repairs) - len(global_breaks),
            "repaired_identities": sorted(global_repairs),
            "broken_identities": sorted(global_breaks),
        },
        "leave_one_group_sensitivity": {
            "baseline_timing_complete_sections": sensitivity_baseline,
            "revised_timing_complete_sections": sensitivity_revised,
            "repaired_sections": len(sensitivity_repairs),
            "broken_sections": len(sensitivity_breaks),
            "net_sections": sensitivity_net,
            "folds": sensitivity_folds,
        },
        "settings": [_setting_summary(evaluation) for evaluation in evaluations],
        "decision": decision,
        "decision_reason": (
            "The descriptive best clears the 25-section timing signal."
            if decision == "continue"
            else "Even the descriptive best does not clear the 25-section timing signal."
        ),
    }


def run_sweep() -> dict[str, object]:
    """Score the fixed setting menu at both reporting tolerances."""
    split = load_development_split(DEVELOPMENT_SPLIT)
    score_groups = load_final_score_groups(GROUP_PATH, split)
    group_by_fixture = {
        video.fixture: group
        for group, score_group in score_groups.items()
        for video in score_group.scored_videos
    }
    videos_by_group = {
        group: score_group.scored_videos for group, score_group in score_groups.items()
    }
    scores = _read_raw_scores(SCORE_PATH)
    source = read_json(SCORE_RESULT_PATH)
    _verify_source_record(source, scores)
    _verify_score_groups(scores, group_by_fixture)
    config = load_baseline_config(CONFIG_PATH)
    predictions = load_development_predictions()
    labels = load_human_labels(LABEL_PATH, split.videos)
    fps_by_fixture = {video.fixture: video.fps for video in split.videos}

    evaluations_by_tolerance: dict[str, list[SettingEvaluation]] = {
        str(tolerance): [] for tolerance in TOLERANCES_AT_30_FPS
    }
    for cutoff in config.score_cutoffs:
        for distance in config.duplicate_distances_at_30_fps:
            prepared = _prepare_setting(
                scores,
                split.videos,
                group_by_fixture,
                predictions.spans,
                cutoff,
                distance,
            )
            for tolerance in TOLERANCES_AT_30_FPS:
                evaluations_by_tolerance[str(tolerance)].append(
                    _evaluate_prepared_setting(
                        prepared,
                        videos_by_group,
                        group_by_fixture,
                        labels,
                        fps_by_fixture,
                        tolerance,
                    )
                )
    results_by_tolerance = {
        str(tolerance): _run_sweep_at_tolerance(
            evaluations_by_tolerance[str(tolerance)],
            videos_by_group,
            tolerance,
        )
        for tolerance in TOLERANCES_AT_30_FPS
    }
    primary = results_by_tolerance[str(PRIMARY_TOLERANCE_AT_30_FPS)]
    return {
        "schema": "contact-detector-setting-sweep/2",
        "status": "complete",
        "run_id": "timing-complete-57-setting-sweep",
        "repository_commit": _repository_commit(),
        "result_type": "Timing-only descriptive development sweep on group-excluded predictions",
        "labels_used": "40-video development labels, for scoring and setting choice",
        "side_limitation": (
            "The compact raw scores do not store player sides for most alternative frames. "
            "Timing-complete sections are not fully correct contact-and-side outputs."
        ),
        "selection_limit": (
            "The leave-one-group checks are sensitivity checks, not outer held-out tests. "
            "Models scoring the other groups may have trained on the omitted group."
        ),
        "inputs": {
            "raw_scores": str(SCORE_PATH.relative_to(REPO_ROOT)),
            "source_record_schema": source["schema"],
            "raw_score_source_commit": source["source_commit"],
            "fixed_spans": [str(path.relative_to(REPO_ROOT)) for path in predictions.paths],
            "settings": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "groups": str(GROUP_PATH.relative_to(REPO_ROOT)),
        },
        "primary_tolerance_at_30_fps": PRIMARY_TOLERANCE_AT_30_FPS,
        **primary,
        "results_by_tolerance_at_30_fps": results_by_tolerance,
    }


def main() -> None:
    """Run the setting sweep and write its compact result."""
    payload = run_sweep()
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "baseline": {
                    name: payload["baseline"][name]
                    for name in (
                        "score_cutoff",
                        "duplicate_distance_at_30_fps",
                        "timing_complete_sections",
                    )
                },
                "global_descriptive_best": {
                    name: payload["global_descriptive_best"][name]
                    for name in (
                        "score_cutoff",
                        "duplicate_distance_at_30_fps",
                        "timing_complete_sections",
                        "repaired_sections",
                        "broken_sections",
                        "net_sections",
                    )
                },
                "leave_one_group_sensitivity": {
                    name: payload["leave_one_group_sensitivity"][name]
                    for name in (
                        "baseline_timing_complete_sections",
                        "revised_timing_complete_sections",
                        "repaired_sections",
                        "broken_sections",
                        "net_sections",
                        "folds",
                    )
                },
                "decision": payload["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
