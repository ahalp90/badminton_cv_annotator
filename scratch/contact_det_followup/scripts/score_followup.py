"""Score revised contact streams against the saved clean test labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_followup.scripts.prediction_io import (
    REPO_ROOT,
    FrozenPredictionPack,
    normalise_side,
    read_json,
)
from scratch.contact_det_full_ds_fit.scripts.score_shuttleset22_test import (
    CleanLabels,
    HumanContact,
    HumanRally,
    player_side_metrics,
    timing_metrics,
    whole_rally_metrics,
)
from scratch.contact_det_full_ds_fit.scripts.summarise_shuttleset22_sections import (
    load_rallies,
    load_sections,
    section_and_rally_counts,
)

DEFAULT_LABELS = (
    REPO_ROOT
    / "scratch/contact_det_full_ds_fit/raw/shuttleset22-test-result/clean_labels.json.gz"
)
LABEL_SCHEMA = "shuttleset22-clean-contact-labels/1"


@dataclass(frozen=True)
class SavedLabels:
    """Clean labels rebuilt from the released scorer record."""

    path: Path
    payload: Mapping[str, Any]
    labels: CleanLabels


def load_saved_test_labels(path: Path = DEFAULT_LABELS) -> SavedLabels:
    """Load the clean labels saved by the fixed 47-video scorer."""
    payload = read_json(path)
    if payload.get("schema") != LABEL_SCHEMA or payload.get("status") != "complete":
        raise ValueError("Saved clean labels are incomplete or have another schema")
    raw_videos = payload.get("videos")
    if not isinstance(raw_videos, list):
        raise TypeError("Saved clean-label videos must be a list")

    rallies_by_fixture: dict[str, tuple[HumanRally, ...]] = {}
    population_by_fixture: dict[str, Mapping[str, int]] = {}
    for raw_video in raw_videos:
        if not isinstance(raw_video, dict):
            raise TypeError("Each saved clean-label video must be an object")
        fixture = str(raw_video["fixture"])
        if fixture in rallies_by_fixture:
            raise ValueError(f"Duplicate clean-label fixture {fixture}")
        raw_rallies = raw_video.get("rallies")
        population = raw_video.get("population")
        if not isinstance(raw_rallies, list) or not isinstance(population, dict):
            raise TypeError(f"{fixture}: rallies and population have another shape")
        rallies: list[HumanRally] = []
        for raw_rally in raw_rallies:
            if not isinstance(raw_rally, dict):
                raise TypeError(f"{fixture}: each rally must be an object")
            raw_contacts = raw_rally.get("contacts")
            if not isinstance(raw_contacts, list):
                raise TypeError(f"{fixture}: rally contacts must be a list")
            contacts: list[HumanContact] = []
            for contact in raw_contacts:
                frame = int(contact["frame"])
                contacts.append(
                    HumanContact(
                        frame,
                        normalise_side(contact.get("side"), fixture, frame),
                    )
                )
            rallies.append(
                HumanRally(
                    set_id=str(raw_rally["set_id"]),
                    rally=int(raw_rally["rally"]),
                    contacts=tuple(contacts),
                    ball_rounds=tuple(int(value) for value in raw_rally["ball_rounds"]),
                    contact_types=tuple(raw_rally["contact_types"]),
                )
            )
        rallies_by_fixture[fixture] = tuple(rallies)
        population_by_fixture[fixture] = MappingProxyType(
            {str(name): int(value) for name, value in population.items()}
        )

    corpus_hash = payload.get("annotation_corpus_sha256")
    tree_hash = payload.get("annotation_tree_sha256")
    if not isinstance(corpus_hash, str) or not isinstance(tree_hash, str):
        raise TypeError("Saved clean-label identities must be strings")
    labels = CleanLabels(
        MappingProxyType(rallies_by_fixture),
        MappingProxyType(population_by_fixture),
        corpus_hash,
        tree_hash,
    )
    return SavedLabels(path, MappingProxyType(payload), labels)


def score_streams(
    labels: CleanLabels,
    spans: Sequence[FixedSpan],
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
) -> dict[str, object]:
    """Calculate the fixed timing, player-side, and whole-rally measures."""
    return {
        "contact_timing": timing_metrics(labels, events_by_fixture),
        "player_side": player_side_metrics(labels, events_by_fixture),
        "whole_rallies": whole_rally_metrics(labels, spans, events_by_fixture),
    }


def clean_section_ids(
    predictions: FrozenPredictionPack,
    saved_labels: SavedLabels,
) -> set[tuple[str, int]]:
    """Return sections that contain exactly one complete labelled rally."""
    sections = load_sections(dict(predictions.payload))
    rallies = load_rallies(dict(saved_labels.payload))
    _section_counts, _rally_counts, clean_ids, _before, _after = section_and_rally_counts(
        sections,
        rallies,
    )
    return clean_ids


def fully_correct_ids(
    result: Mapping[str, object],
    *,
    tolerance: int = 5,
) -> set[tuple[str, int]]:
    """Return section identities marked fully correct by the old scorer."""
    whole_rallies = result["whole_rallies"]
    if not isinstance(whole_rallies, dict):
        raise TypeError("Whole-rally results must be an object")
    by_tolerance = whole_rallies["by_tolerance"]
    if not isinstance(by_tolerance, dict):
        raise TypeError("Whole-rally tolerance results must be an object")
    tolerance_result = by_tolerance[str(tolerance)]
    if not isinstance(tolerance_result, dict):
        raise TypeError("Whole-rally tolerance result must be an object")
    rows = tolerance_result["sections"]
    if not isinstance(rows, list):
        raise TypeError("Whole-rally section rows must be a list")
    return {
        (str(row["fixture"]), int(row["span_id"]))
        for row in rows
        if isinstance(row, dict) and row.get("outcome") == "fully_correct"
    }
