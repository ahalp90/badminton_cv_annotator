"""Provenance gates for the person-aware junction diagnostic."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.annotator.independent_court import run_junctions
from experiments.annotator.independent_court.case_provenance import (
    CaseProvenance,
    ImageKind,
    load_frozen_case_provenance,
)


def provenance(case_id: str, *, same_image: bool) -> CaseProvenance:
    return CaseProvenance(
        case_id,
        ImageKind.SOURCE_FRAME if same_image else ImageKind.COMPOSITE,
        (7,) if same_image else (7, 9, 11),
        7,
    )


def junction_case(case_id: str = "case", boxes: object = ()) -> dict:
    return {
        "id": case_id,
        "dimensions": {"width": 8, "height": 8},
        "segments_px": [[0.0, 1.0, 8.0, 1.0]],
        "bbox_px": boxes,
    }


def frozen_case(case_id: str = "case") -> dict:
    return {"id": case_id, "working_size": [8, 8], "entries": []}


def test_run_case_rejects_unsafe_boxes_before_reading_them() -> None:
    class UnreadableBoxes:
        def __array__(self, *_args, **_kwargs):
            raise AssertionError("bbox_px was read before the provenance gate")

    with pytest.raises(ValueError, match="composite"):
        run_junctions.run_case(
            junction_case(boxes=UnreadableBoxes()),
            frozen_case(),
            provenance("case", same_image=False),
        )


def test_run_case_accepts_same_image_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    measured = []

    def measure(*args):
        measured.append(args[2])
        return {"usable_sites": 0}

    monkeypatch.setattr(run_junctions.junctions, "measure", measure)
    frozen = frozen_case()
    frozen["entries"] = [{
        "id": "geometry",
        "eligible": True,
        "corners_px": [[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]],
    }]
    result = run_junctions.run_case(
        junction_case(boxes=[[1.0, 1.0, 2.0, 2.0]]),
        frozen,
        provenance("case", same_image=True),
    )
    assert result["id"] == "case"
    assert result["entries"][0]["id"] == "geometry"
    assert len(measured) == 1
    assert measured[0].shape == (1, 4)


def test_preflight_rejects_mismatched_replay_and_stripe_sets() -> None:
    inputs = {"cases": [junction_case("replay")]}
    saved = {"records": [frozen_case("stripe")]}
    with pytest.raises(ValueError, match="Replay/stripe case-set mismatch"):
        run_junctions.preflight_cases(inputs, saved, {"replay": provenance("replay", same_image=True)})


def test_preflight_rejects_unknown_provenance_cases() -> None:
    inputs = {"cases": [junction_case("case")]}
    saved = {"records": [frozen_case("case")]}
    with pytest.raises(ValueError, match="Replay/provenance case-set mismatch"):
        run_junctions.preflight_cases(inputs, saved, {"other": provenance("other", same_image=True)})


def test_preflight_checks_every_selected_case_before_execution() -> None:
    inputs = {"cases": [junction_case("safe"), junction_case("unsafe")]}
    saved = {"records": [frozen_case("safe"), frozen_case("unsafe")]}
    records = {
        "safe": provenance("safe", same_image=True),
        "unsafe": provenance("unsafe", same_image=False),
    }
    with pytest.raises(ValueError, match="composite"):
        run_junctions.preflight_cases(inputs, saved, records)


def test_real_marking_pack_is_rejected_before_any_junction_work() -> None:
    root = Path(__file__).resolve().parents[1]
    pack = root / "scratch/court_det_fix/frozen_views/packs/marking_refit_inputs.json.gz"
    records = load_frozen_case_provenance(pack)
    inputs = {"cases": [junction_case(case_id) for case_id in records]}
    saved = {"records": [frozen_case(case_id) for case_id in records]}
    with pytest.raises(ValueError, match="same-image person boxes unavailable"):
        run_junctions.preflight_cases(inputs, saved, records)


def test_replay_must_embed_the_exact_validated_pack(tmp_path: Path) -> None:
    pack = tmp_path / "pack.json.gz"
    pack.write_bytes(b"pinned pack bytes")
    replay = tmp_path / "replay.zip"
    with ZipFile(replay, "w") as archive:
        archive.writestr(run_junctions.REPLAY_PACK_MEMBER, b"other pack bytes")

    with pytest.raises(ValueError, match="must contain the validated pinned pack exactly"):
        run_junctions.require_replay_pack(replay, pack)

    with ZipFile(replay, "w") as archive:
        archive.writestr(run_junctions.REPLAY_PACK_MEMBER, pack.read_bytes())
    run_junctions.require_replay_pack(replay, pack)


def test_old_or_substituted_stripe_results_are_rejected() -> None:
    binding = run_junctions.provenance_binding(Path(next(iter(run_junctions.PACK_MD5_BY_NAME))))
    with pytest.raises(ValueError, match="old or unsupported"):
        run_junctions.validate_stripe_provenance({"schema": "frozen-stripe-observations/1"}, binding)
    substituted = {
        "schema": "frozen-stripe-observations/2",
        "input_provenance": {**binding, "pack_md5": "0" * 32},
    }
    with pytest.raises(ValueError, match="does not match"):
        run_junctions.validate_stripe_provenance(substituted, binding)
