"""Check separate refinement pools, renewed eligibility and empty-frame accounting."""

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.annotator.independent_court import run_refit_selection
from experiments.annotator.independent_court.run_refit_selection import (
    eligible,
    rank_pools,
    summarise,
)
from scratch.court_det_fix.court_detector.image_sources import CaseProvenance, ImageKind


def candidate(identifier: str, model: str, score: float, allowed: bool = True) -> dict:
    return {"id": identifier, "model": model, "eligible": allowed, "stripe_score": score,
            "complete_agreements": 1, "disagreements": 0, "metrics": {"corner_max_error_px": 5}}


def test_each_refinement_pool_retains_starts_and_excludes_the_other_model() -> None:
    entries = [candidate("start", "start", 0.8), candidate("nominal", "nominal_centre", 0.9),
               candidate("fixed", "fixed_position", 1.0), candidate("failed_gate", "fixed_position", 1.1, False)]
    orders = rank_pools(entries)
    assert orders["starts"]["stripe_exclusive"] == ["start"]
    assert orders["nominal_centre"]["stripe_exclusive"] == ["nominal", "start"]
    assert orders["fixed_position"]["stripe_exclusive"] == ["fixed", "start"]
    assert rank_pools(entries[::-1]) == orders


def test_renewed_geometry_and_player_gates_both_apply() -> None:
    assert not eligible({"eligible": False, "reason": "geometry"})
    assert not eligible({"eligible": True, "scheme_eligible": {"original": False}})
    assert eligible({"eligible": True, "scheme_eligible": {"original": True}})


def test_empty_frame_counts_without_an_accurate_pick() -> None:
    entries = [candidate("start", "start", 0.8)]
    records = [{"id": "one", "entries": entries, "orders": rank_pools(entries)},
               {"id": "empty", "entries": [], "orders": rank_pools([])}]
    summary = summarise(records)
    assert summary["frames"] == 2
    assert all(value == 1 for value in summary["useful_eligible_pools"].values())
    assert all(count == 1 for pool in summary["accurate_picks"].values() for count in pool.values())


def test_refit_selection_preflights_all_boxes_before_import_or_output(monkeypatch, tmp_path: Path) -> None:
    pack_name = "marking_refit_inputs.json.gz"
    pack_path = tmp_path / pack_name
    binding = run_refit_selection.provenance_binding(pack_path)
    stripes = {
        "schema": "frozen-stripe-observations/2",
        "input_provenance": binding,
        "records": [{"id": "safe"}, {"id": "unsafe"}],
    }
    stripe_path = tmp_path / "stripes.json.gz"
    stripe_path.write_bytes(gzip.compress(json.dumps(stripes).encode(), mtime=0))
    (tmp_path / "replay.zip").write_bytes(b"replay bytes")
    output_path = tmp_path / "output.json.gz"
    provenance = {
        "safe": CaseProvenance("safe", ImageKind.SOURCE_FRAME, (1,), 1),
        "unsafe": CaseProvenance("unsafe", ImageKind.COMPOSITE, (1, 2), 1),
    }
    monkeypatch.setattr(run_refit_selection, "load_frozen_case_provenance", lambda _path: provenance)
    monkeypatch.setattr(run_refit_selection, "require_replay_pack", lambda *_args: None)
    monkeypatch.setattr(
        run_refit_selection,
        "read_replay_bytes",
        lambda _bytes: ({"cases": [{"id": "safe"}, {"id": "unsafe"}]}, {"records": []}),
    )
    monkeypatch.setattr(
        run_refit_selection.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(AssertionError("legacy measurement imported before preflight")),
    )
    monkeypatch.setattr(
        run_refit_selection.junctions,
        "measure",
        lambda *_args: (_ for _ in ()).throw(AssertionError("junction measurement started before preflight")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refit_selection.py", "--replay", str(tmp_path / "replay.zip"),
         "--provenance-pack", str(pack_path), "--legacy-dir", str(tmp_path / "legacy"),
         "--stripes", str(stripe_path), "--refits", str(tmp_path / "refits.json.gz"),
         "--output", str(output_path)],
    )
    with pytest.raises(ValueError, match="composite"):
        run_refit_selection.main()
    assert not output_path.exists()


def test_refit_helpers_reject_unsafe_boxes_before_prepare_or_measurement() -> None:
    provenance = CaseProvenance("case", ImageKind.COMPOSITE, (1, 2), 1)
    legacy = SimpleNamespace(
        prepare_case=lambda _case: (_ for _ in ()).throw(AssertionError("prepared unsafe boxes")),
    )
    with pytest.raises(ValueError, match="composite"):
        run_refit_selection.verify_starts({"id": "case"}, {}, {}, legacy, provenance)
    with pytest.raises(ValueError, match="composite"):
        run_refit_selection.run_case({"id": "case"}, {}, {}, legacy, {}, {}, provenance)


def test_refit_output_binding_hashes_exact_dependent_bytes() -> None:
    binding = {"pack_filename": "pack", "pack_md5": "0" * 32}
    provenance = run_refit_selection.output_provenance(
        binding, b"replay", b"stripes", b"refits", {"run_alignment.py": "1" * 32},
    )
    assert run_refit_selection.OUTPUT_SCHEMA == "renewed-fixed-refit-selection/2"
    assert provenance["stripe_artefact_md5"] == run_refit_selection.bytes_md5(b"stripes")
    assert provenance["refits_artefact_md5"] == run_refit_selection.bytes_md5(b"refits")
