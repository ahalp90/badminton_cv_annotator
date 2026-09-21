"""Runner regressions: in-memory sensitivity and complete case output."""

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import run_w5
import verifier


def sensitivity_fixture():
    arrays = {}
    candidates = []
    for candidate_index in range(3):
        origin_key = f"G0:{candidate_index}"
        candidates.append({
            "origin_key": origin_key,
            "source_order": 0,
            "origin_index": candidate_index,
            "kind_order": 0,
            "hard_valid": True,
            "gates": {"camera_error": 0.05},
            "historical": {"historical_fullcourt": True, "historical_camera": True},
            "q_geom": 0.8,
            "q_geom_span_weighted": 0.8,
            "exclusive_reverse": 0.8,
            "markings": [
                {"projected_visible_span_px": span}
                for span in (0, 1, 2, None, 4, 5, 6, 7, 8, 9, 10)
            ],
        })
        for marking_index in range(11):
            prefix = f"{origin_key}::marking_{marking_index}"
            ridge = np.array([np.nan, 4, 5, 9, 10, 14, 15, 19, 20, 21], dtype=float)
            if marking_index == candidate_index:
                ridge[:] = np.nan
            if marking_index == 10:
                ridge = np.array([])
            arrays[f"{prefix}_ridge_contrast"] = ridge
            arrays[f"{prefix}_c_support"] = np.linspace(0, 1, len(ridge))
    return candidates, arrays


@pytest.mark.parametrize("output_name", ["case_records", "arrays", "manifest.json"])
def test_existing_outputs_are_not_overwritten(tmp_path, monkeypatch, output_name):
    output = tmp_path / output_name
    output.write_text("preserved evidence")
    monkeypatch.setattr(run_w5, "load_runtime", lambda *args: pytest.fail("must reject before loading data"))
    with pytest.raises(FileExistsError, match="use a new run name"):
        run_w5.run_pilot(tmp_path, tmp_path, ["case"], 1)
    assert output.read_text() == "preserved evidence"


def test_empty_sensitivity_pool_has_no_winner():
    records = run_w5.rank_sensitivity([], {}, {"rank_candidates": verifier.rank_candidates})
    assert len(records) == 4
    assert all(record["rank1_origin_key"] is None for record in records.values())


def test_all_probes_match_scalar_definition_and_read_each_array_once():
    candidates, arrays = sensitivity_fixture()
    reads = Counter()

    class CountedArrays:
        def __getitem__(self, key):
            reads[key] += 1
            return arrays[key]

    for candidate in candidates:
        actual = run_w5.probe_paint_evidence(candidate, CountedArrays())
        for probe_index, probe in enumerate(run_w5.SENSITIVITY_PROBES):
            values = []
            for marking_index in range(11):
                prefix = f"{candidate['origin_key']}::marking_{marking_index}"
                pairs = [
                    (support, ridge)
                    for support, ridge in zip(
                        arrays[f"{prefix}_c_support"], arrays[f"{prefix}_ridge_contrast"], strict=True,
                    )
                    if np.isfinite(ridge)
                ]
                values.append(
                    sum(support * (ridge >= probe) for support, ridge in pairs) / len(pairs)
                    if pairs else None
                )
            directional = []
            weighted = []
            for indices in (range(5), range(5, 11)):
                known = [values[index] for index in indices if values[index] is not None]
                directional.append(sum(known) / len(known))
                with_span = [
                    (values[index], candidate["markings"][index]["projected_visible_span_px"])
                    for index in indices
                    if values[index] is not None
                    and (candidate["markings"][index]["projected_visible_span_px"] or 0) > 0
                ]
                weighted.append(sum(value * span for value, span in with_span) / sum(span for _, span in with_span))
            assert actual[probe_index]["q_paint10"] == pytest.approx(min(directional), abs=1e-15)
            assert actual[probe_index]["q_paint10_span_weighted"] == pytest.approx(min(weighted), abs=1e-15)
    assert set(reads) == set(arrays)
    assert set(reads.values()) == {1}


@pytest.mark.parametrize("eligible", [True, False])
def test_sensitivity_rankings_and_reference_join_need_no_npz(tmp_path, monkeypatch, eligible):
    candidates, arrays = sensitivity_fixture()
    for candidate in candidates:
        candidate["gates"]["camera_error"] = 0.05 if eligible else 1.0
        candidate["corners_px"] = [[0, 0], [1, 0], [1, 1], [0, 1]]
    runtime = {"rank_candidates": verifier.rank_candidates}
    rankings = run_w5.rank_sensitivity(candidates, arrays, runtime)
    for probe_index, probe in enumerate(run_w5.SENSITIVITY_PROBES):
        expected = verifier.rank_candidates([
            run_w5.sensitivity_candidate(candidate, run_w5.probe_paint_evidence(candidate, arrays)[probe_index])
            for candidate in candidates
        ])
        record = rankings[str(int(probe))]
        assert record["provisional_rank"] == expected["provisional_rank"]
        assert record["rank1_was_ungated"] is (not eligible)
        assert record["rank1_origin_key"] == (
            expected["selected_origin_key"] if eligible else expected["ungated_provisional_rank"][0]
        )

    def forbidden_load(*args, **kwargs):
        pytest.fail("sensitivity reporting must not reopen compressed arrays")

    monkeypatch.setattr(np, "load", forbidden_load)
    monkeypatch.setattr(run_w5, "sensitivity_target", lambda *args: (np.asarray(candidates[0]["corners_px"]), "test_target"))
    result = {"case_id": "case", "sensitivity": rankings,
              "review_candidates": {candidate["origin_key"]: candidate for candidate in candidates}}
    packet = run_w5.write_sensitivity(tmp_path, tmp_path, [result], {
        "prepare_view": lambda *args: None,
        "reference_corner_error": verifier.reference_corner_error,
        "jsonable": verifier.jsonable,
    })
    assert packet["cases"]["case"]["probes"]["10"]["control_error"]["maximum"] == 0
    assert json.loads((tmp_path / "p10_sensitivity.json").read_text()) == packet


def test_process_case_keeps_conflicting_gate_candidates_and_writes_result(tmp_path, monkeypatch, capsys):
    candidates, arrays = sensitivity_fixture()
    entry = {
        "candidate_id": "48:6797",
        "corners_px": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "homography_working": np.eye(3).tolist(),
        "gates": {"geometry_valid": True, "camera_error": 0.05, "player_fractions": [1, 1]},
    }
    other = {**entry, "candidate_id": "48:0", "gates": {**entry["gates"], "camera_error": 3.099690687334811}}
    context = SimpleNamespace(case_id="test_case")
    evidence = {
        **candidates[0], "stripe_assignments": [], "q_paint10": 0.7,
        "q_paint10_span_weighted": 0.7,
    }
    raw_arrays = {key.split("::", 1)[1]: value for key, value in arrays.items() if key.startswith("G0:0::")}
    runtime = run_w5.load_verifier(Path(__file__).parent.parent)
    runtime.update({
        "prepare_view": lambda *args: context,
        "source_provenance": lambda *args: {},
        "measure_candidate": lambda *args: (evidence, raw_arrays),
        "CASE_LABELS": {"test_case": "Test case"},
    })
    monkeypatch.setattr(run_w5, "load_runtime", lambda *args: {"verifier": runtime})
    monkeypatch.setattr(run_w5, "load_populations", lambda *args, **kwargs: ([entry], [other], [], {"G0": {}, "line_template": {}}))
    monkeypatch.setattr(run_w5, "attempt_refit", lambda *args: ({}, None, {}))
    monkeypatch.setattr(run_w5, "load_reference", lambda *args: pytest.fail("reference read during automatic ranking"))
    result = run_w5.process_case(tmp_path, "test_case", tmp_path)
    assert result["B"]["selected_origin_key"] == "G0:48:6797"
    assert result["C"]["selected_origin_key"] == "G0:48:6797"
    assert set(result["review_candidates"]) == {"G0:48:6797", "G1:48:0"}
    assert result["automatic_contamination_check"] == {"match": True, "fields": []}
    assert set(result["sensitivity"]) == {"5", "10", "15", "20"}
    saved = verifier.read_json_gz(tmp_path / result["case_record"])
    assert saved["sensitivity"] == result["sensitivity"]
    assert len(saved["parents"]) == 2
    assert (tmp_path / result["array_file"]).exists()
    assert "conflicting geometry groups=1" in capsys.readouterr().out
