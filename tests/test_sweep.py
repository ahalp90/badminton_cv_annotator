from __future__ import annotations

import json
import csv
import argparse
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from annotator.calibration import sweep
from annotator.calibration.fixtures import SSET_01
from annotator.calibration.gt_scoring import RunVideoInputs
from annotator.calibration.scoring import CONTACT_TOLERANCES_BASE30
from annotator.calibration.schemas import CSV_COLUMNS_BY_FILENAME
from annotator.rally_segmentation import ServeStartClose, ServeStartMode
from annotator.types import ContactCandidate, SpanOpen
from annotator.resolve import resolve


def _row(spec: sweep.CandidateSpec, *, covered: int = 100) -> dict[str, object]:
    row: dict[str, object] = {
        "label": spec.label, "covered": covered, "covered_fraction": covered / 113,
        "split": 1, "missed": 1, "spurious_spans": 1, "clean_covered": covered - 1,
        "swallowed_rallies": 0, "max_rallies_in_one_span": 1,
        "strict_align_median": 1.0, "strict_align_p90": 2.0,
        "start_alignment_median": 1.0, "recall_5": 0.8, "precision_raw_5": 0.8,
        "changed_from_defaults": len(spec.overrides_base30), "settings": sweep._settings(spec),
        "_clean_offsets": {("set", 1): 1.0}, "_split_log": [],
    }
    row.update(spec.overrides_base30)
    return row


def test_sweep_uses_the_shared_contact_tolerances() -> None:
    assert sweep.CONTACT_TOLERANCES_BASE30 is CONTACT_TOLERANCES_BASE30


def test_sweep_row_preserves_the_legacy_speed_schema_value() -> None:
    master = pd.DataFrame(
        {
            "vid": [SSET_01.video_id] * 3,
            "set_id": ["synthetic"] * 3,
            "rally": [1, 1, 1],
            "frame_num": [10, 20, 30],
        }
    )
    result = SimpleNamespace(
        spans=[(5, 35)],
        filtered_contacts=[
            ContactCandidate(0, frame, True, True, False)
            for frame in (10, 20, 30)
        ],
    )
    row = sweep._row_for_result(SSET_01, sweep.shipped_spec(), result, master)
    assert row["min_contact_speed"] == sweep.LEGACY_MIN_CONTACT_SPEED == 0.005
    assert row["rest_speed"] == 0.002


def test_boundary_values_resolve_to_frozen_25fps_literals() -> None:
    expected = {
        "rest_speed": (0.002, 0.003, 0.005, 0.01, 0.02),
        "rest_window": (5, 7, 9, 15, 21),
        "end_rest_frames": (20, 30, 45, 60, 75, 90),
        "start_speed": (0.01, 0.015, 0.02, 0.03, 0.05),
        "start_min_frames": (1, 2, 3, 5),
    }
    for key, values in sweep.BOUNDARY_VALUES.items():
        resolved = [getattr(resolve(sweep.BaseAnnotatorConfig(overrides_base30={key: value}), 25).thresholds, key) for value in values]
        assert resolved == list(expected[key])


def test_grids_and_routing_cover_every_key_class() -> None:
    assert len(sweep.build_boundary_grid()) == 3000
    boundary = {key: values[0] for key, values in sweep.BOUNDARY_VALUES.items()}
    assert len(sweep.build_contact_grid(boundary)) == 108
    spec = sweep.CandidateSpec(
        "grid",
        {**boundary, "smooth_window": 4, "gap_state_demotion_bound": 2,
         "quiet_start_window": 3, "threshold_bh": 0.1,
         "stillness_threshold_bh": 0.2, "serve_stillness_window_frames": 4},
        {"span_open": "BACK_FILL", "mode": "TRIM", "close": "BURST"},
    )
    base, serve = sweep._base_and_serve(spec)
    assert base.overrides_base30 is not None
    assert base.overrides_base30["smooth_window"] == 4
    assert base.gap_state_demotion_bound == 2
    assert serve is not None and serve.threshold_bh == 0.1
    assert sweep.serialise_spec(spec)["overrides_base30"]["threshold_bh"] == 0.1


@pytest.mark.parametrize("key", ("unrelated_field", "rest_widnow"))
def test_sweep_rejects_unrelated_and_misspelled_fps_fields(key: str) -> None:
    spec = sweep.CandidateSpec("invalid", {key: 1.0}, {})

    with pytest.raises(ValueError, match=f"cannot route numeric sweep key {key!r}"):
        sweep._base_and_serve(spec)


def test_quality_floor_uses_greatest_coverage(tmp_path, monkeypatch, capsys) -> None:
    low = sweep.CandidateSpec("grid", {key: values[0] for key, values in sweep.BOUNDARY_VALUES.items()}, {})
    high = sweep.CandidateSpec("grid", {key: values[-1] for key, values in sweep.BOUNDARY_VALUES.items()}, {})
    monkeypatch.setattr(sweep, "build_boundary_grid", lambda: [low, high])
    status = sweep.run_sweep(
        fixture=SSET_01, out_dir=tmp_path, phase="boundary", fixture_inputs=object(),
        candidate_runner=lambda *, fixture_inputs, candidate_spec: _row(candidate_spec, covered=50 if candidate_spec is low else 100),
    )
    assert status == 0
    assert "WITHHELD" not in capsys.readouterr().out
    assert (tmp_path / "config_winner.json").exists()


def test_routing_and_fake_score_orchestration(tmp_path, monkeypatch) -> None:
    boundary = sweep.CandidateSpec("grid", {key: values[0] for key, values in sweep.BOUNDARY_VALUES.items()}, {})
    contact = sweep.CandidateSpec("grid", {**boundary.overrides_base30, **{key: values[0] for key, values in sweep.CONTACT_VALUES.items()}}, {})
    monkeypatch.setattr(sweep, "build_boundary_grid", lambda: [boundary])
    monkeypatch.setattr(sweep, "build_contact_grid", lambda _: [contact])
    status = sweep.run_sweep(fixture=SSET_01, out_dir=tmp_path, fixture_inputs=object(), candidate_runner=lambda *, fixture_inputs, candidate_spec: _row(candidate_spec))
    assert status == 0
    for filename in CSV_COLUMNS_BY_FILENAME:
        assert (tmp_path / filename).read_text(encoding="utf-8").splitlines()
    document = json.loads((tmp_path / "config_winner.json").read_text(encoding="utf-8"))
    assert document["meta"]["phases_run"] == ["boundary", "contact"]
    assert document["meta"]["schema_version"] == 1
    assert document["meta"]["tuning_video_ids"] == [SSET_01.video_id]
    assert document["meta"]["input_digests"] == sweep._input_digest_bundle(SSET_01)
    assert document["contact"]["overrides_base30"]["smooth_window"] == 4


def test_withheld_quality_floor_writes_empty_outputs_and_removes_stale_winner(tmp_path, monkeypatch, capsys) -> None:
    spec = sweep.CandidateSpec("grid", {key: values[0] for key, values in sweep.BOUNDARY_VALUES.items()}, {})
    monkeypatch.setattr(sweep, "build_boundary_grid", lambda: [spec])
    stale = tmp_path / "config_winner.json"
    stale.write_text("stale", encoding="utf-8")

    assert sweep.run_sweep(
        fixture=SSET_01, out_dir=tmp_path, phase="boundary", fixture_inputs=object(),
        candidate_runner=lambda *, fixture_inputs, candidate_spec: _row(candidate_spec, covered=0),
    ) == 0
    message = "CALIBRATION VERDICT WITHHELD: best grid coverage is below the quality floor"
    captured = capsys.readouterr()
    assert captured.out.splitlines().count(message) == 1
    assert captured.err.splitlines().count(message) == 1
    assert not stale.exists()
    assert all((tmp_path / filename).read_text(encoding="utf-8").splitlines()[0] == ",".join(columns)
               for filename, columns in CSV_COLUMNS_BY_FILENAME.items())


def test_withheld_contact_writes_empty_outputs_and_removes_stale_winner(tmp_path, monkeypatch, capsys) -> None:
    boundary = {key: values[0] for key, values in sweep.BOUNDARY_VALUES.items()}
    contact = sweep.CandidateSpec("grid", boundary, {})
    monkeypatch.setattr(sweep, "build_contact_grid", lambda _: [contact])
    (tmp_path / "config_winner.json").write_text("stale", encoding="utf-8")

    assert sweep.run_sweep(
        fixture=SSET_01, out_dir=tmp_path, phase="contact", boundary_spec=contact,
        fixture_inputs=object(), candidate_runner=lambda *, fixture_inputs, candidate_spec: {
            **_row(candidate_spec), "recall_5": None, "precision_raw_5": None,
        },
    ) == 0
    message = "CALIBRATION VERDICT WITHHELD: no contact grid row has measurable logical-5 recall and raw precision"
    captured = capsys.readouterr()
    assert captured.out.splitlines().count(message) == 1
    assert captured.err.splitlines().count(message) == 1
    assert not (tmp_path / "config_winner.json").exists()
    assert set(CSV_COLUMNS_BY_FILENAME) == {path.name for path in tmp_path.glob("*.csv")}


def test_boundary_report_frontier_and_allowance_rules() -> None:
    first = _row(sweep.CandidateSpec("grid", {}, {}), covered=10)
    second = {**_row(sweep.CandidateSpec("grid", {"rest_speed": 0.01}, {}), covered=8),
             "split": 0, "_clean_offsets": {("set", 1): 2.0}}
    reference = {**_row(sweep.shipped_spec(), covered=100), "_clean_offsets": {}}
    report = sweep._boundary_report_rows([first, second, reference], 10)
    assert report[0]["coverage_gap_from_best"] == 0
    assert {row["coverage_gap_from_best"] for row in report} == {0, 2}
    assert all(row["rule"] != "shipped_defaults" for row in report)
    assert sweep._contact_frontier([
        {**_row(sweep.CandidateSpec("grid", {"smooth_window": 4}, {})), "recall_5": 0.9, "precision_raw_5": 0.8},
        {**_row(sweep.CandidateSpec("grid", {"smooth_window": 5}, {})), "recall_5": 0.8, "precision_raw_5": 0.9},
        {**_row(sweep.CandidateSpec("grid", {"smooth_window": 9}, {})), "recall_5": 0.7, "precision_raw_5": 0.7},
        {**_row(sweep.CandidateSpec("grid", {"smooth_window": 4}, {})), "recall_5": 0.9, "precision_raw_5": 0.8},
    ])[:2] and len(sweep._contact_frontier([
        {**_row(sweep.CandidateSpec("grid", {"smooth_window": 4}, {})), "recall_5": 0.9, "precision_raw_5": 0.8},
        {**_row(sweep.CandidateSpec("grid", {"smooth_window": 5}, {})), "recall_5": 0.8, "precision_raw_5": 0.9},
        {**_row(sweep.CandidateSpec("grid", {"smooth_window": 9}, {})), "recall_5": 0.7, "precision_raw_5": 0.7},
        {**_row(sweep.CandidateSpec("grid", {"smooth_window": 4}, {})), "recall_5": 0.9, "precision_raw_5": 0.8},
    ])) == 3


def test_stability_deduplicates_boundary_sweeps_and_keeps_none_auxiliary(tmp_path, monkeypatch) -> None:
    boundary = sweep.CandidateSpec("grid", {key: values[0] for key, values in sweep.BOUNDARY_VALUES.items()}, {})
    contact = sweep.CandidateSpec("grid", {**boundary.overrides_base30,
        **{key: values[0] for key, values in sweep.CONTACT_VALUES.items()}}, {})
    calls: list[sweep.CandidateSpec] = []
    monkeypatch.setattr(sweep, "build_boundary_grid", lambda: [boundary])
    monkeypatch.setattr(sweep, "build_contact_grid", lambda _: [contact])

    def runner(*, fixture_inputs, candidate_spec):
        calls.append(candidate_spec)
        return {**_row(candidate_spec), "recall_5": 0.9, "precision_raw_5": 0.9}

    assert sweep.run_sweep(fixture=SSET_01, out_dir=tmp_path, fixture_inputs=object(), candidate_runner=runner) == 0
    assert len(calls) == 4  # boundary grid + boundary reference + live contact grid + contact reference
    stability = list(csv.DictReader((tmp_path / "contact_stability.csv").open(encoding="utf-8")))
    assert all(row["same_winner_as_live"] == "True" for row in stability)
    assert sweep._alignment_rows([{"rule": "r", "_clean_offsets": {}}])[1][0]["n_rallies"] == 0


def test_stability_serialises_empty_auxiliary_row_and_true_match(tmp_path, monkeypatch) -> None:
    first = sweep.CandidateSpec("grid", {key: values[0] for key, values in sweep.BOUNDARY_VALUES.items()}, {})
    second = sweep.CandidateSpec("grid", {key: values[1] for key, values in sweep.BOUNDARY_VALUES.items()}, {})
    contact_first = sweep.CandidateSpec("grid", {
        **first.overrides_base30, **{key: values[0] for key, values in sweep.CONTACT_VALUES.items()},
    }, {})
    monkeypatch.setattr(sweep, "build_boundary_grid", lambda: [first, second])
    monkeypatch.setattr(sweep, "build_contact_grid", lambda boundary: [
        contact_first if boundary == first.overrides_base30 else sweep.CandidateSpec(
            "grid", {**boundary, **{key: values[0] for key, values in sweep.CONTACT_VALUES.items()}}, {},
        )
    ])
    report = [
        {**_row(first), "rule": "rally_id_f1"},
        {**_row(first), "rule": "fewest_merges"},
        {**_row(second), "rule": "coverage_first"},
        {**_row(first), "rule": "tightest_start"},
    ]
    monkeypatch.setattr(sweep, "_boundary_report_rows", lambda rows, n_rallies: report)

    def runner(*, fixture_inputs, candidate_spec):
        if candidate_spec.label == "shipped_defaults":
            return {**_row(candidate_spec), "recall_5": None, "precision_raw_5": None}
        eligible = candidate_spec.overrides_base30 == contact_first.overrides_base30
        return {**_row(candidate_spec), "recall_5": 0.9 if eligible else None,
                "precision_raw_5": 0.9 if eligible else None}

    assert sweep.run_sweep(fixture=SSET_01, out_dir=tmp_path, fixture_inputs=object(), candidate_runner=runner) == 0
    stability = list(csv.DictReader((tmp_path / "contact_stability.csv").open(encoding="utf-8")))
    assert stability[0]["same_winner_as_live"] == "True"
    assert stability[2]["same_winner_as_live"] == "False"
    assert stability[2]["label"] == ""


def test_alignment_split_log_and_csv_serialisation_rules(tmp_path) -> None:
    own, shared = sweep._alignment_rows([
        {"rule": "a", "_clean_offsets": {("s", 1): 1.0, ("s", 2): 3.0}},
        {"rule": "b", "_clean_offsets": {("s", 2): 5.0}},
    ])
    assert [row["n_rallies"] for row in own] == [2, 1]
    assert [row["n_rallies"] for row in shared] == [1, 1]
    assert sweep._csv_value(None) == ""
    with pytest.raises(ValueError, match="not finite"):
        sweep._csv_value(float("nan"))
    spec = sweep.shipped_spec()
    display = sweep._display_config(spec)
    assert display["rest_speed"] == 0.002
    assert display["start_speed"] == pytest.approx(0.015)
    row = {"video_id": 1, "gt_rally_index": 0, "piece_spans": "10-20;30-40"}
    sweep._write_csv(tmp_path / "split.csv", ("video_id", "gt_rally_index", "piece_spans"), [row])
    assert (tmp_path / "split.csv").read_text(encoding="utf-8").splitlines()[1] == "1,0,10-20;30-40"


def test_routing_depth_enums_and_changed_defaults() -> None:
    spec = sweep.CandidateSpec("grid", {
        "gap_state_demotion_bound": 2, "quiet_start_window": 3,
        "threshold_bh": 0.1, "stillness_threshold_bh": 0.2,
        "serve_stillness_window_frames": 4,
    }, {"span_open": "BACK_FILL", "mode": "TRIM", "close": "BURST"})
    base, serve = sweep._base_and_serve(spec)
    assert base.gap_state_demotion_bound == 2 and base.quiet_start_window == 3
    assert base.span_open is SpanOpen.BACK_FILL
    assert serve == sweep.ServeStartConfig(0.1, ServeStartMode.TRIM, ServeStartClose.BURST, 0.2)
    assert sweep._base_and_serve(sweep.CandidateSpec("grid", {}, {}))[0].span_open is SpanOpen.BACK_FILL
    quiet_base, quiet_serve = sweep._base_and_serve(
        sweep.CandidateSpec("quiet", {"quiet_start_window": 3}, {})
    )
    assert quiet_base.quiet_start_window == 3
    assert quiet_base.span_open is None
    assert quiet_serve is None
    for strategies in ({"unknown": "TRIM"}, {"mode": "bad"}, {"close": "bad"}, {"span_open": "bad"}):
        with pytest.raises(ValueError):
            sweep._base_and_serve(sweep.CandidateSpec("grid", {}, strategies))
    shipped = sweep._shipped_base30_values()
    assert sweep._changed_from_defaults(sweep.CandidateSpec("grid", {
        "rest_speed": shipped["rest_speed"], "start_speed": shipped["start_speed"],
        "smooth_window": 4, "contact_impulse_multiple": 4,
        "contact_dedup_radius_frames": 3,
    }, {})) == 1
    for key, values in sweep.CONTACT_VALUES.items():
        for value in values:
            sweep._base_and_serve(sweep.CandidateSpec("grid", {key: value}, {}))


def test_serve_threshold_bh_routes_and_old_threshold_is_closed() -> None:
    old_spec = sweep.CandidateSpec("named", {"threshold": 0.8}, {"mode": "TRIM"})
    with pytest.raises(ValueError, match="cannot route numeric sweep key 'threshold'"):
        sweep._base_and_serve(old_spec)

    new_spec = sweep.CandidateSpec("named", {"threshold_bh": 0.8}, {"mode": "TRIM"})
    _base, serve = sweep._base_and_serve(new_spec)
    assert serve is not None and serve.threshold_bh == 0.8
    assert sweep.ServeStartConfig(threshold_bh=0.8, mode=ServeStartMode.TRIM).threshold_bh == 0.8


def _valid_winner() -> dict[str, object]:
    return {"meta": {"fixture": "sset_01", "phases_run": ["boundary"], "verdict": "issued", "tolerances_base30": [1, 2, 5, 10]},
            "boundary": {"overrides_base30": {key: values[0] for key, values in sweep.BOUNDARY_VALUES.items()}, "strategies": {}}}


@pytest.mark.parametrize("mutate", [
    lambda doc: doc.update({"extra": 1}),
    lambda doc: doc["meta"].update({"extra": 1}),
    lambda doc: doc["meta"].pop("fixture"),
    lambda doc: doc["boundary"].update({"overrides_base30": 1}),
    lambda doc: doc["boundary"]["overrides_base30"].update({"rest_speed": True}),
    lambda doc: doc["boundary"]["overrides_base30"].update({"rest_speed": float("inf")}),
    lambda doc: doc["boundary"]["overrides_base30"].update({"rest_speed": 123}),
])
def test_loader_rejects_malformed_winner_documents(tmp_path, mutate) -> None:
    document = _valid_winner()
    mutate(document)
    path = tmp_path / "config_winner.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        sweep.load_boundary_winner(path, "sset_01")
    with pytest.raises(argparse.ArgumentTypeError):
        sweep._fixture("no-such-fixture")


def test_loader_rejects_duplicate_keys_and_masks(tmp_path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"meta": {}, "meta": {}}', encoding="utf-8")
    with pytest.raises(ValueError):
        sweep.load_boundary_winner(path, "sset_01")
    inputs = RunVideoInputs((np.zeros(4), object(), object(), object(), object()), {}, None, {})  # type: ignore[arg-type]
    for mask in (np.zeros((4, 1), dtype=bool), np.zeros(4, dtype=np.int8), np.zeros(3, dtype=bool), np.ones(4, dtype=bool)):
        mask_path = tmp_path / "mask.npy"
        np.save(mask_path, mask)
        with pytest.raises(ValueError):
            sweep._replace_mask(inputs, mask_path)
    mask_path = tmp_path / "valid.npy"
    np.save(mask_path, np.array([True, False, True, False], dtype=bool))
    assert np.array_equal(
        sweep._replace_mask(inputs, mask_path).keyword["raw_exclusion_mask"],
        [True, False, True, False],
    )


def test_main_classifies_configuration_and_execution_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["sweep", "--fixture", "sset_01", "--out-dir", str(tmp_path), "--phase", "contact"])
    with pytest.raises(SystemExit) as error:
        sweep.main()
    assert error.value.code == 2
    for args in (("--mask-npy", str(tmp_path / "bad.npy")), ("--boundary-winner-json", str(tmp_path / "missing.json"))):
        monkeypatch.setattr(sys, "argv", ["sweep", "--fixture", "sset_01", "--out-dir", str(tmp_path), *args])
        if args[0] == "--mask-npy":
            np.save(tmp_path / "bad.npy", np.ones(2, dtype=bool))
            monkeypatch.setattr(sweep, "build_run_video_inputs", lambda fixture: RunVideoInputs((np.zeros(2), object(), object(), object(), object()), {}, None, {}))
        with pytest.raises(SystemExit) as error:
            sweep.main()
        assert error.value.code == 2
    monkeypatch.setattr(sweep, "build_run_video_inputs", lambda fixture: (_ for _ in ()).throw(RuntimeError("missing root")))
    monkeypatch.setattr(sys, "argv", ["sweep", "--fixture", "sset_01", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        sweep.main()
    assert error.value.code == 2
    monkeypatch.setattr(sweep, "build_run_video_inputs", lambda fixture: object())
    monkeypatch.setattr(sweep, "run_sweep", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("runner failed")))
    assert sweep.main() == 1


def test_failure_continuation_and_pool_initialiser_contract(capsys) -> None:
    specs = [sweep.CandidateSpec("grid", {"rest_speed": value}, {}) for value in (0.1, 0.2, 0.3)]
    def runner(*, fixture_inputs, candidate_spec):
        if candidate_spec is specs[1]:
            raise RuntimeError("boom")
        return _row(candidate_spec)
    rows = sweep._run_candidates(specs, phase="boundary", fixture_inputs=object(), candidate_runner=runner, workers=1)
    assert [row["rest_speed"] for row in rows] == [0.1, 0.3]
    error = capsys.readouterr().err
    assert "boundary grid index 1" in error and "'rest_speed': 0.2" in error and "boom" in error
    with pytest.raises(RuntimeError, match="not initialised"):
        sweep._pool_production_candidate(specs[0])
    with pytest.raises(RuntimeError, match="no boundary grid row succeeded"):
        sweep._run_candidates(specs, phase="boundary", fixture_inputs=object(), candidate_runner=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("all fail")), workers=1)


def test_load_boundary_winner_accepts_provenance_document(tmp_path, monkeypatch) -> None:
    """A boundary-phase winner written since the provenance change must feed a contact run."""
    boundary = {key: values[0] for key, values in sweep.BOUNDARY_VALUES.items()}
    digests = {"fixture-input.npy": "abc123"}
    monkeypatch.setattr(sweep, "_input_digest_bundle", lambda fixture: digests)
    monkeypatch.setattr(sweep, "verify_file", lambda pin: None)
    document = sweep.winner_document(
        "sset_01", ["boundary"], boundary=sweep.winner_spec(boundary, {}),
        schema_version=1, tuning_video_ids=[SSET_01.video_id], input_digests=digests,
    )
    path = tmp_path / "config_winner.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    loaded = sweep.load_boundary_winner(path, "sset_01")
    assert loaded.overrides_base30 == boundary
