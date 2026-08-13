"""Focused contract tests for the fixed annotator measurement."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import annotator.e2e_court_annotator as runner
from annotator.artifact_io import load_npy, open_text_artifact, read_json_object, write_json_object
from annotator.court_evidence import (
    COURT_SCENE_SAMPLE_LIMIT,
    PERSON_COURT_MARGIN,
    SCENE_VALID_MIN_FRACTION,
    CourtEvidenceResult,
    CourtInputs,
    CourtSceneRecord,
)
from annotator.run_video import AnnotatorResult


@pytest.mark.parametrize(
    ('matched', 'unmatched_gt', 'unmatched_candidate', 'expected_f1'),
    ((0, 0, 0, None), (0, 1, 1, 0.0), (1, 1, 3, 1 / 3)),
)
def test_strict_metrics_preserve_missing_zero_and_ordinary_f1(
    matched: int,
    unmatched_gt: int,
    unmatched_candidate: int,
    expected_f1: float | None,
) -> None:
    rows: list[dict[str, object]] = []
    for row_kind, count in (
        ('matched', matched),
        ('unmatched_gt', unmatched_gt),
        ('unmatched_candidate', unmatched_candidate),
    ):
        for _index in range(count):
            rows.append(
                {
                    'tolerance_base30': 5,
                    'tolerance_frames': 4,
                    'row_kind': row_kind,
                    'offset_frames': 0 if row_kind == 'matched' else None,
                }
            )
    assert runner._strict_metrics(rows, tolerance_base30=5, fps=25.0)['f1'] == expected_f1


def _pin(path: str, root: str = "fixtures") -> dict[str, str]:
    return {"path": path, "md5": "0" * 32, "root": root}


def test_configuration_reports_the_executable_measurement_policy() -> None:
    resolved = runner.resolve(runner.BASE_ANNOTATOR_CONFIG, runner.CASES[0].fps)
    configuration = runner._configuration_values(resolved.dead_mask_mode)
    assert configuration['court_samples'] == COURT_SCENE_SAMPLE_LIMIT
    assert configuration['person_margin'] == PERSON_COURT_MARGIN
    assert configuration['scene_threshold'] == SCENE_VALID_MIN_FRACTION
    assert configuration['dead_mask_mode'] == resolved.dead_mask_mode.value
    assert configuration['landing_filter_options'] == {
        'settle_win': runner.LANDING_OPTIONS.settle_win,
        'settle_thr': runner.LANDING_OPTIONS.settle_thr,
        'settle_min': runner.LANDING_OPTIONS.settle_min,
        'carry_win': runner.LANDING_OPTIONS.carry_win,
        'carry_thr': runner.LANDING_OPTIONS.carry_thr,
    }


def _manifest_payload() -> dict[str, object]:
    producers = {key: f"producer:{key}" for key in runner._PRODUCER_KEYS}
    return {
        "schema_version": 1,
        "videos": {
            "sset_01": _pin("videos/sset_01.mp4"),
            "sset_15": _pin("videos/sset_15.mp4"),
            "sset_21": _pin("videos/sset_21.mp4"),
        },
        "track_overrides": {
            "sset_01/tracknet-stride-1": _pin("tracks/sset_01_stride1.npy"),
        },
        "courtkeynet_config": _pin(
            runner.CONFIG_PATH.resolve().relative_to(runner.REPO_ROOT.resolve()).as_posix(), "repo"
        ),
        "courtkeynet_weights": _pin("weights/courtkeynet.safetensors"),
        "producers": producers,
    }


def test_input_manifest_rejects_unknown_missing_and_historical_fields() -> None:
    payload = _manifest_payload()
    payload["old_mask"] = _pin("old_mask.npy")
    with pytest.raises(ValueError, match="fields differ"):
        runner.parse_input_manifest(payload)

    payload = _manifest_payload()
    del payload["track_overrides"]
    with pytest.raises(ValueError, match="fields differ"):
        runner.parse_input_manifest(payload)


@pytest.mark.parametrize(
    "pin",
    [
        {"path": "/tmp/video.mp4", "md5": "0" * 32, "root": "fixtures"},
        {"path": "../video.mp4", "md5": "0" * 32, "root": "fixtures"},
        {"path": "video.mp4", "md5": "A" * 32, "root": "fixtures"},
    ],
)
def test_input_manifest_rejects_bad_pin_paths_and_md5(pin: dict[str, str]) -> None:
    payload = _manifest_payload()
    payload["videos"] = {"sset_01": pin, "sset_15": _pin("sset15.mp4"), "sset_21": _pin("sset21.mp4")}
    with pytest.raises(ValueError):
        runner.parse_input_manifest(payload)


def test_fixed_matrix_and_parent_order_are_deterministic() -> None:
    assert [case.case_id for case in runner.CASES] == [
        "sset_01/tracknet-stride-8",
        "sset_01/tracknet-stride-1",
        "sset_15/tracknet-stride-8",
        "sset_21/tracknet-stride-8",
    ]
    assert [case.tracknet_producer_mode for case in runner.CASES] == [
        "nonoverlap", "weight", "nonoverlap", "nonoverlap",
    ]
    assert [parent for parent in runner.PARENTS for _case in runner.CASES] == (
        ["static_shuttleset_homography"] * 4 + ["detected_ckn_opencv_consensus"] * 4
    )


def test_selected_pin_verification_does_not_use_fixture_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    checked: list[runner.FilePin] = []
    monkeypatch.setattr(runner, "fixtures_root", lambda: Path("/tmp/fixtures"))
    monkeypatch.setattr(runner, "verify_file", checked.append)
    pins = (runner.FilePin(Path("track.npy"), "0" * 32, "fixtures"),)
    runner.verify_selected_pins(pins)
    assert checked == list(pins)


def test_selected_pin_verification_rejects_symlink_escape_before_md5_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixtures"
    outside_root = tmp_path / "outside"
    fixture_root.mkdir()
    outside_root.mkdir()
    (outside_root / "payload.bin").write_bytes(b"outside")
    (fixture_root / "link").symlink_to(outside_root, target_is_directory=True)
    monkeypatch.setattr(runner, "fixtures_root", lambda: fixture_root)
    checked = False

    def fail_if_called(_pin: runner.FilePin) -> None:
        nonlocal checked
        checked = True

    monkeypatch.setattr(runner, "verify_file", fail_if_called)
    pin = runner.FilePin(Path("link/payload.bin"), "0" * 32, "fixtures")
    with pytest.raises(ValueError, match="escapes"):
        runner.verify_selected_pins((pin,))
    assert not checked


def test_selected_pin_verification_preserves_bad_md5_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    (fixture_root / "payload.bin").write_bytes(b"payload")
    monkeypatch.setenv("ANNOTATOR_FIXTURES_ROOT", str(fixture_root))
    pin = runner.FilePin(Path("payload.bin"), "0" * 32, "fixtures")
    with pytest.raises(ValueError, match="md5 mismatch"):
        runner.verify_selected_pins((pin,))


def test_video_metadata_rejects_frame_rate_size_and_count() -> None:
    fixed = runner.CASES[0]
    with pytest.raises(ValueError, match="FPS"):
        runner.validate_video_metadata(fixed, runner.VideoMetadata(30.0, fixed.n_frames, 512, 288))
    with pytest.raises(ValueError, match="frame count"):
        runner.validate_video_metadata(fixed, runner.VideoMetadata(fixed.fps, 1, 512, 288))
    with pytest.raises(ValueError, match="dimensions"):
        runner.validate_video_metadata(fixed, runner.VideoMetadata(fixed.fps, fixed.n_frames, 288, 512))


def test_probe_video_uses_and_closes_a_decoder_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCapture:
        released = False

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            values = {
                runner.cv2.CAP_PROP_FPS: 25.0,
                runner.cv2.CAP_PROP_FRAME_COUNT: 154393.0,
                runner.cv2.CAP_PROP_FRAME_WIDTH: 512.0,
                runner.cv2.CAP_PROP_FRAME_HEIGHT: 288.0,
            }
            return values[property_id]

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(runner.cv2, "VideoCapture", lambda _path: capture)
    assert runner.probe_video(Path("fake.mp4")) == runner.VideoMetadata(25.0, 154393, 512, 288)
    assert capture.released


def test_array_contract_rejects_shape_dtype_and_ndet_count() -> None:
    fixed = runner.CASES[0]
    n_frames, n_slots = fixed.n_frames, 2
    track = np.zeros((n_frames, 3), dtype=float)
    bboxes = np.zeros((n_frames, n_slots, 4), dtype=float)
    scores = np.zeros((n_frames, n_slots), dtype=float)
    scores[:, 1] = np.nan
    kps = np.zeros((n_frames, n_slots, 17, 2), dtype=float)
    ndet = np.ones(n_frames, dtype=np.int64)
    runner._validate_arrays(fixed, track, bboxes, scores, kps, ndet)
    with pytest.raises(ValueError, match="ndet"):
        runner._validate_arrays(fixed, track, bboxes, scores, kps, np.zeros(n_frames, dtype=float))
    with pytest.raises(ValueError, match="finite-score"):
        runner._validate_arrays(fixed, track, bboxes, scores, kps, np.zeros(n_frames, dtype=np.int64))


def test_writers_preserve_headers_nulls_order_and_json_shapes(tmp_path: Path) -> None:
    fixed = runner.CASES[0]
    fixture = next(item for item in runner.FIXTURES if item.name == fixed.fixture_name)
    case = _fake_case(fixed, fixture)
    root = tmp_path / "run"
    root.mkdir()

    runner._write_raw_cuts(case, root)
    with open_text_artifact(root / "shared" / fixed.case_id / "raw_cuts.csv.gz", newline="") as handle:
        raw_rows = list(csv.reader(handle))
    assert raw_rows == [["scene_index", "start_frame", "end_frame"], ["0", "0", "2"]]

    court_result = _fake_court_result(case, runner.PARENTS[0])
    directory = root / runner.PARENTS[0] / fixed.case_id
    runner._write_scene_evidence(directory, court_result)
    with open_text_artifact(directory / "court_scenes.csv.gz", newline="") as handle:
        scene_rows = list(csv.reader(handle))
    assert scene_rows[0] == list(runner.COURT_SCENES_COLUMNS)
    scene_row = dict(zip(scene_rows[0], scene_rows[1]))
    assert scene_row["sampled_frame_indices"] == "[]"
    assert scene_row["quad_source"] == ""
    assert scene_row["raw_tl_x"] == "0.0"
    assert scene_row["raw_br_x"] == "1.0"
    assert scene_row["active_bl_y"] == "1.0"
    assert scene_row["scene_valid"] == "true"
    with open_text_artifact(directory / "scene_rows.csv.gz", newline="") as handle:
        assert next(csv.reader(handle)) == [
            "video_id", "start_frame", "end_frame", "upleft_x", "upleft_y",
            "upright_x", "upright_y", "downleft_x", "downleft_y", "downright_x", "downright_y",
        ]

    result = AnnotatorResult([], [], [], {10: [1], 2: [0]}, [], [], [], [], {}, {}, {}, {}, [])
    runner._write_annotations(directory, result)
    annotation = read_json_object(directory / "annotations.json.gz")
    assert list(annotation) == list(AnnotatorResult._fields)
    assert list(annotation["filtered_by_rally"]) == ["2", "10"]
    bool_path = directory / "bool.csv.gz"
    runner._write_rows(bool_path, ("value",), [{"value": np.bool_(True)}])
    with open_text_artifact(bool_path) as handle:
        assert handle.read().splitlines() == ["value", "true"]


def test_gt_membership_is_checked_as_one_exact_global_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_a = SimpleNamespace(name="a", gt_set_dir=Path("a"))
    fixture_b = SimpleNamespace(name="b", gt_set_dir=Path("b"))
    pins = (
        runner.FilePin(Path("a/one.csv"), "0" * 32, "repo"),
        runner.FilePin(Path("b/two.csv"), "1" * 32, "repo"),
    )
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a/one.csv").write_text("a\n", encoding="utf-8")
    (tmp_path / "b/two.csv").write_text("b\n", encoding="utf-8")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "FIXTURES", (fixture_a, fixture_b))
    monkeypatch.setattr(runner, "SHARED_FILES", pins)
    checked: list[runner.FilePin] = []
    monkeypatch.setattr(runner, "verify_selected_pins", lambda selected: checked.extend(selected))

    result = runner.verify_eligible_gt_files()
    assert result == {"a": (pins[0],), "b": (pins[1],)}
    assert checked == list(pins)
    (tmp_path / "b/extra.csv").write_text("extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="GT CSV set"):
        runner.verify_eligible_gt_files()


def test_setup_failure_returns_one_and_writes_only_terminal_run_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "inputs.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    monkeypatch.setattr(runner, "_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_require_clean_source_tree", lambda: (_ for _ in ()).throw(ValueError("dirty")))
    output_root = tmp_path / "run"
    assert runner.run_annotator_measurement(manifest_path, output_root) == 1
    payload = read_json_object(output_root / "manifest.json.gz")
    assert payload["status"] == "failed"
    assert payload["source_commit"] == "a" * 40
    assert (output_root / "setup_failure.json.gz").is_file()
    assert len(payload["cases"]) == 4
    assert all(case["status"] == "not_run" for case in payload["cases"])
    assert len(payload["configurations"]) == 8
    assert [item["status"] for item in payload["configurations"]] == ["not_run"] * 8
    assert all(item["manifest"] is None and item["failure"] is None for item in payload["configurations"])
    assert not list(output_root.glob("*/**/manifest.json.gz"))


def test_cli_does_not_accept_an_external_output_path(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        runner.main(["--manifest", str(tmp_path / "missing.json"), "--output-root", str(tmp_path / "run")])
    assert error.value.code == 2


def test_inference_failure_keeps_closed_court_evidence_and_terminal_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = runner.CASES[0]
    fixture = next(item for item in runner.FIXTURES if item.name == fixed.fixture_name)
    case = _fake_case(fixed, fixture)
    output_root = tmp_path / "run"
    output_root.mkdir()
    driver = runner.RunDriver(
        tmp_path / "inputs.json", output_root, "cpu", ("runner",), "now", 0.0,
        input_manifest=runner.parse_input_manifest(_manifest_payload()), source_commit="a" * 40,
        resolved_device="cpu", homo_df=pd.DataFrame(), resolution=pd.DataFrame(),
    )
    state = runner.ConfigurationState(
        fixed, runner.PARENTS[0], fixture, case, output_root / runner.PARENTS[0] / fixed.case_id,
        [], "now", 0.0,
    )
    monkeypatch.setattr(
        runner,
        "build_static_court_evidence",
        lambda *_args, **_kwargs: _fake_court_result(case, state.parent),
    )
    monkeypatch.setattr(runner, "run_video", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("inference")))
    runner._run_one_configuration(state, driver)
    assert state.status == "failed"
    assert state.manifest_path is not None
    assert (state.directory / "court_scenes.csv.gz").is_file()
    assert (state.directory / "failure.json.gz").is_file()
    payload = read_json_object(state.manifest_path)
    assert payload["status"] == "failed"
    assert any(item["path"].endswith("court_scenes.csv.gz") for item in payload["artifacts"])


def test_global_gt_failure_retains_inference_outputs_and_marks_inference_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = runner.CASES[0]
    fixture = next(item for item in runner.FIXTURES if item.name == fixed.fixture_name)
    case = _fake_case(fixed, fixture)
    output_root = tmp_path / "run"
    output_root.mkdir()
    driver = runner.RunDriver(
        tmp_path / "inputs.json", output_root, "cpu", ("runner",), "now", 0.0,
        input_manifest=runner.parse_input_manifest(_manifest_payload()), source_commit="a" * 40,
        resolved_device="cpu", master=pd.DataFrame(), courts={},
    )
    state = runner.ConfigurationState(
        fixed, runner.PARENTS[0], fixture, case, output_root / runner.PARENTS[0] / fixed.case_id,
        [], "now", 0.0, result=AnnotatorResult([], [], [], {}, [], [], [], [], {}, {}, {}, {}, []),
        resolved_config=runner.resolve(runner.BASE_ANNOTATOR_CONFIG, fixed.fps),
        status="inference_only",
    )
    state.directory.mkdir(parents=True)
    write_json_object(state.directory / "annotations.json.gz", {})
    driver.configurations = [state]
    monkeypatch.setattr(runner, "verify_eligible_gt_files", lambda: (_ for _ in ()).throw(ValueError("gt")))
    runner._score_configurations(driver)
    assert driver.scoring_failure_path is not None
    assert state.status == "inference_only"
    assert state.manifest_path is not None
    payload = read_json_object(state.manifest_path)
    assert payload["status"] == "inference_only"
    assert (state.directory / "annotations.json.gz").is_file()


def _fake_case(fixed: runner.FixedCase, fixture: runner.Fixture) -> runner.CaseData:
    n_frames, n_slots = 2, 2
    return runner.CaseData(
        fixed,
        fixture,
        runner.FilePin(Path(f"videos/{fixture.name}.mp4"), "0" * 32, "fixtures"),
        runner.FilePin(Path(f"tracks/{fixed.case_id}.npy"), "0" * 32, "fixtures"),
        runner.FilePin(Path("bboxes.npy"), "0" * 32, "fixtures"),
        runner.FilePin(Path("scores.npy"), "0" * 32, "fixtures"),
        runner.FilePin(Path("kps.npy"), "0" * 32, "fixtures"),
        runner.FilePin(Path("ndet.npy"), "0" * 32, "fixtures"),
        Path(f"{fixture.name}.mp4"),
        np.zeros((n_frames, 3), dtype=float),
        np.zeros((n_frames, n_slots, 4), dtype=float),
        np.array([[1.0, np.nan], [1.0, np.nan]]),
        np.zeros((n_frames, n_slots, 17, 2), dtype=float),
        np.ones(n_frames, dtype=np.int64),
        [(0, n_frames)],
        status="succeeded",
    )


def _fake_court_result(case: runner.CaseData, parent: str) -> CourtEvidenceResult:
    info = {"H": np.eye(3), "border_L": 0.0, "border_R": 1.0, "border_U": 0.0, "border_D": 1.0}
    inputs = CourtInputs(
        court_info=info,
        gate_court_info={str(case.fixture.video_id): info},
        net_band=(100.0, 110.0),
        resolution=(1280.0, 720.0),
        gate_resolution_table=pd.DataFrame({"width": [1280.0], "height": [720.0]}, index=[str(case.fixture.video_id)]),
        homography_rows=pd.DataFrame([{
            "video_id": case.fixture.video_id, "start_frame": 0, "end_frame": 2,
            "upleft_x": 0.0, "upleft_y": 0.0, "upright_x": 1.0, "upright_y": 0.0,
            "downleft_x": 0.0, "downleft_y": 1.0, "downright_x": 1.0, "downright_y": 1.0,
        }]),
        landing_error_band_m=0.1,
        active_corners_refpx=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    )
    record = CourtSceneRecord(
        case.fixture.video_id, case.fixed.case_id, parent, 0, 0, 2, (),
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), None, None, None, None,
        2, 1.0, True, None, None,
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    )
    return CourtEvidenceResult(inputs, (record,), np.ones(2, dtype=bool), np.ones(2, dtype=bool), None)


def _install_synthetic_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failed_case_id: str | None = None,
    inference_failure_video_id: int | None = None,
    scoring_failure_video_id: int | None = None,
) -> tuple[Path, dict[str, int]]:
    manifest_path = tmp_path / "inputs.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    monkeypatch.setenv("ANNOTATOR_FIXTURES_ROOT", str(fixture_root))
    monkeypatch.setattr(runner, "_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_require_clean_source_tree", lambda: None)
    monkeypatch.setattr(runner, "verify_selected_pins", lambda _pins: None)
    events = {"detector": 0, "inference": 0, "score": 0}
    synthetic_cases: list[runner.CaseData] = []
    monkeypatch.setattr(
        runner,
        "load_gt_tables",
        lambda: (pd.DataFrame({"vid": []}), pd.DataFrame(), {}, pd.DataFrame()),
    )

    def fake_make_detector(driver: runner.RunDriver) -> None:
        events["detector"] += 1
        driver.detector = object()
        driver.resolved_device = driver.device

    monkeypatch.setattr(runner, "_make_detector", fake_make_detector)
    monkeypatch.setattr(runner, "detect_scene_evidence", lambda *_args: [])
    monkeypatch.setattr(
        runner,
        "build_static_court_evidence",
        lambda case_id, parent, *_args, **_kwargs: _fake_court_result(
            next(case for case in synthetic_cases if case.fixed.case_id == case_id), parent
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_detected_court_evidence",
        lambda case_id, parent, *_args, **_kwargs: _fake_court_result(
            next(case for case in synthetic_cases if case.fixed.case_id == case_id), parent
        ),
    )
    monkeypatch.setattr(runner, "verify_eligible_gt_files", lambda: {})

    inference_fail_seen = False

    def fake_run_video(*_args, capture: runner.RunCapture, video_id: int, **_kwargs) -> AnnotatorResult:
        nonlocal inference_fail_seen
        events["inference"] += 1
        if inference_failure_video_id == video_id and not inference_fail_seen:
            inference_fail_seen = True
            raise ValueError("inference")
        capture.raw_exclusion_mask = np.zeros(2, dtype=bool)
        capture.definitive_exclusion_mask = np.zeros(2, dtype=bool)
        return AnnotatorResult([], [], [], {}, [], [], [], [], {}, {}, {}, {}, [])

    monkeypatch.setattr(runner, "run_video", fake_run_video)
    scoring_fail_seen = False

    def fake_score_video(fixture, *_args):
        nonlocal scoring_fail_seen
        events["score"] += 1
        if scoring_failure_video_id == fixture.video_id and not scoring_fail_seen:
            scoring_fail_seen = True
            raise ValueError("scoring")
        return object()

    monkeypatch.setattr(runner, "score_video", fake_score_video)
    monkeypatch.setattr(runner, "flatten_metrics", lambda _scoring: {})
    monkeypatch.setattr(runner, "_gt_rallies_for_fixture", lambda _master, _fixture: [])

    def fake_load_case(fixed, _manifest, fixture):
        if failed_case_id == fixed.case_id:
            raise ValueError("shared case")
        case = _fake_case(fixed, fixture)
        synthetic_cases.append(case)
        return case

    monkeypatch.setattr(runner, "_load_case", fake_load_case)
    return manifest_path, events


def test_synthetic_successful_assembly_writes_exactly_eight_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "inputs.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    monkeypatch.setattr(runner, "_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_require_clean_source_tree", lambda: None)
    monkeypatch.setattr(runner, "verify_selected_pins", lambda _pins: None)
    gt_loads = 0
    detector_calls = 0
    inference_calls = 0
    synthetic_cases: list[runner.CaseData] = []

    def fake_load_gt_tables():
        nonlocal gt_loads
        gt_loads += 1
        return pd.DataFrame({"vid": []}), pd.DataFrame(), {}, pd.DataFrame()

    monkeypatch.setattr(runner, "load_gt_tables", fake_load_gt_tables)
    def fake_make_detector(driver: runner.RunDriver) -> None:
        nonlocal detector_calls
        detector_calls += 1
        driver.detector = object()
        driver.resolved_device = driver.device

    monkeypatch.setattr(runner, "_make_detector", fake_make_detector)
    monkeypatch.setattr(
        runner,
        "build_static_court_evidence",
        lambda case_id, parent, *_args, **_kwargs: _fake_court_result(
            next(case for case in synthetic_cases if case.fixed.case_id == case_id), parent
        ),
    )
    monkeypatch.setattr(runner, "detect_scene_evidence", lambda *_args: [])
    monkeypatch.setattr(
        runner,
        "build_detected_court_evidence",
        lambda case_id, parent, *_args, **_kwargs: _fake_court_result(
            next(case for case in synthetic_cases if case.fixed.case_id == case_id), parent
        ),
    )
    gt_verified = False

    def fake_verify_gt_files():
        nonlocal gt_verified
        assert inference_calls == 8
        for parent in runner.PARENTS:
            for fixed in runner.CASES:
                directory = output_root / parent / fixed.case_id
                for name in (
                    "raw_replay_mask.npy.xz",
                    "definitive_exclusion_mask.npy.xz",
                    "annotations.json.gz",
                    "landing_horizons.csv.gz",
                ):
                    assert (directory / name).read_bytes()
                np.testing.assert_array_equal(
                    load_npy(directory / "raw_replay_mask.npy.xz"),
                    np.zeros(2, dtype=bool),
                )
        gt_verified = True
        return {}

    monkeypatch.setattr(runner, "verify_eligible_gt_files", fake_verify_gt_files)

    def fake_run_video(
        *_args, capture: runner.RunCapture, base, landing_options, **_kwargs,
    ) -> AnnotatorResult:
        nonlocal inference_calls
        inference_calls += 1
        assert not gt_verified
        assert base is runner.BASE_ANNOTATOR_CONFIG
        assert landing_options is runner.LANDING_OPTIONS
        capture.raw_exclusion_mask = np.zeros(2, dtype=bool)
        capture.definitive_exclusion_mask = np.zeros(2, dtype=bool)
        return AnnotatorResult([], [], [], {}, [], [], [], [], {}, {}, {}, {}, [])

    monkeypatch.setattr(runner, "run_video", fake_run_video)
    def fake_score_video(*_args):
        assert gt_verified
        return object()

    monkeypatch.setattr(runner, "score_video", fake_score_video)
    monkeypatch.setattr(runner, "flatten_metrics", lambda _scoring: {})
    monkeypatch.setattr(runner, "_gt_rallies_for_fixture", lambda _master, _fixture: [])

    def capture_case(fixed, _manifest, fixture):
        case = _fake_case(fixed, fixture)
        synthetic_cases.append(case)
        return case

    monkeypatch.setattr(runner, "_load_case", capture_case)
    output_root = tmp_path / "run"
    assert runner.run_annotator_measurement(manifest_path, output_root) == 0
    assert gt_loads == 1
    assert detector_calls == 1
    assert inference_calls == 8
    assert gt_verified
    assert len(list((output_root / "shared").rglob("raw_cuts.csv.gz"))) == 4
    manifests = sorted(output_root.glob("*/**/manifest.json.gz"))
    assert len(manifests) == 8
    assert (output_root / "manifest.json.gz").is_file()
    assert read_json_object(output_root / "input_manifest.json.gz") == _manifest_payload()
    for plain_pattern in ("*.npy", "*.json", "*.csv"):
        assert not list(output_root.rglob(plain_pattern))
    payload = read_json_object(output_root / "manifest.json.gz")
    assert payload["status"] == "succeeded"
    assert "--output-root" not in payload["command"]
    assert len(payload["configurations"]) == 8
    assert all(item["status"] == "succeeded" for item in payload["configurations"])
    assert payload["environment"]["packages"]["opencv"] == runner.cv2.__version__
    for manifest_path in manifests:
        config_payload = read_json_object(manifest_path)
        assert config_payload["configuration"]["dead_mask_mode"] == (
            config_payload["resolved_annotator_config"]["dead_mask_mode"]
        )
        for artifact in config_payload["artifacts"]:
            path = output_root / artifact["path"]
            assert path.read_bytes().__class__ is bytes
            assert runner._md5_bytes(path.read_bytes()) == artifact["md5"]


def test_shared_case_failure_marks_both_parents_failed_and_keeps_independent_cases_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "run"
    failed_case = runner.CASES[0].case_id
    manifest_path, events = _install_synthetic_runner(
        tmp_path,
        monkeypatch,
        failed_case_id=failed_case,
    )

    assert runner.run_annotator_measurement(manifest_path, output_root) == 3
    payload = read_json_object(output_root / "manifest.json.gz")
    statuses = {item["configuration_id"]: item["status"] for item in payload["configurations"]}
    assert statuses[f"{runner.PARENTS[0]}/{failed_case}"] == "failed"
    assert statuses[f"{runner.PARENTS[1]}/{failed_case}"] == "failed"
    assert all(
        statuses[f"{parent}/{fixed.case_id}"] == "succeeded"
        for parent in runner.PARENTS
        for fixed in runner.CASES[1:]
    )
    assert events["detector"] == 1
    assert events["inference"] == 6


def test_parent_inference_failure_does_not_stop_sibling_or_later_configurations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "run"
    target = runner.CASES[0]
    manifest_path, events = _install_synthetic_runner(
        tmp_path,
        monkeypatch,
        inference_failure_video_id=next(
            fixture.video_id for fixture in runner.FIXTURES if fixture.name == target.fixture_name
        ),
    )

    assert runner.run_annotator_measurement(manifest_path, output_root) == 3
    payload = read_json_object(output_root / "manifest.json.gz")
    statuses = {item["configuration_id"]: item["status"] for item in payload["configurations"]}
    assert statuses[f"{runner.PARENTS[0]}/{target.case_id}"] == "failed"
    assert statuses[f"{runner.PARENTS[1]}/{target.case_id}"] == "succeeded"
    assert events["inference"] == 8
    assert sum(status == "succeeded" for status in statuses.values()) == 7


def test_local_scoring_failure_does_not_stop_later_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "run"
    target = runner.CASES[0]
    target_video_id = next(
        fixture.video_id for fixture in runner.FIXTURES if fixture.name == target.fixture_name
    )
    manifest_path, events = _install_synthetic_runner(
        tmp_path,
        monkeypatch,
        scoring_failure_video_id=target_video_id,
    )

    assert runner.run_annotator_measurement(manifest_path, output_root) == 3
    payload = read_json_object(output_root / "manifest.json.gz")
    statuses = {item["configuration_id"]: item["status"] for item in payload["configurations"]}
    assert statuses[f"{runner.PARENTS[0]}/{target.case_id}"] == "failed"
    assert statuses[f"{runner.PARENTS[1]}/{target.case_id}"] == "succeeded"
    assert events["score"] == 8
