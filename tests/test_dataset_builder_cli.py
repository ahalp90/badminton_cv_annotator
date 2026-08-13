"""One-command coordinator ordering, stop, and resume contracts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pytest

from annotator.run_video import AnnotatorResult
from annotator.video_metadata import VideoMetadata
from dataset_builder import cli, tracknet_input, vision
from dataset_builder.cli import StageExecution, StagePlan
from dataset_builder.manifest import write_run_manifest
from dataset_builder.models import InterpreterIdentity, RunManifest, StageOutcome
from dataset_builder.records import (
    RALLY_RECORD_COLLECTION_SCHEMA,
    RALLY_RECORD_PROJECTION_SCHEMA,
    RALLY_RECORD_SCHEMA,
    RallyRecordArtifacts,
    RallyRecordProjection,
)
from dataset_builder.selection import (
    COMMENTARY_FAILED,
    COMMENTARY_INELIGIBLE,
    COMMENTARY_NO_PAIR,
    COMMENTARY_UNAVAILABLE_TRANSCRIPT,
    load_selection,
)
from dataset_builder.shuttle_quality import summarize_shuttle_quality
from scraper import commentary_cleaning, download_scraped_videos

ORIGINAL_RUN_CLEAN = commentary_cleaning.run_clean


def _write_config(
    path: Path,
    *,
    max_videos: int = 2,
    commentary_enabled: bool = True,
    pose_shards: int = 1,
) -> Path:
    path.write_text(
        f'''[run]
source_dataset = "scraped-professional"
max_videos = {max_videos}
download_workers = 1

[search]
result_count = 5

[search.terms]
match = ["professional singles full match"]

[environment]
tracknet_python = "BADMINTON_TRACKNET_PYTHON"
pose_python = "BADMINTON_POSE_PYTHON"

[models]
tracknet_dir = "src/shared/tracknetv3"
tracknet = "weights/tracknet.pt"
inpaint = ""
court = "weights/court.safetensors"

[vision]
tracknet_workers = 1
tracknet_batch_size = 8
tracknet_stride = 8
tracknet_large_video = true
pose_device = "cuda"
pose_n_max = 16
pose_shards = {pose_shards}
court_device = "cuda"
court_resize_mode = "pad"

[commentary]
enabled = {str(commentary_enabled).lower()}
api_key_environment = "GEMINI_API_KEY"
''',
        encoding="utf-8",
    )
    return path


@dataclass
class _FixtureControl:
    outcomes: dict[str, StageOutcome] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    planned: list[str] = field(default_factory=list)
    executed: list[str] = field(default_factory=list)
    restored: list[str] = field(default_factory=list)
    preflights: int = 0
    secret_value: str | None = None


class _FixtureRuntime:
    def __init__(self, run_dir: Path, control: _FixtureControl) -> None:
        self.run_dir = run_dir
        self.control = control
        self.interpreter = InterpreterIdentity("/fixture/python", "Python 3.12")

    def preflight(self) -> None:
        self.control.preflights += 1

    def plans(self, phase: str, _manifest: RunManifest) -> tuple[StagePlan, ...]:
        self.control.planned.append(phase)
        index = cli.PHASE_ORDER.index(phase)
        dependencies = () if index == 0 else (cli.PHASE_ORDER[index - 1],)
        outcome = self.control.outcomes.get(phase, StageOutcome.PROCESSED)
        artifact = self.run_dir / "fixture" / f"{phase}.txt"
        version = self.control.versions.get(phase, "v1")

        def execute() -> StageExecution:
            self.control.executed.append(phase)
            if outcome is not StageOutcome.PROCESSED:
                suffix = "" if self.control.secret_value is None else self.control.secret_value
                return StageExecution(
                    outcome,
                    {},
                    {"rows": 0},
                    f"fixture {outcome.value}: {suffix}",
                )
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(f"{phase}:{version}\n", encoding="utf-8")
            counts = self.control.counts.get(phase, {"rows": 1})
            return StageExecution(outcome, {"artifact": artifact}, counts)

        def restore() -> None:
            self.control.restored.append(phase)
            if artifact.read_text(encoding="utf-8") != f"{phase}:{version}\n":
                raise ValueError("fixture artifact differs")

        validators = {}
        if outcome is StageOutcome.PROCESSED:
            validators = {
                "artifact": lambda _root: (
                    artifact.read_text(encoding="utf-8") == f"{phase}:{version}\n"
                ),
            }
        return (StagePlan(
            name=phase,
            contract_version=f"{phase}/0.1",
            dependencies=dependencies,
            command=("fixture", phase),
            configuration={"version": version},
            interpreter=self.interpreter,
            model_weights={},
            inputs={},
            execute=execute,
            restore=restore,
            semantic_validators=validators,
            secret_values=(
                () if self.control.secret_value is None else (self.control.secret_value,)
            ),
            failure_outcome=(
                StageOutcome.UNAVAILABLE
                if phase in {"transcript", "triage", "commentary_cleaning", "commentary_pairing"}
                else StageOutcome.FAILED
            ),
            blocks_pipeline=phase in {
                "search", "selection", "download", "assembly", "report",
            },
        ),)


def _factory(control: _FixtureControl):
    def build(_config: cli.BuilderConfig, run_dir: Path, _source_commit: str) -> _FixtureRuntime:
        return _FixtureRuntime(run_dir, control)

    return build


@pytest.fixture(autouse=True)
def stable_source_commit(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_clean_source_commit", lambda _root: "a" * 40)


def test_successful_fixture_run_visits_every_phase_in_order(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "trial.toml")
    control = _FixtureControl()

    result = cli.run_dataset_builder(
        config,
        tmp_path / "run",
        runtime_factory=_factory(control),
    )

    assert control.preflights == 1
    assert control.planned == list(cli.PHASE_ORDER)
    assert control.executed == list(cli.PHASE_ORDER)
    assert [event.name for event in result.events] == list(cli.PHASE_ORDER)
    assert all(event.outcome is StageOutcome.PROCESSED for event in result.events)
    assert result.stopped_after is None


def test_required_failure_records_outcome_and_stops_dependent_phases(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "trial.toml")
    control = _FixtureControl(outcomes={"download": StageOutcome.FAILED})

    result = cli.run_dataset_builder(
        config,
        tmp_path / "run",
        runtime_factory=_factory(control),
    )

    expected = list(cli.PHASE_ORDER[:cli.PHASE_ORDER.index("download") + 1])
    assert control.planned == expected
    assert control.executed == expected
    assert result.stopped_after == "download"
    assert result.manifest.stages[-1].outcome is StageOutcome.FAILED


def test_optional_commentary_unavailability_does_not_stop_visual_phases(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "trial.toml")
    control = _FixtureControl(outcomes={
        "transcript": StageOutcome.UNAVAILABLE,
        "triage": StageOutcome.UNAVAILABLE,
        "commentary_cleaning": StageOutcome.UNAVAILABLE,
        "commentary_pairing": StageOutcome.UNAVAILABLE,
    })

    result = cli.run_dataset_builder(
        config,
        tmp_path / "run",
        runtime_factory=_factory(control),
    )

    assert control.planned == list(cli.PHASE_ORDER)
    assert result.stopped_after is None
    assert result.events[-1].name == "report"
    assert result.events[-1].outcome is StageOutcome.PROCESSED


def test_unchanged_resume_restores_every_stage_without_execution(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "trial.toml")
    run_dir = tmp_path / "run"
    first = _FixtureControl()
    cli.run_dataset_builder(config, run_dir, runtime_factory=_factory(first))
    second = _FixtureControl()

    result = cli.run_dataset_builder(config, run_dir, runtime_factory=_factory(second))

    assert second.executed == []
    assert second.restored == list(cli.PHASE_ORDER)
    assert all(event.reused for event in result.events)
    assert all("integrity" in (event.reason or "") for event in result.events)


def test_changed_stage_configuration_invalidates_only_it_and_dependants(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "trial.toml")
    run_dir = tmp_path / "run"
    cli.run_dataset_builder(config, run_dir, runtime_factory=_factory(_FixtureControl()))
    changed = _FixtureControl(versions={"pose": "v2"})

    result = cli.run_dataset_builder(config, run_dir, runtime_factory=_factory(changed))

    split = cli.PHASE_ORDER.index("pose")
    assert changed.restored == list(cli.PHASE_ORDER[:split])
    assert changed.executed == list(cli.PHASE_ORDER[split:])
    assert [stage.name for stage in result.manifest.stages] == list(cli.PHASE_ORDER)


def test_configuration_is_strict_and_resolves_repo_relative_models(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "trial.toml")

    config = cli.load_builder_config(config_path, repo_root=tmp_path)

    assert config.max_videos == 2
    assert config.search_count == 5
    assert config.tracknet_model == tmp_path / "weights" / "tracknet.pt"
    assert config.inpaint_model is None
    malformed = config_path.read_text(encoding="utf-8").replace(
        "download_workers = 1", "download_workers = 1\nunknown = true",
    )
    config_path.write_text(malformed, encoding="utf-8")
    with pytest.raises(ValueError, match="run fields differ"):
        cli.load_builder_config(config_path, repo_root=tmp_path)


def test_configuration_rejects_zero_max_videos(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "trial.toml", max_videos=0)

    with pytest.raises(ValueError, match="run.max_videos must be positive"):
        cli.load_builder_config(config_path, repo_root=tmp_path)


def test_completed_run_with_no_selected_videos_is_a_terminal_error(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "trial.toml")
    control = _FixtureControl(counts={
        "selection": {"selected": 0},
        "assembly": {"videos": 0, "rallies": 0},
    })

    result = cli.run_dataset_builder(
        config,
        tmp_path / "run",
        runtime_factory=_factory(control),
    )

    assert result.stopped_after is None
    assert result.terminal_error == "selection produced no videos"


def test_dirty_source_refuses_before_runtime_or_run_directory_creation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _write_config(tmp_path / "trial.toml")
    called = False

    def fail_source(_root: Path) -> str:
        raise RuntimeError("tracked files differ from HEAD")

    def forbidden_factory(
        _config: cli.BuilderConfig,
        _run_dir: Path,
        _source_commit: str,
    ) -> _FixtureRuntime:
        nonlocal called
        called = True
        raise AssertionError("runtime must not be built")

    monkeypatch.setattr(cli, "_clean_source_commit", fail_source)
    run_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match="tracked files differ"):
        cli.run_dataset_builder(config, run_dir, runtime_factory=forbidden_factory)

    assert called is False
    assert not run_dir.exists()


def test_stage_errors_redact_registered_secret_values(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "trial.toml")
    secret = "fixture-api-secret"
    control = _FixtureControl(
        outcomes={"transcript": StageOutcome.UNAVAILABLE},
        secret_value=secret,
    )

    result = cli.run_dataset_builder(
        config,
        tmp_path / "run",
        runtime_factory=_factory(control),
    )

    transcript = next(stage for stage in result.manifest.stages if stage.name == "transcript")
    event = next(event for event in result.events if event.name == "transcript")
    assert secret not in (transcript.reason or "")
    assert secret not in (event.reason or "")
    assert "<redacted>" in (transcript.reason or "")


class _ConcreteRuntimeFixture:
    """Fixture external boundaries around the real coordinator runtime."""

    video_id = "fixture-video"

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        with_commentary: bool = True,
        commentary_enabled: bool = True,
        source_basename: str | None = None,
        stale_source_basename: str | None = None,
        commentary_eligible: bool = True,
        with_rally: bool = False,
    ) -> None:
        monkeypatch.syspath_prepend(str(cli.REPO_ROOT / "src" / "bst_x"))
        from dataset_builder import _pipeline_runtime, _runtime_support, _vision_plans

        self.runtime_module = _pipeline_runtime
        self.runtime_support = _runtime_support
        self.vision_plans = _vision_plans
        self.with_commentary = with_commentary
        self.source_basename = source_basename or f"{self.video_id}.mp4"
        self.stale_source_basename = stale_source_basename
        self.commentary_eligible = commentary_eligible
        self.with_rally = with_rally
        self.config_path = _write_config(
            tmp_path / "trial.toml",
            commentary_enabled=commentary_enabled,
        )
        self.run_dir = tmp_path / "run"
        self.workspace = self.run_dir / "workspace"
        self.transcript_dir = self.workspace / "transcripts"
        self.chunk_dir = self.workspace / "chunks"
        self.video_dir = self.workspace / "videos"
        self.scraper_config = self.runtime_module.scraper_config
        self.boundary_calls: list[str] = []
        self.candidate = self._candidate()
        self.pose = self._pose()
        self.court = vision.CourtVision(((0, 3),), object())
        self.tracknet_dir = tmp_path / "tracknet"
        self.tracknet_dir.mkdir()
        (self.tracknet_dir / "batch_predict.py").write_bytes(b"fixture")
        self.tracknet_model = tmp_path / "tracknet.pt"
        self.court_model = tmp_path / "court.safetensors"
        self.tracknet_model.write_bytes(b"fixture tracknet")
        self.court_model.write_bytes(b"fixture court model")
        self._install(monkeypatch)

    @property
    def expected_stage_names(self) -> list[str]:
        video_id = self.video_id
        return [
            "search", "transcript", "triage", "selection", "download",
            f"metadata:{video_id}", "commentary_cleaning", f"tracknet_input:{video_id}",
            f"shuttle:{video_id}",
            f"pose:{video_id}", f"court:{video_id}", f"annotation:{video_id}",
            f"commentary_pairing:{video_id}", f"primitive_projection:{video_id}",
            "assembly", "report",
        ]

    def factory(
        self,
        config: cli.BuilderConfig,
        destination: Path,
        source_commit: str,
    ) -> cli.PipelineRuntime:
        effective = replace(
            config,
            tracknet_dir=self.tracknet_dir,
            tracknet_model=self.tracknet_model,
            court_model=self.court_model,
        )
        runtime = self.runtime_module.DefaultPipelineRuntime(
            effective,
            destination,
            source_commit,
        )

        def preflight() -> None:
            identity = InterpreterIdentity("/fixture/python", "Python 3.12")
            runtime.current_interpreter = identity
            runtime.tracknet_interpreter = identity
            runtime.pose_interpreter = identity
            runtime.ffmpeg_interpreter = identity
            runtime.detector = object()

        runtime.preflight = preflight  # type: ignore[method-assign]
        return runtime

    def _install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        paths = {
            "SCRAPE_DIR": self.workspace,
            "CANDIDATES_CSV": self.workspace / "candidates.csv",
            "TRANSCRIPTS_DIR": self.transcript_dir,
            "CHUNKS_DIR": self.chunk_dir,
            "VIDEOS_DIR": self.video_dir,
        }
        for name, value in paths.items():
            monkeypatch.setattr(self.scraper_config, name, value)
        if self.with_commentary:
            monkeypatch.setenv("GEMINI_API_KEY", "fixture-secret")
        else:
            monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(self.runtime_module.search_index, "build_candidates", self.search)
        monkeypatch.setattr(
            self.runtime_module.transcript_acquisition,
            "run_transcript_acquisition",
            self.transcript,
        )
        monkeypatch.setattr(
            self.runtime_module.relevance_triage,
            "run_relevance_triage",
            self.triage,
        )
        monkeypatch.setattr(
            self.runtime_module.download_scraped_videos,
            "download_all_videos",
            self.download,
        )
        monkeypatch.setattr(self.runtime_module, "probe_video_metadata", self.metadata)
        monkeypatch.setattr(self.runtime_module.commentary_cleaning, "run_clean", self.clean)
        monkeypatch.setattr(self.runtime_module.commentary_cleaning, "run_fine", self.fine)
        monkeypatch.setattr(
            self.vision_plans,
            "create_tracknet_input",
            self.tracknet_input_stage,
        )
        monkeypatch.setattr(self.vision_plans, "extract_all_shuttles", self.tracknet)
        monkeypatch.setattr(self.vision_plans, "extract_rtmlib_pose_stage", self.pose_stage)
        monkeypatch.setattr(vision, "build_detected_court_stage", self.court_stage)
        monkeypatch.setattr(self.runtime_support, "load_court_vision", self.load_court)
        monkeypatch.setattr(
            self.runtime_module,
            "run_full_annotation_stage",
            self.annotation,
        )

    def search(self, *_args: object, **_kwargs: object) -> list[dict[str, object]]:
        self.boundary_calls.append("search")
        rows = [dict(self.candidate)]
        self.scraper_config.write_candidates(rows)
        return rows

    def transcript(self, *_args: object, **_kwargs: object) -> None:
        self.boundary_calls.append("transcript")
        if not self.with_commentary:
            return
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        (self.transcript_dir / f"{self.video_id}.json").write_text(json.dumps({
            "source": "youtube_asr",
            "segments": [{"start": 0.0, "end": 1.0, "text": "fixture call"}],
        }), encoding="utf-8")

    def triage(self, *, rows: list[dict[str, object]]) -> dict[str, bool]:
        self.boundary_calls.append("triage")
        rows[0]["keep"] = "True"
        self.scraper_config.write_candidates(rows)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        (self.chunk_dir / f"{self.video_id}.json").write_text(json.dumps([{
            "chunk_id": f"{self.video_id}_c0",
            "start": 1.0,
            "end": 2.0,
            "text": "fixture call",
        }]), encoding="utf-8")
        return {self.video_id: True}

    def download(self, **kwargs: object) -> list[download_scraped_videos.DownloadOutcome]:
        self.boundary_calls.append("download")
        assert kwargs["selected_video_ids"] == (self.video_id,)
        assert kwargs["accept_silent_video"] is True
        self.video_dir.mkdir(parents=True, exist_ok=True)
        source = self.video_dir / self.source_basename
        source.write_bytes(b"fixture video")
        entry: dict[str, object] = {
            "video_id": self.video_id,
            "title": self.candidate["title"],
            "url": self.candidate["url"],
            "commentary_eligible": self.commentary_eligible,
        }
        entries: dict[str, dict[str, object]] = {}
        if self.stale_source_basename is not None:
            entries[self.stale_source_basename] = dict(entry)
        entries[source.name] = entry
        download_scraped_videos._write_manifest(
            self.video_dir / self.scraper_config.SOURCES_MANIFEST_NAME,
            {"dataset": "scraped-professional", "videos": entries},
        )
        return [download_scraped_videos.DownloadOutcome(
            self.video_id,
            source.name,
            entry,
            False,
        )]

    def metadata(self, source: Path) -> VideoMetadata:
        self.boundary_calls.append("metadata")
        return VideoMetadata(source.resolve(), Fraction(25), 3, 100, 50)

    def clean(self, *, rows: list[dict[str, object]]) -> dict[str, int]:
        self.boundary_calls.append("commentary_cleaning")
        assert rows[0]["keep"] == "True"
        path = self.chunk_dir / f"{self.video_id}.json"
        chunks = json.loads(path.read_text(encoding="utf-8"))
        chunks[0]["text_clean"] = "clean fixture call"
        path.write_text(json.dumps(chunks), encoding="utf-8")
        return {self.video_id: 1}

    def fine(self, *_args: object, **_kwargs: object) -> None:
        return None

    def tracknet_input_stage(
        self, *, source: VideoMetadata, output_dir: Path, **_kwargs: object,
    ) -> tracknet_input.TrackNetInput:
        self.boundary_calls.append("tracknet_input")
        proxy_path, metadata_path = tracknet_input.tracknet_input_paths(source, output_dir)
        proxy_path.parent.mkdir(parents=True, exist_ok=True)
        proxy_path.write_bytes(b"fixture TrackNet input")
        metadata = replace(
            source,
            source_path=proxy_path.resolve(),
            width=tracknet_input.TRACKNET_INPUT_WIDTH,
            height=tracknet_input.TRACKNET_INPUT_HEIGHT,
        )
        vision.save_json_gz(metadata_path, metadata.to_dict())
        return tracknet_input.TrackNetInput(proxy_path, metadata_path, metadata)

    def tracknet(self, **kwargs: object) -> None:
        self.boundary_calls.append("shuttle")
        assert kwargs["enable_inpainting"] is False
        video_paths = kwargs["video_paths"]
        assert isinstance(video_paths, list) and len(video_paths) == 1
        source_stem = Path(video_paths[0]).stem
        output = Path(kwargs["output_csv_dir"]) / f"{source_stem}_ball.csv"
        assert not output.exists()
        output.write_text(
            "Frame,X,Y,Visibility\n0,256,144,1\n1,128,72,1\n2,0,0,0\n",
            encoding="utf-8",
        )
        stride = 8
        sidecar = output.parent / f"{source_stem}_stride{stride}_inpaint_mask.json.gz"
        vision.save_json_gz(sidecar, {
            "schema": "inpaint_fill_mask/1",
            "index_space": "frame",
            "inpaint_status": "disabled",
            "n_rows": 3,
            "eval_mode": "nonoverlap",
            "stride": stride,
            "th_h_px": tracknet_input.TRACKNET_INPUT_HEIGHT * 0.05,
            "tracknet_ckpt": "tracknet.pt",
            "inpaintnet_ckpt": None,
            "input_video": Path(video_paths[0]).name,
            "extracted_utc": "2026-08-13T00:00:00Z",
            "inpaint_selected": [],
        })

    def pose_stage(
        self,
        *,
        output_dir: Path,
        **_kwargs: object,
    ) -> vision.PoseExtraction:
        self.boundary_calls.append("pose")
        artifacts = vision.save_pose_arrays(output_dir, self.pose, 3)
        return vision.PoseExtraction(self.pose, artifacts, ("fixture-pose",))

    def court_stage(
        self,
        *,
        output_dir: Path,
        **_kwargs: object,
    ) -> vision.CourtVision:
        self.boundary_calls.append("court")
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = vision.CourtArtifacts(
            output_dir / vision.COURT_EVIDENCE_FILENAME,
            output_dir / vision.COURT_KEEP_VOTE_FILENAME,
            output_dir / vision.COURT_PRESENT_FILENAME,
        )
        for path in artifacts.as_mapping().values():
            path.write_bytes(b"fixture court")
        self.court = vision.CourtVision(((0, 3),), object(), artifacts)
        return self.court

    def load_court(self, *_args: object, **_kwargs: object) -> vision.CourtVision:
        return self.court

    def annotation(
        self,
        *,
        output_dir: Path,
        **kwargs: object,
    ) -> vision.AnnotationOutput:
        self.boundary_calls.append("annotation")
        result = (
            AnnotatorResult(
                [(0, 3)], [], [], {}, [None], [0], [None], [None], {}, {}, {}, {}, [],
            )
            if self.with_rally
            else AnnotatorResult([], [], [], {}, [], [], [], [], {}, {}, {}, {}, [])
        )
        run = vision.AnnotationRun(
            self.video_id,
            result,
            np.zeros(3, dtype=bool),
            np.zeros(3, dtype=bool),
            summarize_shuttle_quality(
                kwargs["track"],
                kwargs["inpaint_fill_mask"],
                kwargs["guard_codes"],
                frozenset({1, 2, 3}),
            ),
        )
        output = vision.AnnotationOutput(
            run,
            vision.persist_annotation_run(output_dir, run, 3),
        )
        return output

    @classmethod
    def _candidate(cls) -> dict[str, object]:
        return {
            "video_id": cls.video_id,
            "url": f"https://example.test/{cls.video_id}",
            "title": "Fixture singles match",
            "channel": "Fixture channel",
            "duration_s": "3600",
            "upload_date": "20260801",
            "search_term": "professional singles full match",
            "substream": "match",
            "doubles_suspect": "False",
            "duration_suspect": "False",
            "upload_date_suspect": "False",
            "keep": "",
            "triage_verdict": "",
        }

    @staticmethod
    def _pose() -> vision.PoseArrays:
        return vision.PoseArrays(
            kps=np.full((3, 1, 17, 2), np.nan, dtype=np.float32),
            bboxes=np.full((3, 1, 4), np.nan, dtype=np.float32),
            scores=np.full((3, 1), np.nan, dtype=np.float32),
            kp_scores=np.full((3, 1, 17), np.nan, dtype=np.float32),
            ndet=np.zeros(3, dtype=np.int8),
        )


def test_default_runtime_fixture_executes_and_resumes_every_concrete_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch)

    first = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    assert [event.name for event in first.events] == fixture.expected_stage_names
    assert first.stopped_after is None
    stages = {stage.name: stage for stage in first.manifest.stages}
    for stage_name in ("triage", "commentary_cleaning"):
        assert (
            stages[stage_name].configuration["request_timeout_seconds"]
            == fixture.scraper_config.LLM_REQUEST_TIMEOUT_S
        )
    assert dict(stages["commentary_cleaning"].counts) == {"cleaned": 1, "videos": 1}
    selected = load_selection(fixture.run_dir / "selected_videos.csv.gz")
    assert selected[0].commentary_status == COMMENTARY_NO_PAIR
    first_boundary_calls = list(fixture.boundary_calls)

    second = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    assert fixture.boundary_calls == first_boundary_calls
    assert [event.name for event in second.events] == fixture.expected_stage_names
    assert [(event.name, event.reason) for event in second.events if not event.reused] == []


def test_partial_selected_download_fails_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataset_builder import _pipeline_runtime

    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch)
    control = _FixtureControl()
    selected_ids = ("accepted-video", "retry-video")
    attempts = 0

    def download(**kwargs: object) -> list[download_scraped_videos.DownloadOutcome]:
        nonlocal attempts
        attempts += 1
        assert kwargs["selected_video_ids"] == selected_ids
        fixture.video_dir.mkdir(parents=True, exist_ok=True)
        if attempts == 2:
            assert (fixture.video_dir / f"{selected_ids[0]}.mp4").is_file()
        entries: dict[str, dict[str, object]] = {}
        outcomes: list[download_scraped_videos.DownloadOutcome] = []
        for video_id in selected_ids[:attempts]:
            path = fixture.video_dir / f"{video_id}.mp4"
            if not path.exists():
                path.write_bytes(b"fixture video")
            entry: dict[str, object] = dict(
                video_id=video_id,
                title=f"Fixture {video_id}",
                url=f"https://example.test/{video_id}",
                commentary_eligible=True,
            )
            entries[path.name] = entry
            outcomes.append(download_scraped_videos.DownloadOutcome(
                video_id, path.name, entry, False,
            ))
        if attempts == 1:
            outcomes.append(download_scraped_videos.DownloadOutcome(
                selected_ids[1], None, None, True,
            ))
        download_scraped_videos._write_manifest(
            fixture.video_dir / fixture.scraper_config.SOURCES_MANIFEST_NAME,
            {"dataset": "scraped-professional", "videos": entries},
        )
        return outcomes

    monkeypatch.setattr(
        _pipeline_runtime.download_scraped_videos,
        "download_all_videos",
        download,
    )

    candidate_input = fixture.run_dir / "stages" / "triage" / "candidates.csv"
    selection_input = (
        fixture.run_dir / "stages" / "selection" / fixture.runtime_module.SELECTED_VIDEOS_FILENAME
    )
    candidate_input.parent.mkdir(parents=True)
    selection_input.parent.mkdir(parents=True)
    candidate_input.write_text("fixture candidates\n", encoding="utf-8")
    selection_input.write_bytes(b"fixture selection")

    class PartialDownloadRuntime(_FixtureRuntime):
        def __init__(
            self,
            config: cli.BuilderConfig,
            run_dir: Path,
            source_commit: str,
        ) -> None:
            super().__init__(run_dir, control)
            self.download_runtime = _pipeline_runtime.DefaultPipelineRuntime(
                config,
                run_dir,
                source_commit,
            )
            self.download_runtime.current_interpreter = self.interpreter

        def plans(self, phase: str, manifest: RunManifest) -> tuple[StagePlan, ...]:
            if phase != "download":
                return super().plans(phase, manifest)
            self.control.planned.append(phase)
            self.download_runtime.state.selected_ids = selected_ids
            return tuple(self.download_runtime._download_plans(manifest))

    def factory(
        config: cli.BuilderConfig,
        run_dir: Path,
        source_commit: str,
    ) -> PartialDownloadRuntime:
        return PartialDownloadRuntime(config, run_dir, source_commit)

    first = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=factory,
    )
    first_executed = tuple(control.executed)
    second = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=factory,
    )
    first_download = next(event for event in first.events if event.name == "download")
    second_download = next(event for event in second.events if event.name == "download")
    first_recorded = next(stage for stage in first.manifest.stages if stage.name == "download")
    recorded = next(stage for stage in second.manifest.stages if stage.name == "download")

    assert first.stopped_after == "download"
    assert first_download.outcome is StageOutcome.FAILED
    assert dict(first_recorded.counts) == {"selected": 2, "downloaded": 1, "failed": 1}
    assert "metadata" not in first_executed
    assert second.stopped_after is None
    assert second_download.outcome is StageOutcome.PROCESSED
    assert second_download.reused is False
    assert dict(recorded.counts) == {"selected": 2, "downloaded": 2, "failed": 0}
    assert attempts == 2


def test_projection_is_produced_once_and_restored_for_assembly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch, with_rally=True)
    producer = fixture.runtime_module.assemble_rally_records
    calls: list[str] = []

    def assemble_spy(**kwargs: object) -> RallyRecordProjection:
        calls.append(str(kwargs["video_id"]))
        return producer(**kwargs)

    monkeypatch.setattr(fixture.runtime_module, "assemble_rally_records", assemble_spy)

    first = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )
    projection_path = (
        fixture.run_dir
        / "stages"
        / "primitive_projection"
        / fixture.video_id
        / fixture.runtime_module.PROJECTION_FILENAME
    )
    projection_payload = vision.load_json_gz(projection_path)
    records_path = fixture.run_dir / "rally_records.json.gz"
    collection = vision.load_json_gz(records_path)

    assert first.stopped_after is None
    assert calls == [fixture.video_id]
    assert projection_payload["schema"] == RALLY_RECORD_PROJECTION_SCHEMA
    assert projection_payload["source"] == collection["sources"][0]
    assert projection_payload["records"] == collection["records"]
    records_path.unlink()

    resumed = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )
    events = {event.name: event for event in resumed.events}

    assert calls == [fixture.video_id]
    assert events[f"primitive_projection:{fixture.video_id}"].reused is True
    assert events["assembly"].reused is False
    assert events["report"].reused is False
    rebuilt = vision.load_json_gz(records_path)
    assert rebuilt["sources"] == [projection_payload["source"]]
    assert rebuilt["records"] == projection_payload["records"]


def test_corrupt_projection_invalidates_resume_and_is_reproduced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch, with_rally=True)
    producer = fixture.runtime_module.assemble_rally_records
    calls = 0

    def assemble_spy(**kwargs: object) -> RallyRecordProjection:
        nonlocal calls
        calls += 1
        return producer(**kwargs)

    monkeypatch.setattr(fixture.runtime_module, "assemble_rally_records", assemble_spy)
    cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )
    projection_path = (
        fixture.run_dir
        / "stages"
        / "primitive_projection"
        / fixture.video_id
        / fixture.runtime_module.PROJECTION_FILENAME
    )
    payload = vision.load_json_gz(projection_path)
    del payload["records"][0]["contacts"]["accepted"]
    vision.save_json_gz(projection_path, payload)

    resumed = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )
    events = {event.name: event for event in resumed.events}

    assert calls == 2
    assert events[f"primitive_projection:{fixture.video_id}"].reused is False
    assert events["assembly"].reused is False
    assert events["report"].reused is False


def test_assembly_uses_selected_video_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataset_builder import _pipeline_runtime

    config = cli.load_builder_config(_write_config(tmp_path / "trial.toml"))
    run_dir = tmp_path / "run"
    manifest = RunManifest("ordered-run", "2026-08-12T00:00:00Z")
    write_run_manifest(run_dir, manifest)
    runtime = _pipeline_runtime.DefaultPipelineRuntime(config, run_dir, "a" * 40)
    runtime.current_interpreter = InterpreterIdentity("/fixture/python", "Python 3.12")
    runtime.state.selected_ids = ("second", "first")
    runtime.state.active_ids = {"first", "second"}
    runtime.state.projections = {
        "first": RallyRecordProjection("a" * 64, {"video_id": "first"}, ({"row": 1},)),
        "second": RallyRecordProjection("b" * 64, {"video_id": "second"}, ({"row": 2},)),
    }
    captured: list[str] = []

    def write_spy(
        _run_dir: Path,
        _manifest: RunManifest,
        projections: list[RallyRecordProjection],
        **_kwargs: object,
    ) -> RallyRecordArtifacts:
        captured.extend(str(projection.source["video_id"]) for projection in projections)
        return RallyRecordArtifacts(
            run_dir / "rally_records.json.gz",
            run_dir / "run_manifest.json.gz",
        )

    monkeypatch.setattr(_pipeline_runtime, "write_rally_records", write_spy)

    result = runtime._assembly_plans(manifest)[0].execute()

    assert captured == ["second", "first"]
    assert runtime.state.records == [{"row": 2}, {"row": 1}]
    assert result.counts == {"videos": 2, "rallies": 2}


def test_projection_manifest_keeps_upstream_stages_appended_after_retained_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch)
    completed = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    ).manifest
    stages = {stage.name: stage for stage in completed.stages}
    selection = replace(stages["selection"], dependencies=())
    annotation = stages[f"annotation:{fixture.video_id}"]
    projection = stages[f"primitive_projection:{fixture.video_id}"]
    retained_annotation = replace(
        annotation,
        name="annotation:retained",
        dependencies=("selection",),
    )
    retained_projection = replace(
        projection,
        name="primitive_projection:retained",
        dependencies=("annotation:retained",),
    )
    rebuilt_annotation = replace(
        annotation,
        name="annotation:rebuilt",
        dependencies=("selection",),
    )
    reordered = replace(
        completed,
        stages=(selection, retained_annotation, retained_projection, rebuilt_annotation),
    )

    projection_input = fixture.runtime_module._projection_input_manifest(reordered)

    assert [stage.name for stage in projection_input.stages] == [
        "selection",
        "annotation:retained",
        "annotation:rebuilt",
    ]


@pytest.mark.parametrize("change_directory", [False, True], ids=["predictor", "directory"])
def test_tracknet_directory_or_predictor_change_invalidates_shuttle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change_directory: bool,
) -> None:
    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch)
    cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    if change_directory:
        replacement = tmp_path / "replacement-tracknet"
        replacement.mkdir()
        (replacement / "batch_predict.py").write_bytes(b"fixture")
        fixture.tracknet_dir = replacement
    else:
        (fixture.tracknet_dir / "batch_predict.py").write_bytes(b"changed fixture")
    resumed = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    metadata = next(event for event in resumed.events if event.name.startswith("metadata:"))
    shuttle = next(event for event in resumed.events if event.name.startswith("shuttle:"))
    assert metadata.reused is True
    assert shuttle.reused is False


def test_runtime_ignores_absent_stale_source_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(
        tmp_path,
        monkeypatch,
        source_basename=f"{_ConcreteRuntimeFixture.video_id}.mkv",
        stale_source_basename=f"{_ConcreteRuntimeFixture.video_id}.mp4",
    )

    first = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )
    second = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    assert first.stopped_after is None
    assert second.stopped_after is None
    assert all(event.reused for event in second.events)


def test_legacy_source_basename_reaches_shuttle_and_assembly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    basename = f"{_ConcreteRuntimeFixture.video_id} Match Name.mp4"
    fixture = _ConcreteRuntimeFixture(
        tmp_path,
        monkeypatch,
        source_basename=basename,
    )

    first = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )
    second = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    csv_path = (
        fixture.run_dir
        / "stages"
        / "shuttle"
        / fixture.video_id
        / f"{Path(basename).stem}_ball.csv"
    )
    assert first.stopped_after is None
    pairing = next(
        event for event in first.events if event.name.startswith("commentary_pairing:")
    )
    assert pairing.outcome is StageOutcome.PROCESSED
    assert csv_path.is_file()
    assert second.stopped_after is None
    assert [(event.name, event.reason) for event in second.events if not event.reused] == []


def test_default_runtime_uses_visual_fallback_when_commentary_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch, with_commentary=False)

    result = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    assert [event.name for event in result.events] == fixture.expected_stage_names
    outcomes = {event.name: event.outcome for event in result.events}
    assert outcomes["transcript"] is StageOutcome.UNAVAILABLE
    assert outcomes["triage"] is StageOutcome.UNAVAILABLE
    assert outcomes["commentary_cleaning"] is StageOutcome.UNAVAILABLE
    assert "triage" not in fixture.boundary_calls
    assert "commentary_cleaning" not in fixture.boundary_calls
    selected = load_selection(fixture.run_dir / "selected_videos.csv.gz")
    assert selected[0].visual_selected is True
    assert selected[0].selection_source == "metadata_fallback"
    assert selected[0].commentary_status == COMMENTARY_UNAVAILABLE_TRANSCRIPT


def _install_partial_commentary_timeout(
    fixture: _ConcreteRuntimeFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    requests: list[str] = []

    def clean_once(text: str) -> dict[str, object]:
        requests.append(text)
        if text == "fixture call":
            return {
                "text_clean": "clean fixture call",
                "alt_phrasings": ["fixture alternative"]
                * fixture.scraper_config.ALT_PHRASINGS_K,
            }
        raise TimeoutError("commentary request timed out")

    def partial_timeout_clean(*, rows: list[dict[str, object]]) -> dict[str, int]:
        path = fixture.chunk_dir / f"{fixture.video_id}.json"
        chunks = json.loads(path.read_text(encoding="utf-8"))
        chunks.append({
            "chunk_id": f"{fixture.video_id}_c1",
            "start": 2.0,
            "end": 3.0,
            "text": "fixture timeout",
        })
        path.write_text(json.dumps(chunks), encoding="utf-8")
        return ORIGINAL_RUN_CLEAN(rows=rows)

    monkeypatch.setattr(
        fixture.runtime_module.commentary_cleaning,
        "run_clean",
        partial_timeout_clean,
    )
    monkeypatch.setattr(commentary_cleaning, "CHUNKS_DIR", fixture.chunk_dir)
    monkeypatch.setattr(commentary_cleaning, "_clean_once", clean_once)
    monkeypatch.setattr(commentary_cleaning.time, "sleep", lambda _seconds: None)
    return requests


def test_default_runtime_continues_visual_lane_after_partial_commentary_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch, with_rally=True)
    requests = _install_partial_commentary_timeout(fixture, monkeypatch)

    result = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    outcomes = {event.name: event.outcome for event in result.events}
    assert outcomes["commentary_cleaning"] is StageOutcome.UNAVAILABLE
    assert outcomes[f"tracknet_input:{fixture.video_id}"] is StageOutcome.PROCESSED
    assert outcomes[f"annotation:{fixture.video_id}"] is StageOutcome.PROCESSED
    assert outcomes["assembly"] is StageOutcome.PROCESSED
    assert outcomes["report"] is StageOutcome.PROCESSED
    assert requests == [
        "fixture call",
        "fixture timeout",
        "fixture timeout",
        "fixture timeout",
    ]
    assert result.stopped_after is None
    first_selection = load_selection(fixture.run_dir / "selected_videos.csv.gz")
    assert first_selection[0].commentary_status == COMMENTARY_FAILED
    publications = (
        "run_manifest.json.gz",
        "rally_records.json.gz",
        "dataset_builder_report.json.gz",
        "selected_videos.csv.gz",
    )
    first_bytes = {
        name: (fixture.run_dir / name).read_bytes()
        for name in publications
    }

    resumed = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    assert all(event.reused for event in resumed.events)
    assert requests == [
        "fixture call",
        "fixture timeout",
        "fixture timeout",
        "fixture timeout",
    ]
    assert {
        name: (fixture.run_dir / name).read_bytes()
        for name in publications
    } == first_bytes
    resumed_selection = load_selection(fixture.run_dir / "selected_videos.csv.gz")
    assert resumed_selection[0].commentary_status == COMMENTARY_FAILED


def test_unavailable_cleaning_reuse_restores_triage_chunks_for_pairing_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch, with_rally=True)
    requests = _install_partial_commentary_timeout(fixture, monkeypatch)
    cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )
    pairing_path = (
        fixture.run_dir / "stages/commentary_pairing" / fixture.video_id
        / fixture.runtime_module.PAIRING_FILENAME
    )
    first_pairing = vision.load_json_gz(pairing_path)
    first_records = vision.load_json_gz(fixture.run_dir / "rally_records.json.gz")
    pairing_path.unlink()

    resumed = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    events = {event.name: event for event in resumed.events}
    assert events["commentary_cleaning"].reused is True
    assert events[f"commentary_pairing:{fixture.video_id}"].reused is False
    assert len(requests) == 4
    assert vision.load_json_gz(pairing_path)["rows"] == first_pairing["rows"]
    resumed_records = vision.load_json_gz(fixture.run_dir / "rally_records.json.gz")
    assert resumed_records["records"][0]["commentary"] == (
        first_records["records"][0]["commentary"]
    )
    selection = load_selection(fixture.run_dir / "selected_videos.csv.gz")
    assert selection[0].commentary_status == COMMENTARY_FAILED


def test_silent_visual_source_records_ineligible_commentary_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(
        tmp_path,
        monkeypatch,
        commentary_eligible=False,
        with_rally=True,
    )

    result = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    payload = vision.load_json_gz(fixture.run_dir / "rally_records.json.gz")
    records = payload["records"]
    assert payload["schema"] == RALLY_RECORD_COLLECTION_SCHEMA
    assert len(payload["sources"]) == 1
    assert isinstance(records, list) and len(records) == 1
    assert isinstance(records[0], dict)
    assert records[0]["schema"] == RALLY_RECORD_SCHEMA
    commentary = records[0]["commentary"]
    assert isinstance(commentary, dict)
    assert commentary["missing_reason"] == COMMENTARY_INELIGIBLE
    selection = load_selection(fixture.run_dir / "selected_videos.csv.gz")
    assert selection[0].commentary_status == COMMENTARY_INELIGIBLE
    assert result.stopped_after is None


def test_report_reruns_when_an_excluded_video_failure_reason_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch)
    failure_reason = "first metadata failure"

    def fail_metadata(_source: Path) -> VideoMetadata:
        raise RuntimeError(failure_reason)

    monkeypatch.setattr(fixture.runtime_module, "probe_video_metadata", fail_metadata)
    first = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    report = first.manifest.stages[-1]
    assert report.name == "report"
    assert set(report.dependencies) == {stage.name for stage in first.manifest.stages[:-1]}
    assert failure_reason in str(vision.load_json_gz(
        fixture.run_dir / fixture.runtime_module.REPORT_FILENAME,
    )["exclusions"][fixture.video_id])

    failure_reason = "second metadata failure"
    second = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    report_event = next(event for event in second.events if event.name == "report")
    assert report_event.reused is False
    assert failure_reason in str(vision.load_json_gz(
        fixture.run_dir / fixture.runtime_module.REPORT_FILENAME,
    )["exclusions"][fixture.video_id])


def test_vision_exception_reaches_coordinator_outcome_and_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch)

    def fail_pose(**_kwargs: object) -> vision.PoseExtraction:
        raise LookupError("fixture pose boundary failed")

    monkeypatch.setattr(fixture.vision_plans, "extract_rtmlib_pose_stage", fail_pose)

    result = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    expected_reason = "LookupError: fixture pose boundary failed"
    pose_event = next(event for event in result.events if event.name == f"pose:{fixture.video_id}")
    assert pose_event.outcome is StageOutcome.FAILED
    assert pose_event.reason == expected_reason
    assert result.terminal_error == "every selected video was excluded before record assembly"
    report = vision.load_json_gz(fixture.run_dir / fixture.runtime_module.REPORT_FILENAME)
    assert report["exclusions"] == {fixture.video_id: expected_reason}


def test_disabled_commentary_resume_reuses_every_visual_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(
        tmp_path,
        monkeypatch,
        commentary_enabled=False,
    )

    first = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )
    first_calls = list(fixture.boundary_calls)
    second = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    outcomes = {event.name: event.outcome for event in first.events}
    assert outcomes["transcript"] is StageOutcome.SKIPPED
    assert outcomes["triage"] is StageOutcome.SKIPPED
    assert outcomes["commentary_cleaning"] is StageOutcome.SKIPPED
    assert outcomes[f"commentary_pairing:{fixture.video_id}"] is StageOutcome.SKIPPED
    assert fixture.boundary_calls == first_calls
    assert all(event.reused for event in second.events)


def test_unchanged_unavailable_commentary_resume_is_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(
        tmp_path,
        monkeypatch,
        with_commentary=False,
        with_rally=True,
    )
    cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )
    first_calls = list(fixture.boundary_calls)
    publications = (
        "run_manifest.json.gz",
        "rally_records.json.gz",
        "dataset_builder_report.json.gz",
        "selected_videos.csv.gz",
    )
    first_bytes = {
        name: (fixture.run_dir / name).read_bytes()
        for name in publications
    }

    second = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    assert fixture.boundary_calls == first_calls
    assert all(event.reused for event in second.events)
    assert {
        name: (fixture.run_dir / name).read_bytes()
        for name in publications
    } == first_bytes


def test_retry_unavailable_reruns_only_its_dependency_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch, with_commentary=False)
    cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )
    first_calls = list(fixture.boundary_calls)

    second = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
        retry_unavailable=True,
    )

    assert fixture.boundary_calls == [*first_calls, "transcript"]
    reusable_visual = {
        "selection",
        "download",
        f"metadata:{fixture.video_id}",
        f"tracknet_input:{fixture.video_id}",
        f"shuttle:{fixture.video_id}",
        f"pose:{fixture.video_id}",
        f"court:{fixture.video_id}",
        f"annotation:{fixture.video_id}",
    }
    assert all(event.reused for event in second.events if event.name in reusable_visual)
    assert any(
        event.outcome is StageOutcome.UNAVAILABLE and not event.reused
        for event in second.events
    )


def test_retry_unavailable_cli_switch_is_explicit() -> None:
    parser = cli._build_parser()

    default = parser.parse_args(("run", "--config", "trial.toml", "--run-dir", "run"))
    retry = parser.parse_args((
        "run",
        "--config",
        "trial.toml",
        "--run-dir",
        "run",
        "--retry-unavailable",
    ))

    assert default.retry_unavailable is False
    assert retry.retry_unavailable is True


def test_selected_video_without_rallies_is_a_terminal_cli_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch)
    result = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    assert result.stopped_after is None
    assert result.terminal_error == "selected videos produced no rally records"
    assert (fixture.run_dir / fixture.runtime_module.REPORT_FILENAME).is_file()
    monkeypatch.setattr(cli, "run_dataset_builder", lambda *_args, **_kwargs: result)

    exit_status = cli.main(("run", "--config", "trial.toml", "--run-dir", "run"))

    assert exit_status == 1
    assert "selected videos produced no rally records" in capsys.readouterr().err


def test_every_selected_video_excluded_is_a_terminal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch)

    def fail_metadata(_source: Path) -> VideoMetadata:
        raise RuntimeError("fixture metadata failure")

    monkeypatch.setattr(
        fixture.runtime_module,
        "probe_video_metadata",
        fail_metadata,
    )

    result = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    assert result.stopped_after is None
    assert result.terminal_error == "every selected video was excluded before record assembly"
    report = vision.load_json_gz(fixture.run_dir / fixture.runtime_module.REPORT_FILENAME)
    assert report["processed_video_ids"] == []
    assert report["rally_count"] == 0
    projection_path = (
        fixture.run_dir
        / "stages"
        / "primitive_projection"
        / fixture.video_id
        / fixture.runtime_module.PROJECTION_FILENAME
    )
    collection = vision.load_json_gz(fixture.run_dir / "rally_records.json.gz")
    assert not projection_path.exists()
    assert collection["sources"] == []
    assert collection["records"] == []


def test_projection_failure_excludes_video_without_blocking_empty_assembly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch, with_rally=True)

    def fail_projection(**_kwargs: object) -> RallyRecordProjection:
        raise RuntimeError("fixture projection failure")

    monkeypatch.setattr(
        fixture.runtime_module,
        "assemble_rally_records",
        fail_projection,
    )

    result = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )
    events = {event.name: event for event in result.events}
    collection = vision.load_json_gz(fixture.run_dir / "rally_records.json.gz")

    assert events[f"primitive_projection:{fixture.video_id}"].outcome is StageOutcome.FAILED
    assert events["assembly"].outcome is StageOutcome.PROCESSED
    assert events["report"].outcome is StageOutcome.PROCESSED
    assert result.terminal_error == "every selected video was excluded before record assembly"
    assert collection["sources"] == []
    assert collection["records"] == []


def test_required_failure_unpublishes_invalidated_final_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _ConcreteRuntimeFixture(tmp_path, monkeypatch)
    cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )
    video_path = fixture.video_dir / f"{fixture.video_id}.mp4"
    video_path.write_bytes(b"invalidated video")

    def fail_download(**_kwargs: object) -> list[download_scraped_videos.DownloadOutcome]:
        raise RuntimeError("download retry failed")

    monkeypatch.setattr(
        fixture.runtime_module.download_scraped_videos,
        "download_all_videos",
        fail_download,
    )
    resumed = cli.run_dataset_builder(
        fixture.config_path,
        fixture.run_dir,
        runtime_factory=fixture.factory,
    )

    assert resumed.stopped_after == "download"
    assert not (fixture.run_dir / "rally_records.json.gz").exists()
    assert not (fixture.run_dir / fixture.runtime_module.REPORT_FILENAME).exists()
    assert not (fixture.run_dir / "selected_videos.csv.gz").exists()


def test_stage_reset_rejects_a_symlinked_stage_root(tmp_path: Path) -> None:
    from dataset_builder._runtime_support import RuntimeSupport

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside"
    (outside / "search").mkdir(parents=True)
    marker = outside / "search" / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    (run_dir / "stages").symlink_to(outside, target_is_directory=True)
    config = cli.load_builder_config(_write_config(tmp_path / "trial.toml"))
    support = RuntimeSupport(config, run_dir)

    with pytest.raises(ValueError, match="stage root must not be a symlink"):
        support._reset_stage_dir("search")

    assert marker.read_text(encoding="utf-8") == "keep"


def test_mutable_root_validation_rejects_a_symlinked_workspace(
    tmp_path: Path,
) -> None:
    from dataset_builder._runtime_support import RuntimeSupport

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = run_dir / "workspace"
    workspace.symlink_to(outside, target_is_directory=True)
    config = cli.load_builder_config(_write_config(tmp_path / "trial.toml"))
    support = RuntimeSupport(config, run_dir)

    with pytest.raises(ValueError, match="scraper workspace must not be a symlink"):
        support._validate_mutable_roots()


def test_runtime_support_owns_paths_interpreters_and_isolated_typed_state(
    tmp_path: Path,
) -> None:
    from dataset_builder._runtime_support import RuntimeState, RuntimeSupport

    config = cli.load_builder_config(_write_config(tmp_path / "trial.toml"))
    first = RuntimeSupport(config, tmp_path / "first")
    second = RuntimeSupport(config, tmp_path / "second")

    assert first.config is config
    assert first.workspace == tmp_path / "first" / "workspace"
    assert isinstance(first.state, RuntimeState)
    assert first.current_interpreter is None
    assert first.tracknet_interpreter is None
    assert first.pose_interpreter is None
    assert first.ffmpeg_interpreter is None
    first.state.active_ids.add("video")
    assert second.state.active_ids == set()
