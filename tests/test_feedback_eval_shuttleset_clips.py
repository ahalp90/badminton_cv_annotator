"""Planning and cutting ShuttleSet rally clips.

ffmpeg is never run: the runner is injected, so these tests check the spans and
the bookkeeping, which is where a wrong clip would come from.
"""
import pytest

from feedback_eval import shuttleset_clips as clips

VIDEO = "Player_A_Player_B_Open_2021_Final"


def _set_root(tmp_path):
    root = tmp_path / "set"
    (root / VIDEO).mkdir(parents=True)
    (root / VIDEO / "set1.csv").write_text(
        "rally,ball_round,frame_num,type\n"
        "1,1,250.0,serve\n"
        "1,2,300.0,clear\n"
        "1,3,375.0,smash\n"
        "2,1,1000.0,serve\n"
        "2,2,1050.0,net\n"
    )
    return root


MATCHES = {VIDEO: {"id": "7", "video": VIDEO, "url": "https://example.com/v"}}


def _plan(tmp_path, clip_ids=(f"{VIDEO}_s1_r1",), **kwargs):
    options = {"pre_s": 1.0, "post_s": 2.0, "fps": {VIDEO: 25.0}, "matches": MATCHES}
    options.update(kwargs)
    return clips.plan_clips(clip_ids, set_root=_set_root(tmp_path), **options)


def test_parse_clip_id_splits_on_the_last_set_and_rally():
    assert clips.parse_clip_id("A_s1_r2_Open_s3_r14") == ("A_s1_r2_Open", "3", "14")


def test_parse_clip_id_rejects_other_shapes():
    with pytest.raises(clips.ShuttleSetClipError):
        clips.parse_clip_id("abc123_r7")


def test_span_runs_from_first_to_last_stroke_with_padding(tmp_path):
    (plan,) = _plan(tmp_path)
    assert plan.video_id == "7"
    assert plan.start_s == pytest.approx(250 / 25 - 1.0)
    assert plan.end_s == pytest.approx(375 / 25 + 2.0)


def test_start_is_clamped_at_zero(tmp_path):
    (plan,) = _plan(tmp_path, pre_s=60.0)
    assert plan.start_s == 0.0


def test_unknown_rally_fails_with_its_name(tmp_path):
    with pytest.raises(clips.ShuttleSetClipError, match="rally 9"):
        _plan(tmp_path, clip_ids=(f"{VIDEO}_s1_r9",))


def test_excluded_match_fails_with_its_note(tmp_path):
    with pytest.raises(clips.ShuttleSetClipError, match="frame numbers incorrect"):
        _plan(tmp_path, fps={}, excluded={VIDEO: "excluded: all frame numbers incorrect"})


def test_load_fps_separates_excluded_rows(tmp_path):
    path = tmp_path / "meta.csv"
    path.write_text(
        "id,video,url,width,height,fps,note\n"
        f"7,{VIDEO},u,1920,1080,25,\n"
        "9,Gone,u,,,,excluded: video removed\n"
    )
    fps, excluded = clips.load_fps(path)
    assert fps == {VIDEO: 25.0}
    assert excluded == {"Gone": "excluded: video removed"}


def test_find_source_accepts_exact_and_legacy_names(tmp_path):
    (tmp_path / "7 Player_A_Player_B.mp4").write_bytes(b"")
    (tmp_path / "70.mp4").write_bytes(b"")
    assert clips.find_source(tmp_path, "7").name == "7 Player_A_Player_B.mp4"
    assert clips.find_source(tmp_path, "70").name == "70.mp4"
    assert clips.find_source(tmp_path, "8") is None


def test_cut_command_seeks_and_bounds_in_seconds(tmp_path):
    (plan,) = _plan(tmp_path)
    command = clips.cut_command(tmp_path / "src.mp4", plan, tmp_path / "out.mp4")
    assert command[command.index("-ss") + 1] == "9.000"
    assert command[command.index("-t") + 1] == f"{plan.end_s - plan.start_s:.3f}"
    assert "-an" in command


def test_cut_clips_skips_existing_and_reports_missing_sources(tmp_path):
    plans = _plan(tmp_path, clip_ids=(f"{VIDEO}_s1_r1", f"{VIDEO}_s1_r2"))
    videos, out_dir = tmp_path / "raw", tmp_path / "clips"
    out_dir.mkdir()
    (out_dir / f"{VIDEO}_s1_r1.mp4").write_bytes(b"done")

    calls = []
    done, missing = clips.cut_clips(plans, videos_dir=videos, out_dir=out_dir, runner=calls.append)
    assert [p.clip_id for p in done] == [f"{VIDEO}_s1_r1"]
    assert missing == ["7"]
    assert calls == []


def test_cut_clips_writes_through_a_partial_file(tmp_path):
    plans = _plan(tmp_path)
    videos, out_dir = tmp_path / "raw", tmp_path / "clips"
    videos.mkdir()
    (videos / "7.mp4").write_bytes(b"")

    def fake_ffmpeg(command):
        assert command[-1].endswith(".partial.mp4")
        with open(command[-1], "wb") as handle:
            handle.write(b"clip")

    done, missing = clips.cut_clips(plans, videos_dir=videos, out_dir=out_dir, runner=fake_ffmpeg)
    assert missing == []
    assert (out_dir / f"{VIDEO}_s1_r1.mp4").read_bytes() == b"clip"
    assert not list(out_dir.glob("*.partial.mp4"))


def test_write_match_subset_keeps_only_the_needed_rows(tmp_path):
    matches = {**MATCHES, "Other": {"id": "8", "video": "Other", "url": "u"}}
    path = clips.write_match_subset(tmp_path / "m.csv", matches, [VIDEO])
    assert path.read_text().splitlines() == ["id,video,url", f"7,{VIDEO},https://example.com/v"]


def test_plan_carries_the_match_url(tmp_path):
    (plan,) = _plan(tmp_path)
    assert plan.url == "https://example.com/v"


def test_fetch_command_requests_only_the_span_at_720p(tmp_path):
    (plan,) = _plan(tmp_path)
    command = clips.fetch_command(plan, tmp_path / "out.mp4")
    assert command[0] == "yt-dlp"
    assert command[command.index("--download-sections") + 1] == "*9.000-17.000"
    assert "height<=720" in command[command.index("-f") + 1]
    assert command[-1] == plan.url


def test_fetch_clips_skips_existing_and_keeps_going_past_a_failure(tmp_path):
    plans = _plan(tmp_path, clip_ids=(f"{VIDEO}_s1_r1", f"{VIDEO}_s1_r2"))
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    (out_dir / f"{VIDEO}_s1_r1.mp4").write_bytes(b"done")

    def failing(command):
        raise clips.subprocess.CalledProcessError(1, command)

    done, failed = clips.fetch_clips(plans, out_dir=out_dir, runner=failing)
    assert [p.clip_id for p in done] == [f"{VIDEO}_s1_r1"]
    assert failed == [f"{VIDEO}_s1_r2"]


def test_fetch_clips_renames_the_partial_download(tmp_path):
    plans = _plan(tmp_path)
    out_dir = tmp_path / "clips"

    def fake_ytdlp(command):
        with open(command[command.index("-o") + 1], "wb") as handle:
            handle.write(b"clip")

    done, failed = clips.fetch_clips(plans, out_dir=out_dir, runner=fake_ytdlp)
    assert failed == []
    assert (out_dir / f"{VIDEO}_s1_r1.mp4").read_bytes() == b"clip"
