"""The Version A generation loop, without the model.

The generator is injected, so this checks what a two-hour GPU job depends on:
resuming, not losing failures, and output the scorer accepts.
"""
import json

import pytest

from feedback_eval.records import load_predictions
from mllm import batch_feedback as batch


def _clips(tmp_path, n=3):
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir()
    lines = ["clip_id,path,video,video_id,start_s,end_s"]
    for i in range(n):
        (clip_dir / f"c{i}.mp4").write_bytes(b"")
        # recorded as a path on another machine; only the file name is used
        lines.append(f"c{i},/elsewhere/clips/c{i}.mp4,v,1,0.0,1.0")
    (clip_dir / "clips.csv").write_text("\n".join(lines) + "\n")
    return batch.read_clips(clip_dir / "clips.csv")


def test_read_clips_resolves_paths_next_to_the_manifest(tmp_path):
    clips = _clips(tmp_path)
    assert [path.parent for _, path in clips] == [tmp_path / "clips"] * 3


def test_read_clips_refuses_missing_files(tmp_path):
    clips = _clips(tmp_path)
    clips[0][1].unlink()
    with pytest.raises(FileNotFoundError, match="1 clip file"):
        batch.read_clips(tmp_path / "clips" / "clips.csv")


def test_writes_predictions_the_scorer_accepts(tmp_path):
    out = tmp_path / "predictions.jsonl"
    written, failed = batch.run(_clips(tmp_path), out, lambda path, prompt: f"  feedback for {path.stem} ")
    assert (written, failed) == (3, [])
    records = load_predictions(out)
    assert [(r.clip_id, r.feedback) for r in records] == [
        ("c0", "feedback for c0"), ("c1", "feedback for c1"), ("c2", "feedback for c2"),
    ]


def test_resume_skips_clips_already_written(tmp_path):
    clips, out, seen = _clips(tmp_path), tmp_path / "p.jsonl", []
    out.write_text(json.dumps({"clip_id": "c0", "feedback": "earlier"}) + "\n")

    def generate(path, prompt):
        seen.append(path.stem)
        return "new"

    batch.run(clips, out, generate)
    assert seen == ["c1", "c2"]
    assert len(load_predictions(out)) == 3


def test_a_failure_is_left_out_so_a_rerun_retries_it(tmp_path):
    clips, out = _clips(tmp_path), tmp_path / "p.jsonl"

    def flaky(path, prompt):
        if path.stem == "c1":
            raise RuntimeError("CUDA out of memory")
        return "ok"

    written, failed = batch.run(clips, out, flaky)
    assert (written, failed) == (2, ["c1"])
    assert batch.done_clip_ids(out) == {"c0", "c2"}

    batch.run(clips, out, lambda path, prompt: "ok")
    assert batch.done_clip_ids(out) == {"c0", "c1", "c2"}


def test_blank_failures_writes_them_as_empty_feedback(tmp_path):
    clips, out = _clips(tmp_path), tmp_path / "p.jsonl"

    def broken(path, prompt):
        raise RuntimeError("bad clip")

    written, failed = batch.run(clips, out, broken, blank_failures=True)
    assert (written, failed) == (3, [])
    assert {r.feedback for r in load_predictions(out)} == {""}


def test_prompt_is_corrective_and_names_no_fault():
    prompt = batch.PROMPT.lower()
    assert "correction" in prompt
    for leak in ("net", "out", "smash", "drop", "lift", "clear"):
        assert f" {leak} " not in f" {prompt} "
