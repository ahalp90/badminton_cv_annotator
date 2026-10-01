"""Generate Version A coaching feedback for every clip in a clips.csv.

Version A is the pretrained model with no fine-tuning. This loops
InternVideo3-8B-Instruct over the rally clips `feedback_eval.shuttleset_clips`
cut, and writes `predictions.jsonl` in the shape `feedback_eval.score_cli`
reads: one `{"clip_id", "feedback"}` line per clip.

Usage (inside the container; run_batch.sh does this for you):

    python batch_feedback.py --clips /clips/clips.csv --out /out/predictions.jsonl

**The prompt asks for a correction, not a description.** The references are
corrective coaching ("the drop was played into the net; hold the full overhead
action..."), and `run_internvideo3.py`'s "describe this video" prompt produces
broadcast-style description, which is scored against the wrong register. The
prompt does say the rally ended on an error, which is true of every clip in the
reference set (rallies ending in clean winners were excluded), but it never
names the clip's fault: that is the thing being tested.

**Resumable.** Every prediction is written and flushed as soon as it is
generated, and a re-run skips clips already in the output. A GPU job killed at
clip 60 loses one clip, not sixty.

**Failures are not silently dropped.** A clip the model errors on is logged and
left out, so a re-run retries it. When retrying stops helping, pass
`--blank-failures` to write those clips with empty feedback: they then score a
hard zero, which is the honest result. The scorer refuses a predictions file
with clips missing, so a forgotten failure cannot quietly flatter the mean.

**Every setting that changes the output is recorded** in `<out>.meta.json`
(model, revision, prompt, fps, token limit, decoding), because two Version A
runs with different prompts are different experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL = "yanziang/InternVideo3-8B-Instruct"
# The revision COSC595's Issue 38 benchmark pinned and ran on an L40.
REVISION = "c4602918b65225650d152db2850fe34e01d21fcd"

PROMPT = (
    "You are an experienced badminton coach reviewing a rally from a professional "
    "singles match. The rally ends when one player makes an error on the final shot. "
    "Identify that final shot and what went wrong with it, then give the player one "
    "specific technical correction. Answer in two or three sentences of coaching "
    "advice addressed to the player. Do not describe the venue, the score, the "
    "sponsors or what the players are wearing."
)

Generator = Callable[[Path, str], str]


def read_clips(clips_csv: Path) -> list[tuple[str, Path]]:
    """Read clips.csv into (clip_id, path) pairs.

    Each path is resolved next to clips.csv by file name, not taken as written:
    the manifest records the path on the machine that cut the clips, and the
    directory is usually copied somewhere else (a GPU node, a container mount)
    before the model sees it.
    """
    if not clips_csv.is_file():
        raise FileNotFoundError(f"no such clips manifest: {clips_csv}")
    with clips_csv.open(newline="", encoding="utf-8") as handle:
        rows = [(row["clip_id"], clips_csv.parent / Path(row["path"]).name) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"{clips_csv}: no clips")
    missing = [str(path) for _, path in rows if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} clip file(s) missing next to {clips_csv}, e.g. {missing[:3]}")
    return rows


def done_clip_ids(out: Path) -> set[str]:
    """Clip ids already written by an earlier, possibly interrupted, run."""
    if not out.is_file():
        return set()
    done: set[str] = set()
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["clip_id"])
    return done


def run(
    clips: Sequence[tuple[str, Path]],
    out: Path,
    generate: Generator,
    *,
    prompt: str = PROMPT,
    blank_failures: bool = False,
) -> tuple[int, list[str]]:
    """Generate feedback for every clip not already in `out`.

    Returns (clips written this run, clip ids that failed and were not written).
    """
    done = done_clip_ids(out)
    todo = [(clip_id, path) for clip_id, path in clips if clip_id not in done]
    logger.info("%d clip(s), %d already done, %d to go", len(clips), len(done), len(todo))

    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    failed: list[str] = []
    with out.open("a", encoding="utf-8") as handle:
        for index, (clip_id, path) in enumerate(todo, start=1):
            started = time.monotonic()
            try:
                feedback = generate(path, prompt).strip()
            except Exception:  # one bad clip must not end a two-hour GPU job
                logger.exception("%s: generation failed", clip_id)
                failed.append(clip_id)
                continue
            handle.write(json.dumps({"clip_id": clip_id, "feedback": feedback}) + "\n")
            handle.flush()
            written += 1
            logger.info("%d/%d %s (%.1fs)", index, len(todo), clip_id, time.monotonic() - started)

        if blank_failures:
            for clip_id in failed:
                handle.write(json.dumps({"clip_id": clip_id, "feedback": ""}) + "\n")
            written += len(failed)
            failed = []
    return written, failed


class InternVideo3:
    """The model, loaded once. Imports are deferred so the loop is testable without a GPU.

    The input budget follows COSC595's tested adapter
    (`annotator/vlm_scene_benchmark/backends/internvideo3.py`): a cap on frames
    and a fixed per-frame pixel budget. Without them the processor keeps the
    clip's native 720p, the visual prompt runs to tens of thousands of tokens,
    and one 10-second clip took ~3-4 minutes on an A100 with the GPU mostly
    idle. Each clip's sampled frames and token counts are logged so the input
    the model actually saw is on record.
    """

    def __init__(
        self,
        model: str,
        revision: str,
        fps: float,
        max_new_tokens: int,
        *,
        max_frames: int,
        frame_width: int,
        frame_height: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("InternVideo3 needs a CUDA device")
        self.torch = torch
        self.fps = fps
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()
        devices = {parameter.device.type for parameter in self.model.parameters()}
        if devices != {"cuda"}:
            raise RuntimeError(f"model is not wholly on the GPU: {sorted(devices)}")
        self.processor = AutoProcessor.from_pretrained(model, revision=revision, trust_remote_code=True)

        video_processor = self.processor.video_processor
        video_processor.min_frames = 4
        video_processor.max_frames = max_frames
        total_pixels = max_frames * frame_width * frame_height
        video_processor.size = {"shortest_edge": total_pixels, "longest_edge": total_pixels}

    def __call__(self, video: Path, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video), "fps": self.fps},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            fps=self.fps,
            return_metadata=True,
            padding=True,
        )
        metadata = inputs.pop("video_metadata", None)
        if isinstance(metadata, (list, tuple)):
            metadata = metadata[0] if metadata else None
        inputs.convert_to_tensors("pt")
        grid_t, grid_h, grid_w = (int(v) for v in inputs["video_grid_thw"][0].tolist())
        merge = int(self.processor.video_processor.merge_size)
        input_tokens = int(inputs["attention_mask"].sum().item())
        frames = len(metadata.frames_indices) if metadata is not None else "?"

        inputs = inputs.to("cuda:0")
        try:
            with self.torch.inference_mode():
                # Greedy, so a re-run of the same clip gives the same text.
                output = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True
                )
            generated = output[:, inputs["input_ids"].shape[-1]:]
            text = self.processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            logger.info(
                "  %s frames, visual tokens %d, input tokens %d, output tokens %d",
                frames, grid_t * grid_h * grid_w // merge**2, input_tokens, generated.shape[-1],
            )
        finally:
            del inputs
            self.torch.cuda.empty_cache()
        return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=Path, required=True, help="clips.csv from shuttleset_clips")
    parser.add_argument("--out", type=Path, required=True, help="predictions JSONL (appended to; resumable)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--fps", type=float, default=2.0, help="frames sampled per second of clip")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--max-frames", type=int, default=32, help="cap on sampled frames; long rallies are thinned"
    )
    parser.add_argument("--frame-width", type=int, default=448, help="per-frame pixel budget, width")
    parser.add_argument("--frame-height", type=int, default=252, help="per-frame pixel budget, height")
    parser.add_argument("--prompt-file", type=Path, help="replace the built-in prompt")
    parser.add_argument("--limit", type=int, help="only the first N clips (a smoke test)")
    parser.add_argument(
        "--blank-failures",
        action="store_true",
        help="write clips that still fail with empty feedback (they score zero)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    prompt = args.prompt_file.read_text(encoding="utf-8").strip() if args.prompt_file else PROMPT
    clips = read_clips(args.clips)
    if args.limit:
        clips = clips[: args.limit]

    meta_path = args.out.with_suffix(".meta.json")
    meta = {
        "model": args.model,
        "revision": args.revision,
        "prompt": prompt,
        "fps": args.fps,
        "max_new_tokens": args.max_new_tokens,
        "max_frames": args.max_frames,
        "frame_pixels": [args.frame_width, args.frame_height],
        "decoding": "greedy",
        "clips": str(args.clips),
    }
    if meta_path.is_file():
        previous = {k: v for k, v in json.loads(meta_path.read_text()).items() if k in meta}
        if previous != meta:
            raise SystemExit(
                f"{meta_path} records different settings from this run; resuming would mix two "
                "experiments in one file. Use a new --out, or delete both files to start over."
            )

    started = time.monotonic()
    generate = InternVideo3(
        args.model,
        args.revision,
        args.fps,
        args.max_new_tokens,
        max_frames=args.max_frames,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )
    logger.info("model loaded in %.0fs", time.monotonic() - started)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    written, failed = run(clips, args.out, generate, prompt=prompt, blank_failures=args.blank_failures)
    print(f"wrote {written} prediction(s) to {args.out}; {len(done_clip_ids(args.out))} of {len(clips)} done")
    if failed:
        print(f"{len(failed)} clip(s) failed; re-run to retry, or add --blank-failures: {failed[:5]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
