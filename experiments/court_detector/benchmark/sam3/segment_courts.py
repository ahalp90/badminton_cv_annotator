"""Segment court lines in still frames with SAM 3 and save the raw pieces.

Needs a CUDA GPU, Meta's ``sam3`` package and the ``facebook/sam3`` checkpoint in
the Hugging Face cache. Writes one ``.npz`` per frame and prompt; ``merge_pieces.py``
turns those into court regions on any machine.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

PROMPTS = ("court lines", "court markings", "court boundary lines")
# Keep weak pieces too, so merge_pieces.py can choose the confidence cutoff later.
SAVE_FLOOR = 0.1


def prompt_slug(prompt: str) -> str:
    """:return: the prompt as a file-name-safe slug, e.g. ``court_boundary_lines``."""
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+", type=Path, help="frame images to segment")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=SAVE_FLOOR)
    # Without bf16 autocast, SAM 3's vision trunk fails on mixed bf16/fp32 tensors.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for frame_path in args.frames:
            # Stems repeat across source folders, so the folder is part of the name.
            frame_name = f"{frame_path.parent.name}__{frame_path.stem}"
            state = processor.set_image(Image.open(frame_path).convert("RGB"))
            for prompt in PROMPTS:
                output = processor.set_text_prompt(state=state, prompt=prompt)
                # The returned dict is the processor's shared state; the next prompt overwrites it.
                np.savez_compressed(
                    args.out_dir / f"{frame_name}__{prompt_slug(prompt)}.npz",
                    frame_path=str(frame_path),
                    prompt=prompt,
                    scores=output["scores"].float().cpu().numpy(),  # (pieces,)
                    masks=output["masks"][:, 0].cpu().numpy(),  # (pieces, height, width) bool
                )
            print(f"{frame_name} done")


if __name__ == "__main__":
    main()
