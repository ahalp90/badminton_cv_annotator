"""Segment court markings with SAM 3 and save its masks and confidence scores."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+", type=Path, help="frame images to segment")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.5)
    # SAM 3's vision trunk requires autocast for its mixed bf16/fp32 tensors.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for frame_path in args.frames:
            frame_name = f"{frame_path.parent.name}__{frame_path.stem}"
            with Image.open(frame_path) as image:
                state = processor.set_image(image.convert("RGB"))
            output = processor.set_text_prompt(state=state, prompt="court markings")
            np.savez_compressed(
                args.out_dir / f"{frame_name}__court_markings.npz",
                frame_path=str(frame_path),
                prompt="court markings",
                scores=output["scores"].float().cpu().numpy(),  # (pieces,)
                masks=output["masks"][:, 0].cpu().numpy(),  # (pieces, height, width)
            )
            print(f"{frame_name} done")


if __name__ == "__main__":
    main()
