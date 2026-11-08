"""Turn an auto-annotator result into suggestions for the rally labeller.

Run from the repository root, after preparing the video::

    python3 tools/rally_labeller/suggestions_from_annotator.py \\
        path/to/annotator_result.json.gz --video-id E8WW8DFCnwk --name annotator-seed0

The suggestions appear in the labeller as faint rallies and contacts. The
annotator and the labeller both number frames from the start of the source
video, so frames carry over unchanged. The script stops if a suggested frame
falls outside the prepared video, which usually means the result came from a
different file.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path

ANNOTATOR_RESULT_SCHEMA = "annotator-result/0.8"
SUGGESTION_SCHEMA = "rally-suggestions/1"
# Same folder the labeller serves copies from; relative paths count from the repository root.
DEFAULT_MEDIA_ROOT = Path(os.environ.get("RALLY_LABELLER_MEDIA", "local_scratch/rally_labeller"))
SIDE_NAMES = {"Top": "top", "Bot": "bottom", None: None}
# Names become file names, so keep them plain.
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def suggestions_from_result(payload: dict, name: str) -> dict:
    """Convert a saved ``annotator-result/0.8`` payload into a suggestion file.

    :param payload: The parsed annotator result file.
    :param name: Short name shown in the labeller.
    :return: A ``rally-suggestions/1`` payload with inclusive rally ends.
    :raises ValueError: The result uses a different schema.
    """
    if payload.get("schema") != ANNOTATOR_RESULT_SCHEMA:
        raise ValueError(f"expected {ANNOTATOR_RESULT_SCHEMA}, got {payload.get('schema')}")
    rallies: list[dict] = []
    contacts: list[dict] = []
    for rally in payload["result"]["rallies"]:
        sequence = rally["sequence"]
        # The annotator's end frame is exclusive; labels and suggestions use inclusive ends.
        rallies.append({"start_frame": sequence["start_frame"], "end_frame": sequence["end_frame"] - 1})
        for event in sequence["events"]:
            contacts.append({"frame": event["frame"], "side": SIDE_NAMES[event["side"]], "confidence": event["probability"]})
    return {"schema": SUGGESTION_SCHEMA, "name": name, "rallies": rallies, "contacts": contacts}


def check_frames(suggestions: dict, frame_count: int) -> None:
    frames = [rally["end_frame"] for rally in suggestions["rallies"]]
    frames += [contact["frame"] for contact in suggestions["contacts"]]
    if frames and max(frames) >= frame_count:
        raise ValueError(f"suggested frame {max(frames)} is past the prepared video's last frame {frame_count - 1}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result", type=Path, help="annotator_result.json.gz")
    parser.add_argument("--video-id", required=True, help="the labeller's video ID")
    parser.add_argument("--name", required=True, help="short name shown in the labeller")
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT, help=f"default: {DEFAULT_MEDIA_ROOT}")
    args = parser.parse_args(argv)
    if not NAME_PATTERN.match(args.name):
        parser.error("--name may use only letters, digits, '.', '_' and '-'")

    video_dir = args.media_root / args.video_id
    clip_infos = sorted(video_dir.glob("*/video.json.gz"))
    if not clip_infos:
        raise SystemExit(f"no prepared copy of {args.video_id} in {args.media_root}")
    with gzip.open(clip_infos[0], "rt", encoding="utf-8") as file:
        frame_count = json.load(file)["source_frame_count"]
    with gzip.open(args.result, "rt", encoding="utf-8") as file:
        suggestions = suggestions_from_result(json.load(file), args.name)
    check_frames(suggestions, frame_count)

    output = video_dir / "suggestions" / f"{args.name}.json.gz"
    output.parent.mkdir(exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as file:
        json.dump(suggestions, file)
    print(f"wrote {len(suggestions['rallies'])} rallies and {len(suggestions['contacts'])} contacts to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
