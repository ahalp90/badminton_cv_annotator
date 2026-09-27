"""Build the data payload for the feedback-harness readout page.

Every figure on the readout is read out of the harness's own outputs here --
the score runs, the pinned split, the derived faults, the template library --
so the page cannot quietly disagree with the numbers it claims to show. The
exclusion counts in particular are recomputed by re-running the derivation
rather than transcribed from its console output, because a transcribed count
stays plausible-looking forever after the thing it counted has changed.

    PYTHONPATH=src python scripts/feedback_eval/build_readout_payload.py --out payload.json

The page itself is a static snapshot: re-running the scorer does not update a
published readout. Re-run this, re-inline the payload, republish.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from feedback_eval.records import load_references
from feedback_eval.shuttleset_faults import derive_faults, load_match_players
from feedback_eval.splits import load_split
from feedback_eval.templates import load_templates

RUN = Path("experiments/feedback_eval/shuttleset_v1")
DATA = Path("data/feedback_eval")
LIBRARY = DATA / "templates" / "badminton_singles_shuttleset_v1.jsonl"
BASELINES = ("empty", "constant", "random_template", "oracle")


def _aggregates() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in BASELINES:
        raw = json.loads((RUN / f"scores_{name}.json").read_text())
        rescaled_path = RUN / f"scores_{name}_rescaled.json"
        # `empty` is a hard zero by construction and never reaches the scorer,
        # so it has no rescaled run of its own.
        rescaled = (
            json.loads(rescaled_path.read_text())["aggregates"]["mean_f1"]
            if rescaled_path.is_file()
            else 0.0
        )
        out[name] = {
            "raw": round(raw["aggregates"]["mean_f1"], 4),
            "rescaled": round(rescaled, 4),
            "sd": round(raw["aggregates"]["stdev_f1"], 4),
        }
    return out


def _per_clip(name: str, rescaled: bool) -> dict[str, float]:
    path = RUN / f"scores_{name}{'_rescaled' if rescaled else ''}.json"
    if not path.is_file():
        return {}
    return {clip["clip_id"]: clip["f1"] for clip in json.loads(path.read_text())["clips"]}


def _construction() -> list[list]:
    """Re-derive, so these counts are computed rather than copied."""
    _, report = derive_faults(
        Path("data/shuttleset/set"), load_match_players(Path("data/shuttleset/set/match.csv"))
    )
    return [
        ["Reference clips built", report.kept],
        ["Ended in a clean winner", report.opponent_winner],
        ["Ambiguous lose_reason", report.ambiguous_reason],
        ["Point credited to the erring player", report.attribution_inconsistent],
        ["Stroke type unknown", sum(report.unmapped_stroke.values())],
    ]


def build() -> dict:
    split = load_split(DATA / "split_shuttleset_v1.json")
    references = {r.clip_id: r for r in load_references(DATA / "references_shuttleset_v1.jsonl")}
    templates = load_templates(LIBRARY)

    faults: dict[str, tuple[str, str]] = {}
    with (DATA / "shuttleset_clip_faults.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            faults[row["clip_id"]] = (row["template_id"], row["erring_player"])

    scores = {(n, r): _per_clip(n, r) for n in BASELINES for r in (False, True)}
    predictions = {
        baseline: {
            json.loads(line)["clip_id"]: json.loads(line)["feedback"]
            for line in (RUN / f"predictions_{baseline}.jsonl").read_text().splitlines()
            if line.strip()
        }
        for baseline in ("constant", "random_template")
    }

    # Corrections are deduplicated to an index so each clip row stays small:
    # 48 strings shared across 543 clips rather than inlined per clip.
    corrections: list[str] = []
    index: dict[str, int] = {}
    for template in templates.values():
        for correction in template.corrections:
            if correction not in index:
                index[correction] = len(corrections)
                corrections.append(correction)

    clips = []
    for clip_id in split.test_clips:
        record = references[clip_id]
        template_id, erring = faults[clip_id]
        head, _, tail = clip_id.rpartition("_s")
        clips.append(
            {
                "id": clip_id,
                "match": head.replace("_", " "),
                "sr": "s" + tail.replace("_r", " · rally "),
                "p": list(record.player_ids),
                "err": erring,
                "t": template_id,
                "rt": index[predictions["random_template"][clip_id]],
                "sc": {
                    name: [
                        round(scores[(name, False)][clip_id], 4),
                        round(scores[(name, True)].get(clip_id, 0.0), 4),
                    ]
                    for name in ("constant", "random_template", "oracle")
                },
            }
        )

    return {
        "templates": {
            key: {
                "fault": value.fault,
                "area": value.skill_area,
                "corr": [index[c] for c in value.corrections],
            }
            for key, value in templates.items()
        },
        "corrections": corrections,
        "constantText": predictions["constant"][split.test_clips[0]],
        "clips": clips,
        "aggregates": _aggregates(),
        "construction": _construction(),
        "split": {
            "trainClips": len(split.train_clips),
            "testClips": len(split.test_clips),
            "trainPlayers": list(split.train_players),
            "testPlayers": list(split.test_players),
            "discarded": len(split.discarded_clips),
        },
        "faultTotals": dict(Counter(t for t, _ in faults.values())),
    }


TEMPLATE = Path("scripts/feedback_eval/readout_template.html")
PLACEHOLDER = "__PAYLOAD__"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="payload JSON to write")
    parser.add_argument(
        "--html-out",
        type=Path,
        help=f"also inline the payload into {TEMPLATE} and write the finished page here",
    )
    args = parser.parse_args()

    payload = build()
    serialised = json.dumps(payload, separators=(",", ":"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(serialised, encoding="utf-8")
    print(f"wrote {args.out} -- {args.out.stat().st_size:,} bytes")

    if args.html_out:
        template = TEMPLATE.read_text(encoding="utf-8")
        if PLACEHOLDER not in template:
            raise SystemExit(f"{TEMPLATE}: no {PLACEHOLDER} to substitute")
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(template.replace(PLACEHOLDER, serialised), encoding="utf-8")
        print(f"wrote {args.html_out} -- {args.html_out.stat().st_size:,} bytes")

    print(f"{len(payload['clips'])} clip(s), {len(payload['corrections'])} correction(s)")
    print("construction (recomputed):")
    for label, count in payload["construction"]:
        print(f"  {label:<38} {count:>6,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
