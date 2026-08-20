"""Score one validated VLM scene benchmark record against reviewed truth."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from annotator.broadcast_timeline_labels import read_label_csv

from .contracts import read_run_record
from .scoring import score_run_record, write_score_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_record", type=Path)
    parser.add_argument("truth_labels", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = read_run_record(args.run_record)
        summary = score_run_record(record, read_label_csv(args.truth_labels))
        write_score_summary(args.out, summary)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        build_parser().error(str(error))
    print(args.out)
    return 0 if summary["deployment_gate"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
