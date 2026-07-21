"""Run the committed calibration chain from the fixture manifest.

The documented entry point is ``PYTHONPATH=src python -m annotator.calibration.run_cli``.
Entry points own their path setup, so this module adds only the two source roots needed by
the chain after Python has resolved ``annotator`` from ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
_BST_X = _SRC / "bst_x"
if str(_BST_X) not in sys.path:
    sys.path.insert(0, str(_BST_X))

from annotator.calibration.fixtures import FIXTURES, Fixture, fixtures_root  # noqa: E402
from annotator.calibration.gt_scoring import (  # noqa: E402
    flatten_metrics,
    render_table,
    run_fixture,
)


FixtureRunner = Callable[[Fixture], object]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the annotator calibration fixture chain")
    parser.add_argument(
        "--fixtures", nargs="+", metavar="NAME", help="fixture names (default: all)"
    )
    parser.add_argument("--out", type=Path, metavar="DIR", help="write one metrics CSV per fixture")
    return parser


def _validate_registry(registry: Iterable[Fixture]) -> tuple[Fixture, ...]:
    try:
        entries = tuple(registry)
    except (TypeError, ValueError) as exc:
        raise ValueError("fixture registry is not iterable") from exc
    if not entries:
        raise ValueError("fixture registry is empty")
    names: set[str] = set()
    for fixture in entries:
        if not isinstance(fixture, Fixture) or not fixture.name or fixture.name in names:
            raise ValueError("fixture registry is malformed")
        names.add(fixture.name)
    return entries


def _validate_environment() -> None:
    root = fixtures_root()
    if not root.is_dir() or not os.access(root, os.R_OK):
        raise RuntimeError(f"ANNOTATOR_FIXTURES_ROOT is unreadable: {root}")


def _write_metrics(path: Path, metrics: dict[str, int | float | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)


def run_manifest(
    names: Sequence[str] | None = None,
    out: Path | None = None,
    *,
    registry: Iterable[Fixture] = FIXTURES,
    runner: FixtureRunner = run_fixture,
    flattener: Callable[[object], dict[str, int | float | None]] = flatten_metrics,
    renderer: Callable[[dict[str, dict[str, int | float | None]]], str] = render_table,
) -> int:
    """Run selected manifest entries, skipping failures local to one fixture.

    Registry and fixture-root validation happen before the loop.  The runner includes
    digest verification, array loading, the committed-mask ``dead_mask=`` call, and scoring.
    """
    _validate_environment()
    entries = _validate_registry(registry)
    by_name = {fixture.name: fixture for fixture in entries}
    selected = tuple(by_name) if names is None else tuple(names)
    unknown = [name for name in selected if name not in by_name]
    if unknown:
        raise ValueError(f"unknown fixture(s): {', '.join(unknown)}")

    scores: dict[str, dict[str, int | float | None]] = {}
    for name in selected:
        try:
            scores[name] = flattener(runner(by_name[name]))
        except Exception as exc:  # noqa: BLE001 - one bad fixture must not abort the batch
            print(f"{name}: SKIP: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if out is not None:
            _write_metrics(out / f"{name}_metrics.csv", scores[name])

    if scores:
        print(renderer(scores))
    return 0


def main(argv: Sequence[str] | None = None, *, registry: Iterable[Fixture] = FIXTURES,
         runner: FixtureRunner = run_fixture,
         flattener: Callable[[object], dict[str, int | float | None]] = flatten_metrics,
         renderer: Callable[[dict[str, dict[str, int | float | None]]], str] = render_table) -> int:
    args = build_parser().parse_args(argv)
    return run_manifest(args.fixtures, args.out, registry=registry, runner=runner,
                        flattener=flattener, renderer=renderer)


if __name__ == "__main__":
    raise SystemExit(main())
