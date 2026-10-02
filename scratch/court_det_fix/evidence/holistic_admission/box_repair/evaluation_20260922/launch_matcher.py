"""Run the repaired person-observation matcher with the frozen local frame paths."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))

import shared

shared.add_helper_paths()
sys.path.insert(0, str(shared.HELPERS / 'legacy'))

import run_automatic

run_automatic.frame_path = lambda source, _root: shared.frame_path(source)

import line_run_matcher

if __name__ == '__main__':
    line_run_matcher.main()
