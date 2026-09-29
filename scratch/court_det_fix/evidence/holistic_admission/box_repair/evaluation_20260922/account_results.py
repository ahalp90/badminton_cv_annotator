"""Account for the corrected matcher run in its owning evidence directory."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COURT_DET_FIX = HERE.parents[3]


def main() -> None:
    sys.path.insert(0, str(COURT_DET_FIX / "line_identity"))
    import shared

    shared.add_helper_paths()
    sys.path.insert(0, str(COURT_DET_FIX / "frozen_helpers_20260914/legacy"))
    import account

    # Keep the original baseline/control loaders and explicit repair inputs.
    # Only the new run's output directory differs from the historical CLI.
    account.HERE = HERE
    account.main()


if __name__ == "__main__":
    main()
