"""BRIC player-detection defaults."""

from pathlib import Path

# Project-relative default location for the YOLO11n weights.
# runtime/checkpoints/ is gitignored; ultralytics auto-downloads to this path
# on first use. This file lives at <project>/src/bric/perception/players.py, so
# parents[3] = <project>.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_YOLO_WEIGHTS = _PROJECT_ROOT / "runtime" / "checkpoints" / "yolo11" / "yolo11n.pt"
