"""Old-path import shim; module moved to annotator.fps_constants (Stage 2 migration).

Import-only (no -m surface); dies at Stage 7 with its last callers.
"""
from annotator.fps_constants import BASE_FPS as BASE_FPS
from annotator.fps_constants import FpsConstants as FpsConstants
from annotator.fps_constants import probe_fps as probe_fps
from annotator.fps_constants import scale_for_fps as scale_for_fps
