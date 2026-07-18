"""Old-path import shim; module moved to annotator.replay_mask (Stage 2).

Import-only (no -m surface); dies at Stage 7.
"""
from annotator.replay_mask import HOMOGRAPHY_CORNER_COLS as HOMOGRAPHY_CORNER_COLS
from annotator.replay_mask import combine_mask as combine_mask
from annotator.replay_mask import court_absence_signal as court_absence_signal
from annotator.replay_mask import perspective_shift_signal as perspective_shift_signal
from annotator.replay_mask import velocity_drop_signal as velocity_drop_signal
from annotator.replay_mask import main as main
