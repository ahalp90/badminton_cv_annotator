"""Put the experiment modules and the sibling helper directories on the import path."""

import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parents[2]
HELPERS = REPO / 'scratch/court_det_fix/worklog/checks/independent/player_guided/20260914'
# The experiment folder must resolve first, as it does on the remote PYTHONPATH: the
# marking_diagnosis helpers also have a summarise.py.
sys.path[:0] = [str(directory) for directory in (
    EXPERIMENT, HELPERS / 'automatic_axes/svd_fixed', HELPERS / 'automatic_axes', HELPERS / 'axis_matching',
    HELPERS / 'marking_diagnosis', HELPERS / 'vp_pruning', REPO / 'src', REPO)]
