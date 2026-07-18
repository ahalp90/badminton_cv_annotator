"""Old-path import shim; module moved to annotator.doubles_flag (Stage 2).

Import-only (no -m surface); dies at Stage 7.
"""
from annotator.doubles_flag import doubles_flag as doubles_flag
from annotator.doubles_flag import DOUBLES_FLAGS_CSV as DOUBLES_FLAGS_CSV
from annotator.doubles_flag import main as main
