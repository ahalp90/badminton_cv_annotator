"""Old-path import shim; module moved to annotator.composition_mask (Stage 2).

Import-only (no -m surface); dies at Stage 7.
"""
from annotator.composition_mask import build_composition_mask as build_composition_mask
from annotator.composition_mask import detect_cuts as detect_cuts
from annotator.composition_mask import main as main
