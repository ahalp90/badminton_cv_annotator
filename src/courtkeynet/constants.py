"""Shared CourtKeyNet constants, kept torch-free.

Lives apart from wrapper.py so torch-light consumers (the corner annotator) can
import the package without pulling in the model. The wrapper imports what it
needs from here; nothing heavy is imported in return.
"""

# Per-corner peak-confidence floor below which a corner is not trusted (the
# model's peaks are per-corner confidences).
DEFAULT_CORNER_MIN_PEAK_CONF = 0.02
