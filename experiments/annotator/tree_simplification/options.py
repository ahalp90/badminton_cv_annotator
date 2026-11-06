"""Training targets for the production rally proposals.

This module holds the experiment class that pairs each option with its target
label. Runtime proposals use detector evidence; target labels live only in this
training view.
"""

from dataclasses import dataclass

import numpy as np

from annotator.rallies.build_contact_proposals import (
    FEATURE_NAMES,
    RallyOptions,
    build_rally_options,
)
from annotator.training.rally_targets import option_targets
from annotator.training.sequences import SequenceTrainingVideo
from annotator.types import ContactSequence

__all__ = ['FEATURE_NAMES', 'LabelledOptions', 'expand_pool']


@dataclass(frozen=True)
class LabelledOptions:
    sequences: tuple[ContactSequence, ...]
    matrix: np.ndarray
    targets: np.ndarray
    span_rows: dict[int, np.ndarray]
    video: SequenceTrainingVideo

    @property
    def runtime(self) -> RallyOptions:
        """Share proposal arrays with the label-free inference path."""
        return RallyOptions(self.sequences, self.matrix, self.span_rows, self.video.context)


def expand_pool(
    video: SequenceTrainingVideo, max_edits: int = 3, deletion_candidates: int = 6,
) -> LabelledOptions:
    """Build label-free options, then judge their proposed contacts against labels."""
    options = build_rally_options(video.context, max_edits, deletion_candidates)
    targets = option_targets(video, options)
    return LabelledOptions(options.sequences, options.matrix, targets, options.span_rows, video)
