"""Training targets for the production rally proposals.

This module holds the experiment class that pairs each option with its target
label. Runtime proposals use detector evidence; target labels live only in this
training view.
"""

from dataclasses import dataclass

import numpy as np

from annotator.sequence.contacts import ContactSequence
from annotator.sequence.rally_options import (
    FEATURE_NAMES,
    RallyContext,
    RallyOptions,
    build_rally_options,
    feature_names,
    sequence_key,
)
from annotator.training.rally_targets import finish_in_initial_context, option_targets
from annotator.training.sequences import SequenceTrainingVideo

__all__ = ['FEATURE_NAMES', 'LabelledOptions', 'expand_pool', 'feature_names', 'finish_in_initial_context', 'sequence_key']


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
        return RallyOptions(self.sequences, self.matrix, self.span_rows, rally_context(self.video))


def rally_context(video: SequenceTrainingVideo) -> RallyContext:
    return RallyContext(video.fps, video.initial, video.events, video.repairs, video.frame_count)


def expand_pool(video: SequenceTrainingVideo, max_edits: int = 3, deletion_candidates: int = 6) -> LabelledOptions:
    """Build label-free options, then judge them with initial neighbouring spans."""
    options = build_rally_options(rally_context(video), max_edits, deletion_candidates)
    targets = option_targets(video, options)
    return LabelledOptions(options.sequences, options.matrix, targets, options.span_rows, video)
