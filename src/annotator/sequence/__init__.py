"""Refine rally contact sequences with the fitted whole-sequence choosers.

``refine_contact_sequences`` runs the full selected chain for one video from
in-memory arrays. The modules also expose the option and feature builders so
later training can rebuild the same inputs.
"""

from annotator.sequence.contacts import CandidateScores, ContactEvent, ContactSequence
from annotator.sequence.refine import (
    OptionPool,
    RefinedSequences,
    SequenceModels,
    StageChoices,
    refine_contact_sequences,
)

__all__ = [
    "CandidateScores",
    "ContactEvent",
    "ContactSequence",
    "OptionPool",
    "RefinedSequences",
    "SequenceModels",
    "StageChoices",
    "refine_contact_sequences",
]
