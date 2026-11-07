"""Prepared evidence and features shared by rally-model experiments."""

from dataclasses import dataclass

from annotator.contacts.build_features import ContactFeatures
from annotator.inference.prepare_annotation_inputs import PreparedAnnotationInputs
from experiments.annotator.heuristic_only_tune.data import SearchVideo


@dataclass(frozen=True)
class PreparedVideo:
    """Fixed features and labels; fitting code must select the training split."""

    video: SearchVideo
    group: str
    evidence: PreparedAnnotationInputs
    features: ContactFeatures
