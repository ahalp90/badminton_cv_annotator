"""Save and load the trees and preprocessing settings used by annotation."""

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import joblib
import sklearn

from annotator.config import BaseAnnotatorConfig
from annotator.contacts.model import CONTACT_FEATURE_NAMES, ContactModelConfig
from annotator.sequence.refine import SequenceModels

MODEL_SCHEMA = 'annotator-models/1'
MODEL_FILENAME = 'models.joblib'
METADATA_FILENAME = 'metadata.json'


class SideGeometry(StrEnum):
    """Which net position separates the far and near player's feet.

    VIDEO uses one net position for the whole video; SCENE uses the position
    measured in each scene. The fitted bundle preserves the selected mode.
    """

    VIDEO = 'video'
    SCENE = 'scene'


@dataclass(frozen=True)
class AnnotatorModels:
    """The complete fitted annotator, including the rules it was trained with."""

    contact: Any
    sequences: SequenceModels
    confidence: Any
    contact_settings: ContactModelConfig = field(default_factory=ContactModelConfig)
    preprocessing: BaseAnnotatorConfig = field(default_factory=BaseAnnotatorConfig)
    side_geometry: SideGeometry = SideGeometry.VIDEO


def model_files(directory: Path) -> dict[str, Path]:
    """Return the files that annotation resume checks must fingerprint."""
    return {
        'annotator_models': directory / MODEL_FILENAME,
        'annotator_metadata': directory / METADATA_FILENAME,
    }


def save_models(models: AnnotatorModels, directory: Path) -> None:
    """Save all fitted stages together with their runtime and feature order."""
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, directory / MODEL_FILENAME)
    metadata = {
        'schema': MODEL_SCHEMA,
        'scikit_learn': sklearn.__version__,
        'contact_features': list(CONTACT_FEATURE_NAMES),
        'side_geometry': models.side_geometry.value,
    }
    (directory / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + '\n')


def load_models(directory: Path) -> AnnotatorModels:
    """Load a complete bundle fitted with the installed scikit-learn version.

    Historical experiment models belong to their comparison environment. Fit
    and save a fresh bundle with the current runtime before running annotation.
    """
    missing = [path.name for path in model_files(directory).values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f'Annotator model bundle is incomplete at {directory}: missing {", ".join(missing)}. '
            'Supply the model directory produced by the annotator retune.'
        )
    metadata = json.loads((directory / METADATA_FILENAME).read_text())
    if metadata.get('schema') != MODEL_SCHEMA:
        raise ValueError('Unsupported annotator model bundle; fit a fresh bundle with the current training command.')
    trained_version = metadata['scikit_learn']
    if trained_version != sklearn.__version__:
        raise ValueError(
            f'Annotator models were fitted with scikit-learn {trained_version}; '
            f'this runtime has {sklearn.__version__}. Refit the bundle in the current environment.'
        )
    if tuple(metadata['contact_features']) != CONTACT_FEATURE_NAMES:
        raise ValueError('Annotator model feature order differs from the current feature builder; refit the bundle.')
    models = joblib.load(directory / MODEL_FILENAME)
    if not isinstance(models, AnnotatorModels):
        raise TypeError('Model bundle does not contain a complete AnnotatorModels value')
    if models.side_geometry.value != metadata['side_geometry']:
        raise ValueError('Model bundle and metadata disagree about the net position used for player sides')
    return models
