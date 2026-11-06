"""Save and load the trees and preprocessing settings used by annotation."""

import json
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import sklearn

from annotator.config import BaseAnnotatorConfig
from annotator.contacts.score_candidates import (
    CONTACT_FEATURE_NAMES,
    ContactModelConfig,
)
from annotator.rallies.build_contact_proposals import FEATURE_NAMES
from annotator.rallies.refine_rallies import REVIEW_FEATURE_NAMES, RallySelector

MODEL_SCHEMA = 'annotator-three-models/2'
MODEL_FILENAME = 'models.joblib'
METADATA_FILENAME = 'metadata.json'
DEFAULT_MODEL_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / 'experiments/annotator/tree_simplification/results/three_models/models/raw_baseline-seed-0'
)


@dataclass(frozen=True)
class AnnotatorModels:
    """The three fitted stages with their contact and preprocessing settings."""

    contact: Any
    sequences: RallySelector
    confidence: Any
    contact_settings: ContactModelConfig = field(default_factory=ContactModelConfig)
    preprocessing: BaseAnnotatorConfig = field(default_factory=BaseAnnotatorConfig)


def model_files(directory: Path) -> dict[str, Path]:
    """Return the files that annotation resume checks must fingerprint."""
    return {
        'annotator_models': directory / MODEL_FILENAME,
        'annotator_metadata': directory / METADATA_FILENAME,
    }


def fitted_trees(models: AnnotatorModels) -> tuple[Any, ...]:
    return (models.contact, models.sequences.selector, models.confidence)


def selector_contract(models: RallySelector) -> dict[str, Any]:
    return {
        'max_edits': models.max_edits,
        'deletion_candidates': models.deletion_candidates,
        'option_features': list(FEATURE_NAMES),
        'review_features': list(REVIEW_FEATURE_NAMES),
    }


def save_models(models: AnnotatorModels, directory: Path) -> None:
    """Save all fitted stages together with their runtime and feature order."""
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, directory / MODEL_FILENAME)
    metadata = {
        'schema': MODEL_SCHEMA,
        'scikit_learn': sklearn.__version__,
        'contact_features': list(CONTACT_FEATURE_NAMES),
    }
    metadata['rally_selector'] = selector_contract(models.sequences)
    if any(type(tree).__module__.startswith('xgboost') for tree in fitted_trees(models)):
        metadata['xgboost'] = version('xgboost')
    (directory / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + '\n')


def load_models(directory: Path) -> AnnotatorModels:
    """Load the fitted models and check their feature and annotation settings."""
    missing = [path.name for path in model_files(directory).values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f'Annotator model bundle is incomplete at {directory}: missing {", ".join(missing)}. '
            'Supply the model directory produced by the annotator retune.'
        )
    metadata = json.loads((directory / METADATA_FILENAME).read_text())
    if metadata.get('schema') != MODEL_SCHEMA:
        raise ValueError('Unsupported annotator model bundle; fit a fresh bundle with the current training command.')
    if tuple(metadata['contact_features']) != CONTACT_FEATURE_NAMES:
        raise ValueError('Annotator model feature order differs from the current feature builder; refit the bundle.')
    models = joblib.load(directory / MODEL_FILENAME)
    if not isinstance(models, AnnotatorModels):
        raise TypeError('Model bundle does not contain a complete AnnotatorModels value')
    if not isinstance(models.sequences, RallySelector):
        raise TypeError('Rally metadata requires a fitted rally selector')
    if metadata['rally_selector'] != selector_contract(models.sequences):
        raise ValueError('Selector features or proposal settings differ; refit the bundle')
    if models.confidence is None:
        raise ValueError('Model bundle requires a fitted review model')
    return models
