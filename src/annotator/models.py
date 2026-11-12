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

MODEL_SCHEMA = 'annotator-four-models/1'
MODEL_FILENAME = 'models.joblib'
METADATA_FILENAME = 'metadata.json'
DEFAULT_MODEL_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / 'runtime/deployed/annotator'
)


@dataclass(frozen=True)
class AnnotatorModels:
    """The four fitted classifiers and the settings needed to use them.

    ``contact_model`` scores individual frames; ``rally_selector`` chooses rally contact
    lists; ``followup_selector`` corrects split pieces; ``review_model`` estimates
    whether each finished rally is correct.
    Keep preprocessing and contact-selection settings with the trees so
    inference builds the same inputs as training.
    """

    contact_model: Any
    rally_selector: RallySelector
    followup_selector: RallySelector
    review_model: Any
    contact_settings: ContactModelConfig = field(default_factory=ContactModelConfig)
    preprocessing: BaseAnnotatorConfig = field(default_factory=BaseAnnotatorConfig)


def model_files(directory: Path) -> dict[str, Path]:
    """Return model-bundle paths used by the dataset builder's resume checks."""
    return {
        'annotator_models': directory / MODEL_FILENAME,
        'annotator_metadata': directory / METADATA_FILENAME,
    }


def fitted_trees(models: AnnotatorModels) -> tuple[Any, ...]:
    """Contact, first selector, follow-up selector and review classifiers."""
    return (models.contact_model, models.rally_selector.selector, models.followup_selector.selector, models.review_model)


def selector_contract(models: RallySelector) -> dict[str, Any]:
    """Proposal limits and feature names recorded with the saved rally selector."""
    return {
        'max_edits': models.max_edits,
        'deletion_candidates': models.deletion_candidates,
        'option_features': list(FEATURE_NAMES),
        'review_features': list(REVIEW_FEATURE_NAMES),
    }


def save_models(models: AnnotatorModels, directory: Path) -> None:
    """Save the fitted models, preprocessing, feature order and library versions."""
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, directory / MODEL_FILENAME)
    metadata = {
        'schema': MODEL_SCHEMA,
        'scikit_learn': sklearn.__version__,
        'contact_features': list(CONTACT_FEATURE_NAMES),
    }
    metadata['rally_selector'] = selector_contract(models.rally_selector)
    metadata['followup_selector'] = selector_contract(models.followup_selector)
    if any(type(tree).__module__.startswith('xgboost') for tree in fitted_trees(models)):
        metadata['xgboost'] = version('xgboost')
    (directory / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + '\n')


def load_models(directory: Path, *, device: str = 'cuda') -> AnnotatorModels:
    """Load and validate the bundle, selecting the runtime device for its XGBoost trees.

    XGBoost falls back to CPU when no GPU is visible. Pass device='cpu' to keep
    annotation on CPU even when other pipeline stages use a GPU.
    """
    from xgboost import XGBModel

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
    for name, selector in (('rally_selector', models.rally_selector), ('followup_selector', models.followup_selector)):
        if not isinstance(selector, RallySelector):
            raise TypeError(f'Model bundle requires a fitted {name}')
        if metadata[name] != selector_contract(selector):
            raise ValueError(f'{name} features or proposal settings differ; refit the bundle')
    if models.review_model is None:
        raise ValueError('Model bundle requires a fitted review model')
    for tree in fitted_trees(models):
        if isinstance(tree, XGBModel):
            tree.set_params(device=device)
    return models
