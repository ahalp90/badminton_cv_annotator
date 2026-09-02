"""Keep the data dictionary in docs/dataset_v1_schema.md tied to the frozen module."""

from __future__ import annotations

from pathlib import Path

from dataset_builder.schema_v1 import frozen_column_names
from scripts.dataset_v1_dictionary import render_dictionary

DOC_PATH = Path(__file__).resolve().parents[1] / 'docs' / 'dataset_v1_schema.md'
START_MARKER = '<!-- dictionary:start -->'
END_MARKER = '<!-- dictionary:end -->'


def _doc_text() -> str:
    return DOC_PATH.read_text(encoding='utf-8')


def test_embedded_dictionary_matches_the_frozen_schema() -> None:
    """The pasted block must equal the generated dictionary."""
    _, start, remainder = _doc_text().partition(START_MARKER)
    embedded, end, _ = remainder.partition(END_MARKER)
    assert start and end, f'{DOC_PATH.name} is missing the dictionary markers'
    assert embedded.strip() == render_dictionary().strip()


def test_every_frozen_column_name_appears_in_the_doc() -> None:
    """No frozen column may be undocumented."""
    text = _doc_text()
    missing = sorted(
        f'{table}.{column}'
        for table, columns in frozen_column_names().items()
        for column in columns
        if f'`{column}`' not in text
    )
    assert not missing
